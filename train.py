import os
import itertools
import time

import torch
import torch.nn.functional as F
#from compel import Compel, ReturnedEmbeddingsType

from utils.config import parse_args
from utils.get_xlDataLoader import xlDataloader
from utils.init_module import init_trainer, init_model, init_adapter
from utils.adapter import DDCA
import copy
    

def main():
    args = parse_args()

    accelerator = init_trainer(args)
    noise_scheduler, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae, unet, image_encoder, image_proj_model =  \
        init_model(args.pretrained_model_name_or_path, args.image_encoder_path, args.num_tokens)

    image_proj_model = copy.deepcopy(image_proj_model)
    
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)

    unet.set_attn_processor(init_adapter(unet, num_tokens=args.num_tokens))

    weight_dtype = torch.float32

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    image_proj_model.to(accelerator.device, dtype=weight_dtype)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_adapter = DDCA(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)

    params_to_opt = itertools.chain(
            ip_adapter.image_proj_model.parameters(),  
            ip_adapter.adapter_modules.parameters())

    optimizer = torch.optim.AdamW(
            params_to_opt, 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
            )

    train_dataloader = xlDataloader(
            batch_size=args.train_batch_size,
            resolution=args.resolution,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2)
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            #load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    #image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                    clip_images = batch["clip_images"].to(accelerator.device, dtype=weight_dtype)
                    image_embeds = image_encoder(clip_images, output_hidden_states=True).hidden_states[-2]

                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat

                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)

                image_embeds = torch.stack(image_embeds_)


                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    print(f"Epoch {epoch}, global_step {global_step}, time: {time.perf_counter() - begin}s, step_loss: {avg_loss}")
                    accelerator.log({"train_loss": avg_loss}, step=global_step)
                    global_step += 1
            
            
            if accelerator.is_main_process \
                    and accelerator.sync_gradients \
                    and global_step % args.eval_step == 0:

                save_path = os.path.join(args.output_dir, f"{args.run_name}/checkpoint-{global_step}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                save_model = {
                        "image_proj": accelerator.unwrap_model(ip_adapter).image_proj_model.state_dict(),
                        "ip_adapter": accelerator.unwrap_model(ip_adapter).adapter_modules.state_dict()
                    }
                torch.save(save_model, f"{save_path}/model.pth")
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
