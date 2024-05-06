import torch

class DDCA(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)

        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")

        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
        #self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
