from accelerate import Accelerator
import os
import logging
import time
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import torch

from ip_adapter.ip_adapter import Resampler

from ip_adapter.attention_processor import DDCAAttnProcessor 


def init_trainer(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

    log_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.getcwd() + '/Logs/' + log_time + '.log'

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.WARNING)  # 输出到file的log等级的开关
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 3. 创建一个handler用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if args.run_name == "test":
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            #log_with=args.report_to,
            #project_config=accelerator_project_config,
        )
    else :
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=4,
            log_with=args.report_to,
            #project_config=accelerator_project_config,
        )
        accelerator.init_trackers(
            project_name="test",
            init_kwargs={"wandb": {"name": args.run_name}}
        )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    return accelerator

def init_model(pretrained_model_name_or_path, image_encoder_path, num_tokens, dtype=torch.float32):
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    #ip-adapter
    #num_tokens = 4
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )

    return noise_scheduler,tokenizer,tokenizer_2, text_encoder, text_encoder_2,vae,unet, image_encoder, image_proj_model

# init adapter modules
def init_adapter(unet, num_tokens):
    attn_procs = {}
    unet_sd = unet.state_dict()
    #print("init_module: ", unet_sd)
    for name in unet.attn_processors.keys():
        #print(name)

        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None or "1.attentions" in name:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            #print("!!!: ", unet_sd[layer_name + ".to_k_ip.weight"].shape)
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = DDCAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights, strict=False)

    return attn_procs

