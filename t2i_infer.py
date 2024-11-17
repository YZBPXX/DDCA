import pandas as pd
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
#from controlnet_aux.canny import CannyDetector
import torch

#prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"

df = pd.read_csv("/hy-tmp/datasets/data.csv")
negative_prompt = 'low quality, bad quality, sketches'
#prompts = df.
#images = []

# load adapter
adapter = T2IAdapter.from_pretrained("/hy-tmp/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")

# load euler_a scheduler
model_id = '/hy-tmp/stable-diffusion-xl-base-1.0'
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(model_id, adapter=adapter, torch_dtype=torch.float16, variant="fp16", ).to("cuda")
#pipe.enable_xformers_memory_efficient_attention()

negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

for index, row in df.iterrows():
    prompt = row["prompt"]
    image_path = "/hy-tmp/datasets/" + row["image_path"].replace("images", "process_images")

    name = image_path.split('/')[-1]

    image = load_image(image_path)

    gen_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=30,
        guidance_scale=7.5, 
        adapter_conditioning_scale=0.8, 
        adapter_conditioning_factor=1
    ).images[0]
    gen_images.save(f"/hy-tmp/datasets/t2i_infer_images/{name}")
    #gen_images.save(f"test.jpg")

