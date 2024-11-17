from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
import pandas as pd

#prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"

df = pd.read_csv("/hy-tmp/datasets/data.csv")
negative_prompt = 'low quality, bad quality, sketches'

#prompts = df.
#images = []

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    "/hy-tmp/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
#vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "/hy-tmp/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    #vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

for index, row in df.iterrows():
    prompt = row["prompt"]
    image_path = "/hy-tmp/datasets/" + row["image_path"].replace("images", "process_images")

    name = image_path.split('/')[-1]

    image = load_image(image_path)
    #image = np.array(image)
    #image = cv2.Canny(image, 100, 200)
    #image = image[:, :, None]
    #image = np.concatenate([image, image, image], axis=2)
    #image = Image.fromarray(image)

    images = pipe(
        prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

    images[0].save(f"/hy-tmp/datasets/controlnet_infer_images/{name}")

