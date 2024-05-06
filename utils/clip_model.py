import os
import torch
import pandas as pd
#from torch import nn.functional as F
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import numpy as np
import torch.nn

from transformers import AutoProcessor, CLIPProcessor, CLIPModel, CLIPVisionModel,CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

class clipTextImageScore():

    def __init__(self, device, model_path="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, texts, image_paths):
        score = 0
        for text, image_path in zip(texts, image_paths):
            image = Image.open(image_path)
            input_ids = self.processor(text, padding="max_length", truncation=True, return_tensors="pt").to(self.model.device)
            pixel_values = self.processor(images=image, return_tensors="pt").to(self.model.device)
            text_embeds = self.model.get_text_features(**input_ids)
            image_embeds = self.model.get_image_features(**pixel_values)
            for text_embed, image_embed in zip(text_embeds, image_embeds):
                output = F.cosine_similarity(text_embed[None,], image_embed[None,])
                score += output.item()

        return score/len(texts)


class clipImageScore():

    def __init__(self, device, model_path="openai/clip-vit-base-patch32"):
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, image1_paths, image2_paths):
        score = 0
        for image1_path, image2_path in zip(image1_paths, image2_paths):
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
            inputs1 = self.processor(images=image1, return_tensors="pt").to(self.model.device)
            image1_embeds = self.model(**inputs1).image_embeds
            inputs2 = self.processor(images=image2, return_tensors="pt").to(self.model.device)
            image2_embeds = self.model(**inputs2).image_embeds
            for image1_embed, image2_embed in zip(image1_embeds, image2_embeds):
                output = F.cosine_similarity(image1_embed[None], image2_embed[None])
                score += output.item()

        return score/len(image1_paths)



