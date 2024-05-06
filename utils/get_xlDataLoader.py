import torch
import glob
import os
from torchvision import transforms
from transformers import CLIPImageProcessor 
from transformers.models.conditional_detr.modeling_conditional_detr import ConditionalDetrFrozenBatchNorm2d
import numpy as np
import random
from PIL import Image
import pandas as pd

import logging  # 引入logging模块
import os.path
import time
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

logger = logging.getLogger()


class xlDataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        data = pd.read_csv("")
        self.image_paths = data["image_path"].to_list()
        self.prompts = data["prompt"].to_list()


        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()
    def filter_images(self, path):
        try:
            with Image.open(path) as img:
                #width, height = img.size
                aspect_ratio = max(img.size)/min(img.size)
                return aspect_ratio <= 2
        except Exception as e:
            print(f"Error processing image at {path}: {e}")
            return False

    def padding(self, image):
        w, h = image.size
        max_edge = max(w, h)
        image = np.array(image)
        ph, pw = max_edge - h, max_edge - w
        new_image = np.ones([max_edge, max_edge, 3], dtype=np.uint8) * 0
        new_image[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w,:] = image

        image = Image.fromarray(new_image)
        return image
        
    def __getitem__(self, idx):
        #image_path = self.data["image_path"][idx]
        image_path = self.image_paths[idx]
        text = self.prompts[idx]
        
        image = Image.open(image_path).convert("RGB")
        drop_image_embed = 0

        bboxes, _ = app.det_model.detect(np.array(image)[:,:,::-1], max_num=0, metric='default')
        box = list(map(int, bboxes[0][:-1]))
        face = image.crop(box)
        face = self.padding(face)
        angle = random.randint(-45, 45)
        face = face.rotate(angle)

        # original size
        original_width, original_height = image.size
        original_size = torch.tensor([original_height, original_width])
        

        clip_image = self.clip_image_processor(images=face, return_tensors="pt").pixel_values

        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "original_size": original_size,
            "crop_coords_top_left": torch.tensor([0,0]),
            "target_size": original_size
        }
        
    
    def __len__(self):
        return len(self.image_paths)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)

    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }
def xlDataloader(
        batch_size, 
        resolution,
        tokenizer, 
        tokenizer_2):

    train_dataset = xlDataset(tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=resolution)
    print("len train_dataset: ", len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        #num_workers=args.dataloader_num_workers,
    )
    return train_dataloader
