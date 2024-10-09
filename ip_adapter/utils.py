import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

attn_maps = []
#attn_maps = {}
def hook_fn(name):
    #print(name)
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps.append(module.processor.attn_map)
            #attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None
    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]
    return attn_map.cpu()

def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
    global attn_maps
    idx = 0 if instance_or_negative else 1

    net_attn_maps = []
    for i in range(len(attn_maps)):
        with torch.no_grad():
            #print(i, j)
            attn_map = attn_maps[i].detach()
            attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
            attn_map = upscale(attn_map, image_size) 
            net_attn_maps.append(attn_map) 

    attn_maps = []
    net_attn_maps = torch.stack(net_attn_maps, dim=0)
    net_attn_maps = torch.mean(net_attn_maps, dim=0)
    return net_attn_maps

def attnmaps2images(net_attn_maps, image=None):

    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        image = normalized_attn_map

        image = Image.fromarray(image.astype(np.uint8))

        images.append(image)

    return images
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
