from PIL import Image
import torch
import pandas as pd
from tqdm import trange
#from features 
#from dinov2.dinov2.eval.utils import ModelWithIntermediateLayers
#import sys
#from inference import AS
from utils.face_detector import animeFaceDetect
from utils.clip_model import clipImageScore, clipTextImageScore

import numpy as np

import math
import glob

if __name__ == "__main__":
    device = "cuda"
    face_eval_model = clipImageScore(device)
    text_image_eval_model = clipTextImageScore(device) 
    image1_paths = []
    texts = []
    image2_paths = []

    image_score = face_eval_model(image1_paths, image2_paths)
    text_score = text_image_eval_model(texts, image2_paths)
    print(image_score, text_score)
#print("\n".join(image1_paths))
#print("\n".join(image2_paths))

