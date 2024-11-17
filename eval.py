from PIL import Image
import torch
import pandas as pd
from tqdm import trange
#from features 
#from dinov2.dinov2.eval.utils import ModelWithIntermediateLayers
#import sys
#from inference import AS
#from utils.face_detector import animeFaceDetect
from utils.clip_model import clipImageScore, clipTextImageScore

import numpy as np

import math
import glob
import pandas as pd

if __name__ == "__main__":
	df = pd.read_csv("/hy-tmp/datasets/data.csv").head(100)
	prompts = []
	origin_image_paths = []
	controlnet_image_paths = []
	t2i_image_paths = []
	ours_image_paths = []

	for index, row in df.iterrows():
		prompt = row["prompt"]
		origin_image_path = "/hy-tmp/datasets/" + row["image_path"]
		ours_image_path = "/hy-tmp/datasets/" + row["image_path"].replace("images", "ours_images")
		controlnet_image_path = "/hy-tmp/datasets/" + row["image_path"].replace("images", "controlnet_infer_images")
		t2i_image_path = "/hy-tmp/datasets/" + row["image_path"].replace("images", "t2i_infer_images")

		prompts.append(prompt)
		origin_image_paths.append(origin_image_path)
		controlnet_image_paths.append(controlnet_image_path)
		t2i_image_paths.append(t2i_image_path)
		ours_image_paths.append(ours_image_path)
		print(ours_image_path)
		print(t2i_image_path)
		print(controlnet_image_path)

	device = "cuda"
	face_eval_model = clipImageScore(device, model_path="/hy-tmp/models/clip-vit-base-patch32")
	text_image_eval_model = clipTextImageScore(device,  model_path="/hy-tmp/models/clip-vit-base-patch32") 
	#image1_paths = []
	#texts = []
	#image2_paths = []

	#image_score1 = face_eval_model(origin_image_paths, ours_image_paths)
	#image_score2 = face_eval_model(origin_image_paths, controlnet_image_paths)
	#image_score3 = face_eval_model(origin_image_paths, t2i_image_paths)
	#print(image_score1, image_score2, image_score3)
	text_score1 = text_image_eval_model(prompts, ours_image_paths)
	text_score2 = text_image_eval_model(prompts, controlnet_image_paths)
	text_score3 = text_image_eval_model(prompts, t2i_image_paths)
	print(text_score1, text_score2, text_score3)
	#print(image_score, text_score)
#print("\n".join(image1_paths))
#print("\n".join(image2_paths))

