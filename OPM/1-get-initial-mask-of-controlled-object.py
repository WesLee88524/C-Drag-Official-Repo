import os
import cv2
import json
import numpy as np

from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry

model_type = "vit_b"
if model_type == "vit_h":
    sam_checkpoint = "YOUR_PATH/sam_vit_h_4b8939.pth"
    # https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth
elif model_type == "vit_b":
    sam_checkpoint = "YOUR_PATH/sam_vit_b_01ec64.pth"
    # https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth
else:
    raise ValueError(f"Invalid model type: {model_type}")


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)


image_folder = 'data/VOI'
seqs = os.listdir(image_folder)
seqs.sort()

trajectory_path = 'data/VOI-GT'

img_with_initial_point = 'data/VOI-initial-point'
if not os.path.exists(img_with_initial_point):
    os.makedirs(img_with_initial_point)

# 统计时间
import time
import torch
torch.cuda.synchronize()
start = time.time()
for seq in seqs:
    print(f"Processing {seq}, {seqs.index(seq)+1}/{len(seqs)}")
    seq_path = os.path.join(image_folder, seq)
    images = os.listdir(seq_path)
    images.sort()
    image_path = os.path.join(seq_path, images[0])
    image = Image.open(image_path)

    file_path = os.path.join(trajectory_path, seq + '.json')
    data = json.load(open(file_path))
    init_point = data[0][0][0]
    print(init_point)

    # Step-1:use initial point to segment the object
    
    predictor.set_image(cv2.imread(image_path))
    masks, _, _ = predictor.predict(point_coords=np.array([init_point]), point_labels=np.array([1]), multimask_output=False)

    mask = masks[0]
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))

    # Create a completely transparent mask
    red_mask = Image.new('RGBA', image.size, (0, 0, 0, 0))  

    # Add a red semi-transparent mask only in the segmentation area
    mask_image = mask_image.convert("L")
    red_mask.paste((255, 0, 0, 100), (0, 0), mask_image)  

    # Transfer to RGBA mode and merge with mask
    combined_image = Image.alpha_composite(image.convert("RGBA"), red_mask)

    # Transfer to RGB mode and save
    combined_image_rgb = combined_image.convert('RGB')
    save_path = os.path.join(img_with_initial_point, seq + '_segmented_sam_b.jpg')
    combined_image_rgb.save(save_path)


torch.cuda.synchronize()
end = time.time()
print(f"Time: {end-start}")
print(f"Average time: {(end-start)/len(seqs)}")
