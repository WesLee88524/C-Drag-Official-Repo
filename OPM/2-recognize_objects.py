import os
import cv2
import json
import numpy as np

from PIL import Image, ImageDraw
from gpt4v import get_image_description

image_folder = 'data/VOI'

seqs = os.listdir(image_folder)
seqs.sort()
print(seqs)

trajectory_path = 'data/VOI-GT'

img_with_initial_point = 'data/VOI-initial-point'

# time synchronization and time statistics
import time
start = time.time()

for seq in seqs:
    print(f"Processing {seq}, {seqs.index(seq)+1}/{len(seqs)}")
    title = "Analyze the image to identify objects. First, identify the object segmented by a prominent color at the part level, then determine other objects it may contact. List only the categories of these objects as single words, separated by commas, with no additional text, symbols, or line breaks."
    seg_path = os.path.join(img_with_initial_point, seq + '_segmented_sam_b.jpg')
    description = get_image_description(seg_path, title)
    print(description)
    
    words = [word.strip() for word in description[0].replace('\n', '').replace(',', '').split()]

    #  Change the list of words to JSON format
    description_json = json.dumps({"classes": words})

    # print(description_json)
    save_path = os.path.join(image_folder,seq, 'objects_classes.json')
    with open(save_path, 'w') as f:
        f.write(description_json)


end = time.time()
print(f"Time: {end-start}")
print(f"Average time: {(end-start)/len(seqs)}")
