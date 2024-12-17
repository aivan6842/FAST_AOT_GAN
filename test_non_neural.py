from AOT_GAN.src.model.aotgan import InpaintGenerator
import torch
from attrdict import AttrDict
import numpy as np
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from PIL import Image
import cv2 as cv

pct = "1"
save_dir = f"/w/nobackup/385/scratch-space/expires-2024-Dec-23/aivan6842/test/telea/ood/{pct}"
save_dir = "tests/paper"
# test_data_path = "data/x-medium/test"
test_data_path = "/scratch/expires-2024-Dec-23/aivan6842/data/ood3/ood"
test_data_path = "tests/paper"
masks_data_path = f"data/masks_{pct}"
masks_data_path = "tests/paper"

image_paths = ["mountain-scenic-view-stockcake.jpg", "marsh_00002096.jpg"]
masks = ["04259.png", "02014.png"]

for image_path, mask_path in tqdm(zip(image_paths, masks), total=len(image_paths)):
    img = cv.imread(f"{test_data_path}/{image_path}")
    mask = cv.imread(f"{masks_data_path}/{mask_path}", cv.IMREAD_GRAYSCALE)
    
    res = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

    image_name = os.path.basename(image_path).split(".")[0]
    # postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    # postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    cv.imwrite(f"{save_dir}/{image_name}_comp_telea.png", res)
