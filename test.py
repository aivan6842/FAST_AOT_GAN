from AOT_GAN.src.model.aotgan import InpaintGenerator, Discriminator
from AOT_GAN.src.loss.loss import L1, Style, Perceptual, smgan
import torch
from collections import namedtuple
from attrdict import AttrDict
import numpy as np
import cv2
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from AOT_GAN.src.model.common import BaseNetwork
from AOT_GAN.src.model.aotgan import spectral_norm
from AOT_GAN.src.metric.metric import mae, psnr, ssim, fid

device = torch.device("cpu")
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})
teacher_model_path = "AOT_GAN/experiments/places2/G0000000.pt"

train_data_path = "data/x-medium/train"
val_data_path = "data/x-medium/val"
test_data_path = "data/x-medium/test"

student_final_model = "models/student_generator_test_final.pt"


student_generator = InpaintGenerator(half_size_args).to(device)
student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

image_paths = sorted(os.listdir(f"{test_data_path}"))[:10]
masks = sorted(os.listdir(f"data/masks"))[:10]

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

for image_path, mask_path in zip(image_paths, masks):
    image = ToTensor()(Image.open(f"{test_data_path}/{image_path}").convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(f"data/masks/{mask_path}").convert("L"))
    mask = mask.unsqueeze(0)
    image, mask = image.to(device), mask.to(device)
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img, _ = student_generator(image_masked, mask)

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(image_path).split(".")[0]
    postprocess(image_masked[0]).save(f"tests/{image_name}_masked.png")
    postprocess(pred_img[0]).save(f"tests/{image_name}_pred.png")
    postprocess(comp_imgs[0]).save(f"tests/{image_name}_comp.png")
        #res["ssim"] += ssim(images, inpainted_images)
        #res["fid"] += fid(images, inpainted_images, "/home/alex/.cache/torch/hub/checkpoints/inception_v3_google-1a9a5a14.pth")
