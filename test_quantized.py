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
import random
from tqdm import tqdm

device = torch.device("cpu")
half_size_args = AttrDict({"block_num": 8, "rates": [1, 2, 4, 8]})

pct = "3"
test_data_path = "data/x-medium/test"
masks_data_path = f"data/masks_{pct}"
student_final_model = "AOT_GAN/experiments/places2/G0000000.pt"
# student_final_model = "models_first/student_generator_test_final.pt"

student_quantized = "/w/nobackup/385/scratch-space/expires-2024-Dec-05/aivan6842/models/teacher_quantized.pt"


student_model = InpaintGenerator(half_size_args).to(device)
# student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
backend = "qnnpack"
student_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized_student = torch.ao.quantization.prepare(student_model, inplace=False)
model_static_quantized_student = torch.ao.quantization.convert(model_static_quantized_student, inplace=False)
model_static_quantized_student.load_state_dict(torch.load(student_quantized, map_location=device, weights_only=True))


image_paths = sorted(os.listdir(f"{test_data_path}"))
masks = sorted(os.listdir(masks_data_path))

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

for image_path, mask_path in tqdm(zip(image_paths, masks), total=len(image_paths)):
    image = ToTensor()(Image.open(f"{test_data_path}/{image_path}").convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(f"{masks_data_path}/{mask_path}").convert("L"))
    mask = mask.unsqueeze(0)
    image, mask = image.to(device), mask.to(device)
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img, _ = model_static_quantized_student(image_masked, mask)

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(image_path).split(".")[0]
    postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    postprocess(comp_imgs[0]).save(f"tests/{pct}_base/{image_name}_comp.png")
    break