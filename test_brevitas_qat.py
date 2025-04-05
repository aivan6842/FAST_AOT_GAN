import torch
from brevitas.nn import QuantLinear, QuantConv2d
from torch import nn

from AOT_GAN.src.model.aotgan import InpaintGenerator
from attrdict import AttrDict
import torch.nn as nn
from torch.nn.utils import spectral_norm
import os

from AOT_GAN.src.model.common import BaseNetwork

from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import copy

from AOT_GAN.src.model.aotgan import InpaintGenerator
import torch
from attrdict import AttrDict
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
# import torchvision.transforms.functional as F
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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


class QuantInpaintGenerator(BaseNetwork):
    def __init__(self, args):  # 1046
        super(QuantInpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            QuantConv2d(4, 64, 7),
            nn.ReLU(True),
            QuantConv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            QuantConv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), QuantConv2d(64, 3, 3, stride=1, padding=1), nn.Tanh()
        )

        self.init_weights()

        self.activations = []
        for i in range(args.block_num):
            self.middle.register_forward_hook(self.get_activation(f"middle.{i}"))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations.append(output.detach())
        return hook

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x_mid = self.middle(x)
        x = self.decoder(x_mid)
        # x = torch.tanh(x)
        acts = torch.stack(self.activations) if self.activations else torch.tensor([])
        self.activations = []
        return x, acts


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = QuantConv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), QuantConv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), QuantConv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), QuantConv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


device = torch.device("cuda")
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})

pct = "1"
# save_dir = f"/w/nobackup/385/scratch-space/expires-2024-Dec-23/aivan6842/test/ours/ood/{pct}"
save_dir = "tests/paper1"
test_data_path = "data/x-medium/test"
# test_data_path = "/scratch/expires-2024-Dec-23/aivan6842/data/ood3/ood"
test_data_path = "tests/paper"
# masks_data_path = f"data/masks_{pct}"
masks_data_path = "tests/paper"
student_final_model = "AOT_GAN/experiments/places2/G0000000.pt"
student_final_model = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/brev.pt"

##### load model #######
student_generator = QuantInpaintGenerator(half_size_args).to(device)
student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

########################

pct = 6
test_data_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/x-medium/test"
masks_data_path = f"/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/masks_{pct}"
image_paths = os.listdir(test_data_path)
masks = os.listdir(masks_data_path)

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
        pred_img, _ = student_generator(image_masked, mask)

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(image_path).split(".")[0]
    # postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    # postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    postprocess(comp_imgs[0]).save(f"{save_dir}/{image_name}_quant.png")
