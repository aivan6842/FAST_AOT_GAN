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

#### Dataloader def ####
class InpaintingData(Dataset):
    def __init__(self, root_dir: str, masks_dir: str = "data/masks"):
        super(Dataset, self).__init__()
        # images 
        self.images = os.listdir(f"{root_dir}")[:1000]
        self.root_dir = root_dir
        self.masks_dir = masks_dir
        self.masks = os.listdir(masks_dir)
        random.seed(10)

        # augmentation
        self.img_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(512),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),
            ]
        )
        self.mask_trans = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # load image
        image_path = os.path.join(f"{self.root_dir}", self.images[index])
        image = Image.open(image_path).convert("RGB")

        # get mask
        random_idx = random.randint(0, len(self.masks)-1)
        mask_path = os.path.join(f"{self.masks_dir}", self.masks[random_idx])
        mask = Image.open(mask_path).convert("L")

        # augment
        image = self.img_trans(image) * 2.0 - 1.0
        mask = F.to_tensor(self.mask_trans(mask))

        return image, mask, image_path


def train(run_name, 
          student_generator,
          teacher_generator,
          discriminator,
          L1_loss_weight=0.1,
          style_loss_weight=250,
          perceptual_loss_weight=0.1,
          adversarial_loss_weight=0.01,
          distillation_loss_weight=0.1,
          focused_loss_weight=0.2,
          num_epochs = 5,
          gen_lr = 1e-4,
          disc_lr = 1e-4,
          a=0.5,
          b=0.999,
          save_every=3,
          save_dir="models/",
          log_dir="./runs"):
    writer = SummaryWriter(f"{log_dir}/{run_name}")
    iteration = 0

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Create losses
    L1_loss = L1()
    style_loss = Style()
    percetual_loss = Perceptual()
    adversarial_loss = smgan()
    focused_loss = torch.nn.MSELoss()

    # get optimizers
    optimG = torch.optim.AdamW(student_generator.parameters(), lr=gen_lr, betas=(a, b))
    optimD = torch.optim.AdamW(discriminator.parameters(), lr=disc_lr, betas=(a, b))

    print("Beginning Training")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        for i, data in enumerate(tqdm(test_loader)):
            # get batch of data
            images, masks, _ = data
            images, masks = images.to(device), masks.to(device)
            masked_images = (images * (1 - masks).float()) + masks

            predicted_images, student_mids = student_generator(masked_images, masks)
            inpainted_images = (1 - masks) * images + masks * predicted_images

            # losses
            l1_loss_val = L1_loss(predicted_images, images)
            focused_loss_val = focused_loss(predicted_images * masks, images * masks)
            style_loss_val = style_loss(predicted_images, images)
            percetual_loss_val = percetual_loss(predicted_images, images)
            # distillation_loss_val = distillation_loss(student_mids, teacher_mids[1::2])
            adversarial_disc_loss, adversarial_gen_loss = adversarial_loss(discriminator, inpainted_images, images, masks)

            total_loss = (L1_loss_weight * l1_loss_val) + \
                         (style_loss_weight * style_loss_val) + \
                         (perceptual_loss_weight * percetual_loss_val) + \
                         (focused_loss_weight * focused_loss_val) + \
                         (adversarial_loss_weight * adversarial_gen_loss)
        
            optimG.zero_grad()
            optimD.zero_grad()
            total_loss.backward()
            adversarial_disc_loss.backward()
            optimG.step()
            optimD.step()

            writer.add_scalar("Loss/train/generator", adversarial_gen_loss, iteration)
            writer.add_scalar("Loss/train/L1_loss", l1_loss_val, iteration)
            writer.add_scalar("Loss/train/style_loss", style_loss_val, iteration)
            writer.add_scalar("Loss/train/perceptual_loss", percetual_loss_val, iteration)
            writer.add_scalar("Loss/train/focused_loss", focused_loss_val, iteration)
            writer.add_scalar("Loss/train/discriminator", adversarial_disc_loss, iteration)
            writer.add_scalar("Loss/train/total", total_loss, iteration)

            iteration += 1 
    
    return student_generator


#### Get data for calibration ######
test_data_path = "data/x-medium/train"
test = InpaintingData(test_data_path)
test_loader = DataLoader(test, batch_size=1, shuffle=True)

# create models
disc_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/discriminator_up_to_60_percent_mask_final.pt"
disc = Discriminator()
disc.load_state_dict(torch.load(disc_model_path, map_location=device, weights_only=True))


##### load original model #####
student_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/student_generator_up_to_60_percent_mask_final.pt"
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})
student_model = QuantInpaintGenerator(half_size_args)
student_model.load_state_dict(torch.load(student_model_path, map_location=device, weights_only=True))

print(student_model)

student_model.to(device)
disc.to(device)
trained_model = train(run_name="qat_brev_3",
      num_epochs=1,
      student_generator=student_model,
      teacher_generator=None,
      discriminator=disc,
      save_every=2,
      distillation_loss_weight=0.0001,
      gen_lr=1e-6,
      disc_lr=1e-6)

torch.save(trained_model.state_dict(), "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/brev.pt")
