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
from torch.profiler import profile, record_function, ProfilerActivity
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping, QConfigMapping, QConfig, get_default_qconfig, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
from torch.ao.quantization.observer import HistogramObserver, MovingAverageMinMaxObserver, MinMaxObserver, PerChannelMinMaxObserver, FixedQParamsObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import default_fused_per_channel_wt_fake_quant, FusedMovingAvgObsFakeQuantize, FakeQuantize


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


#### Dataloader def ####
class InpaintingData(Dataset):
    def __init__(self, root_dir: str, masks_dir: str = "data/masks"):
        super(Dataset, self).__init__()
        # images 
        self.images = os.listdir(f"{root_dir}")[:100]
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


device = torch.device("cpu")
BATCH_SIZE = 1
test_data_path = "data/x-medium/train"
test = InpaintingData(test_data_path)
test_loader = DataLoader(test, batch_size=1, shuffle=True)


half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})
# student_final_model = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/brev.pt"
student_final_model = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/student_generator_up_to_60_percent_mask_final.pt"
# student_final_model = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/AOT_GAN/experiments/places2/G0000000.pt"
student_generator = InpaintGenerator(half_size_args).to(device)
student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

quant_model = QuantInpaintGenerator(half_size_args)
quant_model.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator = quant_model


#### load model #####
quantized_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/AOT_GAN/experiments/places2/generator_quantized_qat.pth"

student_generator = InpaintGenerator(half_size_args).to(device)
# student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

model_to_quantize = copy.deepcopy(student_generator)
example_inputs = (torch.rand(size=(1,3,512,512)).to(device), torch.rand(size=(1,3,512,512)).to(device))


qconfig_map = get_default_qat_qconfig_mapping()
# qconfig_map = QConfigMapping().set_global(qconfig)

# qconfig_map.set_module_name(name, QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=FixedQParamsObserver.with_args(scale=0.1, zero_point=0)))
prepared_model = prepare_qat_fx(model_to_quantize, qconfig_map, example_inputs)

loaded_quantized_model = convert_fx(prepared_model)
loaded_quantized_model.load_state_dict(torch.load(quantized_model_path, weights_only=True))


pct = 1
test_data_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/x-medium/test"
masks_data_path = f"/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/masks_{pct}"
image_paths = os.listdir(test_data_path)[:11]
masks = os.listdir(masks_data_path)


# warmup
for image_path, mask_path in tqdm(zip(image_paths[:10], masks), total=10):
    image = ToTensor()(Image.open(f"{test_data_path}/{image_path}").convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(f"{masks_data_path}/{mask_path}").convert("L"))
    mask = mask.unsqueeze(0)
    image, mask = image.to(device), mask.to(device)
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img, _ = loaded_quantized_model(image_masked, mask)


# real test
for image_path, mask_path in tqdm(zip([image_paths[-1]], masks), total=1):
    image = ToTensor()(Image.open(f"{test_data_path}/{image_path}").convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(f"{masks_data_path}/{mask_path}").convert("L"))
    mask = mask.unsqueeze(0)
    image, mask = image.to(device), mask.to(device)
    image_masked = image * (1 - mask.float()) + mask

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function("model_inference"):
            loaded_quantized_model(image_masked, mask)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))