import copy

from AOT_GAN.src.model.aotgan import InpaintGenerator
import torch
from attrdict import AttrDict
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping, QConfigMapping, QConfig, get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.observer import HistogramObserver, MovingAverageMinMaxObserver, MinMaxObserver, PerChannelMinMaxObserver, FixedQParamsObserver
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping, QConfigMapping, QConfig, get_default_qconfig, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
from torch.ao.quantization.observer import HistogramObserver, MovingAverageMinMaxObserver, MinMaxObserver, PerChannelMinMaxObserver, FixedQParamsObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import default_fused_per_channel_wt_fake_quant, FusedMovingAvgObsFakeQuantize, FakeQuantize
import random

device = torch.device("cpu")

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


#### load model to quantize #######
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})
student_final_model = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/student_generator_up_to_60_percent_mask_final.pt"

student_generator = InpaintGenerator(half_size_args)
student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

#### Get data for calibration ######
test_data_path = "data/x-medium/test"
test = InpaintingData(test_data_path)
test_loader = DataLoader(test, batch_size=16, shuffle=True)

#### prepare quantize ########
model_to_quantize = copy.deepcopy(student_generator)
example_inputs = (torch.rand(size=(1,3,512,512)), torch.rand(size=(1,1,512,512)))

qconfig_map = get_default_qconfig_mapping()
# qconfig_map = QConfigMapping()
# qconfig = QConfig(
#                     activation=HistogramObserver.with_args(reduce_range=True),
#                     weight=PerChannelMinMaxObserver.with_args(
#                         qscheme=torch.per_channel_symmetric
#                     ),
#                 )
# qconfig_map = QConfigMapping().set_global(qconfig)
for name, module in model_to_quantize.named_modules():
    if "decoder" in name or "decoder" in name:
        qconfig = QConfig(
                activation=MovingAveragePerChannelMinMaxObserver.with_args(
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    ch_axis=1
                ),
                weight=MovingAveragePerChannelMinMaxObserver.with_args(
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    ch_axis=1
                ),
            )
        qconfig_map.set_module_name(name, qconfig)
    else:
        qconfig_map.set_module_name(name, None)

# qconfig_map.set_module_name(name, QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=FixedQParamsObserver.with_args(scale=0.1, zero_point=0)))
prepared_model = prepare_fx(model_to_quantize, qconfig_map, example_inputs)

#### Calibrate #######
def calibrate(model, data_loader):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, masks, _ in tqdm(data_loader):
            images, masks = images.to(device), masks.to(device)
            model(images, masks)

calibrate(prepared_model, test_loader)


### convert model #####
prepared_model.to(torch.device("cpu"))
prepared_model.eval()
quantized_model = convert_fx(prepared_model)

# ##### save model #####
quantized_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/AOT_GAN/experiments/places2/generator_quantized_test_1.pth"
torch.save(quantized_model.state_dict(), quantized_model_path)
