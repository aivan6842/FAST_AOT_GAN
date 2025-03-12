import copy

from AOT_GAN.src.model.aotgan import InpaintGenerator, AOTBlock
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
import random

device = torch.device("cuda")

#### Dataloader def ####
class InpaintingData(Dataset):
    def __init__(self, root_dir: str, masks_dir: str = "data/masks"):
        super(Dataset, self).__init__()
        # images 
        self.images = os.listdir(f"{root_dir}")[:10]
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
half_size_args = AttrDict({"block_num": 8, "rates": [1, 2, 4, 8]})
student_final_model = "AOT_GAN/experiments/places2/G0000000.pt"

student_generator = InpaintGenerator(half_size_args)
student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

#### Get data for calibration ######
test_data_path = "data/x-medium/test"
test = InpaintingData(test_data_path)
test_loader = DataLoader(test, batch_size=16, shuffle=True)

#### prepare quantize ########
model_to_quantize = copy.deepcopy(student_generator)
example_inputs = torch.rand(size=(1,3,512,512))

global_qconfig = get_default_qconfig()
# qconfig_map = QConfigMapping().set_global(global_qconfig)
qconfig_map = QConfigMapping()

for name, module in model_to_quantize.named_modules():
    if "encoder" in name:
        qconfig = QConfig(
                    activation=HistogramObserver.with_args(reduce_range=True),
                    weight=PerChannelMinMaxObserver.with_args(
                        qscheme=torch.per_channel_symmetric
                    ),
                )
        qconfig_map.set_module_name(name, get_default_qconfig())

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
import pdb;pdb.set_trace()

##### save model #####
quantized_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/AOT_GAN/experiments/places2/generator_quantized.pth"
torch.save(quantized_model.state_dict(), quantized_model_path)
