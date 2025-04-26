# === Standard Library ===
import os
import copy
import random

# === Third-Party Libraries ===
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from PIL import Image
from tqdm import tqdm
from attrdict import AttrDict

# === TorchVision ===
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F

# === AOT-GAN Project Modules ===
from AOT_GAN.src.model.aotgan import InpaintGenerator

# === PyTorch Quantization ===
from torch.ao.quantization.qconfig_mapping import (
    get_default_qat_qconfig_mapping
)
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    prepare_qat_fx
)
from quantization.brevitas_quant.common import QuantInpaintGenerator
from inpainting_dataset import InpaintingData

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