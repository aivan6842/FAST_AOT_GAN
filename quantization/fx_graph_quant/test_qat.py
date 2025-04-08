# Standard library
import os
import copy

# Third-party libraries
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from attrdict import AttrDict
from torchvision.transforms import ToTensor

# PyTorch Quantization
from torch.ao.quantization.qconfig_mapping import (
    get_default_qat_qconfig_mapping,
    QConfig,
)
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    prepare_qat_fx,
)
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.fake_quantize import (
    FusedMovingAvgObsFakeQuantize,
    FakeQuantize,
)

# Local imports
from AOT_GAN.src.model.aotgan import InpaintGenerator, AOTBlock

device = torch.device("cpu")
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})

pct = "1"
# save_dir = f"/w/nobackup/385/scratch-space/expires-2024-Dec-23/aivan6842/test/ours/ood/{pct}"
save_dir = "tests/quant_6"
test_data_path = "data/x-medium/test"
# test_data_path = "/scratch/expires-2024-Dec-23/aivan6842/data/ood3/ood"
test_data_path = "tests/paper"
# masks_data_path = f"data/masks_{pct}"
masks_data_path = "tests/paper"
student_final_model = "AOT_GAN/experiments/places2/G0000000.pt"
# student_final_model = "/w/nobackup/385/scratch-space/expires-2024-Dec-23/aivan6842/models/student_generator_up_to_60_percent_mask_45.pt"

#### load model #####
quantized_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/main_models/fx__qat.pth"

student_generator = InpaintGenerator(half_size_args).to(device)
# student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

model_to_quantize = copy.deepcopy(student_generator)
example_inputs = (torch.rand(size=(1,3,512,512)).to(device), torch.rand(size=(1,3,512,512)).to(device))


qconfig_map = get_default_qat_qconfig_mapping()
qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    reduce_range=True,
                    dtype=torch.qint8
                ),
                weight=FakeQuantize.with_args(
                    observer=MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8
                ),
            )
# qconfig_map = QConfigMapping().set_global(qconfig)

# qconfig_map.set_module_name(name, QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=FixedQParamsObserver.with_args(scale=0.1, zero_point=0)))
prepared_model = prepare_qat_fx(model_to_quantize, qconfig_map, example_inputs)

loaded_quantized_model = convert_fx(prepared_model)
loaded_quantized_model.load_state_dict(torch.load(quantized_model_path, weights_only=True))
a = list(loaded_quantized_model.parameters())
import pdb; pdb.set_trace()


pct = 6
test_data_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/x-medium/test"
masks_data_path = f"/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/masks_{pct}"
image_paths = os.listdir(test_data_path)
masks = os.listdir(masks_data_path)

image_paths = ["/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/tests/quant_paper/pond.jpg"]
masks = ["/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/tests/quant_paper/1_00196.png"]

save_dir = "tests/quant_paper"

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

for image_path, mask_path in tqdm(zip(image_paths, masks), total=len(image_paths)):
    image = ToTensor()(Image.open(f"{image_path}").convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(f"{mask_path}").convert("L"))
    mask = mask.unsqueeze(0)
    image, mask = image.to(device), mask.to(device)
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img, _ = loaded_quantized_model(image_masked, mask)
        pred_img_1, _ = student_generator(image_masked, mask)

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(image_path).split(".")[0]
    # postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    # postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    postprocess(comp_imgs[0]).save(f"{save_dir}/{image_name}_fx_qat.png")
