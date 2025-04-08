# Standard library
import copy

# Third-party libraries
import torch
from tqdm import tqdm
from attrdict import AttrDict
from torch.utils.data import DataLoader

# PyTorch Quantization
from torch.ao.quantization.qconfig_mapping import (
    get_default_qconfig_mapping,
    QConfig,
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
)
from torch.ao.quantization.observer import (
    MovingAveragePerChannelMinMaxObserver,
)

# Local imports
from AOT_GAN.src.model.aotgan import InpaintGenerator
from inpainting_dataset import InpaintingData

device = torch.device("cpu")

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
