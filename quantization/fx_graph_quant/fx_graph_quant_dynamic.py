# Third-party libraries
import torch
from attrdict import AttrDict

# PyTorch Quantization
# Local imports
from AOT_GAN.src.model.aotgan import InpaintGenerator

device = torch.device("cuda")

#### load model to quantize #######
half_size_args = AttrDict({"block_num": 8, "rates": [1, 2, 4, 8]})
student_final_model = "AOT_GAN/experiments/places2/G0000000.pt"

student_generator = InpaintGenerator(half_size_args)
# student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
# student_generator.eval()

#### prepare quantize ########
# model_to_quantize = copy.deepcopy(student_generator)

# traced_model = torch.fx.symbolic_trace(model_to_quantize)

# qconfig_mapping = (QConfigMapping()
#     .set_object_type(torch.nn.Conv2d, float_qparams_weight_only_qconfig)
#     .set_object_type(torch.nn.ReLU, float_qparams_weight_only_qconfig)
# )

# example_inputs = torch.rand(size=(1,3,512,512))
# prepared_model = prepare_fx(traced_model, qconfig_mapping, example_inputs)


# ### convert model #####
# prepared_model.to(torch.device("cpu"))
# prepared_model.eval()
# quantized_model = convert_fx(prepared_model)
# import pdb; pdb.set_trace()

quantized_model = torch.ao.quantization.quantize_dynamic(
    student_generator,  # the original model
    {torch.nn.Conv2d, torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)

import pdb; pdb.set_trace()

##### save model #####
quantized_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/AOT_GAN/experiments/places2/generator_quantized_dynamic.pth"
torch.save(quantized_model.state_dict(), quantized_model_path)
