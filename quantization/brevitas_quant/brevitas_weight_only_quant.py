import os

# Third-party libraries
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from tqdm import tqdm
from attrdict import AttrDict

# Local imports
from AOT_GAN.src.model.aotgan import InpaintGenerator
from brevitas_quant.common import QuantInpaintGenerator


##### load original model #####
device = torch.device("cpu")
student_model_path = "AOT_GAN/experiments/places2/G0000000.pt"
student_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/student_generator_up_to_60_percent_mask_final.pt"
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})
student_model = InpaintGenerator(half_size_args)
student_model.load_state_dict(torch.load(student_model_path, map_location=device, weights_only=True))


##### load #####
quant_model = QuantInpaintGenerator(half_size_args)
quant_model.load_state_dict(torch.load(student_model_path, map_location=device, weights_only=True))
# Load weights from the original model

print(quant_model)
quant_model.to(device)

pct = 6
test_data_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/x-medium/test"
masks_data_path = f"/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/data/masks_{pct}"
image_paths = ["beach_00004089.jpg", "valley_00003311.jpg", "valley_00000409.jpg", "beach_00000780.jpg"]
masks = ["02055.png", "05148.png", "06518.png", "04259.png"]
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
        pred_img, _ = quant_model(image_masked, mask)

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(image_path).split(".")[0]
    # postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    # postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    postprocess(comp_imgs[0]).save(f"{save_dir}/{image_name}_brevitas_weight_quant.png")
