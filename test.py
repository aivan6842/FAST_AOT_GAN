from AOT_GAN.src.model.aotgan import InpaintGenerator
import torch
from attrdict import AttrDict
import numpy as np
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from PIL import Image


device = torch.device("cpu")
half_size_args = AttrDict({"block_num": 8, "rates": [1, 2, 4, 8]})

pct = "1"
# save_dir = f"/w/nobackup/385/scratch-space/expires-2024-Dec-23/aivan6842/test/ours/ood/{pct}"
save_dir = "tests/paper1"
test_data_path = "data/x-medium/test"
# test_data_path = "/scratch/expires-2024-Dec-23/aivan6842/data/ood3/ood"
test_data_path = "tests/paper"
# masks_data_path = f"data/masks_{pct}"
masks_data_path = "tests/paper"
student_final_model = "AOT_GAN/experiments/places2/G0000000.pt"
# student_final_model = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/student_generator_up_to_60_percent_mask_final.pt"

##### load model #######
student_generator = InpaintGenerator(half_size_args).to(device)
student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

########################

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

    # postprocess(image_masked[0]).save("/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/tests/quant_paper/clouds_input.png")
    
    with torch.no_grad():
        pred_img, _ = student_generator(image_masked, mask)

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(image_path).split(".")[0]
    # postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    # postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    postprocess(comp_imgs[0]).save(f"/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/tests/quant_paper/pond_baseline.png")
