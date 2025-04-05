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
# student_final_model = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/student_generator_up_to_60_percent_mask_49.pt"

##### load model #######
student_generator = InpaintGenerator(half_size_args).to(device)
student_generator.load_state_dict(torch.load(student_final_model, map_location=device, weights_only=True))
student_generator.eval()

########################

image_paths = ["beach_00004089.jpg", "valley_00003311.jpg", "valley_00000409.jpg", "beach_00000780.jpg"]
masks = ["02055.png", "05148.png", "06518.png", "04259.png"]

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

for image_path, mask_path in tqdm(zip(image_paths, masks), total=len(image_paths)):
    image = ToTensor()(Image.open(f"{test_data_path}/{image_path}").convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(f"{masks_data_path}/{mask_path}").convert("L"))
    mask = mask.unsqueeze(0)
    image, mask = image.to(device), mask.to(device)
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img, _ = student_generator(image_masked, mask)
        import pdb; pdb.set_trace()

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(image_path).split(".")[0]
    # postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    # postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    postprocess(comp_imgs[0]).save(f"{save_dir}/{image_name}_comp_aot.png")
