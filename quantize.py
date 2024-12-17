from AOT_GAN.src.model.aotgan import InpaintGeneratorQuant
import torch
from attrdict import AttrDict
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
import random

from PIL import Image
import numpy as np

torch.set_warn_always(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create models
teacher_model_path = "AOT_GAN/experiments/places2/G0000000.pt"
teacher_model_args = AttrDict({"block_num":8, "rates":[1, 2, 4, 8]})
teacher_model = InpaintGeneratorQuant(teacher_model_args).to(device)
teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device, weights_only=True))
teacher_model.eval()

student_model_path = "/w/nobackup/385/scratch-space/expires-2024-Dec-23/aivan6842/models/student_generator_up_to_60_percent_mask_45.pt"
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})
student_model = InpaintGeneratorQuant(half_size_args).to(device)
student_model.load_state_dict(torch.load(student_model_path, map_location=device, weights_only=True))
student_model.eval()

backend = "qnnpack"
backend = "fbgemm"
teacher_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
student_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
teacher_model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
student_model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
torch.backends.quantized.engine = backend

model_static_quantized_teacher = torch.ao.quantization.prepare(teacher_model, inplace=False)
model_static_quantized_student = torch.ao.quantization.prepare(student_model, inplace=False)

n = 100
test_data_path = "data/x-medium/test"
masks = []
for i in range(1, 7):
    masks += list([(i, mask) for mask in os.listdir(f"data/masks_{i}")])

image_paths = random.sample(os.listdir(f"{test_data_path}"), n)
masks = random.sample(masks, n)
for image_path, (mask_idx, mask_path) in tqdm(zip(image_paths, masks), total=len(image_paths)):
    image = ToTensor()(Image.open(f"{test_data_path}/{image_path}").convert("RGB"))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    mask = ToTensor()(Image.open(f"data/masks_{mask_idx}/{mask_path}").convert("L"))
    mask = mask.unsqueeze(0)
    image, mask = image.to(device), mask.to(device)
    image_masked = image * (1 - mask.float()) + mask

    with torch.no_grad():
        pred_img, _ = model_static_quantized_teacher(image_masked, mask)
        pred_img, _ = model_static_quantized_student(image_masked, mask)

model_static_quantized_teacher = torch.ao.quantization.convert(model_static_quantized_teacher, inplace=False)
model_static_quantized_student = torch.ao.quantization.convert(model_static_quantized_student, inplace=False)

# save_dir = "/w/nobackup/385/scratch-space/expires-2024-Dec-14/aivan6842/models"
# torch.save(model_static_quantized_teacher.state_dict(), os.path.join(save_dir, f"teacher_quantized.pt"))
# torch.save(model_static_quantized_student.state_dict(), os.path.join(save_dir, f"student_quantized.pt"))

device = torch.device("cpu")
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})

pct = "1"
test_data_path = "data/x-medium/test"
masks_data_path = f"data/masks_{pct}"

image_paths = sorted(os.listdir(f"{test_data_path}"))
masks = sorted(os.listdir(masks_data_path))

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
        pred_img_teacher, _ = model_static_quantized_teacher(image_masked, mask)
        pred_img_student, _ = model_static_quantized_student(image_masked, mask)

    comp_imgs_teacher = (1 - mask) * image + mask * pred_img_teacher
    comp_imgs_student = (1 - mask) * image + mask * pred_img_student
    image_name = os.path.basename(image_path).split(".")[0]
    # postprocess(image_masked[0]).save(f"tests/{pct}_base/{image_name}_masked.png")
    # postprocess(pred_img[0]).save(f"tests/{pct}_base/{image_name}_pred.png")
    postprocess(comp_imgs_teacher[0]).save(f"/scratch/expires-2024-Dec-23/aivan6842/test/quant/teacher/{image_name}_comp.png")
    postprocess(comp_imgs_student[0]).save(f"/scratch/expires-2024-Dec-23/aivan6842/test/quant/student/{image_name}_comp.png")