# Standard library
import os
import copy
import numpy as np

# Third-party libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from attrdict import AttrDict

# PyTorch Quantization
from torch.ao.quantization.qconfig_mapping import (
    get_default_qat_qconfig_mapping,
    QConfigMapping,
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
from inpainting_dataset import InpaintingData
from AOT_GAN.src.model.aotgan import InpaintGenerator, Discriminator
from AOT_GAN.src.loss.loss import L1, Style, Perceptual, smgan

device = torch.device("cuda")


#### Get data for calibration ######
test_data_path = "data/x-medium/test"
test = InpaintingData(test_data_path)
test_loader = DataLoader(test, batch_size=1, shuffle=True)
a = next(iter(test_loader))

# create models
student_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/student_generator_up_to_60_percent_mask_final.pt"
half_size_args = AttrDict({"block_num": 4, "rates": [1, 2, 4, 8]})
student_model = InpaintGenerator(half_size_args)
student_model.load_state_dict(torch.load(student_model_path, map_location=device, weights_only=True))

disc_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/models/discriminator_up_to_60_percent_mask_17.pt"
disc = Discriminator()
disc.load_state_dict(torch.load(disc_model_path, map_location=device, weights_only=True))

#### prepare quantize ########
model_to_quantize = copy.deepcopy(student_model)
example_inputs = (torch.rand(size=(1,3,512,512)), torch.rand(size=(1,1,512,512)))

# qconfig_map = get_default_qat_qconfig_mapping()
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
qconfig_map = QConfigMapping().set_global(qconfig)
qconfig_map = get_default_qat_qconfig_mapping()
prepared_model = prepare_qat_fx(model_to_quantize, qconfig_map, example_inputs)

#### model equivalence #######
def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    for _ in range(num_tests):
        x, x1 = torch.rand(size=input_size), torch.rand(size=(1, 1, 512, 512))
        y1, _ = model_1(x, x1)
        y1 = y1.detach().cpu().numpy()
        y2, _ = model_2(x, x1)
        y2 = y2.detach().cpu().numpy()
        ret = np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False)
        if ret == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            raise Exception("NOT EQUIVALENT")


student_model.eval()
prepared_model.eval()
# model_equivalence(model_1=student_model,
#                           model_2=prepared_model, 
#                           device=torch.device("cpu"), 
#                           rtol=1e-01, 
#                           atol=3, 
#                           num_tests=100, 
#                           input_size=(1,3,512,512))

print("Equivalent")
### train model #####
prepared_model.train()
def train(run_name, 
          student_generator,
          teacher_generator,
          discriminator,
          L1_loss_weight=0.1,
          style_loss_weight=250,
          perceptual_loss_weight=0.1,
          adversarial_loss_weight=0.01,
          distillation_loss_weight=0.1,
          focused_loss_weight=0.2,
          num_epochs = 5,
          gen_lr = 1e-4,
          disc_lr = 1e-4,
          a=0.5,
          b=0.999,
          save_every=3,
          save_dir="models/",
          log_dir="./runs"):
    writer = SummaryWriter(f"{log_dir}/{run_name}")
    iteration = 0

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Create losses
    L1_loss = L1()
    style_loss = Style()
    percetual_loss = Perceptual()
    adversarial_loss = smgan()
    focused_loss = torch.nn.MSELoss()

    # get optimizers
    optimG = torch.optim.AdamW(student_generator.parameters(), lr=gen_lr, betas=(a, b))
    optimD = torch.optim.AdamW(discriminator.parameters(), lr=disc_lr, betas=(a, b))

    print("Beginning Training")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        for i, data in enumerate(tqdm(test_loader)):
            # get batch of data
            images, masks, _ = data
            images, masks = images.to(device), masks.to(device)
            masked_images = (images * (1 - masks).float()) + masks

            predicted_images, student_mids = student_generator(masked_images, masks)
            inpainted_images = (1 - masks) * images + masks * predicted_images

            # losses
            l1_loss_val = L1_loss(predicted_images, images)
            focused_loss_val = focused_loss(predicted_images * masks, images * masks)
            style_loss_val = style_loss(predicted_images, images)
            percetual_loss_val = percetual_loss(predicted_images, images)
            # distillation_loss_val = distillation_loss(student_mids, teacher_mids[1::2])
            adversarial_disc_loss, adversarial_gen_loss = adversarial_loss(discriminator, inpainted_images, images, masks)

            total_loss = (L1_loss_weight * l1_loss_val) + \
                         (style_loss_weight * style_loss_val) + \
                         (perceptual_loss_weight * percetual_loss_val) + \
                         (focused_loss_weight * focused_loss_val) + \
                         (adversarial_loss_weight * adversarial_gen_loss)
        
            optimG.zero_grad()
            optimD.zero_grad()
            total_loss.backward()
            adversarial_disc_loss.backward()
            optimG.step()
            optimD.step()

            writer.add_scalar("Loss/train/generator", adversarial_gen_loss, iteration)
            writer.add_scalar("Loss/train/L1_loss", l1_loss_val, iteration)
            writer.add_scalar("Loss/train/style_loss", style_loss_val, iteration)
            writer.add_scalar("Loss/train/perceptual_loss", percetual_loss_val, iteration)
            writer.add_scalar("Loss/train/focused_loss", focused_loss_val, iteration)
            writer.add_scalar("Loss/train/discriminator", adversarial_disc_loss, iteration)
            writer.add_scalar("Loss/train/total", total_loss, iteration)

            iteration += 1 
    
    return student_generator

prepared_model.to(device)
disc.to(device)
trained_model = train(run_name="qat_fused",
      num_epochs=1,
      student_generator=prepared_model,
      teacher_generator=None,
      discriminator=disc,
      save_every=2,
      distillation_loss_weight=0.0001,
      gen_lr=1e-6,
      disc_lr=1e-6)


# trained_model.eval()
trained_model = trained_model.to(torch.device("cpu"))
trained_model.eval()


# convert model
quantized_model = convert_fx(trained_model)

##### save model #####
quantized_model_path = "/w/340/aivan6842/csc2541/AOT_GAN_CSC2541/AOT_GAN/experiments/places2/generator_quantized_qat.pth"
torch.save(quantized_model.state_dict(), quantized_model_path)
