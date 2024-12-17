# MD-AOT-GAN

# Overview
The Multilayer distilled AOT-GAN (MD-AOT-GAN) model was used for my CSC2541 final project. It is a multilayer distilled GAN based on the SOTA AOT-GAN model. Below is an image of the general overview and how the model is structured.
![MD-AOT-GAN Overview](/images/AOT-GAN.png)

The resulting model contains about 2x less parameters. Here are the results of the model.

![Results](image.png)

Note that the majority of the AOT_GAN submodule was provided by Zeng et al (https://github.com/researchmm/AOT-GAN-for-Inpainting). There were a few modifications made to 
1. Support Quantization (AOT_GAN/src/model/aotgan.py)
2. Support multilayer knowledge distillation (AOT_GAN/src/model/aotgan.py)

# Files

## `train.py`
This is the main file which is used for training the MD-AOT-GAN. This performs multilayer knowledge distillation and the parameters for training can be configured here. The main function call is the last function call in the file.

## `test.py`
This file is used to generate samples using your trained model. The `train.py` script will save the file to desired location. Use this location to load your model and generate samples that will be evaluated on.

## `test_non_neural.py`
This is used to test non-neural inpainting methods like telea and naive stokes. Implementation is provided by OpenCV. This script is much like `test.py` as it will generate new samples. 

## `eval.py`
This script is used with arguments `--real_dir /real/dir`, `--fake_dir /fake/dir`, `--metric mae psnr ssim fid`. This is used to evaluate your generated samples against a ground truth.

## `quantize.py`
This script is our attempt at Eager mode Post training dynamic quantization. 