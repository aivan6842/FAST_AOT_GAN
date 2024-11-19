import shutil
import os
from tqdm import tqdm

for mask in tqdm(os.listdir("data/masks_3")):
    for i in range(1, 5):
        shutil.copy(f"data/masks_3/{mask}", f"data/masks_3/{i}_{mask}")

# for mask in os.listdir("data/masks_1"):
#     if "5_0" in mask:
#         os.remove(f"data/masks_1/{mask}")