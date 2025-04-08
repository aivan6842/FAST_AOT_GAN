import os

from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F

class InpaintingData(Dataset):
    def __init__(self, root_dir: str, masks_dir: str = "data/masks"):
        super(Dataset, self).__init__()
        # images 
        self.images = os.listdir(f"{root_dir}")
        self.root_dir = root_dir
        self.masks_dir = masks_dir
        self.masks = os.listdir(masks_dir)
        random.seed(10)

        # augmentation
        self.img_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(512),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),
            ]
        )
        self.mask_trans = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # load image
        image_path = os.path.join(f"{self.root_dir}", self.images[index])
        image = Image.open(image_path).convert("RGB")

        # get mask
        random_idx = random.randint(0, len(self.masks)-1)
        mask_path = os.path.join(f"{self.masks_dir}", self.masks[random_idx])
        mask = Image.open(mask_path).convert("L")

        # augment
        image = self.img_trans(image) * 2.0 - 1.0
        mask = F.to_tensor(self.mask_trans(mask))

        return image, mask, image_path