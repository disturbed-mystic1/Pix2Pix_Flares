from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms


test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
    ]
)


class FlareTest(Dataset):
    def __init__(self, flare_dir, transform=None):
        self.flare_dir = flare_dir

        self.transform = transform
        self.flare_img = os.listdir(flare_dir)

    def __len__(self):
        return len(self.flare_img)

    def __getitem__(self, idx):
        self.flare_img.sort()
        f_img = Image.open(os.path.join(self.flare_dir, self.flare_img[idx])).convert(
            "RGB"
        )

        f_img = self.transform(f_img)

        return f_img
