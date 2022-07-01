from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms


train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


class Flare(Dataset):
    def __init__(self, flare_dir, wf_dir, transform=None):
        self.flare_dir = flare_dir
        self.wf_dir = wf_dir
        self.transform = transform
        self.flare_img = os.listdir(flare_dir)
        self.wf_img = os.listdir(wf_dir)

    def __len__(self):
        return len(self.flare_img)

    def __getitem__(self, idx):
        f_img = Image.open(os.path.join(self.flare_dir, self.flare_img[idx])).convert(
            "RGB"
        )
        for i in self.wf_img:
            if self.flare_img[idx].split(".")[0][4:] == i.split(".")[0]:
                wf_img = Image.open(os.path.join(self.wf_dir, i)).convert("RGB")
                break
        f_img = self.transform(f_img)
        wf_img = self.transform(wf_img)

        return f_img, wf_img
