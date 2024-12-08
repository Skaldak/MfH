import os

import exifread
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def get_focal_length(image_path):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)
        focal_length, width = tags.get("EXIF FocalLengthIn35mmFilm"), tags.get("EXIF ExifImageWidth")
        if focal_length is None:
            length = tags.get("EXIF ExifImageLength")
            focal_length = min(length.values[0], width.values[0])
        else:
            focal_length = focal_length.values[0] / 35 * width.values[0]
        return focal_length


class ToTensor:
    def __init__(self):
        self.size = 480
        self.resize = transforms.Resize(self.size)

    def __call__(self, sample):
        origin, focal = sample["image"], sample["focal"]
        origin = self.to_tensor(origin)
        scale = self.size / min(origin.shape[-2:])
        resized = self.resize(origin)
        sample.update({"image": resized, "origin": origin, "focal": focal * scale})
        return sample

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class WildDataset(Dataset):
    def __init__(self, path):
        self.images = list()
        self.transform = transforms.Compose([ToTensor()])

        if path.lower().endswith((".jpeg", ".jpg", ".png")):
            self.images.append(path)
        else:
            for file in list(os.listdir(path)):
                if file.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.images.append(os.path.join(path, file))
        self.images = sorted(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_path = self.images[idx]
        focal = get_focal_length(sample_path)
        image_path = sample_path
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = np.asarray(image, dtype=np.float32) / 255.0
        sample = {"image": image, "focal": focal, "image_path": sample_path, "dataset": "wild"}
        if self.transform:
            sample = self.transform(sample)
        return sample
