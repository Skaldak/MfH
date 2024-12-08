import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .diode import get_diode_loader
from .eth3d import get_eth3d_loader
from .ibims import get_ibims_loader


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms():
    return transforms.Compose([ToTensor()])


def remove_leading_slash(s):
    if s[0] == "/" or s[0] == "\\":
        return s[1:]
    return s


class ImReader:
    def __init__(self):
        pass

    def open(self, fpath):
        return Image.open(fpath)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, focal = sample["image"], sample["focal"]
        image = self.to_tensor(image)

        depth = sample["depth"]
        has_valid_depth = sample["has_valid_depth"]
        return {
            **sample,
            "image": image,
            "depth": depth,
            "focal": focal,
            "has_valid_depth": has_valid_depth,
            "image_path": sample["image_path"],
            "depth_path": sample["depth_path"],
        }

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
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


class DepthDataLoader:
    def __init__(self, config, transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == "eth3d":
            self.data = get_eth3d_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == "ibims":
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.data = get_diode_loader(config[config.dataset + "_root"], batch_size=1, num_workers=1)
            return

        # nyu and kitti
        if transform is None:
            transform = preprocessing_transforms()

        self.testing_samples = DataLoadPreprocess(config, transform=transform)
        self.data = DataLoader(
            dataset=self.testing_samples,
            batch_size=1,
            shuffle=config.shuffle,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )


class DataLoadPreprocess(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        with open(config.filenames_file_eval, "r") as f:
            if "sample" in config:
                self.filenames = [f.readlines()[config.sample]]
            else:
                self.filenames = f.readlines()

        self.transform = transform
        self.to_tensor = ToTensor()
        self.reader = ImReader()

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}

        data_path = self.config.data_path_eval

        image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]))
        image = np.asarray(self.reader.open(image_path), dtype=np.float32) / 255.0

        gt_path = self.config.gt_path_eval
        depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
        has_valid_depth = False
        try:
            depth_gt = self.reader.open(depth_path)
            has_valid_depth = True
        except IOError:
            depth_gt = False

        if has_valid_depth:
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            if self.config.dataset == "nyu":
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            mask = np.logical_and(depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[
                None, ...
            ]
        else:
            mask = False

        if self.config.do_kb_crop:
            height = image.shape[0]
            width = image.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[top_margin : top_margin + 352, left_margin : left_margin + 1216, :]
            if has_valid_depth:
                depth_gt = depth_gt[top_margin : top_margin + 352, left_margin : left_margin + 1216, :]

        sample = {
            "image": image,
            "depth": depth_gt,
            "focal": focal,
            "has_valid_depth": has_valid_depth,
            "image_path": sample_path.split()[0],
            "depth_path": sample_path.split()[1],
            "mask": mask,
        }

        if "has_valid_depth" in sample and sample["has_valid_depth"]:
            mask = np.logical_and(depth_gt > self.config.min_depth, depth_gt < self.config.max_depth).squeeze()[
                None, ...
            ]
            sample["mask"] = mask

        if self.transform:
            sample = self.transform(sample)

        sample["dataset"] = self.config.dataset
        sample = {
            **sample,
            "image_path": sample_path.split()[0],
            "depth_path": sample_path.split()[1],
        }

        return sample

    def __len__(self):
        return len(self.filenames)
