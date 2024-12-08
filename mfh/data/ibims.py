import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class iBims(Dataset):
    def __init__(self, config):
        root_folder = config.ibims_root
        with open(os.path.join(root_folder, "imagelist.txt"), "r") as f:
            imglist = f.read().split()

        samples = []
        for basename in imglist:
            img_path = os.path.join(root_folder, "rgb", basename + ".png")
            depth_path = os.path.join(root_folder, "depth", basename + ".png")
            valid_mask_path = os.path.join(root_folder, "mask_invalid", basename + ".png")
            transp_mask_path = os.path.join(root_folder, "mask_transp", basename + ".png")
            calib_path = os.path.join(root_folder, "calib", basename + ".txt")

            samples.append((img_path, depth_path, valid_mask_path, transp_mask_path, calib_path))

        self.samples = samples
        # self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x

    def __getitem__(self, idx):
        (
            img_path,
            depth_path,
            valid_mask_path,
            transp_mask_path,
            calib_path,
        ) = self.samples[idx]

        img = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16).astype("float") * 50.0 / 65535

        mask_valid = np.asarray(Image.open(valid_mask_path))
        mask_transp = np.asarray(Image.open(transp_mask_path))

        # depth = depth * mask_valid * mask_transp
        depth = np.where(mask_valid * mask_transp, depth, -1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)
        depth = torch.from_numpy(depth).unsqueeze(0)

        fx, fy, cx, cy = np.loadtxt(calib_path, delimiter=",")
        focal = (fx + fy) / 2

        return dict(
            image=img,
            depth=depth,
            image_path=img_path,
            depth_path=depth_path,
            focal=focal,
            dataset="ibims",
        )

    def __len__(self):
        return len(self.samples)


def get_ibims_loader(config, batch_size=1, **kwargs):
    dataloader = DataLoader(iBims(config), batch_size=batch_size, **kwargs)
    return dataloader
