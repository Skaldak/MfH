import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_camera_parameters(camera_file):
    camera_params = {}
    with open(camera_file, "r") as file:
        for line in file:
            if not line.startswith("#") and line.strip():
                parts = line.split()
                camera_id = int(parts[0])
                f_x, f_y = float(parts[4]), float(parts[5])
                camera_params[camera_id] = (f_x, f_y)
    return camera_params


def map_images_to_cameras(images_file):
    image_to_camera = {}
    with open(images_file, "r") as file:
        for line in file:
            if line.strip().endswith(".JPG"):
                parts = line.split()
                image_id = parts[-1]
                camera_id = int(parts[-2])
                image_to_camera[image_id] = camera_id
    return image_to_camera


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x
        self.resize = transforms.Resize(480)

    def __call__(self, sample):
        image, depth, focal = sample["image"], sample["depth"], sample["focal"]
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)
        scale = 480 / min(image.shape[-2:])
        image = self.resize(image)

        return {
            "image": image,
            "depth": depth,
            "focal": focal * scale,
            "dataset": "eth3d",
        }

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
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


class ETH3D(Dataset):
    def __init__(self, config):
        root_folder = config.eth3d_root
        self.image_files, self.depth_files, self.focal_lengths = [], [], []

        # Read the lines from the text file
        with open(os.path.join(config.eth3d_root, "eth3d_filename_list.txt"), "r") as file:
            lines = file.readlines()

        # Process each line to extract RGB and depth paths
        for line in lines:
            rgb_path, depth_path = line.strip().split(" ")
            self.image_files.append(os.path.join(root_folder, rgb_path))
            self.depth_files.append(os.path.join(root_folder, depth_path))
            calib_file = os.path.join(
                root_folder,
                "/".join(rgb_path.split("/")[:-3] + ["dslr_calibration_jpg", "cameras.txt"]),
            )
            mapping_file = os.path.join(
                root_folder,
                "/".join(rgb_path.split("/")[:-3] + ["dslr_calibration_jpg", "images.txt"]),
            )
            camera_params = load_camera_parameters(calib_file)
            image_to_camera = map_images_to_cameras(mapping_file)
            self.focal_lengths.append(camera_params[image_to_camera["/".join(rgb_path.split("/")[-2:])]])

        # Assuming ToTensor is a transformation that you've defined or imported
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        fx, fy = self.focal_lengths[idx]

        # print(image_path, depth_path)

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = self._read_depth_file(depth_path)

        # depth[depth > 8] = -1
        depth = depth[..., None]

        # print("image:", image.shape, image.min(), image.max())
        # print("depth:",  depth.shape, depth.min(), depth.max())
        return self.transform(dict(image=image, depth=depth, focal=(fx + fy) / 2))

    def __len__(self):
        return len(self.image_files)

    def _read_depth_file(self, depth_path):
        # Read special binary data: https://www.eth3d.net/documentation#format-of-multi-view-data-image-formats
        # if self.is_tar:
        #     if self.tar_obj is None:
        #         self.tar_obj = tarfile.open(self.dataset_dir)
        #     binary_data = self.tar_obj.extractfile("./" + rel_path)
        #     binary_data = binary_data.read()
        #
        # else:
        with open(depth_path, "rb") as file:
            binary_data = file.read()
        # Convert the binary data to a numpy array of 32-bit floats
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()

        depth_decoded[depth_decoded == torch.inf] = 0.0

        depth_decoded = depth_decoded.reshape((4032, 6048))
        # depth_decoded = depth_decoded.astype("uint16") / 5000.0
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and((depth > self.min_depth), (depth < self.max_depth)).bool()
        return valid_mask


def get_eth3d_loader(config, batch_size=1, **kwargs):
    dataloader = DataLoader(ETH3D(config), batch_size=batch_size, **kwargs)
    return dataloader
