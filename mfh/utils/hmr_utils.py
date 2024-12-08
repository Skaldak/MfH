import os
import sys
from pathlib import Path

import torch
from hmr2.configs import CACHE_DIR_4DHUMANS, get_config
from hmr2.datasets import vitdet_dataset
from hmr2.models import HMR2, check_smpl_exists


def recursive_cat(x):
    y = dict()
    for k, v in x.items():
        if isinstance(v[0], dict):
            merged_dict = {sub_key: [vi[sub_key] for vi in v] for sub_key in v[0]}
            y[k] = recursive_cat(merged_dict)
        elif isinstance(v[0], torch.Tensor):
            y[k] = torch.cat(v)
    return y


def load_hmr2(
    checkpoint_path=f"{CACHE_DIR_4DHUMANS}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt",
):
    model_cfg = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg, update_cachedir=True)

    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    check_smpl_exists()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)
    return model, model_cfg


class ViTDetDataset(vitdet_dataset.ViTDetDataset):
    def __init__(self, cfg, img_cv2, boxes, train=False, **kwargs):
        super().__init__(cfg, img_cv2, boxes, train=False, **kwargs)
        self.boxes = boxes
        self.masks = kwargs.get("masks", None)

    def __getitem__(self, idx):
        with open(os.devnull, "w") as devnull:
            stdout, sys.stdout = sys.stdout, devnull
            item = super().__getitem__(idx)
            sys.stdout = stdout

        item["box"] = self.boxes[idx].copy()
        item["mask"] = self.masks[idx].copy() if self.masks is not None else None
        return item
