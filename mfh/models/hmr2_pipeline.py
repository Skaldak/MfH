from pathlib import Path

import hmr2
import numpy as np
import torch
from PIL import Image
from detectron2.config import LazyConfig
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import DEFAULT_CHECKPOINT, download_models
from hmr2.utils import recursive_to
from hmr2.utils.renderer import cam_crop_to_full
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from torch.utils.data import DataLoader

from mfh.utils.hmr_utils import ViTDetDataset, recursive_cat, load_hmr2


class HMR2Pipeline:
    def __init__(self):
        # Setup detectron2
        detectron2_cfg = LazyConfig.load(
            str(Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py")
        )
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(len(detectron2_cfg.model.roi_heads.box_predictors)):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # Download and load checkpoints
        download_models(CACHE_DIR_4DHUMANS)
        model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

        # Setup HMR2.0 model
        self.model, self.model_cfg = model.cuda().eval(), model_cfg

    @torch.inference_mode()
    def __call__(self, image, focal_length=None):
        if isinstance(image, Image.Image):
            image = np.asarray(image)

        det_out = self.detector(image)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        masks = det_instances.pred_masks[valid_idx].cpu().numpy()

        if valid_idx.sum() > 0:
            # Run HMR2.0 on all detected humans
            dataset = ViTDetDataset(self.model_cfg, image, boxes, masks=masks)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_out = {}
            for batch in dataloader:
                batch = recursive_to(batch, "cuda")
                out = self.model(batch)
                scaled_focal_length = (
                    self.model_cfg.EXTRA.FOCAL_LENGTH
                    / self.model_cfg.MODEL.IMAGE_SIZE
                    * batch["img_size"].float().max()
                )
                focal_length = float(scaled_focal_length if focal_length is None else focal_length)
                pred_cam_t_full = cam_crop_to_full(
                    out["pred_cam"],
                    batch["box_center"].float(),
                    batch["box_size"].float(),
                    batch["img_size"].float(),
                    focal_length,
                )
                out = {
                    "focal_length": torch.ones_like(out["focal_length"]) * focal_length,
                    "pred_bboxes": batch["box"],
                    "pred_masks": batch["mask"],
                    "pred_vertices": out["pred_vertices"],
                    "pred_cam_t_full": pred_cam_t_full,
                }
                for k, v in out.items():
                    if k in all_out:
                        all_out[k].append(v)
                    else:
                        all_out[k] = [v]
            all_out = recursive_cat(all_out)
        else:
            all_out = {}

        return all_out
