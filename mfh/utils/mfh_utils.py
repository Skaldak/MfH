import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from submodules.metric_depth.zoedepth.utils.misc import compute_errors


def resize(images, divisible_by=8, size=512, mode="bilinear"):
    *_, h0, w0 = images.shape
    if h0 > w0:
        h, w = h0 * size / w0, size
    else:
        h, w = size, w0 * size / h0
    h, w = (
        int(np.ceil(h / divisible_by)) * divisible_by,
        int(np.ceil(w / divisible_by)) * divisible_by,
    )
    resized_images = F.interpolate(images, size=(h, w), mode=mode)
    return resized_images


def random_mask(image, size=16, min_ratio=0.2, max_ratio=0.8):
    *_, h, w = image.shape
    ws = torch.randint(int(min_ratio * min(h, w)), int(max_ratio * min(h, w)), (size,))
    xs = (torch.rand(size) * (w - ws) + ws / 2).type(torch.int64)
    min_xs, max_xs = xs - ws // 2, xs + ws // 2
    masks = torch.zeros_like(image[:1])[None].repeat_interleave(size, dim=0)
    for idx, (min_x, max_x) in enumerate(zip(min_xs, max_xs)):
        masks[idx, :, :, min_x:max_x] = 1.0
    return masks


def process_depth_mask(depth, mask):
    depth[depth < 0] = torch.inf
    depth = depth.min(0, keepdim=True).values
    mask = (mask.sum(0, keepdim=True) > 0).to(depth)
    depth[depth == torch.inf] = 0
    depth[mask == 0] = 0
    return depth, mask


@torch.inference_mode()
def infer_depth(depth_model, images, **kwargs):
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            return_rel_depth = kwargs.get("return_rel_depth", False)
            pred = pred["rel_depth"] if return_rel_depth else pred.get("metric_depth", pred.get("out", NotImplemented))
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred = depth_model(images, return_intermediate=True, **kwargs)
    pred = get_depth_from_prediction(pred)

    if pred.shape[-2:] != images.shape[-2:]:
        pred = F.interpolate(pred, size=images.shape[-2:], mode="bilinear")

    return pred


def compute_metrics(
    gt,
    pred,
    interpolate=True,
    garg_crop=False,
    eigen_crop=True,
    dataset="nyu",
    min_depth_eval=0.1,
    max_depth_eval=10,
    **kwargs,
):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics."""
    if "config" in kwargs:
        config = kwargs["config"]
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(pred, gt.shape[-2:], mode="bilinear", align_corners=True)

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    eval_mask = np.ones_like(valid_mask)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[
                int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
            ] = 1

        elif eigen_crop:
            if dataset == "kitti":
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            else:
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])
