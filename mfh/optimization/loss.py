import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from submodules.metric_depth.zoedepth.trainers.loss import extract_key, KEY_OUTPUT


class PixelLoss(nn.Module):
    """Pixel-wise loss"""

    def __init__(self, mode="abs_rel"):
        super(PixelLoss, self).__init__()

        def abs_rel_loss(input, target):
            return (F.l1_loss(input, target, reduction="none") / target).nanmean()

        def log_l2_loss(input, target, alpha=1e-7):
            return F.mse_loss(torch.log(input + alpha), torch.log(target + alpha))

        if mode == "abs_rel":
            self.loss_fn = abs_rel_loss
        elif mode == "l1":
            self.loss_fn = nn.L1Loss()
        elif mode == "l2" or mode == "mse":
            self.loss_fn = nn.MSELoss()
        elif mode == "log_l2":
            self.loss_fn = log_l2_loss
        else:
            raise NotImplementedError
        self.name = mode

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode="bilinear", align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            loss = self.loss_fn(input, target)

        if not return_interpolated:
            return loss

        return loss, intr_input
