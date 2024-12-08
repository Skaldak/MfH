import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BinaryMorphology(nn.Module):
    def __init__(self, kernel_size=5, mode=None, device="cuda"):
        """
        n_channels: int
        kernel_size: scalar, the spatial size of the morphological neure.
        type: str, dilate or erode.
        """
        super().__init__()
        assert mode in ["dilate", "erode"], f"Invalid `type` {mode}"

        self.kernel_size = kernel_size
        self.opp_type = mode

        kernel = np.ones((1, 1, kernel_size, kernel_size), dtype=np.float32)
        self.register_buffer("kernel", torch.as_tensor(kernel, device=device))

    def forward(self, x):
        if self.opp_type == "dilate":
            x = F.conv2d(torch.ge(x, 0.5).float(), self.kernel, padding=self.kernel_size // 2)
            x = torch.clamp(x, 0, 1)
        else:  # erode
            x = F.conv2d(torch.lt(x, 0.5).float(), self.kernel, padding=self.kernel_size // 2)
            x = 1 - torch.clamp(x, 0, 1)

        return x


class Dilate2D(BinaryMorphology):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(kernel_size, "dilate", **kwargs)


class Erode2D(BinaryMorphology):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(kernel_size, "erode", **kwargs)
