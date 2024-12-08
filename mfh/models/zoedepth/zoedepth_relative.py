import torch

from mfh.utils.depth_utils import DepthAnythingCore
from submodules.metric_depth.zoedepth.models.depth_model import DepthModel
from submodules.metric_depth.zoedepth.models.model_io import load_state_from_resource


class ZoeDepthRelative(DepthModel):
    def __init__(self, core, inverse_midas=False, **kwargs):
        super().__init__()

        self.core = core
        self.inverse_midas = inverse_midas

    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.

        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)

        """
        b, c, h, w = x.shape
        self.orig_input_width = w
        self.orig_input_height = h

        if hasattr(self.core, "TASK_KEYS"):
            intrinsics = torch.eye(3, device=x.device)[None].repeat(b, 1, 1)
            intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = kwargs["focal"]
            intrinsics[:, 0, 2] = w / 2
            intrinsics[:, 1, 2] = h / 2
            rel_depth = self.core(x, intrinsics.repeat_interleave(b, dim=0)).squeeze(1)
        else:
            rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)
        if self.inverse_midas or kwargs.get("inverse_midas", False):
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-12)
            min_depth = rel_depth.flatten(1).min(1).values[:, None, None]
            max_depth = rel_depth.flatten(1).max(1).values[:, None, None]
            rel_depth = (rel_depth - min_depth) / (max_depth - min_depth)

        intermediate = {"rel_depth": rel_depth[:, None]}
        if kwargs.get("return_intermediate", False):
            return intermediate
        else:
            raise NotImplementedError

    @staticmethod
    def build(
        midas_model_type="DPT_BEiT_L_384",
        pretrained_resource=None,
        use_pretrained_midas=False,
        freeze_midas_bn=True,
        **kwargs
    ):
        core = DepthAnythingCore.build(
            midas_model_type=midas_model_type,
            use_pretrained_midas=use_pretrained_midas,
            fetch_features=True,
            freeze_bn=freeze_midas_bn,
            **kwargs
        )

        model = ZoeDepthRelative(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepthRelative.build(**config)
