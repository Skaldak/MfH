import torch

from submodules.metric_depth.zoedepth.models.base_models.depth_anything import DepthAnythingCore as Core
from submodules.metric_depth.zoedepth.models.base_models.dpt_dinov2.dpt import DPT_DINOv2 as DPT, DPTHead


class DepthAnythingCore(Core):
    @staticmethod
    def build(
        midas_model_type="dinov2_large",
        train_midas=False,
        use_pretrained_midas=True,
        fetch_features=False,
        freeze_bn=True,
        force_keep_ar=False,
        force_reload=False,
        **kwargs,
    ):
        if "img_size" in kwargs:
            kwargs = DepthAnythingCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])
        depth_anything = DPT_DINOv2(out_channels=[256, 512, 1024, 1024], use_clstoken=False)
        encoder = kwargs.get("encoder", "vitl")
        state_dict = torch.load(f"checkpoints/depth_anything_{encoder}14.pth", map_location="cpu", weights_only=True)
        depth_anything.load_state_dict(state_dict)
        kwargs.update({"keep_aspect_ratio": force_keep_ar})
        depth_anything_core = DepthAnythingCore(
            depth_anything,
            trainable=train_midas,
            fetch_features=fetch_features,
            freeze_bn=freeze_bn,
            img_size=img_size,
            **kwargs,
        )
        depth_anything_core.set_output_channels()
        return depth_anything_core


class DPT_DINOv2(DPT):
    def __init__(
        self, encoder="vitl", features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False
    ):
        torch.nn.Module.__init__(self)
        torch.manual_seed(1)
        self.pretrained = torch.hub.load(
            "submodules/torchhub/facebookresearch_dinov2_main",
            f"dinov2_{encoder}14",
            source="local",
            pretrained=False,
        )
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
