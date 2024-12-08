import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoPipelineForInpainting
from hmr2.utils import Renderer
from matplotlib import pyplot as plt
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from tqdm import tqdm

from mfh.data.data_metric import DepthDataLoader
from mfh.models.hmr2_pipeline import HMR2Pipeline
from mfh.models.zoedepth_builder import build_model
from mfh.optimization.morphology import Erode2D
from mfh.optimization.optimizer import AffineOptimizer, linear_regression
from mfh.utils.config import get_config
from mfh.utils.mfh_utils import infer_depth, process_depth_mask, random_mask, resize, compute_metrics
from submodules.metric_depth.zoedepth.utils.misc import RunningAverageDict, colorize, colors


class MfH:
    def __init__(self, dataset, **kwargs):
        self.config = get_config("zoedepth", "eval", dataset, **kwargs)

        print("Setting up relative depth estimation")
        self.depth_model = build_model(self.config).cuda().eval()
        print("Setting up generative painting")
        self.paint_model = AutoPipelineForInpainting.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
        ).to("cuda")
        print("Setting up human mesh recovery")
        self.paint_model.set_progress_bar_config(disable=True)
        self.hmr_model = HMR2Pipeline()

    @torch.no_grad()
    def evaluate(self):
        # prepare evaluation
        test_loader = DepthDataLoader(self.config).data
        metrics = RunningAverageDict()
        output_dir = Path(self.config.output_path)
        metric_dir = output_dir / "metric"
        metric_dir.mkdir(parents=True, exist_ok=True)
        if self.config.visualize:
            visual_dir = output_dir / "visual"
            visual_dir.mkdir(parents=True, exist_ok=True)
        else:
            visual_dir = None

        # evaluation main loop
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            if "has_valid_depth" in sample and not sample["has_valid_depth"]:
                continue

            assert len(sample["image"]) == 1
            image = sample["image"].cuda()
            depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
            focal = sample.get("focal", torch.Tensor([715.0873]).cuda())
            pred = self.infer_image(image, focal=focal)

            # visualization
            if self.config.visualize:
                g = colorize(depth.squeeze().cpu().numpy(), vmin=0, vmax=self.config.max_depth, cmap="magma_r")
                Image.fromarray(g).save(visual_dir / f"{i}_gt.png")
                p = colorize(pred.squeeze().cpu().numpy(), vmin=0, vmax=self.config.max_depth, cmap="magma_r")
                Image.fromarray(p).save(visual_dir / f"{i}_pred.png")

            metric = compute_metrics(depth, pred, config=self.config)
            np.savez(metric_dir / f"batch_{self.config.dataset}_{i}_metrics.npz", **metric)
            metrics.update(metric)
            print(metrics.get_value())

        metrics = metrics.get_value()

        print(f"{colors.fg.green}")
        print(metrics)
        print(f"{colors.reset}")

    @torch.no_grad()
    def infer(self, loader):
        output_dir = Path(self.config.output_path)
        if self.config.visualize:
            visual_dir = output_dir / "visual"
            visual_dir.mkdir(parents=True, exist_ok=True)
        else:
            visual_dir = None

        # inference main loop
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            if "has_valid_depth" in sample and not sample["has_valid_depth"]:
                continue

            image = sample["image"].unsqueeze(0).cuda()
            focal = sample.get("focal", torch.Tensor([715.0873]).cuda())
            pred = self.infer_image(image, focal=focal)
            if self.config.upsample:
                pred = F.interpolate(pred, sample["origin"].shape[1:])

            # visualization
            if self.config.visualize:
                file = os.path.splitext(os.path.basename(sample["image_path"]))[0]
                if "wild" in sample["image_path"]:
                    vmax = int(file[:2])
                    print(sample["image_path"], f"depth range [0, {vmax}]")
                else:
                    vmax = self.config.max_depth
                p = colorize(pred.squeeze().cpu().numpy(), vmin=0, vmax=vmax, cmap="magma_r")
                Image.fromarray(p).save(visual_dir / f"{file}_pred.png")

    def infer_image(self, image, focal=None):
        *_, h0, w0 = image.shape
        scale, size = (self.paint_model.vae_scale_factor, self.paint_model.unet.config.sample_size)
        resized_images = resize(image, divisible_by=scale, size=size * scale)
        *_, h, w = resized_images.shape

        disps = infer_depth(self.depth_model, image, return_rel_depth=True, **self.config)
        focal_default = (
            self.hmr_model.model_cfg.EXTRA.FOCAL_LENGTH / self.hmr_model.model_cfg.MODEL.IMAGE_SIZE * max(h0, w0)
        )
        focal_length = focal_default if focal is None else focal

        R, t = torch.eye(3)[None].to(image), torch.zeros(3)[None].to(image)
        K = torch.as_tensor([[[focal_length, 0, w0 / 2], [0, focal_length, h0 / 2], [0, 0, 1]]]).to(image)
        cameras = cameras_from_opencv_projection(R, t, K, torch.as_tensor([[h0, w0]]))
        settings = RasterizationSettings(image_size=(h0, w0), blur_radius=1e-4, faces_per_pixel=8, bin_size=0)
        erode = Erode2D(kernel_size=3, device=image.device)

        image = resized_images.squeeze(0)
        disp = disps.squeeze(0)

        init_hmr_result = self.hmr_model(image.permute(1, 2, 0).cpu().numpy() * 255, focal_length=focal_length)
        init_human, curr_human = (init_hmr_result["pred_masks"].shape[0] if len(init_hmr_result) > 0 else 0), 0
        aligned_disps, human_images, human_depths, human_masks = [], [], [], []

        while curr_human < self.config.paint_total_size:
            (
                aligned_disps_batch,
                human_images_batch,
                human_depths_batch,
                human_masks_batch,
                curr_human_batch,
            ) = self._paint_batch(image, disp, (h0, w0), (h, w), focal_length, init_human, cameras, settings, erode)
            aligned_disps.extend(aligned_disps_batch)
            human_images.extend(human_images_batch)
            human_depths.extend(human_depths_batch)
            human_masks.extend(human_masks_batch)
            curr_human += curr_human_batch

        aligned_disps = torch.cat(aligned_disps)[:, None]
        human_depths = torch.cat(human_depths)[:, None]
        human_masks = (torch.cat(human_masks)[:, None] * (aligned_disps > 0)).type(torch.bool)

        optimizer = AffineOptimizer(
            device=image.device,
            total_iter=self.config.total_iter,
            loss_fn=self.config.loss_fn,
            scale=self.config.scale,
            offset=self.config.offset,
            inverse=self.config.inverse,
            inverse_gt=self.config.inverse_gt,
        )
        optimizer.optimize(aligned_disps, human_depths, target_mask=human_masks)
        optimized_depth = (1 / optimizer.forward(disp)) if self.config.inverse_gt else optimizer.forward(disp)

        if self.config.debug:
            print("MDE scale", optimizer.scale.data.item(), "MDE offset", optimizer.offset.data.item())
            print("min depth", optimized_depth.min().item(), "max depth", optimized_depth.max().item())

        return optimized_depth[None]

    def _paint_batch(
        self,
        image,
        disp,
        size0,
        size,
        focal_length,
        init_human,
        cameras,
        settings,
        erode,
    ):
        aligned_disps, human_images, human_depths, human_masks = [], [], [], []
        curr_human = 0

        h0, w0 = size0
        h, w = size
        mask_images = random_mask(
            image,
            size=self.config.paint_batch_size,
            min_ratio=self.config.min_ratio,
            max_ratio=self.config.max_ratio,
        )
        painted_images = self.paint_model(
            prompt=np.random.choice(["a man", "a woman"], size=self.config.paint_batch_size).tolist(),
            image=image,
            mask_image=mask_images,
            height=h,
            width=w,
            output_type="np",
            guidance_scale=2e1,
        ).images
        painted_tensors = torch.from_numpy(painted_images).permute(0, 3, 1, 2).to(image)
        painted_tensors = F.interpolate(painted_tensors, size=(h0, w0), mode="bilinear")
        painted_images = painted_tensors.permute(0, 2, 3, 1).cpu().numpy()
        mask_images = F.interpolate(mask_images, size=(h0, w0), mode="nearest").type(torch.bool)
        painted_disps = infer_depth(self.depth_model, painted_tensors, return_rel_depth=True, **self.config)

        for painted_image, mask_image, painted_disp in zip(painted_images, mask_images, painted_disps):
            hmr_result = self.hmr_model(painted_image * 255, focal_length=focal_length)
            if len(hmr_result) <= 0 or hmr_result["pred_masks"].shape[0] <= init_human:
                continue

            if self.config.align:
                scale, offset = linear_regression(
                    painted_disp.flatten(1),
                    disp.flatten(1),
                    mask=(~mask_image * (painted_disp > 0) * (disp > 0)).flatten(1),
                )
                aligned_disp = scale * painted_disp + offset
                if scale <= 0:
                    continue
                if self.config.debug:
                    print("RDE scale", scale.item(), "RDE offset", offset.item())
            else:
                aligned_disp = painted_disp

            vertices = hmr_result["pred_vertices"] + hmr_result["pred_cam_t_full"][:, None]
            faces = torch.as_tensor(self.hmr_model.model.smpl.faces.astype(np.int64))
            faces = faces[None].repeat_interleave(vertices.shape[0], dim=0).to(vertices.device)
            meshes = Meshes(vertices, faces)
            rasterizer = MeshRasterizer(cameras=cameras[[0] * vertices.shape[0]], raster_settings=settings)
            fragments = rasterizer(meshes)
            raster_depth = fragments.zbuf[..., 0]
            raster_mask = erode(hmr_result["pred_masks"][:, None]).squeeze(1) * (raster_depth > 0)
            raster_depth, raster_mask = process_depth_mask(raster_depth, raster_mask)

            aligned_disps.append(aligned_disp)
            human_images.append(painted_image)
            human_depths.append(raster_depth)
            human_masks.append(raster_mask)
            curr_human += 1

            if self.config.debug:
                plt.imshow(painted_image)
                plt.axis("off")
                plt.show()

                renderer = Renderer(self.hmr_model.model_cfg, faces=self.hmr_model.model.smpl.faces)
                overlay = renderer.render_rgba_multiple(
                    hmr_result["pred_vertices"].cpu().numpy(),
                    hmr_result["pred_cam_t_full"].cpu().numpy(),
                    render_res=painted_image.shape[1::-1],
                    focal_length=float(focal_length),
                )
                rendered_image = overlay[..., :3] * overlay[..., 3:] + painted_image * (1 - overlay[..., 3:])
                plt.imshow(np.asarray(rendered_image, dtype=np.float32))
                plt.axis("off")
                plt.show()

                disp_image = aligned_disp.cpu().numpy().squeeze(0)
                disp_image = (disp_image - disp_image.min()) / (disp_image.max() - disp_image.min())
                plt.imshow(disp_image, cmap="gray")
                plt.axis("off")
                plt.show()

        return aligned_disps, human_images, human_depths, human_masks, curr_human
