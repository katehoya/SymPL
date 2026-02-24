import os
import cv2
import sys
sys.path.append("..")
import yaml
from typing import List
import numpy as np
from PIL import Image
import torch
from box import Box
import open3d as o3d
from scipy.stats import mode

# Import modules
import depth_pro
from depth_pro.depth_pro import DepthProConfig
from .vision_utils import resolve_path

class DepthModule:
    def __init__(self, config, device="cuda"):
        self.device = device
        self.config = config

        # Load Depth Pro
        self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
            config=DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=resolve_path(config.depth.ckpt_path),
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            ),
            device=torch.device(self.device),
            precision=torch.float16
        )
        self.depth_model.eval()

    def depth_process_image(
        self,
        image: Image.Image,
        auto_rotate: bool = True, 
        remove_alpha: bool = True
    ):
        '''
        Transform PIL image to depth_pro format
        (modified from depth_pro/utils.py)
        '''
        img_exif = depth_pro.utils.extract_exif(image)
        icc_profile = image.info.get("icc_profile", None)

        # Rotate the image.
        if auto_rotate:
            exif_orientation = img_exif.get("Orientation", 1)
            if exif_orientation == 3:
                image = image.transpose(Image.ROTATE_180)
            elif exif_orientation == 6:
                image = image.transpose(Image.ROTATE_270)
            elif exif_orientation == 8:
                image = image.transpose(Image.ROTATE_90)


        # Convert to numpy array
        image_npy = np.array(image)

        # Convert to RGB if single channel.
        if image_npy.ndim < 3 or image_npy.shape[2] == 1:
            image_npy = np.dstack((image_npy, image_npy, image_npy))

        if remove_alpha:
            image_npy = image_npy[:, :, :3]

        # Extract the focal length from exif data.
        f_35mm = img_exif.get(
            "FocalLengthIn35mmFilm",
            img_exif.get(
                "FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)
            ),
        )
        if f_35mm is not None and f_35mm > 0:
            f_px = depth_pro.utils.fpx_from_f35(image_npy.shape[1], image_npy.shape[0], f_35mm)
        else:
            f_px = None
        
        return image_npy, icc_profile, f_px

    def make_bbox_corners(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        dx: float,
        dy: float,
        dz: float,
    ):
        '''
        Make 8 corners of a bounding box
        '''
        corners_flag = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]

        corners = []
        for flag in corners_flag:
            c = np.array([x_min, y_min, z_min]) + \
                np.array(flag) * \
                np.array([dx, dy, dz])
            corners.append(c)

        return np.array(corners)
    
    def box3D_get_dims(self, bbox3D):
        # Modified from https://github.com/UVA-Computer-Vision-Lab/ovmono3d/blob/main/tools/ovmono3d_geo.py
        x = np.sqrt(np.sum((bbox3D[0] - bbox3D[1]) * (bbox3D[0] - bbox3D[1])))
        y = np.sqrt(np.sum((bbox3D[0] - bbox3D[3]) * (bbox3D[0] - bbox3D[3])))
        z = np.sqrt(np.sum((bbox3D[0] - bbox3D[4]) * (bbox3D[0] - bbox3D[4])))
        
        return np.array([z, y, x])

    def box3D_get_pose(self, bbox3d_a, bbox3d_b):
        # Modified from https://github.com/UVA-Computer-Vision-Lab/ovmono3d/blob/main/tools/ovmono3d_geo.py
        center = np.mean(bbox3d_a, axis=0)
        dim_a = self.box3D_get_dims(bbox3d_a)
        dim_b = self.box3D_get_dims(bbox3d_b)
        bbox3d_a -= center
        bbox3d_b -= center
        U, _, Vt = np.linalg.svd(bbox3d_a.T @ bbox3d_b, full_matrices=True)
        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R
    
    def run_depth_estimation(
        self,
        image: Image.Image,
    ):
        '''
        Run depth estimation
        '''
        # Process image
        image, _, f_px = self.depth_process_image(image)
        image = self.depth_transform(image)

        # Run Depth Pro
        depth_pred = self.depth_model.infer(image, f_px=f_px)
        depth = depth_pred["depth"]
        depth_npy = depth.cpu().numpy().astype(np.float32)

        return depth_npy
    
    
    def run_depth_intrinsic_estimation(
        self,
        image: Image.Image,
    ):
        '''
        Run depth estimation
        '''
        # Process image
        W, H = image.size[:2]

        image, _, f_px = self.depth_process_image(image)
        image = self.depth_transform(image)

        # Run Depth Pro
        depth_pred = self.depth_model.infer(image, f_px=f_px)
        depth = depth_pred["depth"]
        focallength=depth_pred["focallength_px"].cpu().numpy().astype(np.float32)
        depth_npy = depth.cpu().numpy().astype(np.float32)

        K = np.array([[focallength, 0.0, W/2.0],
              [0.0, focallength, H/2.0],
              [0.0, 0.0, 1.0]], dtype=float)

        return depth_npy, K


    def unproject_to_3D(
        self,
        image: Image.Image,
        depth: np.array,
        segment_mask: np.array,
        K: np.array,
        return_box3D: bool = False,    # whether to return 3D bbox (for visualization)
    ):
        '''
        Using detection + segmentation + depth results, unproject each object to 3D points.
        
        NOTE: Modified from 3D bbox extraction pipeline in Ovmono3D [Yao et al., 2024]
        (https://github.com/UVA-Computer-Vision-Lab/ovmono3d/blob/main/tools/ovmono3d_geo.py)
        '''

        # Set camera intrinsic matrix (K)
        image_w, image_h = image.size
        focal_len_ndc = 4.0
        focal_len = focal_len_ndc * image_w
        px, py = image_w / 2, image_h / 2

        # Convert to Open3D format
        depth_o3d = o3d.geometry.Image(depth)
        image_o3d = o3d.geometry.Image(np.array(image).astype(np.uint8))

        # Unproject (Open3D)
        depth_unproj = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=image_o3d,
            depth=depth_o3d,
            depth_scale=1.0,
            depth_trunc=1000.0,
            convert_rgb_to_intensity=False,
        ).depth
        depth_unproj = np.array(depth_unproj)
        
        # Filter out points outside the mask
        ys, xs = np.where(segment_mask > 0.5)
        depth_values = []
        for y, x in zip(ys, xs):
            z = depth_unproj[y, x]
            depth_values.append(z)

        # Find the most frequent depth
        depth_values = np.array(depth_values)
        mode_depth, _ = mode(depth_values, keepdims=True)   # scipy mode function

        # Set a depth threshold (e.g., keep points within ±10% of the mode)
        threshold = 0.1 * mode_depth[0]
        depth_values_filtered = (
            depth_values >= mode_depth[0] - threshold
        ) & (
            depth_values <= mode_depth[0] + threshold
        )

        # Unproject only filtered points
        points_filtered = []
        for i, (y, x) in enumerate(zip(ys, xs)):
            z = depth_unproj[y, x]
            if depth_values_filtered[i]:
                x_3D = z * (x - K[0, 2]) / K[0, 0]
                y_3D = z * (y - K[1, 2]) / K[1, 1]
                points_filtered.append([x_3D, y_3D, z])  # flip
        
        # Compute final 3D coordinates as the median of filtered points
        points_filtered = np.array(points_filtered)
        
        med_x, med_y, med_z = np.median(points_filtered, axis=0)

        pos3D = np.array([med_x, med_y, med_z])
        return pos3D

    import numpy as np
    import open3d as o3d

    def unproject_to_3D_from_bbox(
        self,
        image: Image.Image,                 # PIL.Image
        depth: np.ndarray,     # HxW depth (same scale as unproject step)
        bbox,                  # (x1, y1, x2, y2) in pixel coords
        K: np.ndarray,         # 3x3 intrinsics
        use_center_weight=True,
        depth_band="relative", # "relative" (±p%), or "mad" (±k*MAD)
        band_param=0.12,       # if relative: p=0.12 (±12%); if mad: k=2.5
    ):
        # Convert to Open3D format
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        image_o3d = o3d.geometry.Image(np.array(image).astype(np.uint8))
        depth_unproj = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=image_o3d,
            depth=depth_o3d,
            depth_scale=1.0,
            depth_trunc=1000.0,
            convert_rgb_to_intensity=False,
        ).depth
        depth_unproj = np.asarray(depth_unproj)

        H, W = depth_unproj.shape
        x1, y1, x2, y2 = map(int, bbox)
        x1 = np.clip(x1, 0, W-1); x2 = np.clip(x2, 0, W-1)
        y1 = np.clip(y1, 0, H-1); y2 = np.clip(y2, 0, H-1)

        ys, xs = np.mgrid[y1:y2+1, x1:x2+1]
        zs = depth_unproj[ys, xs]

        # Keep only valid depth (remove 0 or NaN)
        valid = np.isfinite(zs) & (zs > 0)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        if len(zs) == 0:
            return np.array([np.nan, np.nan, np.nan])

        # 1) Estimate depth mode (most frequent bin in histogram)
        bins = max(20, int(np.sqrt(len(zs))))  # appropriate number of bins
        hist, bin_edges = np.histogram(zs, bins=bins)
        mode_bin_idx = np.argmax(hist)
        z_left, z_right = bin_edges[mode_bin_idx], bin_edges[mode_bin_idx+1]
        z_mode = 0.5*(z_left + z_right)

        # 2) Select depth band
        if depth_band == "relative":
            thr = band_param * z_mode
            keep = (zs >= z_mode - thr) & (zs <= z_mode + thr)
        else:  # "mad"
            med_z = np.median(zs)
            mad = np.median(np.abs(zs - med_z)) + 1e-6
            k = band_param
            keep = (zs >= med_z - k*mad) & (zs <= med_z + k*mad)

        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        if len(zs) == 0:
            return np.array([np.nan, np.nan, np.nan])

        # (Optional) 3) Center weighting
        if use_center_weight:
            cx = 0.5*(x1 + x2); cy = 0.5*(y1 + y2)
            # Distance normalized by bbox diagonal length
            diag = np.hypot((x2-x1), (y2-y1)) + 1e-6
            d = np.hypot(xs - cx, ys - cy) / diag
            # Weight increases as it gets closer to the center (e.g., Gaussian bell)
            w = np.exp(- (d**2) / (2*(0.25**2)))  # sigma ~ 0.25 of diag
        else:
            w = np.ones_like(zs, dtype=np.float32)

        # 4) Robust center after back-projection (approximation close to weighted median)
        # Using weighted average with w instead of weighted quantile, stable since outliers are few
        X = zs * (xs - K[0, 2]) / K[0, 0]
        Y = zs * (ys - K[1, 2]) / K[1, 1]
        # Weighted median is ideal, but simply using weighted average + median mixture as backup
        w = w / (w.sum() + 1e-6)
        xw = (X * w).sum(); yw = (Y * w).sum(); zw = (zs * w).sum()

        # Backup: median calculation for interpolation (stabilize when outliers remain)
        x_med, y_med, z_med = np.median(X), np.median(Y), np.median(zs)
        alpha = 0.3  # 70% weighted average, 30% median mixture
        x_final = (1-alpha)*xw + alpha*x_med
        y_final = (1-alpha)*yw + alpha*y_med
        z_final = (1-alpha)*zw + alpha*z_med

        
        return np.array([x_final, y_final, z_final])
        #return np.array([x_med, y_med, z_med])

    def unproject_to_3D_center(
        self,
        image: Image.Image,
        depth: np.array,
        segment_mask: np.array,
        focal_len: np.float32,
    ):
        """
        Calculate 3D center coordinates in camera coordinate system using mask + depth
        """
        import numpy as np
        from scipy.stats import mode
        
        image_w, image_h = image.size
    
        # focal_len_ndc = 4.0
        # focal_len = focal_len_ndc * image_w
        cx, cy = image_w / 2, image_h / 2

        # Pixel coordinates of mask area
        ys, xs = np.where(segment_mask > 0.5)

        # Extract depth values
        depth_values = depth[ys, xs]

        # Representative depth (mode)
        mode_depth, _ = mode(depth_values, keepdims=True)
        threshold = 0.1 * mode_depth[0]

        # Use only pixels within mode ±10% band
        valid = (depth_values >= mode_depth[0] - threshold) & \
                (depth_values <= mode_depth[0] + threshold)
        xs, ys, depth_values = xs[valid], ys[valid], depth_values[valid]

        # Unprojection
        X = depth_values * (xs - cx) / focal_len
        Y = depth_values * (ys - cy) / focal_len
        Z = depth_values

        # Coordinate system flip (fit to camera)
        points_3D = np.stack([X, -Y, -Z], axis=1)

        # Median = representative center coordinates
        pos3D = np.median(points_3D, axis=0)

        return pos3D
