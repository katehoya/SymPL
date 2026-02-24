import os
import io
import sys
import cv2
import matplotlib.pyplot as plt
sys.path.append("..")
import yaml
from typing import List
import numpy as np
from PIL import Image
import torch
from box import Box
import open3d as o3d
from .vision_utils import *
from ..utils import add_message

# Import modules
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as GT
import groundingdino.config.GroundingDINO_SwinT_OGC as groundingdino_config
from segment_anything import SamPredictor, sam_model_registry


class DetectionModule:
    def __init__(
            self, 
            config, 
            device="cuda",
        ):
        self.device = device
        self.config = config
        # Detection parameters
        self.box_threshold = config.detection.box_threshold
        self.text_threshold = config.detection.text_threshold
        self.num_candidates = config.detection.num_candidates

        # Load Grounding DINO
        config_path = groundingdino_config.__file__

        # Resolve checkpoint path robustly
        ckpt_path = resolve_path(config.detection.ckpt_path)

        # Load model
        self.detection_model = load_model(config_path, ckpt_path)
        self.detection_model.to(self.device)


    def detection_process_image(
        self, 
        image: Image.Image,
    ):

        transform = GT.Compose(
            [
                # GT.RandomResize([800], max_size=1333),
                GT.ToTensor(),
                GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_npy = np.asarray(image)
        image_transformed, _ = transform(image, None)

        return image_npy, image_transformed
    
    def run_detection(
        self, 
        image: torch.Tensor,            # after detection_process_image()
        category: str,                # category to detect
    ):

        # print(f"* [INFO] Running detection for {category}...")

        boxes, scores, _ = predict(
            model=self.detection_model,
            image=image,
            caption=category,
            box_threshold=self.config.detection.box_threshold,
            text_threshold=self.config.detection.text_threshold,
        )
        # Get detections with highest scores
        boxes_sorted = sort_by_scores(scores, [boxes])[0]
        boxes_output = boxes_sorted[:self.config.detection.num_candidates]

        return boxes_output
   



    def visualize_crops_with_original(
        self,
        original_img_arr: np.ndarray,
        crop_arr_list: List[np.ndarray],
        max_cols: int = 4,
        fontsize: int = 14,
    ):

        # Full panel image list: [original] + [each crop]
        all_imgs = [original_img_arr] + crop_arr_list

        n = len(all_imgs)
        if n == 0:
            fig = plt.figure()
            return fig

        cols = min(max_cols, n)
        rows = int(np.ceil(n / cols))

        fig_w = cols * 3
        fig_h = rows * 3
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        axes_flat = axes.flatten()

        for slot_idx, (ax, img_arr) in enumerate(zip(axes_flat, all_imgs)):
            ax.imshow(img_arr)
            ax.axis("off")

            if slot_idx > 0:
                # slot_idx-1 is the index in crop_arr_list
                disp_idx = slot_idx - 1  # starting from 0
                ax.text(
                    5, 15,
                    str(disp_idx),
                    color="white",
                    fontsize=fontsize,
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, pad=3),
                )

        # clear remaining axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.tight_layout()
        return fig


    def run_detection_refinement_new(
        self,
        vlm_model,          # VLM model (defined in sympl_pipeline.py)
        image: Image.Image,
        category: str,
        boxes_output: List[List[int]],
        **sympl_args,
    ):

        W, H = image.size

        # relative bbox -> absolute bbox [xmin,ymin,xmax,ymax]
        boxesAbs = [
            [
                int(cxcywh_to_xyxy(box)[0] * W),  # xmin
                int(cxcywh_to_xyxy(box)[1] * H),  # ymin
                int(cxcywh_to_xyxy(box)[2] * W),  # xmax
                int(cxcywh_to_xyxy(box)[3] * H),  # ymax
            ]
            for box in boxes_output
        ]

        # Extract crops
        cropped_images = [
            image.crop((box[0], box[1], box[2], box[3]))
            for box in boxesAbs
        ]
        cropped_images = [np.asarray(crop) for crop in cropped_images]

        # Keep only valid crops
        valid_crops = []
        valid_boxes = []
        for arr, b in zip(cropped_images, boxesAbs):
            if arr.shape[0] > 0 and arr.shape[1] > 0:
                valid_crops.append(arr)
                valid_boxes.append(b)

        cropped_images = valid_crops
        boxesAbs = valid_boxes
        # ================= Visualize candidate crops in existing way from here =================
        fig = visualize_crops(cropped_images)  # Use as is (function that adds indices 0,1,2,...)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        crops_grid = Image.open(buf).convert("RGB")
        plt.close(fig)

        # ================= Add original image only on the leftmost (no label) =================
        # Scale the original image vertically to match crops_grid height and concatenate horizontally.
        grid_h = crops_grid.height
        # Resize full original image to match grid height while maintaining aspect ratio
        orig_w_for_grid = int(image.width * (grid_h / image.height))
        original_resized = image.resize((orig_w_for_grid, grid_h))

        # New canvas: [original (no label)] + [existing crops_grid (maintaining 0,1,2,... labels)]
        combined_w = original_resized.width + crops_grid.width
        combined_h = grid_h
        combined = Image.new("RGB", (combined_w, combined_h), (0, 0, 0))
        combined.paste(original_resized, (0, 0))
        combined.paste(crops_grid, (original_resized.width, 0))

        # This combined image is the final candidate image for the VLM
        candidate_image = combined

        # Downscale as before
        cW, cH = candidate_image.size
        candidate_image = candidate_image.resize((cW // 3, cH // 3))

        # Save (trace)
        if sympl_args.get('trace_save_dir', None) is not None:
            candidate_image.save(
                os.path.join(sympl_args['trace_save_dir'], f"detection_candidates_{category}.png")
            )

        # Update prompt
        '''
        prompt_refinement = f"""
        The first (top-left) panel is the full original image. It is NOT a candidate and has no index.
        Use this original image to understand the overall appearance of the object described as '{category}' — including its shape, color, texture, and approximate location/context.

        Each following panel is a cropped candidate region that may contain the object described as '{category}'.
        Each cropped candidate has a white number label like 0, 1, 2, ...

        Compare each cropped candidate to the object in the original image and select the single cropped candidate that best matches '{category}' in terms of overall visual appearance.

        Answer ONLY with that number (for example: 0 or 1 or 2). Do not include any other text.
        """
        '''
        prompt_refinement = f"""
        The leftmost panel is the full original image. It is NOT a candidate and does not have an index.

        To decide which detection is correct for '{category}', first look at the original image (leftmost) to understand how the target object actually looks in context (shape, color, texture, distinctive parts, etc.).

        To the right of the original image are the cropped candidate regions.
        Each candidate region is labeled with an index number above each cropped image.
        The index number starts at 1 and increases in the order of the candidates. 

        Select the single cropped candidate that best matches '{category}'.
        Answer ONLY the correct index number. Do not include any other text.

        """

        # Compose VLM input messages
        messages = add_message(
            [],
            role="user",
            text=prompt_refinement,
            image=candidate_image
        )

        # Run VLM
        response = vlm_model.process_messages(messages, max_new_tokens=512)
        #print("CHECK", response)

        # Parse selected index from VLM response
        # Now index is the crop's own index (starting from 0) = index of boxesAbs
        selected_idx = 0  # fallback
        max_candidate_idx = min(self.num_candidates, len(boxesAbs)) - 1
        for i in range(0, max_candidate_idx + 1):
            if str(i+1) in str(response):
                selected_idx = i
                break

        # Return the final selected bbox
        return boxesAbs[selected_idx], selected_idx+1, candidate_image



    def run_detection_refinement(
        self,
        vlm_model,
        image: Image.Image,
        category: str,
        boxes_output: List[List[int]],
        conv_history: list = None,
        **sympl_args,
    ):

        W, H = image.size

        MARGIN = 30  # pixels to expand on each side

        def enlarge_and_clip_box(xmin, ymin, xmax, ymax, img_w, img_h, margin):
            """
            Enlarge the box by 'margin' pixels on all sides, then clip to image bounds.
            Returns (xmin2, ymin2, xmax2, ymax2) as integers.
            """
            xmin2 = max(0, xmin - margin)
            ymin2 = max(0, ymin - margin)
            xmax2 = min(img_w, xmax + margin)
            ymax2 = min(img_h, ymax + margin)
            return int(xmin2), int(ymin2), int(xmax2), int(ymax2)


        # Convert relative boxes to absolute boxes
        boxesAbs = [
            [
                int(cxcywh_to_xyxy(box)[0] * W),  # xmin
                int(cxcywh_to_xyxy(box)[1] * H),  # ymin
                int(cxcywh_to_xyxy(box)[2] * W),  # xmax
                int(cxcywh_to_xyxy(box)[3] * H),  # ymax
            ]
            for box in boxes_output
        ]


        boxesAbs_enlarged = []
        for box in boxesAbs:
            xmin, ymin, xmax, ymax = box
            xmin2, ymin2, xmax2, ymax2 = enlarge_and_clip_box(
                xmin, ymin, xmax, ymax,
                W, H,
                MARGIN
            )
            boxesAbs_enlarged.append([xmin2, ymin2, xmax2, ymax2])


        # Get list of cropped images
        cropped_images = [
            image.crop((box[0], box[1], box[2], box[3]))
            #for box in boxesAbs
            for box in boxesAbs_enlarged

        ]
        cropped_images = [
            np.asarray(crop) for crop in cropped_images
        ]

        valid_crops = []
        for arr in cropped_images:
            if arr.shape[0] > 0 and arr.shape[1] > 0:
                valid_crops.append(arr)

        cropped_images = valid_crops

        # Make a grid of cropped images (for input to VLM)
        fig = visualize_crops(cropped_images)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        candidate_image = Image.open(buf).convert("RGB")
        plt.close()
        cW, cH = candidate_image.size
        candidate_image = candidate_image.resize((cW // 3, cH // 3))

        # Save for visualization
        if sympl_args['trace_save_dir'] is not None:
            candidate_image.save(
                os.path.join(sympl_args['trace_save_dir'], f"detection_candidates_{category}.png")
            )

        prompt_refinement = f"""
        The input images are cropped regions from the original image that correspond to the description: '{category}'.
        Examine each image and select the one that best matches the description: '{category}'.

        Note: The cropped images may have been upscaled and can appear less sharp. Consider potential blur or artifacts.
        Choose the most appropriate image based on robust visual/appearance cues (e.g., overall shape, distinctive parts, color pattern, relative proportions), rather than fine details that may be lost due to upscaling.

        Your response must contain only the index number of the selected image.
        
        Note : If multiple images are considered a match, select the one with the lowest index number.
        """
        '''
        

        prompt_refinement = f"""
        The input images are the cropped regions from the original image that correspond to description : '{category}'.
        Look at each of these images and select the one that best matches description : '{category}'.
        Your response should return only the index number of the image you selected.
        Note : If multiple images are considered a match, select the one with the lowest index number.
        """
        # Make conversation
        messages = add_message(
            [], role="user", 
            text=prompt_refinement, image=candidate_image
        )

        # Query VLM
        response = vlm_model.process_messages(messages, max_new_tokens=512)

        if conv_history is not None:
            conv_history += [
                {'text': category, 'image': None},
                {'text': response, 'image': None},
            ]

        # Get the selected index
        selected_idx = 0
        for i in range(min(self.num_candidates, len(boxesAbs))):
            if str(i) in str(response):
                selected_idx = i
                break

        # Return the selected detection for category
        return boxesAbs[selected_idx], conv_history

    def run_segmentation(
        self, 
        image: Image.Image,
        box2d: List[int]       # detected 2D bbox
    ):

        # Convert to BGR numpy array
        image_npy = np.array(image)
        image_bgr = cv2.cvtColor(image_npy, cv2.COLOR_RGB2BGR)

        self.segmentation_predictor.set_image(image_bgr, image_format="BGR")
        box2d = np.array(box2d)
        masks, _, _ = self.segmentation_predictor.predict(box=box2d)
        
        return masks
