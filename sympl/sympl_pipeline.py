import re
import sys
import os
from typing import List, Dict, Tuple, Any, Optional
import importlib
from PIL import Image, ImageDraw
import numpy as np

# Add external module paths
sympl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(sympl_root, "sympl/vision_modules/src/orient_anything"))
sys.path.append(os.path.join(sympl_root, "sympl/vision_modules/src/GroundingDINO"))

# Import vision modules
from .vision_modules.vision_utils import cxcywh_to_xyxy, transform_src_to_tgt, transform_tgt_to_top, calc_Rt, vec_S_to_T
from .vision_modules import DetectionModule, DepthModule, OrientationModule
from .prompts import PromptParser

from .prompt_extraction import get_objects_of_interest, get_reference_viewer, select_category_from_question
from .projection import estimate_scene_geometry, top_view_position, do_perspective_change, do_projection_2d
from .abstraction import do_abstraction
from .bipartition import do_bipartition
from .localization import do_localization
from .utils import symbolic_layout_prompting


def get_vlm_model_class(vlm_type: str):
    try:
        from .vlms import VLM_MODELS
        module_name, class_name = VLM_MODELS[vlm_type]
    except KeyError:
        raise NotImplementedError(f"Unknown VLM type: {vlm_type}")

    full_module_path = f"sympl.vlms.{module_name}"
    module = importlib.import_module(full_module_path)
    
    return getattr(module, class_name)

# Define the sympl pipeline
class SymPL:
    def __init__(
        self,
        config,
        device_vlm: str = 'cuda',
        device_vision: str = 'cuda',
    ):
        self.config = config
        self.device_vlm = device_vlm
        self.device_vision = device_vision
        
        # Import VLM model class
        VLMModel = get_vlm_model_class(config.vlm.type)

        # Initialize VLM model
        self.vlm_model = VLMModel(config.vlm, device_vlm)

        # Initialize vision modules
        self.detection_module = DetectionModule(config, device_vision)
        self.depth_module = DepthModule(config, device_vision)
        self.orientation_module = OrientationModule(config, device_vision)

        # Load instruction parser
        self.prompt_parser = PromptParser(config)
    
    def run_sympl(
        self, 
        image: Image.Image,
        prompt: str,
        reasoning_category: str = None,
        trace_save_dir: str = 'outputs',
        visualize_trace: bool = True,
        return_conv_history: bool = False,
    ):
        sympl_args = {
            'trace_save_dir': trace_save_dir,
            'visualize_trace': visualize_trace,
        }

        assert sympl_args['trace_save_dir'] is not None, "trace_save_dir is required"
        conv_history = [] if return_conv_history else None

        # Stage 1: Spatial Information Extraction
        objs_of_interest, conv_history = get_objects_of_interest(self.vlm_model, self.prompt_parser, image, prompt, conv_history=conv_history)
        if objs_of_interest == None:
            return "fail : cannot find refered objects", conv_history

        ref_viewer, objs_of_interest, conv_history = get_reference_viewer(
            self.vlm_model, self.prompt_parser, image, prompt, 
            objects_of_interest=objs_of_interest, conv_history=conv_history,
        )
        if ref_viewer not in objs_of_interest:
            return "fail : reference viewer not in the list", conv_history


        if reasoning_category is None:
            reasoning_category = select_category_from_question(self.vlm_model, self.prompt_parser, prompt)
        if reasoning_category is None:
            return "fail : reasoning category detection fail", conv_history

        # Stage 2: Projection
        abstract_result = estimate_scene_geometry(
            detection_module=self.detection_module,
            depth_module=self.depth_module,
            orientation_module=self.orientation_module,
            vlm_model=self.vlm_model,
            config=self.config,
            image=image, 
            objects_of_interest=objs_of_interest,
            ref_viewer=ref_viewer,
            conv_history=conv_history,
            **sympl_args,
        )
        if abstract_result == None:
            return "fail : object detection fail", conv_history

        abstract_scene_dict = {'camera': abstract_result}
        abstract_scene_dict['camera']['top_view'] = top_view_position(
            abstract_scene_src=abstract_scene_dict['camera'],
            ref_viewer=ref_viewer,
        )
        abstract_scene_dict['top_view'] = do_perspective_change(
            device_vision=self.device_vision,
            abstract_scene_src=abstract_scene_dict['camera'],
            ref_viewer=ref_viewer,
        )

        u, v, meta, obj_positions_np, ref_position, obj_name_list, is_ref_viewer = do_projection_2d(
            abstract_scene=abstract_scene_dict,
            ref_viewer=ref_viewer,
            category=reasoning_category,
        )

        if not is_ref_viewer:
            return "fail : cannot find reference viewer", conv_history
        if u is None:
            return "fail : only reference viewer detected", conv_history

        # Stage 3: Abstraction
        u, v, meta, obj_positions_np, colors_np, obj_name_list, obj_color_dict, ref_position, is_ref_viewer = do_abstraction(
            u, v, meta, obj_positions_np, ref_position, obj_name_list, is_ref_viewer
        )

        # Stage 4: Bipartition
        partition_specs, u_adj, v_adj = do_bipartition(
            u=u, v=v, colors_np=colors_np, meta=meta,
            category=reasoning_category, ref_color_name=obj_color_dict['ref'],
        )

        # Stage 5: Localization (Render)
        simple_img = do_localization(
            u=u_adj, v=v_adj,
            colors_np=colors_np,
            partition_specs=partition_specs,
        )

        # Stage 6: Symbolic Layout Prompting
        response_sympl = symbolic_layout_prompting(
            vlm_model=self.vlm_model,
            prompt_parser=self.prompt_parser,
            category=reasoning_category,
            prompt=prompt,
            simple_img=simple_img,
            obj_color_dict=obj_color_dict,
            conv_history=conv_history,
            **sympl_args,
        )

        # Save trace if requested
        if visualize_trace and trace_save_dir:
            simple_img.save(os.path.join(trace_save_dir, "simple_img.png"))

        return response_sympl, conv_history