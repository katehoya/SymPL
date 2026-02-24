import numpy as np
from typing import List, Dict
from .utils import sympl_stage

@sympl_stage
def do_abstraction(
    u: np.ndarray,
    v: np.ndarray,
    meta: dict,
    obj_positions_np: np.ndarray,
    ref_position: np.ndarray,
    obj_name_list: List[str],
    is_ref_viewer: bool,
):
    color_values = [
        [255,   0,   0], # red
        [  0,   0, 255], # blue
        [  0, 255,   0], # green
        [255, 255, 255]  # white
    ]
    color_names = ['red', 'blue', 'green', 'white']
    ref_color_name = 'yellow'

    obj_color_dict = {'ref': ref_color_name}
    colors_list = []
    
    for i, name in enumerate(obj_name_list):
        # We assume obj_name_list length matches the color palette provided
        c_idx = i % len(color_names)
        obj_color_dict[name] = color_names[c_idx]
        colors_list.append(color_values[c_idx])
            
    colors_np = np.asarray(colors_list) if len(colors_list) > 0 else np.array([])
    
    return u, v, meta, obj_positions_np, colors_np, obj_name_list, obj_color_dict, ref_position, is_ref_viewer
