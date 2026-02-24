import numpy as np
from typing import List, Dict
from PIL import Image

from .utils import adjust_depth_scale_inplace, move_along_direction_T, sympl_stage, add_message
from .vision_modules.vision_utils import transform_src_to_tgt, transform_tgt_to_top, calc_Rt

@sympl_stage
def estimate_scene_geometry(
    detection_module,
    depth_module,
    orientation_module,
    vlm_model,
    config,
    image: Image.Image, 
    objects_of_interest: List[str],
    ref_viewer: str,
    conv_history: list = None,
    **sympl_args,
):

    abstract_scene_dict = {
        'camera': {
            "position": np.array([0, 0, 0]),
            "orientation": np.array([0, 0, 1]),
        }
    }
    _, image_processed = detection_module.detection_process_image(image)

    # Run depth estimation
    depth_npy, K = depth_module.run_depth_intrinsic_estimation(image)
    for obj_name in objects_of_interest:

        # Run detection
        if obj_name == 'camera':
            continue

        boxes = detection_module.run_detection(image_processed, obj_name)
        if len(boxes) == 0:
            return None

        selected_idx = 0
        if config.detection.use_vlm_refinement:
            box2D, conv_history = detection_module.run_detection_refinement(
                vlm_model,
                image,
                obj_name,
                boxes_output=boxes,
                conv_history = conv_history,
                **sympl_args,
            )
        else:
            box2D = boxes[selected_idx]

        position = depth_module.unproject_to_3D_from_bbox(
            image=image,
            depth=depth_npy,
            bbox=box2D,
            K=K,
        )

        # Run orientation estimation
        if obj_name == ref_viewer:
            orientation, azimuth_deg, polar_deg = orientation_module.run_orientation_estimation(
                image=image,
                category=obj_name,
                bbox=box2D,
                **sympl_args,
            )
            if orientation is None:
                return None

            # Save the results
            abstract_scene_dict[obj_name] = {
                'position': position,
                'orientation': orientation,
            }

        else:
            abstract_scene_dict[obj_name] = {
                'position': position,
                'orientation': None,
            }

    if ref_viewer == "camera":
        abstract_scene_dict = adjust_depth_scale_inplace(abstract_scene_dict, T=10)

    abstract_scene_dict['camera']['intrinsic'] = K
    return abstract_scene_dict

@sympl_stage
def top_view_position(
    abstract_scene_src: Dict,
    ref_viewer: str,
):
    camera_position = abstract_scene_src['camera']['position']
    ref_position = abstract_scene_src[ref_viewer]['position']

    top_view_orientation = np.array([0, 1, 0])

    radius = 0
    if ref_viewer.lower() == 'camera':
        count=1e-6
        for key, value in abstract_scene_src.items():
            if key == 'camera':
                continue
            radius += value['position'][2]
            count+=1
        radius = int(abs(radius) / count)
    else:
        radius = np.linalg.norm(np.array(camera_position) - np.array(ref_position))

    top_view_position = move_along_direction_T(ref_position, -top_view_orientation, radius)

    top_view_abstract = {'position' : top_view_position, 'orientation' : top_view_orientation}

    return top_view_abstract

@sympl_stage  
def do_perspective_change(
    device_vision,
    abstract_scene_src: Dict,
    ref_viewer: str,
):

    origin_tgt = abstract_scene_src[ref_viewer]['position']
    z_axis_tgt = abstract_scene_src[ref_viewer]['orientation']

    camera_position = abstract_scene_src['camera']['position']
    radius = 0
    if ref_viewer.lower() == 'camera':
        count=1e-6
        for key, value in abstract_scene_src.items():
            if key == 'camera':
                continue
            radius += value['position'][2]
            count+=1
        radius = abs(radius) / count
    else:
        radius = np.linalg.norm(np.array(camera_position) - np.array(origin_tgt))

    abstract_scene_tgt = {
        'top_view': {
            'position': np.array([0, 0, 0]),
            'orientation': np.array([0, 0, 1]),
        }
    }
    up_vector_global = -abstract_scene_src['top_view']['orientation']

    position_top = np.array([0, radius, 0])
    orientation_ref = np.array([0, 0, 1])

    points_tgt = {}
    for obj_name, obj_dict in abstract_scene_src.items():
        if obj_name == 'top_view':
            continue

        # Get position and orientation in source perspective
        position_src = obj_dict['position']
        orientation_src = obj_dict['orientation']

        # Transform to target perspective
        position_tgt, orientation_tgt = transform_src_to_tgt(
            position_src, 
            orientation_src, 
            origin_tgt,
            z_axis_tgt,
            up_vector_global=up_vector_global
        )

        if obj_name == ref_viewer:
            orientation_ref = orientation_tgt
            position_top = position_tgt + position_top

        
        points_tgt[obj_name] = position_tgt
    
    R, t = calc_Rt(
        orientation_ref,
        position_top,
        device_vision,
    )

    for obj_name, obj_position in points_tgt.items():
        position_tgt_top = transform_tgt_to_top(
            obj_position, 
            R,
            t,
            device_vision,
        )
        # Store the results
        abstract_scene_tgt[obj_name] = {
            'position': position_tgt_top,
            'orientation' : None,
        }

    return abstract_scene_tgt

def calculate_2d_coordinates(
    points_cam: np.ndarray,
    K,
    image_size: tuple,
    ref_position: np.ndarray | list | tuple | None = None,
):
    def _parse_intrinsics_local(K, image_size):
        W, H = image_size
        if isinstance(K, (list, tuple)) and len(K) == 4:
            fx, fy, cx, cy = map(float, K)
            return fx, fy, cx, cy
        K_ = np.asarray(K, dtype=float)
        if K_.shape == (3, 3) or K_.shape == (4, 4):
            fx, fy = K_[0, 0], K_[1, 1]
            cx, cy = W * 0.5, H * 0.5
            return float(fx), float(fy), float(cx), float(cy)
        fx = fy = 0.5 * (W + H)
        cx, cy = 0.5 * W, 0.5 * H
        return float(fx), float(fy), float(cx), float(cy)

    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    assert pts.shape[0] > 0, "points_cam is empty."

    W, H = image_size
    fx, fy, cx, cy = _parse_intrinsics_local(K, image_size)

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(~np.isfinite(Z)):
        bad = np.where(~np.isfinite(Z))[0][:8].tolist()
        raise ValueError(f"Non-finite Z found. indices={bad}")

    z_min = float(np.min(Z))
    if z_min <= 0:
        z_shift = -z_min + 1e-6
        Z = Z + z_shift
        pts[:, 2] = Z

    u = cx + fx * (X / Z)
    v = cy + fy * (Y / Z)

    if ref_position is not None:
        rx, ry, rz = np.asarray(ref_position, dtype=float).reshape(3,)
        if np.isfinite(rz) and rz > 0:
            ref_uv = (cx + fx * (rx / rz), cy + fy * (ry / rz))
        else:
            ref_uv = None
    else:
        ref_uv = None

    if ref_uv is not None:
        dx = cx - float(ref_uv[0])
        dy = cy - float(ref_uv[1])
        u = u + dx
        v = v + dy
        ref_uv = (cx, cy)

    target_ratio = 1.0 / 2.5
    r_target = target_ratio * min(W, H)
    du = u - cx
    dv = v - cy
    
    r = np.sqrt(du * du + dv * dv)
    r_max = float(np.max(r)) if r.size > 0 else 0.0
    if r_max > 0:
        s = r_target / r_max
    else:
        s = 1.0
    u = cx + s * (u - cx)
    v = cy + s * (v - cy)

    cx_i, cy_i = int(round(cx)), int(round(cy))

    meta = {
        "scale_s": float(s),
        "cx": float(cx), "cy": float(cy),
        "cx_i": cx_i, "cy_i": cy_i,
        "fx": float(fx), "fy": float(fy),
    }
    
    return u, v, meta


@sympl_stage
def do_projection_2d(
    abstract_scene: Dict,
    ref_viewer: str,
    category: str,
):

    # 1. Minimum coordinate extraction for projection
    view = abstract_scene['camera'] if category.lower() == "above_below" else abstract_scene['top_view']

    ref_position = None
    obj_positions = []
    obj_name_list = []
    is_ref_viewer = False
    
    for name, data in view.items():
        if name in ['top_view', 'camera'] and name != ref_viewer:
            continue
        if name == ref_viewer:
            is_ref_viewer = True
            ref_position = data['position']
        else:
            obj_positions.append(data['position'])
            obj_name_list.append(name)
            
    obj_positions_np = np.asarray(obj_positions) if len(obj_positions) > 0 else np.array([])

    if not is_ref_viewer or len(obj_positions_np) == 0:
        return None, None, None, None, None, None, False

    # 2. Project to 2D
    W, H = 512, 512
    intrinsic = abstract_scene['camera']['intrinsic']
    u, v, meta = calculate_2d_coordinates(
        points_cam=obj_positions_np,
        K=intrinsic,
        image_size=(W, H),
        ref_position=ref_position,
    )
    
    return u, v, meta, obj_positions_np, ref_position, obj_name_list, is_ref_viewer
