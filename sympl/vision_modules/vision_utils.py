import cv2
import numpy as np
import torch
import math

import matplotlib.pyplot as plt
from pytorch3d.renderer import look_at_view_transform
import os

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(get_project_root(), path)

def cxcywh_to_xyxy(box):
    #Convert bounding box from 'cxcywh' format to 'xyxy' format.

    cx, cy, w, h = box
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    
    return (xmin, ymin, xmax, ymax)

def convert_pil_to_cv2(pil_image):
    # Convert PIL Image to NumPy array
    img_array = np.array(pil_image)

    # Convert RGB to BGR (OpenCV uses BGR)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return img_bgr

def sort_by_scores(scores, preds):
    #Sort predictions by scores in descending order
    scores = [score.cpu().item() for score in scores]
    scores_indices = np.argsort(-np.array(scores))

    if not scores_indices.tolist() == list(range(len(scores))):
        # print("* [INFO] Sorting by scores in descending order")
        preds = [[t[k] for k in scores_indices] for t in preds]

    return preds

def visualize_crops(crops):
    #Visualize crops as a grid for detection refinement with VLM
    num_crops = len(crops)

    fig, axes = plt.subplots(
        1, num_crops, figsize=(num_crops * 10, 10), squeeze=False
    )

    for i in range(num_crops):
        axes[0, i].imshow(crops[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{i}", fontsize=100)
    plt.tight_layout()

    return fig


def transform_src_to_tgt(
    # Source point
    position_src: np.array,
    orientation_src: np.array,
    # Target perspective (New origin, +z axis)
    origin_tgt: np.array,
    z_axis_tgt: np.array,
    up_vector_global: np.array = np.array([0, -1, 0]),
):

    # Translation source point to origin
    position_src_translated = position_src - origin_tgt

    # Calculate right, up vectors for camera coord. system
    forward_vector = z_axis_tgt / np.linalg.norm(z_axis_tgt)
    
    up_vector = up_vector_global / np.linalg.norm(up_vector_global)
    right_vector = np.cross(forward_vector, up_vector)

    right_vector = right_vector / np.linalg.norm(right_vector)

    rotation_matrix = np.vstack(
        (right_vector, up_vector, forward_vector)
    )

    # Trasform source point to target coord. system (perspective)
    position_src_transformed = rotation_matrix @ position_src_translated

    # Round to 3 decimal places
    position_src_transformed = np.round(position_src_transformed, 3)

    if orientation_src is None:
        return position_src_transformed, None
    orientation_src_transformed = rotation_matrix @ orientation_src
    orientation_src_transformed = np.round(orientation_src_transformed, 3)

    return position_src_transformed, orientation_src_transformed


def normalize_positions(
    positions,
    *,
    render_whole_scene: bool = True,
    cam_fov_deg: float = 60.0,
    box_size: float = 0.0,
    grid_size: float = 1.0,
    min_depth: float = 0.0,
    max_depth: float = 1.0,
):

    positions = [list(map(float, p)) for p in positions]
    if len(positions) == 0:
        return []

    if render_whole_scene:
        z_list = [p[2] for p in positions]
        max_z_obj_pos = positions[int(np.argmax(z_list))]
        max_z = max_z_obj_pos[2]
        if max_z >= 0:
            dist_xy = float(np.linalg.norm(max_z_obj_pos[:2]))
            z_trans = max_z + dist_xy * math.tan(math.radians(cam_fov_deg / 2.0))
            z_trans += box_size
            for i, (x, y, z) in enumerate(positions):
                positions[i] = [x, y, z - z_trans]

    # XY Scaling
    max_x = max(abs(p[0]) for p in positions)
    max_y = max(abs(p[1]) for p in positions)
    max_xy = max(max_x, max_y)
    xy_scale = (grid_size / max_xy) if max_xy > 0 else 1.0

    # Z Normailzation
    z_vals = [p[2] for p in positions]
    min_z, max_z = min(z_vals), max(z_vals)
    z_offset = max_z
    if len(positions) == 1:
        z_scale = 1.0
    else:
        denom = abs(min_z - max_z) + 1e-6
        z_scale = (max_depth - min_depth) / denom
        if z_scale <= 0:
            z_scale = 1.0

    norm_positions = []
    for (x, y, z) in positions:
        nx = x * xy_scale
        ny = y * xy_scale
        nz = (z - z_offset) * z_scale - min_depth
        norm_positions.append([nx, ny, nz])

    return norm_positions

def vec_S_to_T(v_S: np.ndarray) -> np.ndarray:
    # Convert a vector from source coordinate system to target coordinate system
    v_S = np.asarray(v_S, dtype=float).reshape(3,)
    x, y, z = v_S
    return np.array([x, -y, -z], dtype=float)

def calc_Rt(
    orientation_ref: np.array,
    position_top: np.array,
    device: str = 'cuda',
):
    eye = torch.tensor(position_top, dtype=torch.float32, device=device).reshape(1, 3)
    at = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32, device=device).reshape(1, 3)
    up = torch.tensor(orientation_ref, dtype=torch.float32, device=device).reshape(1, 3)

    #print(eye, at, up)
    R, t = look_at_view_transform(
        eye=eye,
        at=at,
        up=up,
        device=device
    )

    return R, t

def transform_tgt_to_top(
    position_tgt: np.array,
    R: torch.tensor,
    t: torch.tensor,
    device: str = 'cuda',
):

    position_tgt_tensor = torch.tensor(position_tgt, dtype=torch.float32, device=device)[None, :]  # (1,3)

    # world -> camera
    position_tgt_top_tensor = position_tgt_tensor @ R.transpose(1, 2) + t[:, None, :]  # (1,1,3)

    position_tgt_top = position_tgt_top_tensor[0, 0].cpu().numpy()
    position_tgt_top = np.round(position_tgt_top, 3)

    return position_tgt_top