import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List
from .utils import add_message, sympl_stage, _fill_halfplane

def render_final_image(
    u: np.ndarray,
    v: np.ndarray,
    colors_np: np.ndarray,
    partition_specs: List[Dict],
    image_size: tuple = (512, 512),
    point_radius: int = 10,
    line_width: int = 3,
):
    W, H = image_size
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Draw partition regions (Bipartition part)
    for spec in partition_specs:
        color = spec.get("color", "yellow")
        named = {
            "yellow": (255, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255),
            "green": (0, 255, 0), "white": (255, 255, 255), "black": (0, 0, 0)
        }
        fill_rgb = named.get(color.lower(), (255, 255, 0)) if isinstance(color, str) else color

        if spec["type"] == "halfplane":
            _fill_halfplane(draw, W, H, spec["a"], spec["b"], spec["c"], spec["want_nonneg"], fill_rgb)
        elif spec["type"] == "rectangle":
            draw.rectangle(spec["coords"], fill=fill_rgb)
        elif spec["type"] == "circle":
            c = spec["center"]
            r = spec["radius"]
            draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=fill_rgb, width=line_width, fill=fill_rgb)

    # 2. Draw object dots (Abstraction part)
    cols = np.asarray(colors_np, dtype=float).reshape(-1, 3)
    for (uu, vv), col in zip(np.stack([u, v], 1), cols):
        uu_i, vv_i = int(round(uu)), int(round(vv))
        if 0 <= uu_i < W and 0 <= vv_i < H:
            inner = [uu_i - (point_radius - 1), vv_i - (point_radius - 1),
                     uu_i + (point_radius - 1), vv_i + (point_radius - 1)]
            draw.ellipse(inner, fill=tuple(np.clip(col, 0, 255).astype(np.uint8).tolist()))

    return img

@sympl_stage
def do_localization(
    u: np.ndarray,
    v: np.ndarray,
    colors_np: np.ndarray,
    partition_specs: List[Dict],
):
    simple_img = render_final_image(u, v, colors_np, partition_specs)
    return simple_img
