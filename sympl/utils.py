from PIL import Image, ImageFont, ImageDraw
from typing import List, Optional, Union, Dict
import textwrap
import math
import numpy as np
import torch
import trimesh
import io, base64
import os

def sympl_stage(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def add_message(
    messages: List[Dict],
    role: str = "user",
    text: str = None,
    image: Image.Image = None,
):

    if image is not None:
        new_message = {
            'role': role,
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': text}
            ]
        }
    else:
        new_message = {
            'role': role,
            'content': [
                {'type': 'text', 'text': text}
            ]
        }

    messages.append(new_message)

    return messages

def create_image_with_text(
    image,
    text,
    fontsize=15,
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
):

    image_width, image_height = image.size

    text_width = int(image_width)
    total_width = image_width + text_width
    text_height = image_height

    image_with_text = Image.new('RGB', (total_width, text_height), 'white')
    image_with_text.paste(image, (0, 0))

    draw = ImageDraw.Draw(image_with_text)

    try:
        font = ImageFont.truetype(font_path, fontsize)
    except IOError:
        font = ImageFont.load_default()

    wrapped_text = textwrap.fill(
        text,
        width=int(text_width / font.getlength('$'))
    )

    padding = 20
    text_x = image_width + padding
    text_y = padding

    draw.text((text_x, text_y), wrapped_text, fill="black", font=font)

    return image_with_text

def visualize_conversation(
    items: List[dict],
    width: int = 1200,
    padding: int = 28,
    row_gap: int = 18,
    image_max_width: int = 220,
    font_path: Optional[str] = None,
    font_size: int = 22,
    text_bg: tuple = (246, 246, 246),
    canvas_bg: tuple = (255, 255, 255),
    text_color: tuple = (20, 20, 20),
    bubble_radius: int = 16,
    output_path: Optional[str] = None
) -> Image.Image:

    def load_font(size: int) -> ImageFont.FreeTypeFont:
        try:
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
        words = text.replace("\n", " \n ").split(" ")
        lines = []
        line = ""
        for w in words:
            if w == "\n":
                lines.append(line)
                line = ""
                continue
            test = w if not line else f"{line} {w}"
            tw, _ = measure_text(draw, test, font)
            if tw <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                ww, _ = measure_text(draw, w, font)
                if ww > max_width:
                    buf = ""
                    for ch in w:
                        tbuf = buf + ch
                        cw, _ = measure_text(draw, tbuf, font)
                        if cw <= max_width:
                            buf = tbuf
                        else:
                            if buf:
                                lines.append(buf)
                            buf = ch
                    line = buf
                else:
                    line = w
        if line:
            lines.append(line)
        return lines

    def rounded_rect(draw, xy, radius, fill):
        try:
            draw.rounded_rectangle(xy, radius=radius, fill=fill)
        except Exception:
            draw.rectangle(xy, fill=fill)

    def load_image(img: Union[str, Image.Image, None]) -> Optional[Image.Image]:
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        return Image.open(img).convert("RGBA")

    tmp = Image.new("RGB", (width, 200), canvas_bg)
    tmp_draw = ImageDraw.Draw(tmp)
    font = load_font(font_size)
    line_height = max(font.getbbox("Ag")[3] - font.getbbox("Ag")[1], font_size)
    line_gap = max(4, int(font_size * 0.2))

    measured_rows = []
    total_height = padding

    for idx, item in enumerate(items):
        text = str(item.get("text", ""))
        img = load_image(item.get("image"))

        if img is not None:
            left_block_w = image_max_width
            gutter = 16
            text_area_w = width - (padding * 2) - left_block_w - gutter
        else:
            left_block_w = 0
            gutter = 0
            text_area_w = width - (padding * 2)

        lines = wrap_text(tmp_draw, text, font, text_area_w)
        text_h = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        text_h = max(text_h, line_height)

        img_h = 0
        img_w = 0
        if img is not None:
            iw, ih = img.size
            scale = min(image_max_width / iw, 1.0)
            img_w = int(iw * scale)
            img_h = int(ih * scale)
            max_img_h = min(400, text_h * 2)
            if img_h > max_img_h:
                scale = max_img_h / img_h
                img_w = int(img_w * scale)
                img_h = int(img_h * scale)

        row_h = max(text_h, img_h) + padding
        measured_rows.append({
            "lines": lines, "text_h": text_h, "img": img, "img_w": img_w,
            "img_h": img_h, "left_block_w": left_block_w, "text_area_w": text_area_w,
            "row_h": row_h, "gutter": gutter
        })
        total_height += row_h + row_gap

    total_height += padding - row_gap
    canvas = Image.new("RGB", (width, total_height), canvas_bg)
    draw = ImageDraw.Draw(canvas)

    y = padding
    for idx, row in enumerate(measured_rows):
        img = row["img"]
        img_w, img_h = row["img_w"], row["img_h"]
        row_h = row["row_h"]
        gutter = row["gutter"]
        x = padding
        if img is not None and img_w > 0 and img_h > 0:
            img_resized = img.resize((img_w, img_h), Image.LANCZOS)
            mask = Image.new("L", (img_w, img_h), 0)
            mdraw = ImageDraw.Draw(mask)
            rr = min(16, img_w // 8, img_h // 8)
            mdraw.rounded_rectangle([0, 0, img_w, img_h], radius=rr, fill=255)
            img_y = y + (row_h - img_h) // 2
            canvas.paste(img_resized, (x, img_y), mask)
            x += img_w + gutter

        bubble_w = row["text_area_w"]
        bubble_h = row["text_h"] + padding // 2
        bubble_x0 = x
        bubble_y0 = y + (row_h - bubble_h) // 2
        bubble_x1 = min(bubble_x0 + bubble_w, width - padding)
        bubble_y1 = bubble_y0 + bubble_h

        if idx < len(measured_rows) - 2:
            rounded_rect(draw, (bubble_x0, bubble_y0, bubble_x1, bubble_y1), bubble_radius, text_bg)
        else:
            rounded_rect(draw, (bubble_x0, bubble_y0, bubble_x1, bubble_y1), bubble_radius, (246, 246, 255))

        tx = bubble_x0 + padding // 2
        ty = bubble_y0 + padding // 4
        for i, line in enumerate(row["lines"]):
            draw.text((tx, ty + i * (line_height + line_gap)), line, font=font, fill=text_color)
        y += row_h + row_gap

    if output_path:
        canvas.save(output_path)
    return canvas

def vec_S_to_T(v_S: np.ndarray) -> np.ndarray:
    v_S = np.asarray(v_S, dtype=float).reshape(3,)
    x, y, z = v_S
    return np.array([x, -y, -z], dtype=float)

def _line_rect_intersections(a: float, b: float, c: float, W: int, H: int):
    eps = 1e-9
    pts = []
    if abs(b) > eps:
        y = (-c - a*0.0) / b
        if -eps <= y <= H-1+eps: pts.append((0.0, float(y)))
    if abs(b) > eps:
        y = (-c - a*(W-1)) / b
        if -eps <= y <= H-1+eps: pts.append((float(W-1), float(y)))
    if abs(a) > eps:
        x = (-c - b*0.0) / a
        if -eps <= x <= W-1+eps: pts.append((float(x), 0.0))
    if abs(a) > eps:
        x = (-c - b*(H-1)) / a
        if -eps <= x <= W-1+eps: pts.append((float(x), float(H-1)))
    uniq = []
    for p in pts:
        if not any(abs(p[0]-q[0])<1e-6 and abs(p[1]-q[1])<1e-6 for q in uniq):
            uniq.append(p)
    return uniq[:2]

def _fill_halfplane(draw, W: int, H: int, a: float, b: float, c: float, want_nonneg: bool, fill_rgb):
    corners = [(0.0,0.0),(W-1.0,0.0),(W-1.0,H-1.0),(0.0,H-1.0)]
    def sgn(xy): return a*xy[0] + b*xy[1] + c
    corner_keep = [xy for xy in corners if (sgn(xy) >= 0) == want_nonneg]
    inters = _line_rect_intersections(a,b,c,W,H)
    if len(corner_keep) + len(inters) < 3:
        if len(corner_keep) == 4:
            draw.rectangle([(0,0),(W-1,H-1)], fill=fill_rgb)
        return
    poly = corner_keep + inters
    cxp = sum(p[0] for p in poly)/len(poly)
    cyp = sum(p[1] for p in poly)/len(poly)
    poly.sort(key=lambda p: math.atan2(p[1]-cyp, p[0]-cxp))
    draw.polygon(poly, fill=fill_rgb)

def move_along_direction_T(p_T: np.ndarray, d_S: np.ndarray, s: float, *, eps: float = 1e-12) -> np.ndarray:
    p_T = np.asarray(p_T, dtype=float).reshape(3,)
    d_T = vec_S_to_T(d_S)
    n = np.linalg.norm(d_T)
    if n < eps:
        raise ValueError("Direction vector norm is too small")
    return p_T + s * (d_T / n)

def _parse_intrinsics(K, image_size):
    W, H = image_size
    if isinstance(K, dict):
        return float(K['fx']), float(K['fy']), float(K['cx']), float(K['cy'])
    arr = np.asarray(K, dtype=float)
    if arr.shape == (3,3):
        return float(arr[0,0]), float(arr[1,1]), float(arr[0,2]), float(arr[1,2])
    if arr.size == 4:
        fx, fy, cx, cy = arr.ravel()
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError("Unsupported K format")

def adjust_depth_scale_inplace(obj_dict, T=2.0, eps=1e-12):
    for k, v in obj_dict.items():
        if k == 'camera': continue
        if "position" in v and v["position"] is not None:
            p = np.asarray(v["position"], dtype=float).ravel()
            if p.size >= 3:
                p[2] *= T
                v["position"] = p
    return obj_dict

def symbolic_layout_prompting(
    vlm_model, prompt_parser, category: str, prompt: str,
    simple_img: Image.Image, obj_color_dict: Dict,
    conv_history: list = None, **sympl_args,
):
    response_sympl = None
    if 'left_right' in category.lower():
        obj_name = list(obj_color_dict.keys())
        obj_num = sum(1 for x in obj_name if x != "ref")
        if obj_num == 1:
            obj_color = list(obj_color_dict.values())[1]
            simple_prompt = prompt_parser.get_prompt_by_type("lr_simple_one").format(obj=obj_color)
        elif obj_num == 2:
            obj_color = list(obj_color_dict.values())
            simple_prompt = prompt_parser.get_prompt_by_type("lr_simple_two").format(obj_1=obj_color[1], obj_2=obj_color[2])
        else: return "fail"
        messages = add_message([], role='user', image=simple_img, text=simple_prompt)
        res = vlm_model.process_messages(messages)
        if obj_num == 1:
            response_sympl = "Left" if 'yellow' in res.lower() else ("Right" if 'black' in res.lower() else "fail")
        elif obj_num == 2:
            if list(obj_color_dict.values())[1].lower() in res.lower(): response_sympl = obj_name[1]
            elif list(obj_color_dict.values())[2].lower() in res.lower(): response_sympl = obj_name[2]
            else: response_sympl = "fail"

    elif 'closer' in category.lower() or 'facing' in category.lower() or 'front_behind' in category.lower() or 'above_below' in category.lower():
        obj_name = list(obj_color_dict.keys())
        obj_color = list(obj_color_dict.values())
        if len(obj_color) != 3: return "fail"

        type_map = {'closer': 'closer_simple', 'facing': 'facing_simple', 'front_behind': 'fb_simple', 'above_below': 'ab_simple'}
        found_cat = next((c for c in type_map if c in category.lower()), None)
        if not found_cat: return "fail"

        simple_prompt = prompt_parser.get_prompt_by_type(type_map[found_cat]).format(obj_1=obj_color[1], obj_2=obj_color[2])
        res = vlm_model.process_messages(add_message([], image=simple_img, text=simple_prompt))

        if found_cat in ['closer', 'facing']:
            if obj_color[1].lower() in res.lower(): response_sympl = obj_name[1]
            elif obj_color[2].lower() in res.lower(): response_sympl = obj_name[2]
            else: response_sympl = "fail"
        elif found_cat in ['front_behind', 'above_below']:
            if obj_color[1].lower() in res.lower(): response_sympl = obj_name[1]
            elif obj_color[2].lower() in res.lower(): response_sympl = obj_name[2]
            else: response_sympl = "fail"

    elif 'visibility' in category.lower():
        obj_color = list(obj_color_dict.values())
        if len(obj_color) < 2: return "fail"
        simple_prompt = prompt_parser.get_prompt_by_type("visibility_simple").format(obj=obj_color[1])
        res = vlm_model.process_messages(add_message([], image=simple_img, text=simple_prompt))
        response_sympl = "Yes, Visible" if 'yellow' in res.lower() else ("Not" if 'black' in res.lower() else "fail")

    if conv_history is not None:
        conv_history += [{'text': category, 'image': None}, {'text': str(response_sympl), 'image': simple_img}]
    return response_sympl