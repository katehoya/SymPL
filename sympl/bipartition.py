import numpy as np

from .utils import sympl_stage

def get_bipartition_parameter(
    u: np.ndarray,
    v: np.ndarray,
    colors: np.ndarray,
    meta: dict,
    image_size: tuple,
    category: str = "",
    ref_color: str | tuple = "yellow",
    point_radius: int = 4,
    line_width: int = 3,
):
    import math

    W, H = image_size
    cx, cy = meta["cx"], meta["cy"]
    cx_i, cy_i = meta["cx_i"], meta["cy_i"]

    cat = (category or "").strip().lower()

    def _max_radius_from_center(cx, cy, ux, uy, W, H, margin_px):
        INF = 1e18
        tx = INF
        ty = INF
        if abs(ux) > 1e-9:
            if ux > 0:
                tx = (W - 1 - margin_px - cx) / ux
            else:
                tx = (margin_px - cx) / ux
        if abs(uy) > 1e-9:
            if uy > 0:
                ty = (H - 1 - margin_px - cy) / uy
            else:
                ty = (margin_px - cy) / uy
        tmax = min([t for t in (tx, ty) if t > 0] + [INF])
        return max(0.0, tmax)

    def _enforce_min_center_separation(p1_xy, p2_xy, cx, cy, W, H,
                                       point_radius, line_width,
                                       min_sep=30.0):
        x1, y1 = float(p1_xy[0]), float(p1_xy[1])
        x2, y2 = float(p2_xy[0]), float(p2_xy[1])

        dx1, dy1 = x1 - cx, y1 - cy
        dx2, dy2 = x2 - cx, y2 - cy
        r1 = math.hypot(dx1, dy1)
        r2 = math.hypot(dx2, dy2)
        if r1 < 1e-9:
            ux1, uy1 = 0.0, -1.0
        else:
            ux1, uy1 = dx1 / r1, dy1 / r1
        if r2 < 1e-9:
            ux2, uy2 = 0.0, -1.0
        else:
            ux2, uy2 = dx2 / r2, dy2 / r2

        margin_px = point_radius + max(1, int(line_width/2)) + 1
        rmin = float(margin_px)
        r1_max = _max_radius_from_center(cx, cy, ux1, uy1, W, H, margin_px)
        r2_max = _max_radius_from_center(cx, cy, ux2, uy2, W, H, margin_px)

        if r1 <= r2:
            rs, usx, usy, rs_max, idx_small = r1, ux1, uy1, r1_max, 1
            rl, ulx, uly, rl_max, idx_large = r2, ux2, uy2, r2_max, 2
        else:
            rs, usx, usy, rs_max, idx_small = r2, ux2, uy2, r2_max, 2
            rl, ulx, uly, rl_max, idx_large = r1, ux1, uy1, r1_max, 1

        cur_diff = abs(rl - rs)
        if cur_diff >= min_sep - 1e-6:
            return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))

        need = min_sep - cur_diff
        cap_in  = max(0.0, rs - rmin)
        cap_out = max(0.0, rl_max - rl) 

        d_small = min(need, cap_in)             
        d_large = min(need - d_small, cap_out)  

        rs_new = rs - d_small
        rl_new = rl + d_large

        if idx_small == 1:
            x1n, y1n = cx + usx * rs_new, cy + usy * rs_new
            x2n, y2n = cx + ulx * rl_new, cy + uly * rl_new
        else:
            x2n, y2n = cx + usx * rs_new, cy + usy * rs_new
            x1n, y1n = cx + ulx * rl_new, cy + uly * rl_new

        return (int(round(x1n)), int(round(y1n))), (int(round(x2n)), int(round(y2n)))

    cols = np.asarray(colors, dtype=float).reshape(-1, 3)

    partition_specs = []

    if cat == "left_right":
        uv = np.stack([u, v], 1)
        n = uv.shape[0]
        if n >= 2:
            x_left  = float(np.min(uv[:, 0]))
            x_right = float(np.max(uv[:, 0]))
            mx = 0.5 * (x_left + x_right)

            a, b, c = 1.0, 0.0, -mx
            want_nonneg = False
            partition_specs.append({"type": "halfplane", "a": a, "b": b, "c": c, "want_nonneg": want_nonneg, "color": ref_color})
        else:
            partition_specs.append({"type": "rectangle", "coords": [(0, 0), (cx_i, H - 1)], "color": ref_color})

    elif cat == "visibility":
        partition_specs.append({"type": "rectangle", "coords": [(0, 0), (W - 1, cy_i)], "color": ref_color})

    elif cat == "above_below":
        uv = np.stack([u, v], 1)
        n = uv.shape[0]
        if n >= 2:
            top_index    = int(np.argmin(uv[:, 1]))
            bottom_index = int(np.argmax(uv[:, 1]))
            y_top    = float(uv[top_index, 1])
            y_bottom = float(uv[bottom_index, 1])

            my = 0.5 * (y_top + y_bottom)

            a, b, c = 0.0, 1.0, -my
            want_nonneg = False 
            partition_specs.append({"type": "halfplane", "a": a, "b": b, "c": c, "want_nonneg": want_nonneg, "color": ref_color})
        else:
            partition_specs.append({"type": "rectangle", "coords": [(0, 0), (W - 1, cy_i)], "color": ref_color})

    elif cat == "closer" or cat == "front_behind":
        uv = np.stack([u, v], 1)
        n = uv.shape[0]
        if n >= 2:
            (x1, y1), (x2, y2) = (float(uv[0,0]), float(uv[0,1])), (float(uv[1,0]), float(uv[1,1]))
            (x1, y1), (x2, y2) = _enforce_min_center_separation(
                (x1, y1), (x2, y2), cx, cy, W, H, point_radius, line_width, min_sep=30.0
            )
            u[0], v[0] = x1, y1
            u[1], v[1] = x2, y2

        dists = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
        pos = dists[np.isfinite(dists)]
        if pos.size > 0:
            r_draw = float(np.mean(pos))
            r_limit = (min(W, H) / 2.0) - max(1.0, line_width / 2.0) - point_radius
            if r_limit > 0:
                r_draw = min(r_draw, r_limit)
            r_i = int(round(r_draw))
            partition_specs.append({"type": "circle", "center": (int(cx), int(cy)), "radius": r_i, "color": ref_color, "line_width": line_width})

    elif cat == "facing":
        R0 = min(W, H) / 3.0
        dirs = []
        for (uu, vv) in np.stack([u, v], 1):
            dx, dy = float(uu - cx), float(vv - cy)
            L = (dx*dx + dy*dy) ** 0.5
            if L < 1e-6:
                ux, uy = 0.0, -1.0
            else:
                ux, uy = dx / L, dy / L
            px = cx + R0 * ux
            py = cy + R0 * uy
            dirs.append((px, py))

        used = min(len(dirs), len(cols), 2)
        yc = (int(round(cx)), int(round(cy - R0)))

        if used >= 2:
            (p1x, p1y), (p2x, p2y) = _enforce_min_center_separation(
                dirs[0], dirs[1], float(yc[0]), float(yc[1]), W, H, point_radius, line_width, min_sep=30.0
            )
            dirs[0], dirs[1] = (p1x, p1y), (p2x, p2y)
            u[0], v[0] = p1x, p1y
            u[1], v[1] = p2x, p2y

        dlist = []
        for k in range(used):
            px, py = dirs[k]
            dlist.append(((px - yc[0])**2 + (py - yc[1])**2) ** 0.5)
        r_yellow = float(np.mean(dlist)) if dlist else R0

        partition_specs.append({"type": "circle", "center": yc, "radius": r_yellow, "color": ref_color, "line_width": line_width})

    return partition_specs, u, v


@sympl_stage
def do_bipartition(
    u: np.ndarray,
    v: np.ndarray,
    colors_np: np.ndarray,
    meta: dict,
    category: str,
    ref_color_name: str,
):
    W = 512
    H = 512
    partition_specs, u_adj, v_adj = get_bipartition_parameter(
        u=u,
        v=v,
        colors=colors_np,
        meta=meta,
        image_size=(W, H),
        category=category,
        ref_color=ref_color_name,
        point_radius=10,
    )
    return partition_specs, u_adj, v_adj
