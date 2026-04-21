"""
Image drawing utilities using only Pillow (no cv2/OpenCV needed).
Draws AI-detected crack bounding boxes onto images.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64


SEVERITY_COLORS_RGB = {
    "CRITICAL": (220,  40,  40),
    "HIGH":     (255, 140,  30),
    "MEDIUM":   (230, 200,  30),
    "LOW":      ( 50, 200,  60),
    "UNKNOWN":  (120, 120, 120),
}

SEVERITY_LABELS_AR = {
    "CRITICAL": "خطير",
    "HIGH":     "عالي",
    "MEDIUM":   "متوسط",
    "LOW":      "منخفض",
    "UNKNOWN":  "غير محدد",
}


def _draw_dashed_rect(draw, xy, color, width=2, dash=12):
    """Draw a dashed rectangle using line segments."""
    x1, y1, x2, y2 = xy
    segments = [
        [(x1, y1), (x2, y1)],
        [(x2, y1), (x2, y2)],
        [(x2, y2), (x1, y2)],
        [(x1, y2), (x1, y1)],
    ]
    for (sx, sy), (ex, ey) in segments:
        length = max(abs(ex - sx), abs(ey - sy))
        steps = max(1, length // (dash * 2))
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 0.5) / steps
            ax = int(sx + (ex - sx) * t0)
            ay = int(sy + (ey - sy) * t0)
            bx = int(sx + (ex - sx) * t1)
            by = int(sy + (ey - sy) * t1)
            draw.line([(ax, ay), (bx, by)], fill=color, width=width)


def draw_ai_detections(image_np, cracks):
    """
    Draw bounding boxes on image using AI-detected crack coordinates.
    Uses only Pillow — no OpenCV/cv2 required.

    - Dual-confirmed cracks: solid thick border + corner accents
    - Single-model cracks: dashed thinner border
    """
    img = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    w_img, h_img = img.size
    scale = min(w_img, h_img) / 1000.0

    for crack in cracks:
        bbox = crack.get("bbox", {})
        if not bbox:
            continue

        nx = float(bbox.get("x", 0))
        ny = float(bbox.get("y", 0))
        nw = float(bbox.get("width", 0))
        nh = float(bbox.get("height", 0))

        x1 = max(0, int(nx * w_img))
        y1 = max(0, int(ny * h_img))
        x2 = min(w_img - 1, int((nx + nw) * w_img))
        y2 = min(h_img - 1, int((ny + nh) * h_img))

        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        severity  = crack.get("severity", "UNKNOWN")
        color     = SEVERITY_COLORS_RGB.get(severity, SEVERITY_COLORS_RGB["UNKNOWN"])
        dual      = crack.get("dual_confirmed", crack.get("_dual_confirmed", False))
        crack_id  = crack.get("id", "?")
        sev_label = SEVERITY_LABELS_AR.get(severity, severity)

        box_thick = max(2, int(4 * scale))

        # Translucent fill
        fill_alpha = 30
        draw.rectangle([x1, y1, x2, y2],
                       fill=(*color, fill_alpha),
                       outline=None)

        if dual:
            # Solid thick border
            draw.rectangle([x1, y1, x2, y2],
                           outline=(*color, 255),
                           width=box_thick + 1)
            # Inner highlight
            draw.rectangle([x1 + 2, y1 + 2, x2 - 2, y2 - 2],
                           outline=(255, 255, 255, 120),
                           width=1)
            # Corner accents
            corner_len = max(10, int(22 * scale))
            ct = max(2, box_thick + 1)
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                dx = 1 if cx == x1 else -1
                dy = 1 if cy == y1 else -1
                draw.line([(cx, cy), (cx + dx * corner_len, cy)],
                          fill=(*color, 255), width=ct)
                draw.line([(cx, cy), (cx, cy + dy * corner_len)],
                          fill=(*color, 255), width=ct)
        else:
            # Dashed border for single-model
            _draw_dashed_rect(draw, (x1, y1, x2, y2),
                              (*color, 220),
                              width=max(1, box_thick - 1))

        # Label
        dual_dot = " ●" if dual else " ○"
        label    = f"#{crack_id} {sev_label}{dual_dot}"

        font_size = max(14, int(18 * scale))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                      font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox_txt = draw.textbbox((0, 0), label, font=font)
        tw = bbox_txt[2] - bbox_txt[0]
        th = bbox_txt[3] - bbox_txt[1]
        pad = max(3, int(5 * scale))

        lx1, lx2 = x1, x1 + tw + 2 * pad
        ly2 = y1
        ly1 = y1 - th - 2 * pad
        if ly1 < 0:
            ly1 = y1
            ly2 = y1 + th + 2 * pad

        # Label background
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(*color, 230))
        # Label text
        text_y = ly1 + pad
        draw.text((lx1 + pad, text_y), label, fill=(255, 255, 255), font=font)

    return np.array(img)


def image_to_base64(image_np):
    """Convert numpy RGB image to base64 JPEG string."""
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_for_api(image_np, max_size=2000):
    """Resize image if too large, preserving aspect ratio. Uses only Pillow."""
    img = Image.fromarray(image_np)
    h, w = image_np.shape[:2]
    if max(h, w) <= max_size:
        return image_np
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img_resized)
