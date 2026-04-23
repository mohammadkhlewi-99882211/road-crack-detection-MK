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

DAMAGE_TYPE_AR = {
    "STRUCTURAL_CRACK": "شرخ إنشائي",
    "SURFACE_CRACK":    "شرخ سطحي",
    "PAINT_PEELING":    "تقشر دهان",
    "SPALLING":         "تقشر خرساني",
}


def _draw_dashed_rect(draw, xy, color, width=2, dash=12):
    x1, y1, x2, y2 = xy
    segments = [
        [(x1, y1), (x2, y1)],
        [(x2, y1), (x2, y2)],
        [(x2, y2), (x1, y2)],
        [(x1, y2), (x1, y1)],
    ]
    for (sx, sy), (ex, ey) in segments:
        length = max(abs(ex - sx), abs(ey - sy))
        steps  = max(1, length // (dash * 2))
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 0.5) / steps
            ax = int(sx + (ex - sx) * t0);  ay = int(sy + (ey - sy) * t0)
            bx = int(sx + (ex - sx) * t1);  by = int(sy + (ey - sy) * t1)
            draw.line([(ax, ay), (bx, by)], fill=color, width=width)


def draw_ai_detections(image_np, cracks):
    """
    Draw bounding boxes for detected cracks/damage on the image.
    Uses the normalised bbox (x, y, width, height) computed from
    start_point / end_point by ai_analyzer, so the box always covers
    the full crack path.
    """
    img  = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    w_img, h_img = img.size
    scale = min(w_img, h_img) / 1000.0

    for crack in cracks:
        bbox = crack.get("bbox", {})
        if not bbox:
            continue

        nx = float(bbox.get("x",     0))
        ny = float(bbox.get("y",     0))
        nw = float(bbox.get("width", 0))
        nh = float(bbox.get("height",0))

        # تحويل الإحداثيات المُعيَّرة إلى بكسل على الصورة الفعلية
        x1 = max(0,          int(nx           * w_img))
        y1 = max(0,          int(ny           * h_img))
        x2 = min(w_img - 1,  int((nx + nw)   * w_img))
        y2 = min(h_img - 1,  int((ny + nh)   * h_img))

        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        severity    = crack.get("severity", "UNKNOWN")
        damage_type = crack.get("damage_type", "")
        color       = SEVERITY_COLORS_RGB.get(severity, SEVERITY_COLORS_RGB["UNKNOWN"])
        crack_id    = crack.get("id", "?")
        sev_label   = SEVERITY_LABELS_AR.get(severity, severity)
        type_label  = DAMAGE_TYPE_AR.get(damage_type, "")
        box_thick   = max(2, int(4 * scale))

        # تعبئة شفافة
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 28), outline=None)

        # حدود المستطيل — خط متصل دائماً (أوضح للمستخدم)
        draw.rectangle([x1, y1, x2, y2],
                       outline=(*color, 255),
                       width=box_thick)

        # إضافة خطوط ركنية لتوضيح الزوايا
        corner = max(12, int(24 * scale))
        ct     = box_thick + 1
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = 1 if cx == x1 else -1
            dy = 1 if cy == y1 else -1
            draw.line([(cx, cy), (cx + dx * corner, cy)],
                      fill=(*color, 255), width=ct)
            draw.line([(cx, cy), (cx, cy + dy * corner)],
                      fill=(*color, 255), width=ct)

        # ── التسمية ──────────────────────────────────────────────────────────
        label_parts = [f"#{crack_id}"]
        if type_label:
            label_parts.append(type_label)
        label_parts.append(sev_label)
        label = "  ".join(label_parts)

        font_size = max(13, int(17 * scale))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox_txt = draw.textbbox((0, 0), label, font=font)
        tw = bbox_txt[2] - bbox_txt[0]
        th = bbox_txt[3] - bbox_txt[1]
        pad = max(3, int(5 * scale))

        lx1, lx2 = x1, x1 + tw + 2 * pad
        ly1, ly2 = y1 - th - 2 * pad, y1
        if ly1 < 0:                      # إذا خرج من حدود الصورة للأعلى
            ly1, ly2 = y1, y1 + th + 2 * pad

        draw.rectangle([lx1, ly1, lx2, ly2], fill=(*color, 230))
        draw.text((lx1 + pad, ly1 + pad), label,
                  fill=(255, 255, 255), font=font)

    return np.array(img)


def image_to_base64(image_np):
    """Convert numpy RGB image to base64 JPEG string."""
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_for_api(image_np, max_size=2000):
    """Resize image preserving aspect ratio. Uses Pillow only."""
    img    = Image.fromarray(image_np)
    h, w   = image_np.shape[:2]
    if max(h, w) <= max_size:
        return image_np
    scale  = max_size / max(h, w)
    img_r  = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img_r)
