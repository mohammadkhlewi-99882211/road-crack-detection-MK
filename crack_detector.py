"""
crack_detector.py
-----------------
• image_to_base64     — numpy → base64 JPEG
• resize_for_api      — تصغير الصورة مع الحفاظ على النسبة
• refine_bbox         — تحسين bbox بمعالجة الصورة فعلياً (الجوهر)
• draw_ai_detections  — رسم المستطيلات على الصورة
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64


# ─────────────────────────────────────────────────────────────────────────────
#  ألوان حسب الخطورة
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  refine_bbox  ← الجوهر الجديد
# ─────────────────────────────────────────────────────────────────────────────

def refine_bbox(image_np, rough_bbox, expand=0.15):
    """
    يأخذ bbox تقريبياً من النموذج (قد يغطي جزءاً فقط من الشرخ)
    ويوسّعه أولاً ثم يطبّق معالجة الصورة داخل المنطقة الموسّعة
    ليجد الحدود الحقيقية للشرخ.

    الخوارزمية:
    1. وسّع الـ bbox بنسبة expand (15%) في كل اتجاه
    2. استخرج منطقة البحث من الصورة
    3. حوّل إلى تدرج رمادي وطبّق Canny edge detection
    4. ابحث عن أبعد نقطتين في حواف الشرخ لتحديد طرفيه
    5. أعد bbox يغطي الشرخ كاملاً
    """
    h_img, w_img = image_np.shape[:2]

    # ── 1. توسيع منطقة البحث ─────────────────────────────────────────────────
    nx  = float(rough_bbox.get("x",     0))
    ny  = float(rough_bbox.get("y",     0))
    nw  = float(rough_bbox.get("width", 0.1))
    nh  = float(rough_bbox.get("height",0.1))

    # المركز
    cx = nx + nw / 2
    cy = ny + nh / 2

    # أبعاد منطقة البحث الموسّعة
    half_w = max(nw / 2 + expand, 0.12)
    half_h = max(nh / 2 + expand, 0.12)

    sx1 = max(0.0, cx - half_w)
    sy1 = max(0.0, cy - half_h)
    sx2 = min(1.0, cx + half_w)
    sy2 = min(1.0, cy + half_h)

    px1 = int(sx1 * w_img);  px2 = int(sx2 * w_img)
    py1 = int(sy1 * h_img);  py2 = int(sy2 * h_img)

    if px2 - px1 < 10 or py2 - py1 < 10:
        # منطقة البحث صغيرة جداً — أعد bbox الموسّع فقط
        pad = 0.03
        return {"x":     round(max(0.0, nx - pad),       4),
                "y":     round(max(0.0, ny - pad),       4),
                "width": round(min(1.0 - nx, nw + pad*2),4),
                "height":round(min(1.0 - ny, nh + pad*2),4)}

    # ── 2. استخراج المنطقة وتحويلها ─────────────────────────────────────────
    region = image_np[py1:py2, px1:px2]

    # تدرج رمادي
    gray = np.mean(region, axis=2).astype(np.uint8)

    # ── 3. Canny edge detection بسيط بالـ numpy ──────────────────────────────
    # Sobel بسيط
    gy = np.zeros_like(gray, dtype=float)
    gx = np.zeros_like(gray, dtype=float)
    gy[1:-1, :] = gray[2:, :].astype(float) - gray[:-2, :].astype(float)
    gx[:, 1:-1] = gray[:, 2:].astype(float) - gray[:, :-2].astype(float)
    magnitude = np.sqrt(gx**2 + gy**2)

    # عتبة: أعلى 15% من القيم كـ edges
    threshold = np.percentile(magnitude, 85)
    edges = magnitude > threshold

    # ── 4. أبعد نقطتين في الحواف ─────────────────────────────────────────────
    ys, xs = np.where(edges)

    if len(xs) < 10:
        # لا حواف كافية — أعد منطقة البحث الموسّعة
        return {"x":     round(sx1, 4),
                "y":     round(sy1, 4),
                "width": round(sx2 - sx1, 4),
                "height":round(sy2 - sy1, 4)}

    # الحدود الفعلية للحواف داخل المنطقة
    min_x_local = int(np.percentile(xs,  2))
    max_x_local = int(np.percentile(xs, 98))
    min_y_local = int(np.percentile(ys,  2))
    max_y_local = int(np.percentile(ys, 98))

    # تحويل إلى إحداثيات الصورة الكاملة ثم تطبيع
    pad_px = max(4, int(0.02 * min(w_img, h_img)))

    final_x1 = max(0,      px1 + min_x_local - pad_px)
    final_y1 = max(0,      py1 + min_y_local - pad_px)
    final_x2 = min(w_img,  px1 + max_x_local + pad_px)
    final_y2 = min(h_img,  py1 + max_y_local + pad_px)

    result = {
        "x":      round(final_x1 / w_img, 4),
        "y":      round(final_y1 / h_img, 4),
        "width":  round((final_x2 - final_x1) / w_img, 4),
        "height": round((final_y2 - final_y1) / h_img, 4),
    }

    print(f"[REFINE] rough={rough_bbox} → refined={result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  رسم المستطيلات
# ─────────────────────────────────────────────────────────────────────────────

def _draw_dashed_rect(draw, xy, color, width=2, dash=10):
    x1, y1, x2, y2 = xy
    for (sx, sy), (ex, ey) in [
        [(x1,y1),(x2,y1)], [(x2,y1),(x2,y2)],
        [(x2,y2),(x1,y2)], [(x1,y2),(x1,y1)]
    ]:
        length = max(abs(ex-sx), abs(ey-sy))
        steps  = max(1, length // (dash*2))
        for i in range(steps):
            t0 = i/steps;  t1 = (i+0.5)/steps
            draw.line([
                (int(sx+(ex-sx)*t0), int(sy+(ey-sy)*t0)),
                (int(sx+(ex-sx)*t1), int(sy+(ey-sy)*t1))
            ], fill=color, width=width)


def draw_ai_detections(image_np, cracks):
    """
    يرسم bbox لكل شرخ مكتشف.
    يستدعي refine_bbox أولاً لتوسيع الـ bbox ليغطي الشرخ كاملاً.
    """
    img  = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w_img, h_img = img.size
    scale = min(w_img, h_img) / 1000.0

    for crack in cracks:
        raw_bbox = crack.get("bbox", {})
        if not raw_bbox:
            continue

        # ← تحسين الـ bbox بمعالجة الصورة
        bbox = refine_bbox(image_np, raw_bbox, expand=0.15)

        nx = float(bbox["x"]);     ny = float(bbox["y"])
        nw = float(bbox["width"]); nh = float(bbox["height"])

        x1 = max(0,         int(nx       * w_img))
        y1 = max(0,         int(ny       * h_img))
        x2 = min(w_img - 1, int((nx+nw)  * w_img))
        y2 = min(h_img - 1, int((ny+nh)  * h_img))

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
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 25))

        # حدود متصلة
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=box_thick)

        # أركان مميّزة
        corner = max(12, int(22 * scale))
        ct = box_thick + 1
        for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            dx = 1 if cx == x1 else -1
            dy = 1 if cy == y1 else -1
            draw.line([(cx,cy),(cx+dx*corner,cy)], fill=(*color,255), width=ct)
            draw.line([(cx,cy),(cx,cy+dy*corner)], fill=(*color,255), width=ct)

        # تسمية
        parts = [f"#{crack_id}"]
        if type_label:
            parts.append(type_label)
        parts.append(sev_label)
        label = "  ".join(parts)

        font_size = max(13, int(17 * scale))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        bb   = draw.textbbox((0,0), label, font=font)
        tw   = bb[2]-bb[0];  th = bb[3]-bb[1]
        pad  = max(3, int(5*scale))
        lx1, lx2 = x1, x1 + tw + 2*pad
        ly1, ly2 = y1 - th - 2*pad, y1
        if ly1 < 0:
            ly1, ly2 = y1, y1 + th + 2*pad

        draw.rectangle([lx1,ly1,lx2,ly2], fill=(*color,230))
        draw.text((lx1+pad, ly1+pad), label, fill=(255,255,255), font=font)

    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
#  دوال مساعدة
# ─────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_np):
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_for_api(image_np, max_size=2000):
    img  = Image.fromarray(image_np)
    h, w = image_np.shape[:2]
    if max(h, w) <= max_size:
        return image_np
    scale = max_size / max(h, w)
    img_r = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return np.array(img_r)
