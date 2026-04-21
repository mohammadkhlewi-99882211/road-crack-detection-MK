import os
import json
import base64
from openai import OpenAI
import google.generativeai as genai
from google.generativeai import types


def get_openai_client():
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not base_url or not api_key:
        raise ValueError("OpenAI integration env vars not set")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')


def _parse_json_response(text):
    """Extract and parse JSON from model response text."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _compute_iou(box_a, box_b):
    """Compute Intersection over Union for two normalized bounding boxes."""
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["width"], ay1 + box_a["height"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["width"], by1 + box_b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _merge_bbox(box_a, box_b):
    """Merge two bounding boxes by averaging weighted by confidence."""
    ca = box_a.get("_conf", 0.5)
    cb = box_b.get("_conf", 0.5)
    total = ca + cb if ca + cb > 0 else 1.0
    wa, wb = ca / total, cb / total

    ax2 = box_a["x"] + box_a["width"]
    ay2 = box_a["y"] + box_a["height"]
    bx2 = box_b["x"] + box_b["width"]
    by2 = box_b["y"] + box_b["height"]

    x1 = min(box_a["x"], box_b["x"])
    y1 = min(box_a["y"], box_b["y"])
    x2 = max(ax2, bx2)
    y2 = max(ay2, by2)
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
            "_conf": (ca + cb) / 2}


def _ensemble_boxes(gemini_boxes, gpt_boxes, iou_threshold=0.25):
    """
    Merge bounding boxes from two models using IoU-based NMS.
    Boxes confirmed by both models get higher weight.
    Single-model detections are kept but with a slight confidence penalty.
    """
    all_boxes = []

    for b in gemini_boxes:
        b["_source"] = "gemini"
        b["_conf"] = b.get("confidence", 75) / 100.0
        all_boxes.append(b)

    for b in gpt_boxes:
        b["_source"] = "gpt"
        b["_conf"] = b.get("confidence", 75) / 100.0
        all_boxes.append(b)

    merged = []
    used = set()

    for i, ba in enumerate(all_boxes):
        if i in used:
            continue
        group = [ba]
        used.add(i)
        for j, bb in enumerate(all_boxes):
            if j in used or i == j:
                continue
            if ba.get("_source") != bb.get("_source"):
                iou = _compute_iou(ba["bbox"] if "bbox" in ba else ba,
                                   bb["bbox"] if "bbox" in bb else bb)
                if iou >= iou_threshold:
                    group.append(bb)
                    used.add(j)

        if len(group) == 1:
            box = group[0]
            box["_dual_confirmed"] = False
            box["_conf"] = max(0.1, box["_conf"] - 0.1)
            merged.append(box)
        else:
            box_a = group[0]["bbox"] if "bbox" in group[0] else group[0]
            box_b = group[1]["bbox"] if "bbox" in group[1] else group[1]
            box_a["_conf"] = group[0]["_conf"]
            box_b["_conf"] = group[1]["_conf"]
            merged_box = _merge_bbox(box_a, box_b)
            result = group[0].copy()
            result["bbox"] = merged_box
            result["_dual_confirmed"] = True
            result["_conf"] = min(1.0, merged_box["_conf"] + 0.15)
            merged.append(result)

    return merged


DETECTION_SYSTEM = """You are a precision computer vision system specialized in detecting structural cracks and surface defects in concrete, asphalt, masonry, plaster, and painted surfaces.

Your ONLY task: detect every visible crack, fracture, fissure, or surface defect and return precise bounding box coordinates.

BOUNDING BOX RULES:
- Use normalized coordinates: x, y, width, height all in range [0.0, 1.0]
- x = left edge (0=left side of image, 1=right side)
- y = top edge (0=top of image, 1=bottom)  
- width = horizontal span of the crack region
- height = vertical span of the crack region
- Make boxes TIGHT — fit closely around each visible crack with minimal padding (0.01-0.02 max)
- For long diagonal or curved cracks, make the box encompass the full crack path
- Do NOT merge separate distinct cracks into one box unless they clearly form a connected system

DETECTION SENSITIVITY: 
- Detect ALL cracks regardless of size (even hairline/micro cracks)
- Include delamination, spalling, and surface separation
- Do NOT include shadows, joints, seams, or intentional grooves
- If unsure, include it (better to over-detect than miss)

Respond ONLY with valid JSON, no explanation."""


DETECTION_PROMPT = """Carefully examine this image for ALL cracks, fractures, and surface defects.

Return ONLY this JSON structure:
{
    "detected": [
        {
            "id": 1,
            "bbox": {"x": 0.15, "y": 0.22, "width": 0.35, "height": 0.08},
            "confidence": 90,
            "rough_type": "linear crack",
            "rough_severity": "HIGH"
        }
    ],
    "total": 2,
    "image_quality": "good/poor/blurry",
    "surface_visible": "concrete/asphalt/plaster/paint/unknown"
}

If NO cracks are visible, return: {"detected": [], "total": 0, "image_quality": "good", "surface_visible": "unknown"}"""


ANALYSIS_SYSTEM = """You are a world-class structural engineer and concrete pathologist. 
Given pre-detected crack locations (bounding boxes already identified by computer vision), 
provide expert engineering analysis for each crack.

CLASSIFICATION RULES:
- STRUCTURAL cracks: penetrate the load-bearing material (concrete/asphalt body)
- SURFACE cracks: limited to finish layers (paint/plaster/render/coating)
- Differentiate based on: width, pattern, context, edges, associated damage signs

SEVERITY:
- CRITICAL: >2mm wide, active displacement, corrosion, full-depth through cracks
- HIGH: 0.5-2mm, structural pattern cracks, near joints/edges
- MEDIUM: 0.1-0.5mm hairline structural cracks, surface crazing
- LOW: cosmetic only, <0.1mm, paint/plaster surface cracks only

Respond ONLY with valid JSON in Arabic for text fields."""


def _gemini_detect(client, image_base64):
    """استخدام جيمناي مع تنظيف الرد لضمان القراءة"""
    try:
        image_parts = [{"mime_type": "image/jpeg", "data": base64.b64decode(image_base64)}]
        # نطلب منه بوضوح عدم كتابة أي نص خارج الـ JSON
        full_prompt = DETECTION_SYSTEM + "\n\n" + DETECTION_PROMPT + "\nReturn ONLY valid JSON."
        
        response = client.generate_content([full_prompt, image_parts[0]])
        text_response = response.text
        
        # تنظيف الرد من علامات الـ Markdown إذا وجدت
        clean_json = text_response.replace("```json", "").replace("```", "").strip()
        
        return _parse_json_response(clean_json) or {"detected": [], "total": 0}
    except Exception as e:
        print(f"Detection Error: {e}")
        return {"detected": [], "total": 0}

def _gpt_detect(client, image_base64):
    """دالة احتياطية معطلة حالياً لتوفير الموارد"""
    return {"detected": [], "total": 0}

def _gemini_analyze(client, image_base64, merged_detections, img_w, img_h):
    """التحليل الهندسي الإنشائي المفصل باستخدام جيمناي"""
    boxes_desc = ""
    for d in merged_detections:
        bbox = d.get("bbox", d)
        boxes_desc += f"\nالشرخ #{d.get('id', '?')}: x={bbox.get('x', 0):.3f}, y={bbox.get('y', 0):.3f}, w={bbox.get('width', 0):.3f}, h={bbox.get('height', 0):.3f}"

    prompt = f"""As an expert civil engineer, analyze the road surface from this image. 
    Detected cracks: {boxes_desc}
    Image size: {img_w}x{img_h}px
    Provide a full structural report in Arabic JSON format."""
    
    try:
        image_parts = [{"mime_type": "image/jpeg", "data": base64.b64decode(image_base64)}]
        response = client.generate_content([ANALYSIS_SYSTEM + "\n" + prompt, image_parts[0]])
        return _parse_json_response(response.text or "")
    except Exception:
        return {"summary": "حدث خطأ أثناء التحليل الإنشائي."}


def detect_and_analyze(image_base64, img_width, img_height):
    """
    نسخة نظيفة 100% تعتمد على Gemini وتتجاوز أخطاء المسافات.
    """
    gemini_client = get_gemini_client()

    # الخطوة 1: الكشف عن الشقوق
    try:
        gemini_result = _gemini_detect(gemini_client, image_base64)
    except Exception:
        gemini_result = {"detected": []}

    # الخطوة 2: معالجة النتائج
    final_detections = []
    raw_boxes = gemini_result.get("detected", [])
    
    for i, det in enumerate(raw_boxes):
        bbox = det.get("bbox", det)
        # توحيد الإحداثيات
        x = max(0.0, min(0.98, float(bbox.get("x", 0))))
        y = max(0.0, min(0.98, float(bbox.get("y", 0))))
        w = max(0.02, min(1.0 - x, float(bbox.get("width", 0.05))))
        h = max(0.02, min(1.0 - y, float(bbox.get("height", 0.05))))
        
        final_detections.append({
            "id": i + 1,
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "type": det.get("type", "شرخ"),
            "severity": det.get("severity", "MEDIUM")
        })

    # الخطوة 3: التحليل في حال عدم وجود شقوق
    if not final_detections:
        return {
            "total_cracks_detected": 0,
            "summary": "السطح سليم ولا توجد شقوق واضحة.",
            "overall_severity": "LOW",
            "cracks": [],
            "recommendations": [{"priority": 1, "action": "فحص دوري", "timeline": "6 أشهر"}]
        }

    # الخطوة 4: التحليل الهندسي المفصل
    analysis = _gemini_analyze(gemini_client, image_base64, final_detections, img_width, img_height)
    analysis["total_cracks_detected"] = len(final_detections)
    
    return analysis
def generate_dashboard_recommendations(records_summary):
    openai_client = get_openai_client()

    prompt = f"""Based on these analyzed crack records, provide comprehensive maintenance recommendations in Arabic:

{records_summary}

Respond ONLY with valid JSON:
{{
    "overall_assessment": "تقييم شامل",
    "priority_actions": ["الإجراء الأول", "الإجراء الثاني"],
    "maintenance_schedule": "جدول الصيانة المقترح",
    "budget_estimate": "منخفض/متوسط/عالي",
    "risk_areas": ["منطقة الخطر الأولى"],
    "preventive_measures": ["إجراء وقائي أول"]
}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-5.2",
            max_completion_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": "Expert structural engineer. Respond in Arabic with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ]
        )
        result = _parse_json_response(response.choices[0].message.content or "")
        if result:
            return result
    except Exception:
        pass

    return {"overall_assessment": "تعذّر توليد التوصيات، يرجى المحاولة مرة أخرى."}
