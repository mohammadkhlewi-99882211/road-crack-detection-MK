import os
import json
import base64
import re
import google.generativeai as genai
from PIL import Image
import io


# ─────────────────────────────────────────────
#  Client initialisation
# ─────────────────────────────────────────────

def _configure_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)


def get_gemini_client(temperature=0.1):
    _configure_gemini()
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=genai.GenerationConfig(
            max_output_tokens=8192,
            temperature=temperature,
            candidate_count=1,
        )
    )


# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────

def _parse_json_response(text):
    """
    Robust JSON extractor. Tries multiple strategies:
    1. Direct parse after stripping markdown fences
    2. Find outermost { } block
    3. Regex to find JSON-like structure
    """
    if not text:
        return None

    # Strategy 1: strip markdown fences
    cleaned = text.strip()
    for fence in ["```json", "```JSON", "```"]:
        cleaned = cleaned.replace(fence, "")
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find outermost { }
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    # Strategy 3: fix common issues (trailing commas, single quotes)
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned[start:end])
        fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        pass

    return None


def _image_to_pil(image_base64):
    """Convert base64 string to PIL Image."""
    img_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ─────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────

DETECTION_PROMPT = """You are an AI vision system for detecting cracks in concrete, asphalt, and masonry surfaces.

TASK: Detect ALL visible cracks, fractures, fissures, and surface defects in this image.

OUTPUT FORMAT - Return ONLY this exact JSON, nothing else, no markdown, no explanation:
{
    "detected": [
        {
            "id": 1,
            "bbox": {"x": 0.10, "y": 0.20, "width": 0.30, "height": 0.10},
            "confidence": 85,
            "rough_type": "linear crack",
            "rough_severity": "HIGH"
        }
    ],
    "total": 1,
    "image_quality": "good",
    "surface_visible": "concrete"
}

BOUNDING BOX RULES:
- All values normalized between 0.0 and 1.0
- x = left edge, y = top edge, width = horizontal span, height = vertical span
- Fit boxes tightly around each crack
- Each separate crack gets its own box

DETECTION RULES:
- Detect ALL cracks regardless of size, even hairline cracks
- Include spalling and delamination
- Exclude: shadows, joints, intentional grooves, seams

If absolutely no cracks exist return: {"detected": [], "total": 0, "image_quality": "good", "surface_visible": "unknown"}

IMPORTANT: Your entire response must be ONLY the JSON object above. No text before or after."""


ANALYSIS_PROMPT_TEMPLATE = """You are a structural engineer analyzing pre-detected cracks.

Detected crack locations:
{boxes_desc}

Image size: {img_w}x{img_h}px

Provide expert analysis. Return ONLY this JSON (no markdown, Arabic text fields):
{{
    "summary": "ملخص الحالة في 2-3 جمل",
    "overall_severity": "CRITICAL",
    "overall_confidence": 85,
    "material_type": "خرسانة مسلحة",
    "surface_condition": "وصف حالة السطح",
    "environmental_factors": "العوامل البيئية",
    "cracks": [
        {{
            "id": 1,
            "bbox": {{"x": 0.10, "y": 0.20, "width": 0.30, "height": 0.10}},
            "type": "شرخ طولي",
            "category": "structural",
            "is_structural": true,
            "estimated_width_mm": "1.0-2.0",
            "estimated_length_cm": "20-30",
            "depth_assessment": "متوسط",
            "severity": "HIGH",
            "confidence": 85,
            "description": "وصف الشرخ",
            "cause_analysis": "تحليل السبب",
            "progression_risk": "متوسط",
            "immediate_action": "الإجراء المطلوب"
        }}
    ],
    "recommendations": [
        {{
            "priority": 1,
            "action": "الإجراء",
            "timeline": "خلال أسبوع",
            "estimated_cost_level": "متوسط",
            "details": "التفاصيل"
        }}
    ],
    "monitoring_plan": "خطة المتابعة",
    "professional_consultation_required": true,
    "notes": "ملاحظات إضافية"
}}

Rules:
- overall_severity must be one of: CRITICAL, HIGH, MEDIUM, LOW
- Use EXACT bbox coordinates from the detected cracks above
- All text fields must be in Arabic
- Return ONLY the JSON, nothing else"""


# ─────────────────────────────────────────────
#  Detection with retries
# ─────────────────────────────────────────────

def _gemini_detect(image_base64, max_retries=3):
    """
    Detect cracks using Gemini with multiple retry attempts.
    Each retry slightly adjusts temperature to get different responses.
    """
    pil_image = _image_to_pil(image_base64)

    for attempt in range(max_retries):
        try:
            temp = 0.1 + (attempt * 0.1)
            client = get_gemini_client(temperature=temp)
            response = client.generate_content([DETECTION_PROMPT, pil_image])

            if not response or not response.text:
                print(f"[Detection] Attempt {attempt+1}: Empty response")
                continue

            raw_text = response.text
            print(f"[Detection] Attempt {attempt+1} raw response (first 300 chars): {raw_text[:300]}")

            result = _parse_json_response(raw_text)
            if result and "detected" in result:
                detected = result.get("detected", [])
                print(f"[Detection] Attempt {attempt+1}: Found {len(detected)} cracks")
                return result

            print(f"[Detection] Attempt {attempt+1}: Could not parse JSON")

        except Exception as e:
            print(f"[Detection] Attempt {attempt+1} error: {e}")

    print("[Detection] All attempts failed, returning empty result")
    return {"detected": [], "total": 0, "image_quality": "unknown", "surface_visible": "unknown"}


# ─────────────────────────────────────────────
#  Analysis with retries
# ─────────────────────────────────────────────

def _gemini_analyze(image_base64, final_detections, img_w, img_h, max_retries=2):
    """Run detailed structural analysis via Gemini."""
    pil_image = _image_to_pil(image_base64)

    boxes_desc = ""
    for d in final_detections:
        bbox = d.get("bbox", {})
        boxes_desc += (
            f"\nCrack #{d.get('id', '?')}: "
            f"x={bbox.get('x', 0):.3f}, y={bbox.get('y', 0):.3f}, "
            f"w={bbox.get('width', 0):.3f}, h={bbox.get('height', 0):.3f} | "
            f"confidence: {int(d.get('_conf', 0.7) * 100)}% | "
            f"type: {d.get('rough_type', 'crack')} | "
            f"severity: {d.get('rough_severity', 'MEDIUM')}"
        )

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        boxes_desc=boxes_desc,
        img_w=img_w,
        img_h=img_h
    )

    for attempt in range(max_retries):
        try:
            temp = 0.2 + (attempt * 0.1)
            client = get_gemini_client(temperature=temp)
            response = client.generate_content([prompt, pil_image])

            if not response or not response.text:
                continue

            raw_text = response.text
            print(f"[Analysis] Attempt {attempt+1} raw (first 200 chars): {raw_text[:200]}")

            result = _parse_json_response(raw_text)
            if result and ("summary" in result or "cracks" in result):
                return result

        except Exception as e:
            print(f"[Analysis] Attempt {attempt+1} error: {e}")

    # Fallback: build basic analysis from detections
    return _build_fallback_analysis(final_detections)


def _build_fallback_analysis(final_detections):
    """Build a basic analysis result when Gemini analysis fails."""
    severities = [d.get("rough_severity", "MEDIUM") for d in final_detections]
    priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    overall_sev = "MEDIUM"
    for sev in priority_order:
        if sev in severities:
            overall_sev = sev
            break

    cracks = []
    for det in final_detections:
        cracks.append({
            "id": det["id"],
            "bbox": det["bbox"],
            "type": det.get("rough_type", "شرخ"),
            "category": "structural",
            "is_structural": True,
            "estimated_width_mm": "غير محدد",
            "estimated_length_cm": "غير محدد",
            "depth_assessment": "غير محدد",
            "severity": det.get("rough_severity", "MEDIUM"),
            "confidence": int(det.get("_conf", 0.7) * 100),
            "description": "تم الكشف عن هذا الشرخ بواسطة نظام الرؤية الذكية",
            "cause_analysis": "يتطلب فحصاً ميدانياً لتحديد السبب",
            "progression_risk": "متوسط",
            "immediate_action": "فحص ميداني من قبل مهندس متخصص"
        })

    return {
        "summary": f"تم اكتشاف {len(final_detections)} شرخ/شقوق في السطح. يُنصح بالفحص الميداني.",
        "overall_severity": overall_sev,
        "overall_confidence": 70,
        "material_type": "غير محدد",
        "surface_condition": "يوجد شروخ تستوجب الفحص",
        "environmental_factors": "غير محدد",
        "cracks": cracks,
        "recommendations": [
            {
                "priority": 1,
                "action": "فحص ميداني عاجل",
                "timeline": "خلال أسبوع",
                "estimated_cost_level": "متوسط",
                "details": "يُنصح بمراجعة مهندس إنشائي لتقييم الشروخ المكتشفة"
            }
        ],
        "monitoring_plan": "متابعة الشروخ كل شهر",
        "professional_consultation_required": True,
        "notes": ""
    }


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def detect_and_analyze(image_base64, img_width, img_height):
    """
    Full crack detection and analysis pipeline:
    1. Detect cracks with bounding boxes (Gemini, up to 3 attempts)
    2. Normalise and validate bounding boxes
    3. Run expert structural analysis (Gemini, up to 2 attempts)
    """

    # ── Step 1: Detection ──────────────────────────────────────────────
    gemini_result = _gemini_detect(image_base64, max_retries=3)
    raw_boxes = gemini_result.get("detected", [])

    # Normalise bbox structure
    for b in raw_boxes:
        if "bbox" not in b:
            b["bbox"] = {
                "x": b.pop("x", 0),
                "y": b.pop("y", 0),
                "width": b.pop("width", 0),
                "height": b.pop("height", 0)
            }
        b["_conf"] = b.get("confidence", 75) / 100.0

    # ── Step 2: Clip & renumber ────────────────────────────────────────
    final_detections = []
    for i, det in enumerate(raw_boxes):
        bbox = det.get("bbox", {})
        x = max(0.0, min(0.98, float(bbox.get("x", 0))))
        y = max(0.0, min(0.98, float(bbox.get("y", 0))))
        w = max(0.02, min(1.0 - x, float(bbox.get("width", 0.05))))
        h = max(0.02, min(1.0 - y, float(bbox.get("height", 0.05))))
        det["bbox"] = {"x": x, "y": y, "width": w, "height": h}
        det["id"] = i + 1
        final_detections.append(det)

    total_detected = len(final_detections)
    print(f"[Pipeline] Total detections after normalisation: {total_detected}")

    # ── Step 3: No cracks ─────────────────────────────────────────────
    if total_detected == 0:
        surface = gemini_result.get("surface_visible", "غير محدد")
        return {
            "total_cracks_detected": 0,
            "summary": "لم يتم الكشف عن أي شروخ أو شقوق في هذه الصورة. السطح يبدو بحالة جيدة.",
            "overall_severity": "LOW",
            "overall_confidence": 90,
            "material_type": surface,
            "surface_condition": "السطح في حالة جيدة بدون شروخ مرئية",
            "environmental_factors": "",
            "cracks": [],
            "recommendations": [
                {
                    "priority": 1,
                    "action": "الصيانة الوقائية الدورية",
                    "timeline": "كل 6-12 شهراً",
                    "estimated_cost_level": "منخفض",
                    "details": "قم بإجراء فحص دوري للحفاظ على الحالة الجيدة للسطح"
                }
            ],
            "monitoring_plan": "فحص بصري كل 6 أشهر",
            "professional_consultation_required": False,
            "notes": "",
            "_detection_info": {"gemini_detected": 0, "merged": 0}
        }

    # ── Step 4: Analysis ───────────────────────────────────────────────
    analysis = _gemini_analyze(image_base64, final_detections, img_width, img_height)

    analysis["total_cracks_detected"] = total_detected
    analysis["_detection_info"] = {
        "gemini_detected": total_detected,
        "merged_total": total_detected
    }

    # Sync bbox from detections into analysis cracks
    if "cracks" in analysis and analysis["cracks"]:
        for i, crack in enumerate(analysis["cracks"]):
            if i < len(final_detections):
                crack["bbox"] = final_detections[i]["bbox"]
            crack["id"] = i + 1
    else:
        analysis["cracks"] = []
        for det in final_detections:
            analysis["cracks"].append({
                "id": det["id"],
                "bbox": det["bbox"],
                "type": det.get("rough_type", "شرخ"),
                "category": "structural",
                "is_structural": True,
                "estimated_width_mm": "غير محدد",
                "estimated_length_cm": "غير محدد",
                "depth_assessment": "غير محدد",
                "severity": det.get("rough_severity", "MEDIUM"),
                "confidence": int(det.get("_conf", 0.7) * 100),
                "description": "تم الكشف عن هذا الشرخ بواسطة نظام الرؤية الذكية",
                "cause_analysis": "يتطلب فحصاً ميدانياً لتحديد السبب",
                "progression_risk": "متوسط",
                "immediate_action": "فحص ميداني"
            })

    return analysis


# ─────────────────────────────────────────────
#  Dashboard recommendations
# ─────────────────────────────────────────────

def generate_dashboard_recommendations(records_summary):
    """Generate maintenance recommendations for multiple records using Gemini."""
    prompt = f"""Based on these analyzed crack records, provide comprehensive maintenance recommendations in Arabic.

Records:
{records_summary}

Return ONLY this JSON (no markdown, Arabic text):
{{
    "overall_assessment": "تقييم شامل للوضع",
    "priority_actions": ["الإجراء الأول", "الإجراء الثاني"],
    "maintenance_schedule": "جدول الصيانة المقترح",
    "budget_estimate": "متوسط",
    "risk_areas": ["منطقة الخطر الأولى"],
    "preventive_measures": ["إجراء وقائي أول"]
}}"""

    try:
        client = get_gemini_client(temperature=0.3)
        response = client.generate_content([prompt])
        if response and response.text:
            result = _parse_json_response(response.text)
            if result:
                return result
    except Exception as e:
        print(f"[Dashboard] Error: {e}")

    return {"overall_assessment": "تعذّر توليد التوصيات، يرجى المحاولة مرة أخرى."}
