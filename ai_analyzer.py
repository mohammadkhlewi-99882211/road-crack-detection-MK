import os
import json
import base64
import google.generativeai as genai


# ─────────────────────────────────────────────
#  Client initialisation
# ─────────────────────────────────────────────

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=genai.GenerationConfig(
            max_output_tokens=8192,
            temperature=0.1,
        )
    )


# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────

def _parse_json_response(text):
    """Extract and parse the first JSON object found in model response text."""
    if not text:
        return None
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _image_part(image_base64):
    """Convert base64 image string to Gemini-compatible image part."""
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64decode(image_base64)
    }


# ─────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────

DETECTION_SYSTEM = """You are a precision computer vision system specialized in detecting structural cracks and surface defects in concrete, asphalt, masonry, plaster, and painted surfaces.

Your ONLY task: detect every visible crack, fracture, fissure, or surface defect and return precise bounding box coordinates.

BOUNDING BOX RULES:
- Use normalized coordinates: x, y, width, height all in range [0.0, 1.0]
- x = left edge (0=left side of image, 1=right side)
- y = top edge (0=top of image, 1=bottom)
- width = horizontal span of the crack region
- height = vertical span of the crack region
- Make boxes TIGHT -- fit closely around each visible crack with minimal padding (0.01-0.02 max)
- For long diagonal or curved cracks, make the box encompass the full crack path
- Do NOT merge separate distinct cracks into one box unless they clearly form a connected system

DETECTION SENSITIVITY:
- Detect ALL cracks regardless of size (even hairline/micro cracks)
- Include delamination, spalling, and surface separation
- Do NOT include shadows, joints, seams, or intentional grooves
- If unsure, include it (better to over-detect than miss)

Respond ONLY with valid JSON, no explanation, no markdown."""


DETECTION_PROMPT = """Carefully examine this image for ALL cracks, fractures, and surface defects.

Return ONLY this JSON structure (no markdown, no explanation):
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
    "image_quality": "good",
    "surface_visible": "concrete"
}

If NO cracks are visible, return exactly:
{"detected": [], "total": 0, "image_quality": "good", "surface_visible": "unknown"}"""


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

Respond ONLY with valid JSON (no markdown, no explanation). Use Arabic for all text fields."""


# ─────────────────────────────────────────────
#  Detection
# ─────────────────────────────────────────────

def _gemini_detect(client, image_base64):
    """Run crack detection via Gemini vision model."""
    try:
        full_prompt = DETECTION_SYSTEM + "\n\n" + DETECTION_PROMPT
        response = client.generate_content([
            full_prompt,
            _image_part(image_base64)
        ])
        result = _parse_json_response(response.text or "")
        return result if result else {"detected": [], "total": 0}
    except Exception as e:
        print(f"[Detection Error] {e}")
        return {"detected": [], "total": 0}


# ─────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────

def _gemini_analyze(client, image_base64, merged_detections, img_w, img_h):
    """Run detailed structural engineering analysis via Gemini."""
    boxes_desc = ""
    for d in merged_detections:
        bbox = d.get("bbox", d)
        boxes_desc += (
            f"\nالشرخ #{d.get('id', '?')}: "
            f"x={bbox.get('x', 0):.3f}, y={bbox.get('y', 0):.3f}, "
            f"w={bbox.get('width', 0):.3f}, h={bbox.get('height', 0):.3f} | "
            f"ثقة أولية: {int(d.get('_conf', 0.7) * 100)}%"
        )

    prompt = f"""These crack regions were detected by an AI vision system:
{boxes_desc}

Image dimensions: {img_w}x{img_h}px

Provide expert structural engineering analysis. Return ONLY this JSON (Arabic text fields, no markdown):
{{
    "summary": "ملخص 2-3 جمل",
    "overall_severity": "CRITICAL/HIGH/MEDIUM/LOW",
    "overall_confidence": 88,
    "material_type": "نوع المادة",
    "surface_condition": "وصف حالة السطح",
    "environmental_factors": "العوامل البيئية المرئية",
    "cracks": [
        {{
            "id": 1,
            "bbox": {{"x": 0.15, "y": 0.22, "width": 0.35, "height": 0.08}},
            "type": "نوع الشرخ",
            "category": "structural/surface/cosmetic/shrinkage/settlement/thermal/corrosion/fatigue",
            "is_structural": true,
            "estimated_width_mm": "1.5-2.0",
            "estimated_length_cm": "25-30",
            "depth_assessment": "سطحي/متوسط/عميق",
            "severity": "CRITICAL/HIGH/MEDIUM/LOW",
            "confidence": 90,
            "description": "وصف دقيق",
            "cause_analysis": "تحليل السبب",
            "progression_risk": "عالي/متوسط/منخفض",
            "immediate_action": "الإجراء الفوري"
        }}
    ],
    "recommendations": [
        {{
            "priority": 1,
            "action": "الإجراء",
            "timeline": "الجدول الزمني",
            "estimated_cost_level": "منخفض/متوسط/عالي",
            "details": "التفاصيل"
        }}
    ],
    "monitoring_plan": "خطة المراقبة",
    "professional_consultation_required": true,
    "notes": "ملاحظات"
}}

Use the EXACT bbox coordinates provided above for each crack (do not change them)."""

    try:
        full_prompt = ANALYSIS_SYSTEM + "\n\n" + prompt
        response = client.generate_content([
            full_prompt,
            _image_part(image_base64)
        ])
        result = _parse_json_response(response.text or "")
        if result:
            return result
    except Exception as e:
        print(f"[Analysis Error] {e}")

    return {
        "summary": "تعذّر إكمال التحليل المفصّل، يرجى المحاولة مرة أخرى.",
        "overall_severity": "UNKNOWN",
        "overall_confidence": 0,
        "cracks": [],
        "recommendations": []
    }


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def detect_and_analyze(image_base64, img_width, img_height):
    """
    Full crack detection and analysis pipeline using Gemini:
    1. Detect cracks and get bounding boxes
    2. Normalise and validate bounding boxes
    3. Run expert structural analysis
    """
    client = get_gemini_client()

    # Step 1: Detection
    gemini_result = _gemini_detect(client, image_base64)
    raw_boxes = gemini_result.get("detected", [])

    # Normalise bbox keys (handle flat or nested formats)
    for b in raw_boxes:
        if "bbox" not in b:
            b["bbox"] = {
                "x": b.pop("x", 0),
                "y": b.pop("y", 0),
                "width": b.pop("width", 0),
                "height": b.pop("height", 0)
            }
        b["_conf"] = b.get("confidence", 75) / 100.0

    # Step 2: Clip & renumber
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

    # Step 3: No cracks found
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

    # Step 4: Detailed analysis
    analysis = _gemini_analyze(client, image_base64, final_detections, img_width, img_height)

    analysis["total_cracks_detected"] = total_detected
    analysis["_detection_info"] = {
        "gemini_detected": total_detected,
        "merged_total": total_detected
    }

    # Ensure crack list is present and bboxes match detections
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
    client = get_gemini_client()

    prompt = f"""Based on these analyzed crack records, provide comprehensive maintenance recommendations in Arabic:

{records_summary}

Respond ONLY with valid JSON (no markdown):
{{
    "overall_assessment": "تقييم شامل",
    "priority_actions": ["الإجراء الأول", "الإجراء الثاني"],
    "maintenance_schedule": "جدول الصيانة المقترح",
    "budget_estimate": "منخفض/متوسط/عالي",
    "risk_areas": ["منطقة الخطر الأولى"],
    "preventive_measures": ["إجراء وقائي أول"]
}}"""

    try:
        system = "Expert structural engineer. Respond in Arabic with valid JSON only."
        response = client.generate_content([system + "\n\n" + prompt])
        result = _parse_json_response(response.text or "")
        if result:
            return result
    except Exception as e:
        print(f"[Dashboard Recommendations Error] {e}")

    return {"overall_assessment": "تعذّر توليد التوصيات، يرجى المحاولة مرة أخرى."}
