import os
import json
import base64
import re
import io
import traceback
from PIL import Image
from groq import Groq

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def _parse_json(text):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            try:
                fixed = re.sub(r",\s*([}\]])", r"\1", m.group())
                return json.loads(fixed)
            except Exception:
                pass
    return None


def _b64_to_pil(image_base64):
    data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    max_px = 1024
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _fix_bbox(bbox):
    """
    تصحيح الـ bbox مع padding يضمن إحاطة الشرخ كاملاً في أي اتجاه.
    """
    x = max(0.0, min(0.95, float(bbox.get("x", 0))))
    y = max(0.0, min(0.95, float(bbox.get("y", 0))))
    w = float(bbox.get("width", 0.1))
    h = float(bbox.get("height", 0.1))

    # حد أدنى مطلق لكل بُعد
    w = max(0.04, w)
    h = max(0.04, h)

    # padding نسبي: 3% من حجم الصورة في كل اتجاه
    pad_x = 0.03
    pad_y = 0.03

    x = max(0.0, x - pad_x)
    y = max(0.0, y - pad_y)
    w = min(1.0 - x, w + pad_x * 2)
    h = min(1.0 - y, h + pad_y * 2)

    return {"x": round(x, 4), "y": round(y, 4),
            "width": round(w, 4), "height": round(h, 4)}


# ─────────────────────────────────────────────
#  Prompt الكشف — محسّن للتمييز بين أنواع الأضرار
# ─────────────────────────────────────────────

DETECT_PROMPT = """You are an expert computer vision system specialized in structural damage assessment.
Your task: carefully inspect this image and detect ALL damage with precise bounding boxes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFICATION GUIDE — READ CAREFULLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. STRUCTURAL_CRACK (شرخ إنشائي عميق)
   - Penetrates deep into the material (concrete, asphalt, masonry)
   - Has DARK, SHARP, well-defined edges
   - Shows depth/shadow inside the crack
   - Width varies but usually > 0.3mm
   - Cannot be rubbed off or scratched away
   - Often accompanied by displacement or offset on either side
   - Severity: HIGH or CRITICAL

2. SURFACE_CRACK (شرخ سطحي / تشقق)
   - Limited to the top surface layer only
   - Thin, fine lines — no visible depth
   - Common: map cracking / crazing patterns (شبكة خيوط رفيعة)
   - Caused by shrinkage, thermal expansion, aging
   - No displacement between crack edges
   - Severity: LOW or MEDIUM

3. PAINT_PEELING (تقشر دهان / طلاء)
   - Flaking or peeling of PAINT or COATING only
   - Shows a different color layer underneath
   - Edges are irregular and lifting
   - No structural damage to the base material
   - Often caused by moisture, UV, or poor adhesion
   - Severity: LOW (cosmetic only)

4. SPALLING (تقشر خرساني / تفتت)
   - Chunks of concrete or asphalt breaking away
   - Exposes aggregate or rebar underneath
   - Rough, uneven surface with material loss
   - More serious than paint peeling
   - Severity: MEDIUM or HIGH

5. SHADOW / JOINT / GROOVE (ظل / درز / أخدود)
   - DO NOT detect these as cracks
   - Shadows: change based on lighting angle, soft edges
   - Construction joints: straight, uniform, intentional
   - Grooves: regular pattern, smooth edges

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIGHTING CONDITIONS — IMPORTANT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Dark areas are NOT always cracks — check for sharp defined edges
- A crack has CONSISTENT width along its path (not just a dark patch)
- Shadows change with lighting but cracks do not
- If unsure between shadow and crack: look for edge sharpness and continuity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOUNDING BOX RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All values normalized 0.0–1.0 (fraction of image size):
- x = left edge of the damage zone
- y = top edge of the damage zone
- width = full horizontal span of the damage from left to right
- height = full vertical span of the damage from top to bottom
- The bbox must FULLY COVER the entire crack or damage from end to end

DIRECTION EXAMPLES:
- Vertical crack top-to-bottom: x≈0.45, y≈0.02, width≈0.10, height≈0.95
- Horizontal crack left-to-right: x≈0.02, y≈0.48, width≈0.95, height≈0.08
- Diagonal crack: x≈0.10, y≈0.10, width≈0.75, height≈0.80
- Small local crack: x≈0.30, y≈0.25, width≈0.20, height≈0.25

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — JSON ONLY, NO MARKDOWN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"detected":[{"id":1,"bbox":{"x":0.35,"y":0.05,"width":0.12,"height":0.88},"confidence":90,"damage_type":"STRUCTURAL_CRACK","rough_severity":"HIGH","orientation":"vertical","notes":"deep crack with visible shadow"}],"total":1,"image_quality":"good","surface_visible":"concrete","lighting_conditions":"normal"}

damage_type must be one of: STRUCTURAL_CRACK / SURFACE_CRACK / PAINT_PEELING / SPALLING
rough_severity must be one of: CRITICAL / HIGH / MEDIUM / LOW
orientation must be one of: vertical / horizontal / diagonal / irregular

If no real damage found: {"detected":[],"total":0,"image_quality":"good","surface_visible":"unknown","lighting_conditions":"normal"}

FINAL CHECK before responding:
- Is each detected item a REAL crack/damage or just a shadow/joint?
- Does each bbox fully cover the damage from start to finish?
- Is the damage_type correctly classified?

Output JSON only:"""


# ─────────────────────────────────────────────
#  Prompt التحليل الهندسي
# ─────────────────────────────────────────────

ANALYZE_PROMPT_TPL = """You are a senior structural engineer with expertise in concrete pathology and infrastructure assessment.

Pre-detected damage locations (from AI vision system):
{boxes}

Image dimensions: {w}x{h}px

Your task: provide detailed engineering analysis for each detected damage item.

SEVERITY SCALE:
- CRITICAL: Structural integrity at risk. Immediate action required. (e.g., full-depth crack, rebar exposed, active displacement)
- HIGH: Significant structural damage. Action within days. (e.g., wide crack >2mm, deep spalling)
- MEDIUM: Moderate damage. Monitor and repair soon. (e.g., surface crack 0.3-2mm, minor spalling)
- LOW: Cosmetic/surface only. Routine maintenance. (e.g., paint peeling, hairline surface cracks <0.3mm)

DAMAGE CATEGORY DEFINITIONS:
- structural: Penetrates load-bearing material
- surface: Limited to top layer, no structural concern
- cosmetic: Paint, coating, or finish layer only
- shrinkage: Fine map cracking from drying/curing
- settlement: Caused by differential movement/subsidence
- thermal: From temperature expansion/contraction cycles
- corrosion: Related to rebar corrosion (rust stains visible)
- fatigue: From repeated loading (common in roads/bridges)
- spalling: Loss of surface material chunks

Output ONLY the JSON below with Arabic text fields. No markdown. No explanation outside the JSON.

{{"summary":"ملخص شامل للحالة الإنشائية في 3-4 جمل يذكر أنواع الأضرار ودرجة خطورتها","overall_severity":"HIGH","overall_confidence":85,"material_type":"نوع المادة","surface_condition":"وصف تفصيلي لحالة السطح","environmental_factors":"العوامل البيئية المرئية مثل الرطوبة والتعرية","cracks":[{{"id":1,"bbox":{{"x":0.35,"y":0.05,"width":0.12,"height":0.88}},"type":"شرخ إنشائي رأسي عميق","damage_type":"STRUCTURAL_CRACK","category":"structural","is_structural":true,"estimated_width_mm":"2-3","estimated_length_cm":"45-50","depth_assessment":"عميق — يخترق طبقة الخرسانة الأساسية","severity":"HIGH","confidence":88,"orientation":"رأسي","description":"وصف دقيق للشرخ وخصائصه المرئية","cause_analysis":"التحليل المحتمل لسبب نشوء الشرخ","progression_risk":"عالي — يتوقع تطور سريع مع الأحمال والرطوبة","immediate_action":"حقن الشرخ بالإيبوكسي وفحص الأسباب الجذرية"}}],"recommendations":[{{"priority":1,"action":"وصف الإجراء","timeline":"الجدول الزمني المقترح","estimated_cost_level":"منخفض/متوسط/عالي","details":"تفاصيل تقنية للتنفيذ"}}],"monitoring_plan":"خطة المراقبة والمتابعة الدورية","professional_consultation_required":true,"notes":"ملاحظات إضافية للمهندس الميداني"}}

RULES:
- Use EXACT bbox values from the detected damage above — do NOT change them
- overall_severity = CRITICAL/HIGH/MEDIUM/LOW only
- For PAINT_PEELING: is_structural=false, category="cosmetic", severity=LOW
- For SURFACE_CRACK: is_structural=false, category="surface", severity=LOW or MEDIUM
- For STRUCTURAL_CRACK: is_structural=true, severity=HIGH or CRITICAL
- For SPALLING: is_structural depends on depth, severity=MEDIUM or HIGH
- All text fields must be in Arabic
- JSON only, no text outside the JSON:"""


def _detect(image_base64):
    print("[DETECT] starting...")
    try:
        client = _get_client()
        pil_img = _b64_to_pil(image_base64)
        img_bytes = _pil_to_bytes(pil_img)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        print(f"[DETECT] image ready, model={MODEL}")
    except Exception as e:
        print(f"[DETECT] setup error: {e}")
        traceback.print_exc()
        return {"detected": [], "total": 0}

    for attempt in range(3):
        try:
            print(f"[DETECT {attempt+1}] calling API...")
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                            },
                            {
                                "type": "text",
                                "text": DETECT_PROMPT
                            }
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=round(0.05 + attempt * 0.1, 2),
            )
            raw = response.choices[0].message.content or ""
            print(f"[DETECT {attempt+1}] raw={raw[:500]}")
            result = _parse_json(raw)
            if result and isinstance(result.get("detected"), list):
                print(f"[DETECT {attempt+1}] OK — {len(result['detected'])} items found")
                return result
            print(f"[DETECT {attempt+1}] parse failed")
        except Exception as e:
            print(f"[DETECT {attempt+1}] error: {e}")
            traceback.print_exc()

    return {"detected": [], "total": 0}


def _analyze(image_base64, detections, img_w, img_h):
    print("[ANALYZE] starting...")
    try:
        client = _get_client()
        pil_img = _b64_to_pil(image_base64)
        img_bytes = _pil_to_bytes(pil_img)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"[ANALYZE] setup error: {e}")
        return None

    boxes = ""
    for d in detections:
        b = d.get("bbox", {})
        boxes += (
            f"#{d['id']}: x={b.get('x',0):.3f} y={b.get('y',0):.3f} "
            f"w={b.get('width',0):.3f} h={b.get('height',0):.3f} | "
            f"type={d.get('damage_type', d.get('rough_type','STRUCTURAL_CRACK'))} | "
            f"severity={d.get('rough_severity','MEDIUM')} | "
            f"orientation={d.get('orientation','unknown')} | "
            f"notes={d.get('notes','')}\n"
        )

    prompt = ANALYZE_PROMPT_TPL.format(boxes=boxes, w=img_w, h=img_h)

    for attempt in range(2):
        try:
            print(f"[ANALYZE {attempt+1}] calling API...")
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=round(0.15 + attempt * 0.1, 2),
            )
            raw = response.choices[0].message.content or ""
            print(f"[ANALYZE {attempt+1}] raw={raw[:300]}")
            result = _parse_json(raw)
            if result and "summary" in result:
                return result
        except Exception as e:
            print(f"[ANALYZE {attempt+1}] error: {e}")
            traceback.print_exc()

    return None


def detect_and_analyze(image_base64, img_width, img_height):
    print("[PIPELINE] starting detect_and_analyze")
    raw_result = _detect(image_base64)
    raw_boxes = raw_result.get("detected", [])

    # توحيد bbox وتصحيحه
    for b in raw_boxes:
        if "bbox" not in b:
            b["bbox"] = {
                "x": b.pop("x", 0), "y": b.pop("y", 0),
                "width": b.pop("width", 0.1), "height": b.pop("height", 0.1)
            }
        b["_conf"] = float(b.get("confidence", 75)) / 100.0

    final = []
    for i, det in enumerate(raw_boxes):
        det["bbox"] = _fix_bbox(det["bbox"])
        det["id"] = i + 1
        print(f"[PIPELINE] item #{i+1}: type={det.get('damage_type','?')} "
              f"severity={det.get('rough_severity','?')} bbox={det['bbox']}")
        final.append(det)

    total = len(final)
    print(f"[PIPELINE] total: {total}")

    if total == 0:
        return {
            "total_cracks_detected": 0,
            "summary": "لم يتم الكشف عن أي شروخ أو أضرار. السطح يبدو بحالة جيدة.",
            "overall_severity": "LOW", "overall_confidence": 90,
            "material_type": raw_result.get("surface_visible", "غير محدد"),
            "surface_condition": "السطح في حالة جيدة بدون أضرار مرئية",
            "environmental_factors": "",
            "cracks": [],
            "recommendations": [{
                "priority": 1,
                "action": "الصيانة الوقائية الدورية",
                "timeline": "كل 6-12 شهراً",
                "estimated_cost_level": "منخفض",
                "details": "فحص دوري للحفاظ على الحالة الجيدة للسطح"
            }],
            "monitoring_plan": "فحص بصري كل 6 أشهر",
            "professional_consultation_required": False,
            "notes": "",
            "_detection_info": {"gemini_detected": 0}
        }

    analysis = _analyze(image_base64, final, img_width, img_height)

    # Fallback إذا فشل التحليل
    if analysis is None:
        sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        all_sevs = [d.get("rough_severity", "MEDIUM") for d in final]
        overall = next((s for s in sev_order if s in all_sevs), "MEDIUM")
        analysis = {
            "summary": f"تم اكتشاف {len(final)} ضرر في السطح. يُنصح بالفحص الميداني.",
            "overall_severity": overall,
            "overall_confidence": 65,
            "material_type": "غير محدد",
            "surface_condition": "يوجد أضرار تستوجب الفحص",
            "environmental_factors": "",
            "cracks": [],
            "recommendations": [{
                "priority": 1,
                "action": "فحص ميداني عاجل",
                "timeline": "خلال أسبوع",
                "estimated_cost_level": "متوسط",
                "details": "مراجعة مهندس إنشائي لتقييم الأضرار المكتشفة"
            }],
            "monitoring_plan": "متابعة الأضرار شهرياً",
            "professional_consultation_required": True,
            "notes": ""
        }

    analysis["total_cracks_detected"] = total
    analysis["_detection_info"] = {"gemini_detected": total}

    # مزامنة الـ bbox المُصحَّح دائماً
    if analysis.get("cracks"):
        for i, crack in enumerate(analysis["cracks"]):
            if i < len(final):
                crack["bbox"] = final[i]["bbox"]
            crack["id"] = i + 1
    else:
        analysis["cracks"] = [
            {
                "id": d["id"],
                "bbox": d["bbox"],
                "type": d.get("damage_type", "شرخ"),
                "damage_type": d.get("damage_type", "STRUCTURAL_CRACK"),
                "category": "surface" if d.get("damage_type") in ["PAINT_PEELING", "SURFACE_CRACK"] else "structural",
                "is_structural": d.get("damage_type") not in ["PAINT_PEELING", "SURFACE_CRACK"],
                "estimated_width_mm": "غير محدد",
                "estimated_length_cm": "غير محدد",
                "depth_assessment": "غير محدد",
                "severity": d.get("rough_severity", "MEDIUM"),
                "confidence": int(d.get("_conf", 0.7) * 100),
                "description": "تم الكشف بواسطة الذكاء الاصطناعي",
                "cause_analysis": "يتطلب فحصاً ميدانياً",
                "progression_risk": "متوسط",
                "immediate_action": "فحص ميداني"
            }
            for d in final
        ]

    return analysis


def generate_dashboard_recommendations(records_summary):
    client = _get_client()
    prompt = (
        "Based on these crack and damage records give maintenance recommendations in Arabic.\n"
        "Return ONLY JSON no markdown:\n"
        '{"overall_assessment":"تقييم شامل","priority_actions":["إجراء 1","إجراء 2"],'
        '"maintenance_schedule":"جدول الصيانة","budget_estimate":"متوسط",'
        '"risk_areas":["منطقة خطر"],"preventive_measures":["إجراء وقائي"]}\n\n'
        f"Records:\n{records_summary}\n\nJSON only:"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        result = _parse_json(response.choices[0].message.content or "")
        if result:
            return result
    except Exception as e:
        print(f"[DASHBOARD] error: {e}")
    return {"overall_assessment": "تعذّر توليد التوصيات، يرجى المحاولة مرة أخرى."}
