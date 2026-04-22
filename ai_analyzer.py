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


def _fix_bbox(bbox, min_size=0.05):
    """
    تصحيح الـ bbox:
    - ضمان حد أدنى للحجم حتى لا يظهر كخط
    - للشقوق الرأسية: width صغير وheight كبير — هذا صحيح، لا نعكسه
    - للشقوق الأفقية: height صغير وwidth كبير — هذا صحيح أيضاً
    - المشكلة فقط عندما يكون كلاهما صغير جداً
    """
    x = max(0.0, min(0.95, float(bbox.get("x", 0))))
    y = max(0.0, min(0.95, float(bbox.get("y", 0))))
    w = float(bbox.get("width", 0.05))
    h = float(bbox.get("height", 0.05))

    # إذا كان الـ bbox صغير جداً في كلا الاتجاهين — وسّعه
    if w < min_size and h < min_size:
        w = min_size
        h = min_size

    # إذا كان أحد الأبعاد صغير جداً مقارنة بالآخر بشكل غير منطقي
    # (مثلاً width=0.01 وheight=0.6 — هذا شرخ رأسي وهو صحيح)
    # لكن إذا width=0.01 وheight=0.01 — هذا خطأ
    # الحد الأدنى المطلق لكل بُعد هو 0.03
    w = max(0.03, min(1.0 - x, w))
    h = max(0.03, min(1.0 - y, h))

    # إضافة padding بسيط (2%) حول الشرخ لضمان الإحاطة الكاملة
    pad = 0.02
    x = max(0.0, x - pad)
    y = max(0.0, y - pad)
    w = min(1.0 - x, w + pad * 2)
    h = min(1.0 - y, h + pad * 2)

    return {"x": x, "y": y, "width": w, "height": h}


DETECT_PROMPT = """Analyze this image and detect ALL visible cracks, fractures, and surface defects.

Respond with ONLY a JSON object - no text, no markdown, no explanation.

CRITICAL BOUNDING BOX INSTRUCTIONS:
- Values are normalized 0.0 to 1.0 (fraction of image width/height)
- x = left edge of the FULL crack from start to end
- y = top edge of the FULL crack from start to end
- width = horizontal distance from leftmost to rightmost point of the FULL crack
- height = vertical distance from topmost to bottommost point of the FULL crack
- The bbox must FULLY CONTAIN the entire crack path from one end to the other
- For VERTICAL cracks: height will be LARGE (e.g. 0.6-0.9), width will be small (e.g. 0.05-0.15)
- For HORIZONTAL cracks: width will be LARGE (e.g. 0.5-0.9), height will be small
- For DIAGONAL cracks: both width and height will be moderate to large
- NEVER use height < 0.1 for a crack that runs vertically through the image
- NEVER use width < 0.1 for a crack that runs horizontally through the image

Example for a vertical crack running most of the image height:
{"detected":[{"id":1,"bbox":{"x":0.35,"y":0.05,"width":0.12,"height":0.88},"confidence":92,"rough_type":"vertical crack","rough_severity":"HIGH"}],"total":1,"image_quality":"good","surface_visible":"concrete"}

Example for a diagonal crack:
{"detected":[{"id":1,"bbox":{"x":0.10,"y":0.08,"width":0.75,"height":0.80},"confidence":88,"rough_type":"diagonal crack","rough_severity":"HIGH"}],"total":1,"image_quality":"good","surface_visible":"asphalt"}

If no cracks found: {"detected":[],"total":0,"image_quality":"good","surface_visible":"unknown"}

Output JSON only:"""


ANALYZE_PROMPT_TPL = """You are a structural engineer. Analyze these cracks found in the image.

Detected cracks:
{boxes}

Image size: {w}x{h}px

Output ONLY this JSON structure with Arabic text fields. No markdown. No explanation.

{{"summary":"ملخص الحالة","overall_severity":"HIGH","overall_confidence":85,"material_type":"أسفلت","surface_condition":"وصف السطح","environmental_factors":"","cracks":[{{"id":1,"bbox":{{"x":0.1,"y":0.05,"width":0.12,"height":0.88}},"type":"شرخ رأسي","category":"structural","is_structural":true,"estimated_width_mm":"1-2","estimated_length_cm":"20-30","depth_assessment":"متوسط","severity":"HIGH","confidence":85,"description":"وصف","cause_analysis":"سبب","progression_risk":"متوسط","immediate_action":"إجراء"}}],"recommendations":[{{"priority":1,"action":"إجراء","timeline":"أسبوع","estimated_cost_level":"متوسط","details":"تفاصيل"}}],"monitoring_plan":"خطة","professional_consultation_required":true,"notes":""}}

IMPORTANT: Use the EXACT bbox values from the detected cracks above. Do NOT change them.
overall_severity = CRITICAL/HIGH/MEDIUM/LOW only. JSON only:"""


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
                temperature=round(0.1 + attempt * 0.15, 2),
            )
            raw = response.choices[0].message.content or ""
            print(f"[DETECT {attempt+1}] raw={raw[:500]}")
            result = _parse_json(raw)
            if result and isinstance(result.get("detected"), list):
                print(f"[DETECT {attempt+1}] OK — {len(result['detected'])} cracks")
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
        boxes += (f"#{d['id']}: x={b.get('x',0):.3f} y={b.get('y',0):.3f} "
                  f"w={b.get('width',0):.3f} h={b.get('height',0):.3f} "
                  f"| {d.get('rough_type','crack')} | {d.get('rough_severity','MEDIUM')}\n")

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
                temperature=round(0.2 + attempt * 0.1, 2),
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
            b["bbox"] = {"x": b.pop("x", 0), "y": b.pop("y", 0),
                         "width": b.pop("width", 0.05), "height": b.pop("height", 0.05)}
        b["_conf"] = float(b.get("confidence", 75)) / 100.0

    final = []
    for i, det in enumerate(raw_boxes):
        # تطبيق الإصلاح الذكي للـ bbox
        det["bbox"] = _fix_bbox(det["bbox"])
        det["id"] = i + 1
        print(f"[PIPELINE] crack #{i+1} bbox: {det['bbox']}")
        final.append(det)

    total = len(final)
    print(f"[PIPELINE] total detections: {total}")

    if total == 0:
        return {
            "total_cracks_detected": 0,
            "summary": "لم يتم الكشف عن أي شروخ أو شقوق. السطح يبدو بحالة جيدة.",
            "overall_severity": "LOW", "overall_confidence": 90,
            "material_type": raw_result.get("surface_visible", "غير محدد"),
            "surface_condition": "السطح في حالة جيدة", "environmental_factors": "",
            "cracks": [],
            "recommendations": [{"priority": 1, "action": "الصيانة الوقائية الدورية",
                                  "timeline": "كل 6-12 شهراً", "estimated_cost_level": "منخفض",
                                  "details": "فحص دوري للحفاظ على الحالة الجيدة"}],
            "monitoring_plan": "فحص بصري كل 6 أشهر",
            "professional_consultation_required": False, "notes": "",
            "_detection_info": {"gemini_detected": 0}
        }

    analysis = _analyze(image_base64, final, img_width, img_height)
    if analysis is None:
        sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        all_sevs = [d.get("rough_severity", "MEDIUM") for d in final]
        overall = next((s for s in sev_order if s in all_sevs), "MEDIUM")
        analysis = {
            "summary": f"تم اكتشاف {len(final)} شرخ في السطح.",
            "overall_severity": overall, "overall_confidence": 70,
            "material_type": "غير محدد", "surface_condition": "يوجد شروخ",
            "environmental_factors": "", "cracks": [], "recommendations": [],
            "monitoring_plan": "", "professional_consultation_required": True, "notes": ""
        }

    analysis["total_cracks_detected"] = total
    analysis["_detection_info"] = {"gemini_detected": total}

    # مزامنة الـ bbox من الكشف إلى التحليل مع تطبيق الإصلاح
    if analysis.get("cracks"):
        for i, crack in enumerate(analysis["cracks"]):
            if i < len(final):
                crack["bbox"] = final[i]["bbox"]  # استخدم bbox المُصحَّح دائماً
            crack["id"] = i + 1
    else:
        analysis["cracks"] = [
            {"id": d["id"], "bbox": d["bbox"],
             "type": d.get("rough_type", "شرخ"), "category": "structural",
             "is_structural": True, "estimated_width_mm": "غير محدد",
             "estimated_length_cm": "غير محدد", "depth_assessment": "غير محدد",
             "severity": d.get("rough_severity", "MEDIUM"),
             "confidence": int(d.get("_conf", 0.7) * 100),
             "description": "تم الكشف بواسطة الذكاء الاصطناعي",
             "cause_analysis": "يتطلب فحصاً ميدانياً",
             "progression_risk": "متوسط", "immediate_action": "فحص ميداني"}
            for d in final
        ]

    return analysis


def generate_dashboard_recommendations(records_summary):
    client = _get_client()
    prompt = (
        "Based on these crack records give maintenance recommendations in Arabic.\n"
        "Return ONLY JSON no markdown:\n"
        '{"overall_assessment":"تقييم","priority_actions":["إجراء"],'
        '"maintenance_schedule":"الجدول","budget_estimate":"متوسط",'
        '"risk_areas":["منطقة"],"preventive_measures":["إجراء وقائي"]}\n\n'
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
