import os
import json
import base64
import re
import io
import traceback
import requests
from PIL import Image
from groq import Groq

# ─────────────────────────────────────────────
#  النماذج
# ─────────────────────────────────────────────

GROQ_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
ROBOFLOW_MODEL   = "crack-detection-y5kyg-3ywwl/1"
ROBOFLOW_VERSION = 1

# ─────────────────────────────────────────────
#  Clients
# ─────────────────────────────────────────────

def _get_groq():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)

def _get_roboflow_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise ValueError("ROBOFLOW_API_KEY is not set")
    return key


# ─────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────

def _b64_to_pil(image_base64):
    data = base64.b64decode(image_base64)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    return img

def _pil_to_bytes(pil_img, max_px=1024):
    w, h = pil_img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), pil_img.size   # bytes, (w, h)

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


# ─────────────────────────────────────────────
#  STEP 1 — Roboflow: الكشف الدقيق عن الشروخ
# ─────────────────────────────────────────────

def _detect_roboflow(image_base64):
    """
    يستخدم Roboflow YOLO للكشف الدقيق عن الشروخ.
    يُرجع قائمة من الـ detections بـ bbox دقيق.
    """
    print("[ROBOFLOW] starting detection...")
    try:
        rf_key  = _get_roboflow_key()
        pil_img = _b64_to_pil(image_base64)
        img_bytes, (img_w, img_h) = _pil_to_bytes(pil_img, max_px=1024)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        url = (f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}"
               f"?api_key={rf_key}&confidence=30&overlap=50")

        response = requests.post(
            url,
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        print(f"[ROBOFLOW] response: {json.dumps(result)[:500]}")

        detections = []
        rf_w = result.get("image", {}).get("width",  img_w)
        rf_h = result.get("image", {}).get("height", img_h)

        for i, pred in enumerate(result.get("predictions", [])):
            # Roboflow يُرجع المركز (x,y) والعرض والارتفاع بالبكسل
            cx = pred["x"] / rf_w
            cy = pred["y"] / rf_h
            bw = pred["width"]  / rf_w
            bh = pred["height"] / rf_h

            # تحويل من مركز إلى زاوية علوية يسرى
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            w  = min(1.0 - x1, bw)
            h  = min(1.0 - y1, bh)

            detections.append({
                "id":             i + 1,
                "bbox":           {"x": round(x1,4), "y": round(y1,4),
                                   "width": round(w,4), "height": round(h,4)},
                "confidence":     round(pred.get("confidence", 0.5) * 100),
                "damage_type":    "STRUCTURAL_CRACK",
                "rough_severity": "HIGH",
                "class":          pred.get("class", "crack"),
                "_conf":          pred.get("confidence", 0.5),
            })

        print(f"[ROBOFLOW] detected {len(detections)} cracks")
        return detections, img_w, img_h

    except Exception as e:
        print(f"[ROBOFLOW] error: {e}")
        traceback.print_exc()
        return [], 0, 0


# ─────────────────────────────────────────────
#  STEP 2 — Groq: التحليل الهندسي العميق
# ─────────────────────────────────────────────

ANALYZE_PROMPT_TPL = """أنت مهندس إنشائي أول متخصص في علم أمراض الخرسانة وتقييم البنية التحتية.

قام نظام YOLO بالكشف عن الشروخ التالية في الصورة:
{boxes}

أبعاد الصورة: {w}×{h} بكسل
إجمالي الشروخ المكتشفة: {total}

مهمتك: تقديم تحليل هندسي احترافي شامل لكل شرخ مكتشف.

أخرج فقط JSON التالي بحقول عربية. لا markdown خارج JSON.

{{"summary":"ملخص شامل للحالة الإنشائية في 4-5 جمل يذكر أنواع الأضرار وأسبابها المحتملة ودرجة خطورتها الإجمالية","overall_severity":"HIGH","overall_confidence":88,"material_type":"نوع المادة الإنشائية","surface_condition":"وصف تفصيلي دقيق لحالة السطح","environmental_factors":"العوامل البيئية المرئية كالرطوبة والتعرية والتمدد الحراري","cracks":[{{"id":1,"bbox":{{"x":0.1,"y":0.2,"width":0.3,"height":0.4}},"type":"شرخ إنشائي قطري عميق","damage_type":"STRUCTURAL_CRACK","category":"structural","is_structural":true,"estimated_width_mm":"2-4","estimated_length_cm":"35-45","depth_assessment":"عميق — يخترق طبقة الخرسانة الأساسية بعمق يُقدَّر بـ 15-25 مم","severity":"HIGH","confidence":88,"description":"وصف دقيق ومفصل للشرخ وخصائصه المرئية وشكله واتجاهه","cause_analysis":"التحليل الهندسي للسبب المحتمل: إجهاد الانحناء، الحركة التفاضلية، التمدد الحراري، إلخ","progression_risk":"عالي — يتوقع تطور سريع في حال عدم المعالجة خاصة مع الرطوبة والأحمال المتكررة","immediate_action":"حقن الشرخ بالإيبوكسي أو حشوات البولي يوريثان وإزالة الأسباب الجذرية"}}],"recommendations":[{{"priority":1,"action":"الإجراء الأول والأهم","timeline":"خلال 48 ساعة","estimated_cost_level":"متوسط","details":"تفاصيل تقنية دقيقة للتنفيذ الميداني"}},{{"priority":2,"action":"الإجراء الثاني","timeline":"خلال أسبوع","estimated_cost_level":"منخفض","details":"تفاصيل"}}],"monitoring_plan":"خطة مراقبة تفصيلية: التردد، الأدوات المستخدمة، المؤشرات الحرجة التي تستوجب تدخلاً فورياً","professional_consultation_required":true,"structural_risk_assessment":"تقييم شامل لمخاطر الهيكل الإنشائي وسلامة المنشأة","notes":"ملاحظات إضافية مهمة للمهندس الميداني"}}

قواعد مهمة:
- استخدم قيم bbox المُعطاة بالضبط دون تعديل
- overall_severity = CRITICAL / HIGH / MEDIUM / LOW فقط
- كل الحقول النصية بالعربية
- قدّم تحليلاً هندسياً حقيقياً وليس وصفاً عاماً
- JSON فقط:"""


def _analyze_groq(image_base64, detections, img_w, img_h):
    """
    يستخدم Groq + LLaMA Vision للتحليل الهندسي العميق
    بناءً على الشروخ المكتشفة من Roboflow.
    """
    print("[GROQ] starting engineering analysis...")
    try:
        client  = _get_groq()
        pil_img = _b64_to_pil(image_base64)
        img_bytes, _ = _pil_to_bytes(pil_img, max_px=1024)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"[GROQ] setup error: {e}")
        return None

    boxes = ""
    for d in detections:
        b = d["bbox"]
        boxes += (f"#{d['id']}: bbox=(x={b['x']:.3f}, y={b['y']:.3f}, "
                  f"w={b['width']:.3f}, h={b['height']:.3f}) | "
                  f"نوع={d.get('class','crack')} | "
                  f"ثقة={d.get('confidence',75)}%\n")

    prompt = ANALYZE_PROMPT_TPL.format(
        boxes=boxes, w=img_w, h=img_h, total=len(detections)
    )

    for attempt in range(2):
        try:
            print(f"[GROQ] analysis attempt {attempt+1}...")
            resp = _get_groq().chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=4096,
                temperature=round(0.15 + attempt * 0.1, 2),
            )
            raw    = resp.choices[0].message.content or ""
            print(f"[GROQ] raw={raw[:300]}")
            result = _parse_json(raw)
            if result and "summary" in result:
                print("[GROQ] analysis OK")
                return result
        except Exception as e:
            print(f"[GROQ] attempt {attempt+1} error: {e}")
            traceback.print_exc()

    return None


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def detect_and_analyze(image_base64, img_width, img_height):
    print("[PIPELINE] ═══ START ═══")

    # ── الخطوة 1: Roboflow — الكشف الدقيق ──────────────────────────────────
    detections, rf_w, rf_h = _detect_roboflow(image_base64)
    img_w = rf_w or img_width
    img_h = rf_h or img_height
    total = len(detections)
    print(f"[PIPELINE] Roboflow found {total} cracks")

    # ── لا شروخ ─────────────────────────────────────────────────────────────
    if total == 0:
        return {
            "total_cracks_detected": 0,
            "summary": "لم يتم الكشف عن أي شروخ أو أضرار. السطح يبدو بحالة جيدة.",
            "overall_severity": "LOW", "overall_confidence": 95,
            "material_type": "غير محدد",
            "surface_condition": "السطح في حالة جيدة بدون أضرار مرئية",
            "environmental_factors": "",
            "cracks": [],
            "recommendations": [{"priority": 1,
                                  "action": "الصيانة الوقائية الدورية",
                                  "timeline": "كل 6-12 شهراً",
                                  "estimated_cost_level": "منخفض",
                                  "details": "فحص بصري دوري للحفاظ على الحالة الجيدة"}],
            "monitoring_plan": "فحص بصري كل 6 أشهر",
            "professional_consultation_required": False,
            "structural_risk_assessment": "لا توجد مخاطر إنشائية مرئية",
            "notes": "",
            "_detection_info": {"roboflow_detected": 0, "gemini_detected": 0},
        }

    # ── الخطوة 2: Groq — التحليل الهندسي ────────────────────────────────────
    analysis = _analyze_groq(image_base64, detections, img_w, img_h)

    # ── Fallback إذا فشل التحليل ─────────────────────────────────────────────
    if analysis is None:
        analysis = {
            "summary": f"تم اكتشاف {total} شرخ في السطح بواسطة نظام YOLO. يُنصح بالفحص الميداني.",
            "overall_severity": "HIGH",
            "overall_confidence": 75,
            "material_type": "غير محدد",
            "surface_condition": "يوجد شروخ تستوجب الفحص",
            "environmental_factors": "",
            "cracks": [],
            "recommendations": [{"priority": 1,
                                  "action": "فحص ميداني عاجل",
                                  "timeline": "خلال أسبوع",
                                  "estimated_cost_level": "متوسط",
                                  "details": "مراجعة مهندس إنشائي لتقييم الشروخ المكتشفة"}],
            "monitoring_plan": "متابعة الشروخ شهرياً",
            "professional_consultation_required": True,
            "structural_risk_assessment": "يتطلب تقييماً ميدانياً",
            "notes": "",
        }

    analysis["total_cracks_detected"] = total
    analysis["_detection_info"] = {
        "roboflow_detected": total,
        "gemini_detected": total,
    }

    # ── مزامنة الـ bbox من Roboflow (لا نثق بما يُعيده النموذج) ──────────────
    if analysis.get("cracks"):
        for i, crack in enumerate(analysis["cracks"]):
            if i < len(detections):
                crack["bbox"] = detections[i]["bbox"]
            crack["id"] = i + 1
    else:
        analysis["cracks"] = [
            {"id":                  d["id"],
             "bbox":                d["bbox"],
             "type":                "شرخ إنشائي",
             "damage_type":         "STRUCTURAL_CRACK",
             "category":            "structural",
             "is_structural":       True,
             "estimated_width_mm":  "غير محدد",
             "estimated_length_cm": "غير محدد",
             "depth_assessment":    "يتطلب فحصاً ميدانياً",
             "severity":            "HIGH",
             "confidence":          d.get("confidence", 75),
             "description":         "تم الكشف بواسطة نظام YOLO",
             "cause_analysis":      "يتطلب فحصاً ميدانياً",
             "progression_risk":    "متوسط",
             "immediate_action":    "فحص ميداني"}
            for d in detections
        ]

    print(f"[PIPELINE] ═══ DONE — {total} cracks ═══")
    return analysis


# ─────────────────────────────────────────────
#  Dashboard
# ─────────────────────────────────────────────

def generate_dashboard_recommendations(records_summary):
    prompt = (
        "بناءً على سجلات الشروخ التالية، قدّم توصيات صيانة شاملة واحترافية باللغة العربية.\n"
        "أرجع JSON فقط بدون markdown:\n"
        '{"overall_assessment":"تقييم شامل ومفصل",'
        '"priority_actions":["إجراء 1","إجراء 2","إجراء 3"],'
        '"maintenance_schedule":"جدول صيانة مفصل",'
        '"budget_estimate":"تقدير الميزانية",'
        '"risk_areas":["منطقة خطر 1","منطقة خطر 2"],'
        '"preventive_measures":["إجراء وقائي 1","إجراء وقائي 2"]}\n\n'
        f"السجلات:\n{records_summary}\n\nJSON فقط:"
    )
    try:
        client = _get_groq()
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        result = _parse_json(resp.choices[0].message.content or "")
        if result:
            return result
    except Exception as e:
        print(f"[DASHBOARD] error: {e}")
    return {"overall_assessment": "تعذّر توليد التوصيات، يرجى المحاولة مرة أخرى."}
