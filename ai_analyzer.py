import os
import json
import base64
import re
import io
import traceback
import requests
from PIL import Image
from groq import Groq

# ───────────────────────
#  هذا القسم يلي استدعيت في النماذج اللي دربتها من Roboflow ونموذج جروك للتحليل
# ─────────────────────────────────────────────

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

ROBOFLOW_MODELS = [
    {"model": "crack-moovo-0ujw8",           "version": 1},
    {"model": "crack-vudec-v04jc",           "version": 1},
    {"model": "crack-detection-y5kyg-3ywwl", "version": 1},
]

IOU_THRESHOLD = 0.4   # حد التداخل لدمج النتائج المتكررة


# ───────────────────────...
#  التحقق من الهوية وأمان الاتصال
# ─────────────

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
#  هي الفقرة اللي بتحلل الصور ويتحولها لكيانات برمجية وتفحص الحجم والتوافق...الخ
# ─────────────────────────────────────────────

def _b64_to_pil(image_base64):
    data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def _pil_to_bytes(pil_img, max_px=1024):
    w, h = pil_img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), pil_img.size

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
                return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group()))
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────
#  هون دكتور انا واجهت مشاكل لما كان بدي ادمج 3 نماذج سوا لانو ممكن الصور تتداخل ويصير يعطي نتائج كارثية جدا متداخلة بشكل كبير 
# فاستعنت بدالة iou 
# وصارت تحتفظ بالموثوقية الاعلى دائما
# ─────────────────────────────────────────────

def _iou(b1, b2):
    """حساب نسبة التداخل بين bbox واحد"""
    x1 = max(b1["x"], b2["x"])
    y1 = max(b1["y"], b2["y"])
    x2 = min(b1["x"] + b1["width"],  b2["x"] + b2["width"])
    y2 = min(b1["y"] + b1["height"], b2["y"] + b2["height"])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    a1 = b1["width"] * b1["height"]
    a2 = b2["width"] * b2["height"]
    return inter / (a1 + a2 - inter)

def _merge_detections(all_detections):
    """
    دمج نتائج عدة نماذج وحذف التكرار بناءً على الدالة.
    الشرخ اللي يكتشفه أكثر من نموذج يحصل على أولوية أعلى.
    """
    merged = []
    for det in all_detections:
        duplicate = False
        for m in merged:
            if _iou(det["bbox"], m["bbox"]) > IOU_THRESHOLD:
                
                if det["_conf"] > m["_conf"]:
                    m.update(det)
                m["_model_count"] = m.get("_model_count", 1) + 1
                duplicate = True
                break
        if not duplicate:
            det["_model_count"] = 1
            merged.append(det)

   
    for i, d in enumerate(merged):
        d["id"] = i + 1
        d["dual_confirmed"] = d.get("_model_count", 1) > 1

    print(f"[MERGE] {len(all_detections)} raw → {len(merged)} unique")
    return merged


# ─────────────────────────────────────────────
#هي الفقرة هي الفقرة الي رح تبعت الصور للنماذج ال3 اللي دربناها من روبو فلو 
# ─────────────────────────────────────────────

def _call_single_model(img_b64, img_w, img_h, model_name, version, rf_key):
    """استدعاء نموذج Roboflow واحد وإرجاع نتائجه."""
    url = (f"https://detect.roboflow.com/{model_name}/{version}"
           f"?api_key={rf_key}&confidence=25&overlap=50")
    try:
        resp = requests.post(
            url,
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        rf_w = result.get("image", {}).get("width",  img_w)
        rf_h = result.get("image", {}).get("height", img_h)

        detections = []
        for pred in result.get("predictions", []):
            cx = pred["x"] / rf_w
            cy = pred["y"] / rf_h
            bw = pred["width"]  / rf_w
            bh = pred["height"] / rf_h
            x1 = max(0.0, cx - bw/2)
            y1 = max(0.0, cy - bh/2)
            w  = min(1.0 - x1, bw)
            h  = min(1.0 - y1, bh)
            detections.append({
                "bbox":           {"x": round(x1,4), "y": round(y1,4),
                                   "width": round(w,4), "height": round(h,4)},
                "confidence":     round(pred.get("confidence", 0.5) * 100),
                "damage_type":    "STRUCTURAL_CRACK",
                "rough_severity": "HIGH",
                "class":          pred.get("class", "crack"),
                "_conf":          pred.get("confidence", 0.5),
                "_source_model":  model_name,
            })
        print(f"[ROBOFLOW] {model_name}: {len(detections)} cracks")
        return detections
    except Exception as e:
        print(f"[ROBOFLOW] {model_name} error: {e}")
        return []


def _detect_roboflow(image_base64):
    """يستدعي كل النماذج ويدمج النتائج."""
    print("[ROBOFLOW] starting multi-model detection...")
    try:
        rf_key  = _get_roboflow_key()
        pil_img = _b64_to_pil(image_base64)
        img_bytes, (img_w, img_h) = _pil_to_bytes(pil_img, max_px=1024)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"[ROBOFLOW] setup error: {e}")
        traceback.print_exc()
        return [], 0, 0

    all_detections = []
    for m in ROBOFLOW_MODELS:
        dets = _call_single_model(
            img_b64, img_w, img_h,
            m["model"], m["version"], rf_key
        )
        all_detections.extend(dets)

    merged = _merge_detections(all_detections)
    return merged, img_w, img_h


# ─────────────────────────────────────────────
# هي الفقرة يلي رح تلاقي فيها دكتور الأمر اللي وجهنا لجروك مشان يصير قادر على اعطاء تحليل دقيق

ANALYZE_PROMPT_TPL = """أنت مهندس إنشائي أول متخصص في علم أمراض الخرسانة وتقييم البنية التحتية.

قام نظام YOLO متعدد النماذج بالكشف عن الشروخ التالية في الصورة:
{boxes}

أبعاد الصورة: {w}×{h} بكسل
إجمالي الشروخ المكتشفة: {total}

مهمتك: تقديم تحليل هندسي احترافي شامل.

أخرج فقط JSON التالي بحقول عربية. لا markdown خارج JSON.

{{"summary":"ملخص شامل 4-5 جمل","overall_severity":"HIGH","overall_confidence":88,"material_type":"نوع المادة","surface_condition":"وصف السطح","environmental_factors":"العوامل البيئية","cracks":[{{"id":1,"bbox":{{"x":0.1,"y":0.2,"width":0.3,"height":0.4}},"type":"شرخ إنشائي","damage_type":"STRUCTURAL_CRACK","category":"structural","is_structural":true,"estimated_width_mm":"2-4","estimated_length_cm":"35-45","depth_assessment":"عميق","severity":"HIGH","confidence":88,"description":"وصف الشرخ","cause_analysis":"سبب الشرخ","progression_risk":"عالي","immediate_action":"الإجراء الفوري"}}],"recommendations":[{{"priority":1,"action":"الإجراء","timeline":"خلال 48 ساعة","estimated_cost_level":"متوسط","details":"التفاصيل"}}],"monitoring_plan":"خطة المراقبة","professional_consultation_required":true,"structural_risk_assessment":"تقييم المخاطر","notes":""}}

قواعد:
- استخدم قيم bbox المُعطاة بالضبط
- overall_severity = CRITICAL/HIGH/MEDIUM/LOW فقط
- كل الحقول النصية بالعربية
- JSON فقط:"""


def _analyze_groq(image_base64, detections, img_w, img_h):
    print("[GROQ] starting engineering analysis...")
    try:
        pil_img = _b64_to_pil(image_base64)
        img_bytes, _ = _pil_to_bytes(pil_img, max_px=1024)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"[GROQ] setup error: {e}")
        return None

    boxes = ""
    for d in detections:
        b = d["bbox"]
        confirmed = "✓✓ مؤكد من نماذج متعددة" if d.get("dual_confirmed") else "✓ نموذج واحد"
        boxes += (f"#{d['id']}: bbox=(x={b['x']:.3f}, y={b['y']:.3f}, "
                  f"w={b['width']:.3f}, h={b['height']:.3f}) | "
                  f"نوع={d.get('class','crack')} | "
                  f"ثقة={d.get('confidence',75)}% | {confirmed}\n")

    prompt = ANALYZE_PROMPT_TPL.format(
        boxes=boxes, w=img_w, h=img_h, total=len(detections)
    )

    for attempt in range(2):
        try:
            print(f"[GROQ] attempt {attempt+1}...")
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
            raw = resp.choices[0].message.content or ""
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
#  وهي هي الدالة اللي بتطلع الصورة لمخرجات فيها تصنيف الحالة ودرجة خطورتها والتوصيات
# ─────────────────────────────────────────────

def detect_and_analyze(image_base64, img_width, img_height):
    print("[PIPELINE] ═══ START ═══")

    detections, rf_w, rf_h = _detect_roboflow(image_base64)
    img_w = rf_w or img_width
    img_h = rf_h or img_height
    total = len(detections)
    print(f"[PIPELINE] total unique cracks: {total}")

    if total == 0:
        return {
            "total_cracks_detected": 0,
            "summary": "لم يتم الكشف عن أي شروخ أو أضرار. السطح يبدو بحالة جيدة.",
            "overall_severity": "LOW", "overall_confidence": 95,
            "material_type": "غير محدد",
            "surface_condition": "السطح في حالة جيدة بدون أضرار مرئية",
            "environmental_factors": "", "cracks": [],
            "recommendations": [{"priority": 1, "action": "الصيانة الوقائية الدورية",
                                  "timeline": "كل 6-12 شهراً", "estimated_cost_level": "منخفض",
                                  "details": "فحص بصري دوري للحفاظ على الحالة الجيدة"}],
            "monitoring_plan": "فحص بصري كل 6 أشهر",
            "professional_consultation_required": False,
            "structural_risk_assessment": "لا توجد مخاطر إنشائية مرئية",
            "notes": "",
            "_detection_info": {"roboflow_detected": 0, "gemini_detected": 0},
        }

    analysis = _analyze_groq(image_base64, detections, img_w, img_h)

    if analysis is None:
        sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        all_sevs  = [d.get("rough_severity", "MEDIUM") for d in detections]
        overall   = next((s for s in sev_order if s in all_sevs), "HIGH")
        analysis  = {
            "summary": f"تم اكتشاف {total} شرخ بواسطة نظام YOLO متعدد النماذج.",
            "overall_severity": overall, "overall_confidence": 75,
            "material_type": "غير محدد", "surface_condition": "يوجد شروخ",
            "environmental_factors": "", "cracks": [],
            "recommendations": [{"priority": 1, "action": "فحص ميداني عاجل",
                                  "timeline": "خلال أسبوع", "estimated_cost_level": "متوسط",
                                  "details": "مراجعة مهندس إنشائي"}],
            "monitoring_plan": "متابعة شهرية",
            "professional_consultation_required": True,
            "structural_risk_assessment": "يتطلب تقييماً ميدانياً", "notes": "",
        }

    analysis["total_cracks_detected"] = total
    analysis["_detection_info"] = {
        "roboflow_detected": total,
        "gemini_detected": total,
        "models_used": len(ROBOFLOW_MODELS),
    }

    if analysis.get("cracks"):
        for i, crack in enumerate(analysis["cracks"]):
            if i < len(detections):
                crack["bbox"] = detections[i]["bbox"]
                crack["dual_confirmed"] = detections[i].get("dual_confirmed", False)
            crack["id"] = i + 1
    else:
        analysis["cracks"] = [
            {"id": d["id"], "bbox": d["bbox"],
             "type": "شرخ إنشائي", "damage_type": "STRUCTURAL_CRACK",
             "category": "structural", "is_structural": True,
             "estimated_width_mm": "غير محدد", "estimated_length_cm": "غير محدد",
             "depth_assessment": "يتطلب فحصاً ميدانياً", "severity": "HIGH",
             "confidence": d.get("confidence", 75),
             "dual_confirmed": d.get("dual_confirmed", False),
             "description": "تم الكشف بواسطة نظام YOLO متعدد النماذج",
             "cause_analysis": "يتطلب فحصاً ميدانياً",
             "progression_risk": "متوسط", "immediate_action": "فحص ميداني"}
            for d in detections
        ]

    print(f"[PIPELINE] ═══ DONE — {total} cracks ═══")
    return analysis


# ─────────────────────────────────────────────
#  وهذا هو الداشبورد اللي قادر يعطي توصيات وتقارير بناءً على البيانات اللي عنده
# ─────────────────────────────────────────────

def generate_dashboard_recommendations(records_summary):
    prompt = (
        "بناءً على سجلات الشروخ التالية، قدّم توصيات صيانة شاملة باللغة العربية.\n"
        "أرجع JSON فقط بدون markdown:\n"
        '{"overall_assessment":"تقييم شامل","priority_actions":["إجراء 1","إجراء 2"],'
        '"maintenance_schedule":"جدول الصيانة","budget_estimate":"متوسط",'
        '"risk_areas":["منطقة خطر"],"preventive_measures":["إجراء وقائي"]}\n\n'
        f"السجلات:\n{records_summary}\n\nJSON فقط:"
    )
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048, temperature=0.3,
        )
        result = _parse_json(resp.choices[0].message.content or "")
        if result:
            return result
    except Exception as e:
        print(f"[DASHBOARD] error: {e}")
    return {"overall_assessment": "تعذّر توليد التوصيات، يرجى المحاولة مرة أخرى."}
