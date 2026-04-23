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


def _points_to_bbox(x1, y1, x2, y2, pad=0.03):
    """تحويل نقطتي البداية والنهاية إلى bbox يحيط بالشرخ كاملاً."""
    left   = max(0.0, min(x1, x2) - pad)
    top    = max(0.0, min(y1, y2) - pad)
    right  = min(1.0, max(x1, x2) + pad)
    bottom = min(1.0, max(y1, y2) + pad)
    return {
        "x":      round(left,         4),
        "y":      round(top,          4),
        "width":  round(right - left, 4),
        "height": round(bottom - top, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DETECT PROMPT
# ─────────────────────────────────────────────────────────────────────────────

DETECT_PROMPT = """You are an expert structural damage detection system.

Inspect this image and detect ALL damage: cracks, spalling, paint peeling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAMAGE TYPES — CLASSIFY CAREFULLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRUCTURAL_CRACK — deep crack penetrating the material
  • Dark, sharp, well-defined edges with visible depth or shadow inside
  • Width > 0.3 mm, cannot be scratched away
  • May show displacement or offset on either side
  • Severity: HIGH or CRITICAL

SURFACE_CRACK — shallow crack in top layer only
  • Fine thin lines, no visible depth, map/crazing patterns
  • No displacement between edges
  • Severity: LOW or MEDIUM

PAINT_PEELING — paint or coating flaking off only
  • Shows a different colour underneath, edges lifting
  • No structural damage to base material
  • Severity: LOW

SPALLING — chunks of material breaking away
  • Rough surface with material loss, may expose aggregate or rebar
  • Severity: MEDIUM or HIGH

DO NOT detect: shadows, construction joints, intentional grooves, uniform texture.
Shadows have soft edges. Cracks have sharp, consistent edges along their full length.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO REPORT LOCATION — READ CAREFULLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For every crack or damage area provide exactly TWO points:

  start_point  →  one end of the crack / one corner of the damage area
  end_point    →  the OPPOSITE end of the crack / opposite corner

All x, y values are normalised 0.0–1.0 (fraction of image width / height).
x=0 is left edge, x=1 is right edge, y=0 is top edge, y=1 is bottom edge.

These two points must span the FULL extent of the damage — from the very
beginning to the very end, not just a small section of it.

Examples:
  Diagonal crack (top-left → bottom-right):
    start_point: {"x":0.05,"y":0.08}   end_point: {"x":0.88,"y":0.92}

  Vertical crack (top → bottom):
    start_point: {"x":0.47,"y":0.02}   end_point: {"x":0.53,"y":0.97}

  Horizontal crack (left → right):
    start_point: {"x":0.02,"y":0.50}   end_point: {"x":0.96,"y":0.54}

  Small local spalling:
    start_point: {"x":0.30,"y":0.25}   end_point: {"x":0.55,"y":0.50}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ONLY, NO MARKDOWN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{"detected":[{"id":1,"start_point":{"x":0.05,"y":0.08},"end_point":{"x":0.88,"y":0.92},"confidence":90,"damage_type":"STRUCTURAL_CRACK","rough_severity":"HIGH","notes":"deep diagonal crack with shadow inside"}],"total":1,"image_quality":"good","surface_visible":"concrete"}

damage_type : STRUCTURAL_CRACK / SURFACE_CRACK / PAINT_PEELING / SPALLING
rough_severity : CRITICAL / HIGH / MEDIUM / LOW

If no real damage: {"detected":[],"total":0,"image_quality":"good","surface_visible":"unknown"}

Before outputting — verify:
1. Is each item a REAL crack, not a shadow or joint?
2. Do start_point and end_point mark the FULL length of the damage?
3. Is damage_type correct?

JSON only:"""


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYZE PROMPT
# ─────────────────────────────────────────────────────────────────────────────

ANALYZE_PROMPT_TPL = """You are a senior structural engineer specialising in concrete pathology.

Pre-detected damage (AI vision):
{boxes}

Image: {w}×{h} px

Output ONLY the JSON below. Arabic text fields. No markdown outside the JSON.

{{"summary":"ملخص شامل في 3-4 جمل","overall_severity":"HIGH","overall_confidence":85,"material_type":"نوع المادة","surface_condition":"وصف السطح","environmental_factors":"العوامل البيئية","cracks":[{{"id":1,"bbox":{{"x":0.05,"y":0.08,"width":0.86,"height":0.87}},"type":"شرخ إنشائي قطري عميق","damage_type":"STRUCTURAL_CRACK","category":"structural","is_structural":true,"estimated_width_mm":"2-3","estimated_length_cm":"45-55","depth_assessment":"عميق يخترق طبقة الخرسانة","severity":"HIGH","confidence":88,"description":"وصف الشرخ","cause_analysis":"سبب محتمل","progression_risk":"عالي","immediate_action":"إجراء فوري"}}],"recommendations":[{{"priority":1,"action":"الإجراء","timeline":"الجدول","estimated_cost_level":"متوسط","details":"التفاصيل"}}],"monitoring_plan":"خطة المراقبة","professional_consultation_required":true,"notes":""}}

Rules:
- Use bbox values exactly as provided — do NOT change them
- overall_severity = CRITICAL / HIGH / MEDIUM / LOW only
- PAINT_PEELING → is_structural=false, category="cosmetic", severity=LOW
- SURFACE_CRACK → is_structural=false, category="surface"
- STRUCTURAL_CRACK → is_structural=true
- All text in Arabic. JSON only:"""


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _detect(image_base64):
    print("[DETECT] starting")
    try:
        client  = _get_client()
        pil_img = _b64_to_pil(image_base64)
        img_b64 = base64.b64encode(_pil_to_bytes(pil_img)).decode()
    except Exception as e:
        print(f"[DETECT] setup error: {e}")
        return {"detected": [], "total": 0}

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": DETECT_PROMPT},
                ]}],
                max_tokens=2048,
                temperature=round(0.05 + attempt * 0.1, 2),
            )
            raw    = resp.choices[0].message.content or ""
            print(f"[DETECT {attempt+1}] raw={raw[:500]}")
            result = _parse_json(raw)
            if result and isinstance(result.get("detected"), list):
                print(f"[DETECT {attempt+1}] OK — {len(result['detected'])} items")
                return result
            print(f"[DETECT {attempt+1}] parse failed")
        except Exception as e:
            print(f"[DETECT {attempt+1}] error: {e}")
            traceback.print_exc()

    return {"detected": [], "total": 0}


def _analyze(image_base64, detections, img_w, img_h):
    print("[ANALYZE] starting")
    try:
        client  = _get_client()
        pil_img = _b64_to_pil(image_base64)
        img_b64 = base64.b64encode(_pil_to_bytes(pil_img)).decode()
    except Exception as e:
        print(f"[ANALYZE] setup error: {e}")
        return None

    boxes = ""
    for d in detections:
        b = d["bbox"]
        boxes += (f"#{d['id']}: bbox x={b['x']:.3f} y={b['y']:.3f} "
                  f"w={b['width']:.3f} h={b['height']:.3f} | "
                  f"type={d.get('damage_type','STRUCTURAL_CRACK')} | "
                  f"severity={d.get('rough_severity','MEDIUM')} | "
                  f"notes={d.get('notes','')}\n")

    prompt = ANALYZE_PROMPT_TPL.format(boxes=boxes, w=img_w, h=img_h)

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=4096,
                temperature=round(0.15 + attempt * 0.1, 2),
            )
            raw    = resp.choices[0].message.content or ""
            print(f"[ANALYZE {attempt+1}] raw={raw[:300]}")
            result = _parse_json(raw)
            if result and "summary" in result:
                return result
        except Exception as e:
            print(f"[ANALYZE {attempt+1}] error: {e}")
            traceback.print_exc()

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_analyze(image_base64, img_width, img_height):
    print("[PIPELINE] start")
    raw_result = _detect(image_base64)
    raw_boxes  = raw_result.get("detected", [])

    final = []
    for i, det in enumerate(raw_boxes):
        det["_conf"] = float(det.get("confidence", 75)) / 100.0
        det["id"]    = i + 1

        sp = det.get("start_point", {})
        ep = det.get("end_point",   {})

        if sp and ep and "x" in sp and "x" in ep:
            # ✅ نقطتان — نحسب الـ bbox بالكود (أدق بكثير)
            bbox = _points_to_bbox(
                float(sp.get("x", 0)), float(sp.get("y", 0)),
                float(ep.get("x", 0)), float(ep.get("y", 0)),
                pad=0.025,
            )
        elif "bbox" in det:
            # bbox مباشر من النموذج — نضيف padding فقط
            b  = det["bbox"]
            px = 0.025
            x  = max(0.0, float(b.get("x", 0))     - px)
            y  = max(0.0, float(b.get("y", 0))     - px)
            w  = min(1.0 - x, float(b.get("width",  0.1)) + px * 2)
            h  = min(1.0 - y, float(b.get("height", 0.1)) + px * 2)
            bbox = {"x": round(x,4), "y": round(y,4),
                    "width": round(max(0.04,w),4), "height": round(max(0.04,h),4)}
        else:
            print(f"[PIPELINE] item #{i+1} skipped — no coordinates")
            continue

        det["bbox"] = bbox
        print(f"[PIPELINE] #{i+1} {det.get('damage_type','?')} "
              f"{det.get('rough_severity','?')} bbox={bbox}")
        final.append(det)

    total = len(final)
    print(f"[PIPELINE] total valid: {total}")

    # ── لا أضرار ──────────────────────────────────────────────────────────────
    if total == 0:
        return {
            "total_cracks_detected": 0,
            "summary": "لم يتم الكشف عن أي شروخ أو أضرار. السطح يبدو بحالة جيدة.",
            "overall_severity": "LOW", "overall_confidence": 90,
            "material_type": raw_result.get("surface_visible", "غير محدد"),
            "surface_condition": "السطح في حالة جيدة", "environmental_factors": "",
            "cracks": [],
            "recommendations": [{"priority": 1, "action": "الصيانة الوقائية الدورية",
                                  "timeline": "كل 6-12 شهراً", "estimated_cost_level": "منخفض",
                                  "details": "فحص دوري للحفاظ على الحالة الجيدة"}],
            "monitoring_plan": "فحص بصري كل 6 أشهر",
            "professional_consultation_required": False, "notes": "",
            "_detection_info": {"gemini_detected": 0},
        }

    # ── التحليل ───────────────────────────────────────────────────────────────
    analysis = _analyze(image_base64, final, img_width, img_height)

    if analysis is None:
        sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        all_sevs  = [d.get("rough_severity", "MEDIUM") for d in final]
        overall   = next((s for s in sev_order if s in all_sevs), "MEDIUM")
        analysis  = {
            "summary": f"تم اكتشاف {len(final)} ضرر في السطح. يُنصح بالفحص الميداني.",
            "overall_severity": overall, "overall_confidence": 65,
            "material_type": "غير محدد", "surface_condition": "يوجد أضرار",
            "environmental_factors": "", "cracks": [],
            "recommendations": [{"priority": 1, "action": "فحص ميداني عاجل",
                                  "timeline": "خلال أسبوع", "estimated_cost_level": "متوسط",
                                  "details": "مراجعة مهندس إنشائي"}],
            "monitoring_plan": "متابعة شهرية",
            "professional_consultation_required": True, "notes": "",
        }

    analysis["total_cracks_detected"] = total
    analysis["_detection_info"] = {"gemini_detected": total}

    # ── مزامنة bbox المحسوب بالكود — لا نثق بما يُعيده النموذج في التحليل ────
    if analysis.get("cracks"):
        for i, crack in enumerate(analysis["cracks"]):
            if i < len(final):
                crack["bbox"] = final[i]["bbox"]   # ← الـ bbox المحسوب دائماً
            crack["id"] = i + 1
    else:
        analysis["cracks"] = [
            {"id": d["id"], "bbox": d["bbox"],
             "type": d.get("damage_type", "شرخ"),
             "damage_type": d.get("damage_type", "STRUCTURAL_CRACK"),
             "category": "surface" if d.get("damage_type") in
                         ("PAINT_PEELING", "SURFACE_CRACK") else "structural",
             "is_structural": d.get("damage_type") not in ("PAINT_PEELING", "SURFACE_CRACK"),
             "estimated_width_mm": "غير محدد", "estimated_length_cm": "غير محدد",
             "depth_assessment": "غير محدد",
             "severity": d.get("rough_severity", "MEDIUM"),
             "confidence": int(d.get("_conf", 0.7) * 100),
             "description": "تم الكشف بواسطة الذكاء الاصطناعي",
             "cause_analysis": "يتطلب فحصاً ميدانياً",
             "progression_risk": "متوسط", "immediate_action": "فحص ميداني"}
            for d in final
        ]

    return analysis


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def generate_dashboard_recommendations(records_summary):
    client = _get_client()
    prompt = (
        "Based on these crack records give maintenance recommendations in Arabic.\n"
        "Return ONLY JSON, no markdown:\n"
        '{"overall_assessment":"تقييم شامل","priority_actions":["إجراء 1","إجراء 2"],'
        '"maintenance_schedule":"جدول الصيانة","budget_estimate":"متوسط",'
        '"risk_areas":["منطقة خطر"],"preventive_measures":["إجراء وقائي"]}\n\n'
        f"Records:\n{records_summary}\n\nJSON only:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048, temperature=0.3,
        )
        result = _parse_json(resp.choices[0].message.content or "")
        if result:
            return result
    except Exception as e:
        print(f"[DASHBOARD] error: {e}")
    return {"overall_assessment": "تعذّر توليد التوصيات، يرجى المحاولة مرة أخرى."}
