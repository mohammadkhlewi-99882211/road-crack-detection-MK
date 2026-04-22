import os
import json
import base64
import google.generativeai as genai

def get_gemini_client():
    """تهيئة عميل Gemini باستخدام مفتاح API من متغيرات البيئة."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("الرجاء ضبط متغير البيئة GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    # استخدام موديل gemini-1.5-pro لقدراته العالية في تحليل الصور
    return genai.GenerativeModel('gemini-1.5-pro')

def _parse_json_response(text):
    """استخراج وتحليل JSON من نص استجابة النموذج."""
    if not text:
        return None
    text = text.strip()
    # تنظيف علامات Markdown البرمجية إذا وجدت
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```json"):
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:-1])
    
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None

# تعليمات نظام الكشف
DETECTION_SYSTEM = """أنت نظام رؤية حاسوبية دقيق متخصص في اكتشاف الشقوق الإنشائية وعيوب الأسطح في الخرسانة والأسفلت والمباني.
مهمتك الوحيدة: اكتشاف كل شرخ أو كسر مرئي وإرجاع إحداثيات دقيقة للصناديق المحيطة (Bounding Boxes).

قواعد الصناديق المحيطة:
- استخدم إحداثيات معيرة (Normalized): x, y, width, height جميعها في النطاق [0.0, 1.0].
- x = الحافة اليسرى (0 = أقصى اليسار، 1 = أقصى اليمين).
- y = الحافة العلوية (0 = أعلى الصورة، 1 = أسفلها).
- width = العرض الأفقي لمنطقة الشرخ.
- height = الارتفاع الرأسي لمنطقة الشرخ.
- اجعل الصناديق ضيقة (Tight) حول الشرخ مباشرة.

حساسية الاكتشاف:
- اكتشف جميع الشقوق بغض النظر عن حجمها (حتى الشقوق الشعرية الدقيقة).
- لا تشمل الظلال أو الفواصل الإنشائية المتعمدة.
"""

DETECTION_PROMPT = """افحص هذه الصورة بعناية لاستخراج جميع الشقوق والكسور.
أرجع النتيجة فقط بتنسيق JSON التالي:
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
    "total": 1,
    "surface_visible": "concrete/asphalt/etc"
}

إذا لم توجد شقوق، أرجع: {"detected": [], "total": 0, "surface_visible": "unknown"}"""

# تعليمات نظام التحليل الهندسي
ANALYSIS_SYSTEM = """أنت مهندس إنشائي خبير وخبير في أمراض الخرسانة.
بناءً على مواقع الشقوق المكتشفة، قدم تحليلاً هندسياً دقيقاً باللغة العربية.

تصنيف الشقوق:
- إنشائية (Structural): تخترق جسم المادة الحاملة.
- سطحية (Surface): تقتصر على طبقات التشطيب.

الخطورة (Severity):
- حرجة (CRITICAL): عرض > 2 مم، إزاحة نشطة، صدأ حديد.
- عالية (HIGH): 0.5 - 2 مم، أنماط إنشائية.
- متوسطة (MEDIUM): 0.1 - 0.5 مم، شقوق شعرية إنشائية.
- منخفضة (LOW): تجميلية فقط، < 0.1 مم.
"""

def _gemini_detect(model, image_base64):
    """استخدام Gemini لاكتشاف الشقوق وتحديد مواقعها."""
    try:
        image_data = base64.b64decode(image_base64)
        contents = [
            {"mime_type": "image/jpeg", "data": image_data},
            DETECTION_SYSTEM + "\n\n" + DETECTION_PROMPT
        ]
        response = model.generate_content(contents)
        return _parse_json_response(response.text) or {"detected": [], "total": 0}
    except Exception as e:
        print(f"Error in detection: {e}")
        return {"detected": [], "total": 0}

def _gemini_analyze(model, image_base64, detections, img_w, img_h):
    """استخدام Gemini لإجراء التحليل الهندسي المفصل باللغة العربية."""
    boxes_desc = ""
    for d in detections:
        bbox = d.get("bbox", {})
        boxes_desc += (
            f"\nالشرخ #{d.get('id')}: "
            f"x={bbox.get('x'):.3f}, y={bbox.get('y'):.3f}, "
            f"w={bbox.get('width'):.3f}, h={bbox.get('height'):.3f}"
        )

    prompt = f"""تم اكتشاف مناطق الشقوق التالية بواسطة نظام الرؤية:
{boxes_desc}

أبعاد الصورة: {img_w}x{img_h} بكسل.

قدم تحليلاً هندسياً إنشائياً خبيراً. أرجع النتيجة فقط بتنسيق JSON (باللغة العربية للحقول النصية):
{{
    "summary": "ملخص شامل للحالة",
    "overall_severity": "CRITICAL/HIGH/MEDIUM/LOW",
    "overall_confidence": 90,
    "material_type": "نوع المادة",
    "surface_condition": "وصف حالة السطح",
    "cracks": [
        {{
            "id": 1,
            "type": "نوع الشرخ",
            "category": "structural/surface",
            "is_structural": true,
            "estimated_width_mm": "1.5",
            "severity": "HIGH",
            "description": "وصف دقيق للشرخ",
            "cause_analysis": "تحليل السبب المحتمل",
            "immediate_action": "الإجراء الفوري المطلوب"
        }}
    ],
    "recommendations": [
        {{
            "priority": 1,
            "action": "الإجراء المقترح",
            "timeline": "الجدول الزمني",
            "details": "تفاصيل إضافية"
        }}
    ],
    "monitoring_plan": "خطة المراقبة المقترحة"
}}"""

    try:
        image_data = base64.b64decode(image_base64)
        contents = [
            {"mime_type": "image/jpeg", "data": image_data},
            ANALYSIS_SYSTEM + "\n\n" + prompt
        ]
        response = model.generate_content(contents)
        result = _parse_json_response(response.text)
        if result:
            return result
    except Exception as e:
        print(f"Error in analysis: {e}")
    
    return {"summary": "تعذر إجراء التحليل التفصيلي حالياً."}

def detect_and_analyze(image_base64, img_width, img_height):
    """الدالة الرئيسية التي تجمع بين الاكتشاف والتحليل."""
    model = get_gemini_client()
    
    # 1. الاكتشاف
    detection_result = _gemini_detect(model, image_base64)
    raw_boxes = detection_result.get("detected", [])
    
    # معالجة وتصحيح الإحداثيات
    final_detections = []
    for i, det in enumerate(raw_boxes):
        bbox = det.get("bbox", det)
        x = max(0.0, min(0.98, float(bbox.get("x", 0))))
        y = max(0.0, min(0.98, float(bbox.get("y", 0))))
        w = max(0.02, min(1.0 - x, float(bbox.get("width", 0.05))))
        h = max(0.02, min(1.0 - y, float(bbox.get("height", 0.05))))
        
        final_detections.append({
            "id": i + 1,
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "rough_type": det.get("rough_type", "شرخ"),
            "rough_severity": det.get("rough_severity", "MEDIUM")
        })
    
    # 2. التحقق من وجود نتائج
    if not final_detections:
        return {
            "total_cracks_detected": 0,
            "summary": "لم يتم الكشف عن أي شروخ واضحة في الصورة. السطح يبدو بحالة جيدة.",
            "overall_severity": "LOW",
            "cracks": [],
            "recommendations": [
                {"priority": 1, "action": "مراقبة دورية", "timeline": "6-12 شهر", "details": "الحفاظ على نظافة السطح ومراقبته"}
            ]
        }
    
    # 3. التحليل الهندسي
    analysis = _gemini_analyze(model, image_base64, final_detections, img_width, img_height)
    
    # دمج بيانات الصناديق المحيطة مع نتائج التحليل
    analysis["total_cracks_detected"] = len(final_detections)
    if "cracks" in analysis:
        for i, crack in enumerate(analysis["cracks"]):
            if i < len(final_detections):
                crack["bbox"] = final_detections[i]["bbox"]
    else:
        # بناء قائمة الشقوق في حال فشل التحليل في توليدها
        analysis["cracks"] = []
        for det in final_detections:
            analysis["cracks"].append({
                "id": det["id"],
                "bbox": det["bbox"],
                "type": det["rough_type"],
                "severity": det["rough_severity"],
                "description": "تم الكشف عن الشرخ بواسطة نظام الرؤية"
            })
            
    return analysis

def generate_dashboard_recommendations(records_summary):
    """توليد توصيات شاملة بناءً على سجلات متعددة."""
    model = get_gemini_client()
    
    prompt = f"""بناءً على ملخص سجلات الشقوق التالية، قدم توصيات صيانة شاملة باللغة العربية:
{records_summary}

أرجع النتيجة فقط بتنسيق JSON:
{{
    "overall_assessment": "تقييم شامل للحالة العامة",
    "priority_actions": ["إجراء 1", "إجراء 2"],
    "maintenance_schedule": "جدول الصيانة المقترح",
    "budget_estimate": "منخفض/متوسط/عالي",
    "risk_areas": ["منطقة 1"],
    "preventive_measures": ["إجراء وقائي 1"]
}}"""

    try:
        response = model.generate_content(prompt)
        return _parse_json_response(response.text)
    except Exception:
        return {"overall_assessment": "تعذر توليد التوصيات العامة حالياً."}
