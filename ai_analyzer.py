import os
import json
import base64
import re
import google.generativeai as genai
from google.generativeai import types

def get_gemini_client():
    """
    تهيئة عميل Gemini باستخدام مفتاح API من متغيرات البيئة.
    يتم استخدام نموذج gemini-1.5-pro لقدراته العالية في تحليل الصور والاكتشاف البصري الدقيق.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("الرجاء ضبط متغير البيئة GEMINI_API_KEY لضمان عمل التطبيق.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

def _parse_json_response(text):
    """
    استخراج وتحليل JSON من نص استجابة النموذج بمرونة عالية جداً.
    يقوم بتنظيف النص من أي علامات Markdown أو نصوص خارج نطاق الـ JSON.
    """
    if not text:
        return None
    
    text = text.strip()
    # البحث عن أول { وآخر } لضمان استخراج الـ JSON فقط
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # محاولة تنظيف إضافية للفواصل الزائدة التي قد يضيفها النموذج
            try:
                cleaned = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(cleaned)
            except:
                pass
    return None

# --- تعليمات النظام المتخصصة (System Prompts) ---

DETECTION_SYSTEM = """أنت نظام رؤية حاسوبية فائق الدقة متخصص في اكتشاف العيوب الإنشائية.
مهمتك الأساسية هي مسح الصورة واكتشاف كل شرخ (Crack) أو كسر أو تلف في الأسطح الخرسانية أو الأسفلتية.

يجب عليك إرجاع إحداثيات دقيقة لكل شرخ تكتشفه باستخدام نظام الصناديق المحيطة (Bounding Boxes).
- استخدم الإحداثيات المعيرة (Normalized) في النطاق [0, 1000].
- التنسيق المطلوب لكل صندوق هو: [ymin, xmin, ymax, xmax].
- 0 يمثل الحافة العلوية/اليسرى، و 1000 يمثل الحافة السفلية/اليمنى.

كن حساساً جداً واكتشف حتى الشقوق الشعرية الدقيقة (Hairline Cracks).
"""

ANALYSIS_SYSTEM = """أنت مهندس إنشائي خبير متخصص في تقييم سلامة المباني والطرق.
بناءً على الشقوق المكتشفة في الصورة، يجب عليك تقديم تحليل هندسي مفصل باللغة العربية يشمل:
1. نوع الشرخ (طولي، عرضي، تماسيح، شعري، إلخ).
2. تصنيف الشرخ (إنشائي Structural أو سطحي Surface).
3. مدى الخطورة (حرجة، عالية، متوسطة، منخفضة).
4. الأسباب المحتملة للشرخ (هبوط، تمدد حراري، أحمال زائدة، إلخ).
5. الإجراءات الفورية والتوصيات الفنية للصيانة.
"""

def detect_and_analyze(image_base64, img_width, img_height):
    """
    الدالة الرئيسية والكاملة لاكتشاف الشقوق وتحليلها هندسياً.
    تستخدم Gemini 1.5 Pro لإجراء العملية في خطوة واحدة متكاملة لضمان الدقة.
    """
    model = get_gemini_client()
    
    # برومبت مكثف واحترافي يجمع بين الاكتشاف والتحليل
    full_prompt = f"""
    افحص هذه الصورة بدقة هندسية واكتشف جميع الشقوق والعيوب.
    
    أولاً: الاكتشاف البصري (Detection):
    - حدد موقع كل شرخ بدقة باستخدام إحداثيات [ymin, xmin, ymax, xmax] في نطاق 0-1000.
    
    ثانياً: التحليل الهندسي (Engineering Analysis):
    - لكل شرخ مكتشف، قدم تحليلاً كاملاً باللغة العربية.
    
    يجب أن يكون الرد بتنسيق JSON حصراً كما يلي:
    {{
      "detected_cracks": [
        {{
          "id": 1,
          "box_2d": [ymin, xmin, ymax, xmax],
          "type": "نوع الشرخ (مثلاً: شرخ طولي إنشائي)",
          "severity": "CRITICAL/HIGH/MEDIUM/LOW",
          "is_structural": true/false,
          "width_estimate_mm": "تقدير العرض بالمليمتر",
          "description": "وصف هندسي دقيق للحالة",
          "cause": "السبب المحتمل للشرخ",
          "action": "الإجراء الفوري المطلوب"
        }}
      ],
      "overall_report": {{
        "summary": "ملخص شامل للحالة الإنشائية للسطح",
        "material_type": "نوع المادة (خرسانة، أسفلت، إلخ)",
        "overall_severity": "CRITICAL/HIGH/MEDIUM/LOW",
        "confidence_score": 95,
        "recommendations": [
          {{
            "priority": 1,
            "title": "عنوان التوصية",
            "details": "تفاصيل التوصية الفنية",
            "timeline": "الجدول الزمني المقترح"
          }}
        ],
        "monitoring_plan": "خطة المراقبة المقترحة"
      }}
    }}
    
    إذا لم تجد أي شقوق، أرجع قائمة "detected_cracks" فارغة مع ملخص يؤكد سلامة السطح.
    """

    try:
        # تحويل الصورة إلى تنسيق Blob المتوافق مع Gemini
        image_bytes = base64.b64decode(image_base64)
        image_blob = {
            'mime_type': 'image/jpeg',
            'data': image_bytes
        }
        
        # إعدادات التوليد المتقدمة لضمان استجابة احترافية
        generation_config = {
            "temperature": 0.1, # حرارة منخفضة جداً لضمان الدقة وعدم "التخيل"
            "top_p": 0.95,
            "max_output_tokens": 8192, # زيادة التوكينز للسماح بتقارير مفصلة
            "response_mime_type": "application/json" # إجبار الموديل على إخراج JSON سليم
        }
        
        # إرسال الطلب لـ Gemini
        response = model.generate_content(
            contents=[DETECTION_SYSTEM + "\n" + ANALYSIS_SYSTEM + "\n" + full_prompt, image_blob],
            generation_config=generation_config
        )
        
        # تحليل الرد
        raw_result = _parse_json_response(response.text)
        
        if not raw_result:
            raise ValueError("لم يتمكن النموذج من توليد استجابة JSON صالحة.")
            
        # معالجة النتائج وتحويلها لتنسيق التطبيق (x, y, width, height)
        final_cracks = []
        for i, item in enumerate(raw_result.get("detected_cracks", [])):
            box = item.get("box_2d", [])
            if len(box) == 4:
                ymin, xmin, ymax, xmax = box
                # تحويل من نظام 0-1000 إلى 0-1.0
                x = xmin / 1000.0
                y = ymin / 1000.0
                w = (xmax - xmin) / 1000.0
                h = (ymax - ymin) / 1000.0
                
                # تصحيح القيم لضمان بقائها داخل حدود الصورة
                x = max(0.0, min(0.99, x))
                y = max(0.0, min(0.99, y))
                w = max(0.01, min(1.0 - x, w))
                h = max(0.01, min(1.0 - y, h))
                
                final_cracks.append({
                    "id": item.get("id", i + 1),
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "type": item.get("type", "شرخ"),
                    "severity": item.get("severity", "MEDIUM"),
                    "is_structural": item.get("is_structural", True),
                    "estimated_width_mm": item.get("width_estimate_mm", "غير محدد"),
                    "description": item.get("description", ""),
                    "cause_analysis": item.get("cause", ""),
                    "immediate_action": item.get("action", ""),
                    "confidence": 90
                })
        
        # بناء الرد النهائي المتوافق مع واجهة المستخدم في تطبيقك
        report = raw_result.get("overall_report", {})
        final_output = {
            "total_cracks_detected": len(final_cracks),
            "summary": report.get("summary", "تم تحليل الصورة بنجاح."),
            "overall_severity": report.get("overall_severity", "LOW"),
            "material_type": report.get("material_type", "غير محدد"),
            "overall_confidence": report.get("confidence_score", 90),
            "cracks": final_cracks,
            "recommendations": []
        }
        
        # إضافة التوصيات بالتنسيق المتوقع
        for rec in report.get("recommendations", []):
            final_output["recommendations"].append({
                "priority": rec.get("priority", 1),
                "action": rec.get("title", ""),
                "details": rec.get("details", ""),
                "timeline": rec.get("timeline", "غير محدد")
            })
            
        # إضافة خطة المراقبة والملاحظات الإضافية
        final_output["monitoring_plan"] = report.get("monitoring_plan", "فحص دوري كل 6 أشهر.")
        
        return final_output

    except Exception as e:
        print(f"Error in ai_analyzer: {str(e)}")
        # إرجاع رد فارغ آمن في حالة الخطأ لضمان عدم توقف التطبيق
        return {
            "total_cracks_detected": 0,
            "summary": f"حدث خطأ فني أثناء التحليل: {str(e)}",
            "overall_severity": "UNKNOWN",
            "cracks": [],
            "recommendations": []
        }

def generate_dashboard_recommendations(records_summary):
    """
    توليد توصيات شاملة للوحة التحكم بناءً على ملخص السجلات.
    دالة احترافية تدعم اتخاذ القرار.
    """
    model = get_gemini_client()
    prompt = f"""
    بناءً على سجلات الشقوق التالية المستخرجة من تطبيق الفحص، قدم تقريراً إدارياً وتوصيات صيانة شاملة باللغة العربية.
    
    السجلات:
    {records_summary}
    
    أرجع الرد بتنسيق JSON فقط:
    {{
      "overall_assessment": "تقييم شامل للحالة العامة للمنشأة/الطريق",
      "priority_actions": ["إجراء 1", "إجراء 2"],
      "maintenance_schedule": "جدول الصيانة المقترح للمرحلة القادمة",
      "budget_estimate": "منخفض/متوسط/عالي",
      "risk_areas": ["المناطق الأكثر خطورة"],
      "preventive_measures": ["إجراءات وقائية لمنع تكرار الشقوق"]
    }}
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return _parse_json_response(response.text)
    except:
        return {"overall_assessment": "تعذر توليد التوصيات العامة حالياً بسبب خطأ فني."}
