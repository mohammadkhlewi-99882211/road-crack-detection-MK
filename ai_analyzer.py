import os
import json
import base64
import re
from google import genai
from google.genai import types

def get_gemini_client():
    """تهيئة عميل Gemini باستخدام المكتبة الجديدة google-genai المتوافقة مع Streamlit Cloud."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("الرجاء ضبط متغير البيئة GEMINI_API_KEY")
    
    # تهيئة العميل الجديد
    client = genai.Client(api_key=api_key)
    return client

def _parse_json_response(text):
    """استخراج وتحليل JSON من نص استجابة النموذج بمرونة عالية."""
    if not text:
        return None
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(cleaned)
            except:
                pass
    return None

def detect_and_analyze(image_base64, img_width, img_height):
    """
    الدالة الرئيسية المطابقة لمتطلبات app.py والمحدثة للمكتبة الجديدة.
    تستخدم Gemini 1.5 Pro لاكتشاف الشقوق بدقة متناهية.
    """
    client = get_gemini_client()
    
    # برومبت متخصص يركز على الاكتشاف والتحليل الهندسي باللغة العربية
    prompt = """
    أنت خبير هندسي متخصص في فحص المنشآت. مهمتك هي اكتشاف كل شرخ في الصورة.
    
    يجب أن يكون الرد بتنسيق JSON فقط ويحتوي على الحقول التالية باللغة العربية:
    {
      "total_cracks_detected": عدد الشروخ,
      "summary": "ملخص عام للحالة الإنشائية",
      "overall_severity": "CRITICAL/HIGH/MEDIUM/LOW",
      "overall_confidence": 95,
      "material_type": "نوع المادة (خرسانة/أسفلت/إلخ)",
      "surface_condition": "وصف حالة السطح",
      "environmental_factors": "العوامل البيئية المرئية",
      "cracks": [
        {
          "id": 1,
          "box_2d": [ymin, xmin, ymax, xmax],
          "type": "نوع الشرخ",
          "category": "تصنيف الشرخ",
          "is_structural": true/false,
          "estimated_width_mm": "1.5",
          "estimated_length_cm": "20",
          "depth_assessment": "تقييم العمق",
          "severity": "CRITICAL/HIGH/MEDIUM/LOW",
          "confidence": 90,
          "description": "وصف دقيق",
          "cause_analysis": "تحليل السبب",
          "progression_risk": "عالي/متوسط/منخفض",
          "immediate_action": "الإجراء الفوري"
        }
      ],
      "recommendations": [
        {
          "priority": 1,
          "action": "الإجراء المقترح",
          "timeline": "الجدول الزمني",
          "estimated_cost_level": "منخفض/متوسط/عالي",
          "details": "تفاصيل إضافية"
        }
      ],
      "monitoring_plan": "خطة المراقبة",
      "professional_consultation_required": true/false,
      "notes": "ملاحظات إضافية"
    }
    
    ملاحظة هامة للاكتشاف:
    - استخدم إحداثيات [ymin, xmin, ymax, xmax] في نطاق 0-1000 لتحديد مكان كل شرخ في حقل 'box_2d'.
    - ابحث عن أدق الشقوق والكسور.
    """

    try:
        # فك تشفير الصورة وإرسالها باستخدام المكتبة الجديدة
        image_bytes = base64.b64decode(image_base64)
        
        # استدعاء Gemini 1.5 Pro باستخدام المكتبة الجديدة
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
                response_mime_type="application/json"
            )
        )
        
        result = _parse_json_response(response.text)
        if not result:
            return {"total_cracks_detected": 0, "summary": "لم يتم العثور على شقوق.", "cracks": []}
            
        # تحويل إحداثيات Gemini Spatial إلى bbox المتوقع في app.py
        processed_cracks = []
        for i, crack in enumerate(result.get("cracks", [])):
            box = crack.get("box_2d", [])
            if len(box) == 4:
                ymin, xmin, ymax, xmax = box
                # تحويل إلى نظام bbox (x, y, width, height) بنسبة 0-1.0
                crack["bbox"] = {
                    "x": xmin / 1000.0,
                    "y": ymin / 1000.0,
                    "width": (xmax - xmin) / 1000.0,
                    "height": (ymax - ymin) / 1000.0
                }
            
            # التأكد من وجود الحقول التي يطلبها app.py
            crack["id"] = crack.get("id", i + 1)
            crack["confidence"] = crack.get("confidence", 90)
            crack["is_structural"] = crack.get("is_structural", crack.get("severity") in ["HIGH", "CRITICAL"])
            processed_cracks.append(crack)
            
        result["cracks"] = processed_cracks
        result["total_cracks_detected"] = len(processed_cracks)
        
        return result

    except Exception as e:
        print(f"Error in Gemini Analysis: {e}")
        return {"total_cracks_detected": 0, "summary": f"خطأ تقني: {str(e)}", "cracks": []}

def generate_dashboard_recommendations(records_summary):
    """توصيات لوحة التحكم المطابقة لـ app.py باستخدام المكتبة الجديدة."""
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[f"بناءً على السجلات التالية، قدم توصيات صيانة شاملة باللغة العربية بتنسيق JSON:\n{records_summary}"],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return _parse_json_response(response.text)
    except:
        return {"overall_assessment": "تعذر توليد التوصيات."}
