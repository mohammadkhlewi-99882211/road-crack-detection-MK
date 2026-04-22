import os
import json
import base64
import re
import google.generativeai as genai

def get_gemini_client():
    """تهيئة عميل Gemini باستخدام مفتاح API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("الرجاء ضبط متغير البيئة GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

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
    الدالة الرئيسية المطابقة تماماً لمتطلبات ملف app.py.
    تستخدم Gemini 1.5 Pro لاكتشاف الشقوق وتحليلها.
    """
    model = get_gemini_client()
    
    # برومبت متخصص يركز على المصطلحات التي يتوقعها ملف app.py
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
          "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1},
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
    
    ملاحظة هامة جداً للاكتشاف:
    - استخدم إحداثيات [ymin, xmin, ymax, xmax] في نطاق 0-1000 لتحديد مكان كل شرخ.
    - سأقوم أنا بتحويلها لاحقاً لـ bbox.
    - ابحث عن أدق الشقوق.
    """

    try:
        image_bytes = base64.b64decode(image_base64)
        image_blob = {'mime_type': 'image/jpeg', 'data': image_bytes}
        
        # نطلب من الموديل إرجاع الإحداثيات بنظام Gemini Spatial [ymin, xmin, ymax, xmax]
        # ثم نقوم بتحويلها في الكود لـ bbox المتوقع في app.py
        spatial_prompt = prompt + "\n\nلكل شرخ، أضف حقل 'box_2d': [ymin, xmin, ymax, xmax] بإحداثيات 0-1000."
        
        response = model.generate_content(
            contents=[spatial_prompt, image_blob],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json"
            }
        )
        
        result = _parse_json_response(response.text)
        if not result:
            return {"total_cracks_detected": 0, "summary": "فشل التحليل.", "cracks": []}
            
        # تحويل إحداثيات Gemini Spatial إلى bbox المتوقع في app.py
        processed_cracks = []
        for i, crack in enumerate(result.get("cracks", [])):
            # نحاول الحصول على الإحداثيات من box_2d أو من الحقل المباشر
            box = crack.get("box_2d", [])
            if len(box) == 4:
                ymin, xmin, ymax, xmax = box
                crack["bbox"] = {
                    "x": xmin / 1000.0,
                    "y": ymin / 1000.0,
                    "width": (xmax - xmin) / 1000.0,
                    "height": (ymax - ymin) / 1000.0
                }
            
            # التأكد من وجود كافة الحقول التي يطلبها app.py لتجنب أخطاء العرض
            crack["id"] = crack.get("id", i + 1)
            crack["confidence"] = crack.get("confidence", 90)
            crack["is_structural"] = crack.get("is_structural", crack.get("severity") in ["HIGH", "CRITICAL"])
            processed_cracks.append(crack)
            
        result["cracks"] = processed_cracks
        result["total_cracks_detected"] = len(processed_cracks)
        
        return result

    except Exception as e:
        print(f"Error: {e}")
        return {"total_cracks_detected": 0, "summary": f"خطأ: {str(e)}", "cracks": []}

def generate_dashboard_recommendations(records_summary):
    """توصيات لوحة التحكم المطابقة لـ app.py."""
    model = get_gemini_client()
    try:
        response = model.generate_content(
            f"بناءً على السجلات التالية، قدم توصيات صيانة شاملة باللغة العربية بتنسيق JSON:\n{records_summary}",
            generation_config={"response_mime_type": "application/json"}
        )
        return _parse_json_response(response.text)
    except:
        return {"overall_assessment": "تعذر توليد التوصيات."}
