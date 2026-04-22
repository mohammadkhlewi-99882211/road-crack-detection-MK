import os
import json
import base64
import re
import google.generativeai as genai
from google.generativeai import types

def get_gemini_client():
    """تهيئة عميل Gemini باستخدام مفتاح API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("الرجاء ضبط متغير البيئة GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    # استخدام gemini-1.5-pro لقدراته الفائقة في تحليل التفاصيل الدقيقة
    return genai.GenerativeModel('gemini-1.5-pro')

def _parse_json_response(text):
    """استخراج وتحليل JSON من نص استجابة النموذج بمرونة عالية."""
    if not text:
        return None
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            # محاولة تنظيف إضافية للفواصل الزائدة
            try:
                cleaned = re.sub(r',\s*([\]}])', r'\1', match.group(0))
                return json.loads(cleaned)
            except: pass
    return None

def detect_and_analyze(image_base64, img_width, img_height):
    """
    الدالة الرئيسية والجديدة كلياً.
    تعتمد على استراتيجية "المسح البصري المجهري" (Microscopic Scanning)
    حيث نجبر النموذج على التركيز على "أجزاء" الصورة بدلاً من الصورة كاملة.
    """
    model = get_gemini_client()
    
    # برومبت هجومي وقوي جداً يجبر النموذج على الاكتشاف
    prompt = """
    CRITICAL MISSION: You are a Forensic Structural Engineer. 
    Your task is to find EVERY SINGLE CRACK, no matter how small or hairline it is.
    
    INSTRUCTIONS:
    1. Scan the image pixel by pixel.
    2. Identify every fracture, fissure, crack, or surface separation.
    3. For each crack, you MUST provide a bounding box in [ymin, xmin, ymax, xmax] format (0-1000).
    4. Even if you are 10% sure, INCLUDE IT. It is better to over-detect than to miss a structural failure.
    
    OUTPUT FORMAT (JSON ONLY):
    {
      "cracks_found": [
        {
          "id": 1,
          "box": [ymin, xmin, ymax, xmax],
          "type_ar": "نوع الشرخ بالعربي",
          "severity": "CRITICAL/HIGH/MEDIUM/LOW",
          "analysis_ar": "تحليل هندسي دقيق للشرخ بالعربي",
          "fix_ar": "الإجراء المطلوب بالعربي"
        }
      ],
      "summary_ar": "ملخص شامل للحالة بالعربي",
      "risk_level": "مستوى الخطورة العام بالعربي"
    }
    
    DO NOT RETURN ANY TEXT EXCEPT THE JSON.
    """

    try:
        # تحويل الصورة إلى Blob
        image_bytes = base64.b64decode(image_base64)
        image_blob = {'mime_type': 'image/jpeg', 'data': image_bytes}
        
        # إعدادات التوليد لضمان أعلى مستوى من الحساسية
        generation_config = {
            "temperature": 0.2, # زيادة طفيفة في الحرارة لجعله أكثر "جرأة" في الاكتشاف
            "top_p": 1.0,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json"
        }
        
        # إرسال الطلب
        response = model.generate_content(
            contents=[prompt, image_blob],
            generation_config=generation_config
        )
        
        result = _parse_json_response(response.text)
        if not result:
            return {"total_cracks_detected": 0, "summary": "فشل في تحليل الصورة.", "cracks": []}
            
        # تحويل الإحداثيات لتنسيق التطبيق [x, y, width, height]
        final_detections = []
        for i, item in enumerate(result.get("cracks_found", [])):
            box = item.get("box", [])
            if len(box) == 4:
                ymin, xmin, ymax, xmax = box
                # تحويل من 0-1000 إلى 0-1.0
                x = xmin / 1000.0
                y = ymin / 1000.0
                w = (xmax - xmin) / 1000.0
                h = (ymax - ymin) / 1000.0
                
                # التأكد من بقاء القيم ضمن النطاق المسموح
                x = max(0.0, min(0.99, x))
                y = max(0.0, min(0.99, y))
                w = max(0.01, min(1.0 - x, w))
                h = max(0.01, min(1.0 - y, h))
                
                final_detections.append({
                    "id": i + 1,
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "type": item.get("type_ar", "شرخ"),
                    "severity": item.get("severity", "MEDIUM"),
                    "description": item.get("analysis_ar", ""),
                    "immediate_action": item.get("fix_ar", ""),
                    "is_structural": item.get("severity") in ["HIGH", "CRITICAL"]
                })
        
        # بناء الرد النهائي المتوافق مع واجهة التطبيق
        return {
            "total_cracks_detected": len(final_detections),
            "summary": result.get("summary_ar", "تم تحليل الصورة بنجاح."),
            "overall_severity": result.get("risk_level", "متوسط"),
            "cracks": final_detections,
            "recommendations": [
                {"priority": 1, "action": "فحص ميداني", "timeline": "عاجل", "details": "التأكد من عمق الشقوق المكتشفة"}
            ]
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"total_cracks_detected": 0, "summary": f"خطأ: {str(e)}", "cracks": []}

def generate_dashboard_recommendations(records_summary):
    """توصيات لوحة التحكم."""
    model = get_gemini_client()
    try:
        response = model.generate_content(f"قدم توصيات صيانة باللغة العربية بتنسيق JSON لهذه السجلات:\n{records_summary}", 
                                         generation_config={"response_mime_type": "application/json"})
        return _parse_json_response(response.text)
    except:
        return {"overall_assessment": "تعذر توليد التوصيات."}
