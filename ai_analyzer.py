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
    # استخدام gemini-1.5-pro لقدراته الفائقة في تحليل الصور والاكتشاف البصري
    return genai.GenerativeModel('gemini-1.5-pro')

def _parse_json_response(text):
    """استخراج وتحليل JSON من نص استجابة النموذج بمرونة عالية."""
    if not text:
        return None
    
    # تنظيف النص من علامات Markdown البرمجية
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # محاولة إيجاد أول { وآخر }
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            # محاولة تنظيف إضافية إذا فشل التحليل (مثل الفواصل الزائدة)
            try:
                # محاولة بسيطة لإصلاح بعض أخطاء JSON الشائعة
                cleaned = re.sub(r',\s*}', '}', text[start:end])
                cleaned = re.sub(r',\s*]', ']', cleaned)
                return json.loads(cleaned)
            except:
                pass
    return None

def detect_and_analyze(image_base64, img_width, img_height):
    """
    الدالة الرئيسية لاكتشاف الشقوق وتحليلها.
    تستخدم برومبت واحد مكثف يركز على الاكتشاف البصري الدقيق.
    """
    model = get_gemini_client()
    
    # برومبت متخصص جداً في الاكتشاف البصري ورسم المربعات
    # نطلب من النموذج إرجاع الإحداثيات بتنسيق [ymin, xmin, ymax, xmax] وهو المعيار لـ Gemini Spatial
    prompt = """
    أنت خبير في الرؤية الحاسوبية والهندسة الإنشائية. مهمتك هي فحص الصورة بدقة واكتشاف كل شرخ (Crack) أو كسر في الخرسانة أو الأسفلت.
    
    المهمة الأولى (الاكتشاف):
    - حدد موقع كل شرخ بدقة متناهية.
    - لكل شرخ، ارسم صندوقاً محيطاً (Bounding Box) يحيط بالشرخ تماماً.
    - استخدم الإحداثيات المعيرة (Normalized) من 0 إلى 1000.
    - التنسيق المطلوب للصندوق: [ymin, xmin, ymax, xmax] حيث 0 هو الأعلى/اليسار و 1000 هو الأسفل/اليمين.
    
    المهمة الثانية (التحليل):
    - لكل شرخ، حدد نوعه (طولي، عرضي، شعري، إلخ) ومدى خطورته (منخفضة، متوسطة، عالية، حرجة).
    - قدم توصيات هندسية باللغة العربية.
    
    يجب أن يكون الرد بتنسيق JSON فقط كما يلي:
    {
      "detected": [
        {
          "id": 1,
          "box_2d": [ymin, xmin, ymax, xmax],
          "type": "نوع الشرخ باللغة العربية",
          "severity": "HIGH/MEDIUM/LOW/CRITICAL",
          "description": "وصف دقيق للشرخ باللغة العربية",
          "action": "الإجراء المطلوب"
        }
      ],
      "summary": "ملخص شامل للحالة الإنشائية باللغة العربية",
      "overall_severity": "الخطورة العامة",
      "recommendations": ["توصية 1", "توصية 2"]
    }
    
    إذا لم تجد أي شقوق، أرجع "detected": [] ولكن ابذل قصارى جهدك لاكتشاف حتى الشقوق الصغيرة جداً.
    """

    try:
        image_data = base64.b64decode(image_base64)
        contents = [
            {"mime_type": "image/jpeg", "data": image_data},
            prompt
        ]
        
        # إعدادات التوليد لضمان دقة النتائج
        generation_config = {
            "temperature": 0.1, # حرارة منخفضة لزيادة الدقة وعدم التخيل
            "top_p": 0.95,
            "max_output_tokens": 4096,
        }
        
        response = model.generate_content(contents, generation_config=generation_config)
        result = _parse_json_response(response.text)
        
        if not result:
            raise ValueError("فشل في تحليل رد النموذج كـ JSON")
            
        # تحويل إحداثيات Gemini [ymin, xmin, ymax, xmax] إلى تنسيق التطبيق [x, y, width, height]
        final_cracks = []
        for i, item in enumerate(result.get("detected", [])):
            box = item.get("box_2d", [])
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
                
                final_cracks.append({
                    "id": item.get("id", i + 1),
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "type": item.get("type", "شرخ"),
                    "severity": item.get("severity", "MEDIUM"),
                    "description": item.get("description", ""),
                    "immediate_action": item.get("action", ""),
                    "is_structural": item.get("severity") in ["HIGH", "CRITICAL"]
                })
        
        # بناء الرد النهائي المتوافق مع واجهة التطبيق
        final_response = {
            "total_cracks_detected": len(final_cracks),
            "summary": result.get("summary", "تم تحليل الصورة بنجاح."),
            "overall_severity": result.get("overall_severity", "MEDIUM"),
            "cracks": final_cracks,
            "recommendations": []
        }
        
        # تحويل التوصيات إلى التنسيق المتوقع
        for i, rec in enumerate(result.get("recommendations", [])):
            final_response["recommendations"].append({
                "priority": i + 1,
                "action": rec,
                "timeline": "فوري" if "حرجة" in str(result.get("overall_severity")) else "خلال شهر",
                "details": "بناءً على التحليل البصري للشقوق"
            })
            
        return final_response

    except Exception as e:
        print(f"Error in detect_and_analyze: {e}")
        return {
            "total_cracks_detected": 0,
            "summary": "حدث خطأ أثناء معالجة الصورة. يرجى التأكد من جودة الصورة ومفتاح API.",
            "overall_severity": "UNKNOWN",
            "cracks": [],
            "recommendations": []
        }

def generate_dashboard_recommendations(records_summary):
    """توليد توصيات لوحة التحكم."""
    model = get_gemini_client()
    prompt = f"بناءً على السجلات التالية، قدم خطة صيانة شاملة باللغة العربية بتنسيق JSON:\n{records_summary}"
    try:
        response = model.generate_content(prompt)
        return _parse_json_response(response.text)
    except:
        return {"overall_assessment": "تعذر توليد التوصيات."}
