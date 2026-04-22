import streamlit as st
import numpy as np
from PIL import Image
import os
import json
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# TEMP DIAGNOSTIC - REMOVE AFTER TEST
import os
api_key = os.environ.get("GEMINI_API_KEY", "")
print(f"[DIAG] API Key: {'SET — ' + api_key[:8] if api_key else 'NOT SET'}")
try:
    from google import genai
    import google.genai as gm
    print(f"[DIAG] google-genai version: {gm.__version__}")
    client = genai.Client(api_key=api_key)
    from google.genai import types
    r = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Part(text="say hi")],
        config=types.GenerateContentConfig(max_output_tokens=10)
    )
    print(f"[DIAG] API works: {r.text}")
except Exception as e:
    print(f"[DIAG] ERROR: {e}")



from crack_detector import draw_ai_detections, image_to_base64, resize_for_api
from ai_analyzer import detect_and_analyze, generate_dashboard_recommendations
from database import save_record, get_all_records, get_record, delete_record, get_statistics

st.set_page_config(
    page_title="CrackVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #2563EB;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: #60A5FA; font-size: 2.3rem; margin-bottom: 0.3rem; }
    .main-header p { color: #94A3B8; font-size: 1.05rem; }

    .severity-CRITICAL {
        background: linear-gradient(135deg,#7F1D1D,#991B1B);
        color:#FCA5A5; padding:0.6rem 1.2rem; border-radius:8px;
        border-left:4px solid #EF4444; font-weight:bold; display:inline-block;
    }
    .severity-HIGH {
        background: linear-gradient(135deg,#7C2D12,#9A3412);
        color:#FDBA74; padding:0.6rem 1.2rem; border-radius:8px;
        border-left:4px solid #F97316; font-weight:bold; display:inline-block;
    }
    .severity-MEDIUM {
        background: linear-gradient(135deg,#713F12,#92400E);
        color:#FDE047; padding:0.6rem 1.2rem; border-radius:8px;
        border-left:4px solid #EAB308; font-weight:bold; display:inline-block;
    }
    .severity-LOW {
        background: linear-gradient(135deg,#14532D,#166534);
        color:#86EFAC; padding:0.6rem 1.2rem; border-radius:8px;
        border-left:4px solid #22C55E; font-weight:bold; display:inline-block;
    }
    .severity-UNKNOWN {
        background:#1E293B; color:#94A3B8; padding:0.6rem 1.2rem;
        border-radius:8px; border-left:4px solid #6B7280;
        font-weight:bold; display:inline-block;
    }
    .metric-card {
        background: linear-gradient(135deg,#1E293B,#1a2535);
        padding:1.5rem; border-radius:14px; border:1px solid #334155;
        text-align:center; margin-bottom:0.5rem;
    }
    .metric-card h3 { color:#94A3B8; font-size:0.82rem; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.05em; }
    .metric-card .value { color:#60A5FA; font-size:2.1rem; font-weight:bold; }

    .rec-card {
        background:#1E293B; padding:1.2rem 1.4rem; border-radius:10px;
        border-left:4px solid #2563EB; margin-bottom:0.9rem;
    }
    .crack-expander { background:#1E293B; border-radius:10px; }
    .info-row { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:0.4rem; }
    .badge {
        display:inline-block; padding:0.15rem 0.55rem; border-radius:20px;
        font-size:0.78rem; font-weight:600; background:#2563EB22; color:#60A5FA;
        border:1px solid #2563EB44;
    }
    .badge-green { background:#22C55E22; color:#86EFAC; border-color:#22C55E44; }
    .badge-red { background:#EF444422; color:#FCA5A5; border-color:#EF444444; }
    .badge-yellow { background:#EAB30822; color:#FDE047; border-color:#EAB30844; }
    .section-divider { border:none; border-top:1px solid #334155; margin:1.5rem 0; }
    div[data-testid="stSidebar"] { background-color:#0A111E; }
    .stButton > button { border-radius:8px; font-weight:600; }
    .upload-area { border:2px dashed #334155; border-radius:12px; padding:2rem; text-align:center; }
</style>
""", unsafe_allow_html=True)

SEVERITY_AR = {
    "CRITICAL": "⛔ خطير جداً",
    "HIGH": "🔴 عالي",
    "MEDIUM": "🟡 متوسط",
    "LOW": "🟢 منخفض",
    "UNKNOWN": "⬜ غير محدد"
}

SEVERITY_COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH": "#F97316",
    "MEDIUM": "#EAB308",
    "LOW": "#22C55E",
    "UNKNOWN": "#6B7280"
}


def severity_badge_html(severity):
    label = SEVERITY_AR.get(severity, severity)
    css = f"severity-{severity}" if severity in SEVERITY_AR else "severity-UNKNOWN"
    return f'<div class="{css}">{label}</div>'


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔍 CrackVision AI")
        st.markdown("نظام تحليل الشروخ بالذكاء الاصطناعي")
        st.markdown("---")
        page = st.radio(
            "الاقسام",
            ["📷 تحليل صورة جديدة", "📂 سجل التحليلات", "📊 لوحة المعلومات"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        stats = get_statistics()
        st.markdown(f"**اجمالي التحليلات:** {stats['total']}")
        critical = stats['by_severity'].get('CRITICAL', 0)
        high = stats['by_severity'].get('HIGH', 0)
        if critical + high > 0:
            st.warning(f"⚠️ {critical + high} حالة تستوجب الانتباه")
        st.markdown("---")
        st.markdown(
            "<div style='color:#475569;font-size:0.75rem;text-align:center'>مدعوم بـ GPT-5.2 Vision</div>",
            unsafe_allow_html=True
        )
    return page


def render_analysis_page():
    st.markdown("""
    <div class="main-header">
        <h1>📷 تحليل الشروخ والشقوق</h1>
        <p>ارفع صورة للخرسانة او الطريق ليقوم الذكاء الاصطناعي بالكشف عن الشروخ وتحليلها</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "اختر صورة للتحليل (JPG، PNG، BMP، WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="اختر صورة واضحة للسطح. كلما كانت الصورة اوضح كانت النتائج ادق."
    )

    if uploaded_file is None:
        st.markdown("""
        <div style='background:#1E293B;border-radius:12px;padding:2rem;text-align:center;border:2px dashed #334155;margin-top:1rem;'>
            <p style='color:#64748B;font-size:1.1rem;'>قم برفع صورة للبدء بالتحليل</p>
            <p style='color:#475569;font-size:0.9rem;'>يدعم التطبيق: الخرسانة المسلحة، الأسفلت، الجدران، الأسقف، الأرضيات</p>
        </div>
        """, unsafe_allow_html=True)
        return

    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_np_resized = resize_for_api(image_np, max_size=2000)
    h_orig, w_orig = image_np_resized.shape[:2]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### الصورة الأصلية")
        st.image(image_np, width="stretch")

    img_b64 = image_to_base64(image_np_resized)

    with st.spinner("🔍 جاري التحليل بالذكاء الاصطناعي... قد يستغرق هذا 15-30 ثانية"):
        analysis = detect_and_analyze(img_b64, w_orig, h_orig)

    cracks = analysis.get("cracks", [])
    total_detected = analysis.get("total_cracks_detected", len(cracks))

    result_image = draw_ai_detections(image_np_resized, cracks)

    with col2:
        st.markdown(f"#### نتيجة الكشف ({total_detected} شرخ/شق)")
        st.image(result_image, width="stretch")

    # Detection model info panel
    det_info = analysis.get("_detection_info", {})
    if det_info:
        g_count  = det_info.get("gemini_detected", 0)
        gpt_count = det_info.get("gpt_detected", 0)
        dual_count = det_info.get("dual_confirmed", 0)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f"""<div class="metric-card">
                <h3>كشف Gemini</h3>
                <div class="value" style="color:#60A5FA;">{g_count}</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class="metric-card">
                <h3>كشف GPT-5.2</h3>
                <div class="value" style="color:#A78BFA;">{gpt_count}</div>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class="metric-card">
                <h3>مؤكد من كلا النموذجين</h3>
                <div class="value" style="color:#22C55E;">{dual_count}</div>
            </div>""", unsafe_allow_html=True)
        with d4:
            st.markdown(f"""<div class="metric-card">
                <h3>إجمالي بعد الدمج</h3>
                <div class="value" style="color:#F59E0B;">{total_detected}</div>
            </div>""", unsafe_allow_html=True)
        st.caption("● مربع سميك = مؤكد من كلا النموذجين  |  ○ مربع رفيع = مكتشف بنموذج واحد")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    overall_sev = analysis.get("overall_severity", "UNKNOWN")
    overall_conf = analysis.get("overall_confidence", 0)
    material = analysis.get("material_type", "غير محدد")
    surface_cond = analysis.get("surface_condition", "")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>مستوى الخطورة</h3>
            {severity_badge_html(overall_sev)}
        </div>""", unsafe_allow_html=True)
    with col_m2:
        conf_color = "#EF4444" if overall_conf < 60 else "#EAB308" if overall_conf < 80 else "#22C55E"
        st.markdown(f"""
        <div class="metric-card">
            <h3>نسبة الثقة</h3>
            <div class="value" style="color:{conf_color};">{overall_conf}%</div>
        </div>""", unsafe_allow_html=True)
    with col_m3:
        sev_color = SEVERITY_COLORS.get(overall_sev, "#6B7280")
        st.markdown(f"""
        <div class="metric-card">
            <h3>عدد الشروخ المكتشفة</h3>
            <div class="value" style="color:{sev_color};">{total_detected}</div>
        </div>""", unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>نوع المادة</h3>
            <div class="value" style="font-size:1rem;padding-top:0.4rem;">{material}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if analysis.get("summary"):
        st.markdown("#### 📋 الملخص العام")
        st.markdown(f"""
        <div class="rec-card" style="border-left-color:#60A5FA;">
            <p style="font-size:1.05rem;line-height:1.9;color:#CBD5E1;">{analysis['summary']}</p>
        </div>""", unsafe_allow_html=True)

    info_cols = st.columns(2)
    with info_cols[0]:
        if surface_cond:
            st.markdown(f"🔲 **حالة السطح:** {surface_cond}")
    with info_cols[1]:
        env = analysis.get("environmental_factors", "")
        if env:
            st.markdown(f"🌤 **العوامل البيئية:** {env}")

    if total_detected == 0:
        st.success("✅ لم يتم الكشف عن شروخ أو شقوق في هذه الصورة. السطح يبدو بحالة جيدة.")
    elif cracks:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### 🔬 تفاصيل الشروخ المكتشفة")

        for crack in cracks:
            sev = crack.get("severity", "UNKNOWN")
            sev_label = SEVERITY_AR.get(sev, sev)
            crack_type = crack.get("type", "غير محدد")
            crack_id = crack.get("id", "?")
            is_struct = crack.get("is_structural", False)
            struct_icon = "🔴 انشائي" if is_struct else "🟢 سطحي/تجميلي"
            conf_val = crack.get("confidence", 0)
            dual_conf = crack.get("dual_confirmed", crack.get("_dual_confirmed", False))
            dual_tag  = " ✅" if dual_conf else " ○"

            with st.expander(
                f"الشرخ #{crack_id} — {crack_type} | {sev_label} | ثقة: {conf_val}%{dual_tag}",
                expanded=(sev in ["CRITICAL", "HIGH"])
            ):
                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown(f"**النوع:** {crack_type}")
                    st.markdown(f"**التصنيف:** {crack.get('category', 'غير محدد')}")
                    st.markdown(
                        f"**نوع الضرر:** <span class='badge {'badge-red' if is_struct else 'badge-green'}'>{struct_icon}</span>",
                        unsafe_allow_html=True
                    )
                    dual_label = "✅ مؤكد من Gemini + GPT-5.2" if dual_conf else "○ مكتشف بنموذج واحد"
                    dual_color = "#22C55E" if dual_conf else "#94A3B8"
                    st.markdown(
                        f"**حالة الكشف:** <span style='color:{dual_color};font-weight:600'>{dual_label}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**عرض الشرخ (تقديري):** {crack.get('estimated_width_mm', 'غير محدد')} مم")
                    st.markdown(f"**طول الشرخ (تقديري):** {crack.get('estimated_length_cm', 'غير محدد')} سم")
                with c_right:
                    st.markdown(f"**تقييم العمق:** {crack.get('depth_assessment', 'غير محدد')}")
                    st.markdown(f"**نسبة الثقة:** {conf_val}%")
                    prog_risk = crack.get("progression_risk", "غير محدد")
                    risk_badge = "badge-red" if prog_risk == "عالي" else "badge-yellow" if prog_risk == "متوسط" else "badge-green"
                    st.markdown(
                        f"**خطر التفاقم:** <span class='badge {risk_badge}'>{prog_risk}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(severity_badge_html(sev), unsafe_allow_html=True)

                st.markdown(f"**📝 الوصف:** {crack.get('description', '')}")
                st.markdown(f"**🔎 تحليل السبب:** {crack.get('cause_analysis', '')}")
                if crack.get("immediate_action"):
                    st.warning(f"**⚡ الإجراء الفوري:** {crack['immediate_action']}")

    if analysis.get("recommendations"):
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### 💡 التوصيات")
        for rec in analysis["recommendations"]:
            priority = rec.get("priority", 0)
            timeline = rec.get("timeline", "غير محدد")
            cost = rec.get("estimated_cost_level", "غير محدد")
            priority_icon = "🔴" if priority == 1 else "🟡" if priority <= 2 else "🟢"
            st.markdown(f"""
            <div class="rec-card">
                <p style="font-weight:bold;color:#60A5FA;margin-bottom:0.5rem;">
                    {priority_icon} الأولوية {priority}: {rec.get('action', '')}
                </p>
                <p style="color:#94A3B8;margin:0;">
                    ⏱ <strong>الجدول الزمني:</strong> {timeline} &nbsp;|&nbsp;
                    💰 <strong>مستوى التكلفة:</strong> {cost}
                </p>
                <p style="margin-top:0.5rem;color:#CBD5E1;">{rec.get('details', '')}</p>
            </div>""", unsafe_allow_html=True)

    if analysis.get("monitoring_plan"):
        st.info(f"📅 **خطة المراقبة:** {analysis['monitoring_plan']}")

    if analysis.get("professional_consultation_required"):
        st.warning("⚠️ يُنصح بالتشاور مع مهندس إنشائي متخصص للفحص الميداني الدقيق.")

    if analysis.get("notes"):
        st.markdown(f"📌 **ملاحظات:** {analysis['notes']}")

    if analysis.get("raw_response"):
        with st.expander("تفاصيل الاستجابة الخام"):
            st.text(analysis["raw_response"])

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_save1, col_save2 = st.columns([3, 1])
    with col_save1:
        save_notes = st.text_input("ملاحظات إضافية (اختياري)", placeholder="أضف أي ملاحظات...")
    with col_save2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 حفظ في السجل", type="primary", use_container_width=True):
            os.makedirs("data/uploads", exist_ok=True)
            os.makedirs("data/results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in uploaded_file.name)
            img_path = f"data/uploads/{timestamp}_{safe_name}"
            result_path = f"data/results/{timestamp}_result.jpg"
            Image.fromarray(image_np).save(img_path)
            Image.fromarray(result_image).save(result_path)
            record_id = save_record(
                image_filename=uploaded_file.name,
                image_path=img_path,
                result_image_path=result_path,
                num_cracks=total_detected,
                overall_severity=overall_sev,
                overall_confidence=overall_conf,
                material_type=material,
                analysis_data=analysis,
                notes=save_notes
            )
            st.success(f"✅ تم حفظ التحليل بنجاح! (رقم السجل: {record_id})")


def render_history_page():
    st.markdown("""
    <div class="main-header">
        <h1>📂 سجل التحليلات</h1>
        <p>عرض وإدارة جميع التحليلات السابقة</p>
    </div>
    """, unsafe_allow_html=True)

    records = get_all_records()

    if not records:
        st.info("لا توجد تحليلات محفوظة بعد. قم بتحليل صورة أولاً ثم احفظها.")
        return

    st.markdown(f"**إجمالي التحليلات:** {len(records)}")
    st.markdown("---")

    for record in records:
        with st.container():
            col_img, col_info, col_actions = st.columns([1, 2.5, 1])

            with col_img:
                displayed = False
                for path_key in ["result_image_path", "image_path"]:
                    p = record.get(path_key)
                    if p and os.path.exists(p):
                        st.image(p, width="stretch")
                        displayed = True
                        break
                if not displayed:
                    st.markdown("🖼 لا توجد صورة")

            with col_info:
                sev = record.get("overall_severity", "UNKNOWN")
                st.markdown(f"**📁 الملف:** {record['image_filename']}")
                st.markdown(f"**📅 التاريخ:** {record['created_at'][:19].replace('T', ' ')}")
                st.markdown(f"**🔢 عدد الشروخ:** {record['num_cracks']}")
                st.markdown(severity_badge_html(sev), unsafe_allow_html=True)
                conf = record.get("overall_confidence", 0)
                st.markdown(f"**📊 نسبة الثقة:** {conf}%")
                if record.get("material_type"):
                    st.markdown(f"**🏗 نوع المادة:** {record['material_type']}")
                if record.get("notes"):
                    st.markdown(f"📌 {record['notes']}")

            with col_actions:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 عرض", key=f"view_{record['id']}", use_container_width=True):
                    st.session_state["view_record_id"] = record["id"]
                    st.session_state["show_detail"] = True
                if st.button("🗑 حذف", key=f"del_{record['id']}", use_container_width=True):
                    delete_record(record["id"])
                    if "view_record_id" in st.session_state and st.session_state["view_record_id"] == record["id"]:
                        del st.session_state["view_record_id"]
                        st.session_state.pop("show_detail", None)
                    st.rerun()

        st.markdown("---")

    if st.session_state.get("show_detail") and "view_record_id" in st.session_state:
        record = get_record(st.session_state["view_record_id"])
        if record:
            st.markdown(f"### 🔬 تفاصيل التحليل — #{record['id']}")
            c1, c2 = st.columns(2)
            with c1:
                if record.get("image_path") and os.path.exists(record["image_path"]):
                    st.markdown("**الصورة الأصلية:**")
                    st.image(record["image_path"], width="stretch")
            with c2:
                if record.get("result_image_path") and os.path.exists(record["result_image_path"]):
                    st.markdown("**نتيجة الكشف:**")
                    st.image(record["result_image_path"], width="stretch")

            if record.get("analysis_json"):
                analysis = json.loads(record["analysis_json"])
                if analysis.get("summary"):
                    st.markdown(f"""
                    <div class="rec-card" style="border-left-color:#60A5FA;">
                        <p style="font-size:1.05rem;line-height:1.8;">{analysis['summary']}</p>
                    </div>""", unsafe_allow_html=True)
                if analysis.get("cracks"):
                    for crack in analysis["cracks"]:
                        with st.expander(f"الشرخ #{crack.get('id')} — {crack.get('type', '')}"):
                            st.json(crack)
                if analysis.get("recommendations"):
                    st.markdown("**التوصيات:**")
                    for rec in analysis["recommendations"]:
                        st.markdown(f"""
                        <div class="rec-card">
                            <p><strong>الأولوية {rec.get('priority', '')}:</strong> {rec.get('action', '')}</p>
                            <p style="color:#94A3B8;">{rec.get('details', '')}</p>
                        </div>""", unsafe_allow_html=True)

            if st.button("✖ إغلاق التفاصيل"):
                st.session_state.pop("view_record_id", None)
                st.session_state.pop("show_detail", None)
                st.rerun()


def render_dashboard_page():
    st.markdown("""
    <div class="main-header">
        <h1>📊 لوحة المعلومات</h1>
        <p>نظرة شاملة على جميع التحليلات مع توصيات متكاملة</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_statistics()

    if stats["total"] == 0:
        st.info("لا توجد بيانات بعد. قم بتحليل بعض الصور أولاً لعرض الإحصائيات.")
        return

    critical = stats['by_severity'].get('CRITICAL', 0)
    high = stats['by_severity'].get('HIGH', 0)
    medium = stats['by_severity'].get('MEDIUM', 0)
    low = stats['by_severity'].get('LOW', 0)
    alert_count = critical + high

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>إجمالي التحليلات</h3>
            <div class="value">{stats['total']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>متوسط الشروخ/صورة</h3>
            <div class="value">{stats['avg_cracks']}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>متوسط الثقة</h3>
            <div class="value">{stats['avg_confidence']}%</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        alert_color = "#EF4444" if alert_count > 0 else "#22C55E"
        st.markdown(f"""
        <div class="metric-card">
            <h3>حالات حرجة / عالية</h3>
            <div class="value" style="color:{alert_color};">{alert_count}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### توزيع مستويات الخطورة")
        sev_data = {
            "⛔ خطير جداً": critical,
            "🔴 عالي": high,
            "🟡 متوسط": medium,
            "🟢 منخفض": low
        }
        sev_data = {k: v for k, v in sev_data.items() if v > 0}
        if sev_data:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(sev_data.keys()),
                values=list(sev_data.values()),
                marker_colors=["#EF4444", "#F97316", "#EAB308", "#22C55E"],
                hole=0.45,
                textinfo="label+percent+value",
                textfont=dict(size=13)
            )])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0", size=12),
                height=320,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("لا توجد بيانات")

    with chart_col2:
        st.markdown("#### توزيع أنواع المواد")
        mat_data = {k if k else "غير محدد": v for k, v in stats.get("by_material", {}).items()}
        if mat_data:
            mat_df = pd.DataFrame(mat_data.items(), columns=["المادة", "العدد"])
            fig_bar = px.bar(
                mat_df, x="المادة", y="العدد",
                color_discrete_sequence=["#2563EB"],
                text="العدد"
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                height=320,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(gridcolor="#1E293B")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("لا توجد بيانات")

    if stats.get("recent") and len(stats["recent"]) > 1:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### تطور الحالات عبر الزمن")
        timeline_data = []
        sev_values = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
        for r in reversed(stats["recent"]):
            timeline_data.append({
                "التاريخ": r["created_at"][:10],
                "مستوى الخطورة": sev_values.get(r.get("overall_severity", ""), 0),
                "عدد الشروخ": r.get("num_cracks", 0),
                "الملف": r.get("image_filename", "")
            })
        tl_df = pd.DataFrame(timeline_data)
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=tl_df["التاريخ"], y=tl_df["عدد الشروخ"],
            mode="lines+markers+text",
            name="عدد الشروخ",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=8, color="#2563EB"),
            text=tl_df["عدد الشروخ"],
            textposition="top center"
        ))
        fig_tl.add_trace(go.Scatter(
            x=tl_df["التاريخ"], y=tl_df["مستوى الخطورة"],
            mode="lines+markers",
            name="مستوى الخطورة",
            line=dict(color="#EF4444", width=2, dash="dot"),
            marker=dict(size=8, color="#EF4444"),
            yaxis="y2"
        ))
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(title="عدد الشروخ", gridcolor="#1E293B"),
            yaxis2=dict(title="الخطورة (4=حرج)", overlaying="y", side="right", gridcolor="#1E293B"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🤖 التوصيات الشاملة بالذكاء الاصطناعي")

    if st.button("🔄 توليد تقرير وتوصيات شاملة", type="primary", use_container_width=True):
        with st.spinner("جاري تحليل جميع السجلات وتوليد التقرير الشامل..."):
            records = get_all_records()
            severity_name = {"CRITICAL": "خطير جداً", "HIGH": "عالي", "MEDIUM": "متوسط", "LOW": "منخفض"}
            lines = []
            for r in records:
                sev_ar = severity_name.get(r.get("overall_severity", ""), "غير محدد")
                lines.append(
                    f"• {r['image_filename']}: {r['num_cracks']} شروخ | الخطورة: {sev_ar} | "
                    f"الثقة: {r.get('overall_confidence', 0)}% | المادة: {r.get('material_type', 'غير محدد')}"
                )
            summary_txt = "\n".join(lines)
            summary_txt += f"\n\nإجمالي: {stats['total']} تحليل"
            summary_txt += f" | حالات حرجة: {critical} | عالية: {high} | متوسطة: {medium} | منخفضة: {low}"

            recs = generate_dashboard_recommendations(summary_txt)

            if recs.get("overall_assessment"):
                st.markdown(f"""
                <div class="rec-card" style="border-left-color:#60A5FA;">
                    <p style="font-weight:bold;color:#60A5FA;margin-bottom:0.5rem;">📋 التقييم الشامل</p>
                    <p style="line-height:1.9;color:#CBD5E1;">{recs['overall_assessment']}</p>
                </div>""", unsafe_allow_html=True)

            left_col, right_col = st.columns(2)
            with left_col:
                if recs.get("priority_actions"):
                    st.markdown("**⚡ الإجراءات ذات الأولوية:**")
                    for action in recs["priority_actions"]:
                        st.markdown(f"- {action}")
                if recs.get("maintenance_schedule"):
                    st.info(f"📅 **جدول الصيانة:** {recs['maintenance_schedule']}")
                if recs.get("budget_estimate"):
                    st.markdown(f"💰 **تقدير الميزانية:** {recs['budget_estimate']}")
            with right_col:
                if recs.get("risk_areas"):
                    st.markdown("**⚠️ المناطق الأكثر خطورة:**")
                    for area in recs["risk_areas"]:
                        st.warning(area)
                if recs.get("preventive_measures"):
                    st.markdown("**🛡 الإجراءات الوقائية:**")
                    for measure in recs["preventive_measures"]:
                        st.markdown(f"- {measure}")


def main():
    page = render_sidebar()
    if page == "📷 تحليل صورة جديدة":
        render_analysis_page()
    elif page == "📂 سجل التحليلات":
        render_history_page()
    elif page == "📊 لوحة المعلومات":
        render_dashboard_page()


if __name__ == "__main__":
    main()
