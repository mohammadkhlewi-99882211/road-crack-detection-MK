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

from crack_detector import draw_ai_detections, image_to_base64, resize_for_api
from ai_analyzer import detect_and_analyze, generate_dashboard_recommendations
from database import save_record, get_all_records, get_record, delete_record, get_statistics

# هون الواجهة دكتور ورح تلاقي ببداية كل فقرة شو بتعالج بالضبط

# استخدمت هون مكتبة Streamlit لانو خبرت حضرتك انو ما نزلت مكتبات على جهازي وايضا استعنت ب css
st.set_page_config(
    page_title="CrackVision AI — محمد خليوي",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050D1A 0%, #0A1628 60%, #0D1F3C 100%);
        border-right: 1px solid #1E3A5F;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .sidebar-logo h2 {
        color: #38BDF8;
        font-size: 1.4rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: 0.03em;
    }
    .sidebar-logo p {
        color: #64748B;
        font-size: 0.78rem;
        margin: 0.2rem 0 0 0;
    }
    .sidebar-stat {
        background: rgba(30,58,95,0.4);
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sidebar-stat .lbl { color: #64748B; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .sidebar-stat .val { color: #38BDF8; font-size: 1.5rem; font-weight: 700; }
    .author-tag {
        text-align: center;
        padding: 0.8rem 0.5rem;
        color: #334155;
        font-size: 0.72rem;
        border-top: 1px solid #1E3A5F;
        margin-top: 1rem;
    }
    .author-tag span { color: #38BDF8; font-weight: 700; }

    /* ── Page header ── */
    .page-header {
        background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 100%);
        border: 1px solid #1E3A5F;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .page-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .page-header h1 { color: #F1F5F9; font-size: 1.9rem; font-weight: 900; margin: 0 0 0.3rem 0; }
    .page-header p  { color: #64748B; font-size: 0.95rem; margin: 0; }
    .page-header .tag {
        display: inline-block;
        background: rgba(56,189,248,0.1);
        border: 1px solid rgba(56,189,248,0.3);
        color: #38BDF8;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* ── Severity badges ── */
    .sev-CRITICAL { background:linear-gradient(135deg,#7F1D1D,#991B1B); color:#FCA5A5; padding:0.5rem 1.1rem; border-radius:8px; border-left:4px solid #EF4444; font-weight:700; display:inline-block; font-size:0.9rem; }
    .sev-HIGH     { background:linear-gradient(135deg,#7C2D12,#9A3412); color:#FDBA74; padding:0.5rem 1.1rem; border-radius:8px; border-left:4px solid #F97316; font-weight:700; display:inline-block; font-size:0.9rem; }
    .sev-MEDIUM   { background:linear-gradient(135deg,#713F12,#92400E); color:#FDE047; padding:0.5rem 1.1rem; border-radius:8px; border-left:4px solid #EAB308; font-weight:700; display:inline-block; font-size:0.9rem; }
    .sev-LOW      { background:linear-gradient(135deg,#14532D,#166534); color:#86EFAC; padding:0.5rem 1.1rem; border-radius:8px; border-left:4px solid #22C55E; font-weight:700; display:inline-block; font-size:0.9rem; }
    .sev-UNKNOWN  { background:#1E293B; color:#94A3B8; padding:0.5rem 1.1rem; border-radius:8px; border-left:4px solid #6B7280; font-weight:700; display:inline-block; font-size:0.9rem; }

    /* ── Metric card ── */
    .mcard {
        background: linear-gradient(135deg, #0D1F3C, #0A1628);
        border: 1px solid #1E3A5F;
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
        position: relative;
        overflow: hidden;
    }
    .mcard::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1E3A5F, #38BDF8, #1E3A5F);
        border-radius: 0 0 14px 14px;
    }
    .mcard .lbl { color: #475569; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.5rem; }
    .mcard .val { color: #38BDF8; font-size: 2rem; font-weight: 900; line-height: 1; }
    .mcard .sub { color: #334155; font-size: 0.72rem; margin-top: 0.3rem; }

    /* ── Detection result banner ── */
    .det-banner {
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        margin: 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .det-banner.found  { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); }
    .det-banner.none   { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.3); }
    .det-banner .icon  { font-size: 2.2rem; }
    .det-banner .title { font-size: 1.15rem; font-weight: 700; }
    .det-banner .sub   { font-size: 0.85rem; color: #64748B; }

    /* ── Rec card ── */
    .rec-card {
        background: #0D1F3C;
        border: 1px solid #1E3A5F;
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.8rem;
    }

    /* ── Badge ── */
    .badge { display:inline-block; padding:0.15rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .badge-blue   { background:rgba(56,189,248,0.1);  color:#38BDF8;  border:1px solid rgba(56,189,248,0.3); }
    .badge-green  { background:rgba(34,197,94,0.1);   color:#86EFAC;  border:1px solid rgba(34,197,94,0.3); }
    .badge-red    { background:rgba(239,68,68,0.1);   color:#FCA5A5;  border:1px solid rgba(239,68,68,0.3); }
    .badge-yellow { background:rgba(234,179,8,0.1);   color:#FDE047;  border:1px solid rgba(234,179,8,0.3); }

    .section-divider { border:none; border-top:1px solid #1E3A5F; margin:1.5rem 0; }
    .stButton > button { border-radius:8px; font-weight:600; }

    /* ── Expander ── */
    details { background: #0D1F3C !important; border: 1px solid #1E3A5F !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

SEVERITY_AR = {
    "CRITICAL": "⛔ خطير جداً",
    "HIGH":     "🔴 عالي",
    "MEDIUM":   "🟡 متوسط",
    "LOW":      "🟢 منخفض",
    "UNKNOWN":  "⬜ غير محدد"
}
SEVERITY_COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH":     "#F97316",
    "MEDIUM":   "#EAB308",
    "LOW":      "#22C55E",
    "UNKNOWN":  "#6B7280"
}


def sev_badge(severity):
    label = SEVERITY_AR.get(severity, severity)
    css   = f"sev-{severity}" if severity in SEVERITY_AR else "sev-UNKNOWN"
    return f'<span class="{css}">{label}</span>'


# ─────────────────────────────────────────────
#  Sidebar
# ─────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2>🏗 CrackVision AI</h2>
            <p>نظام تحليل الشروخ الإنشائية</p>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["🔬 تحليل صورة جديدة", "📂 سجل التحليلات", "📊 لوحة المعلومات"],
            label_visibility="collapsed"
        )
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        stats = get_statistics()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="sidebar-stat">
                <div class="lbl">إجمالي</div>
                <div class="val">{stats['total']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            alert = stats['by_severity'].get('CRITICAL',0) + stats['by_severity'].get('HIGH',0)
            color = "#EF4444" if alert > 0 else "#22C55E"
            st.markdown(f"""<div class="sidebar-stat">
                <div class="lbl">تنبيهات</div>
                <div class="val" style="color:{color};">{alert}</div>
            </div>""", unsafe_allow_html=True)

        if alert > 0:
            st.warning(f"⚠️ {alert} حالة تستوجب الانتباه")

        st.markdown("""
        <div class="author-tag">
            صُمِّم وبُرمج بواسطة<br>
            <span>محمد خليوي</span>
        </div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────
#  Analysis Page
# ...............

def render_analysis_page():
    st.markdown("""
    <div class="page-header">
        <div class="tag">🔬 تحليل ذكي متعدد النماذج</div>
        <h1>تحليل الشروخ والشقوق الإنشائية</h1>
        <p>ارفع صورة للخرسانة أو الطريق ليقوم النظام بالكشف عن الشروخ وتقديم تقرير هندسي احترافي</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "اختر صورة للتحليل (JPG، PNG، BMP، WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.markdown("""
        <div style='background:#0D1F3C;border-radius:14px;padding:3rem 2rem;text-align:center;
                    border:2px dashed #1E3A5F;margin-top:1rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>🏗</div>
            <p style='color:#475569;font-size:1.05rem;margin:0;'>قم برفع صورة للبدء بالتحليل</p>
            <p style='color:#334155;font-size:0.85rem;margin:0.5rem 0 0;'>
                يدعم النظام: الخرسانة المسلحة · الأسفلت · الجدران · الأسقف · الأرضيات
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np  = np.array(image_pil)
    image_np_resized = resize_for_api(image_np, max_size=2000)
    h_orig, w_orig   = image_np_resized.shape[:2]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### الصورة الأصلية")
        st.image(image_np, use_container_width=True)

    img_b64 = image_to_base64(image_np_resized)

    with st.spinner("🔍 جاري التحليل بالذكاء الاصطناعي... قد يستغرق 15-45 ثانية"):
        analysis = detect_and_analyze(img_b64, w_orig, h_orig)

    cracks         = analysis.get("cracks", [])
    total_detected = analysis.get("total_cracks_detected", len(cracks))
    result_image   = draw_ai_detections(image_np_resized, cracks)

    with col2:
        st.markdown(f"#### نتيجة الكشف ({total_detected} شرخ/شق)")
        st.image(result_image, use_container_width=True)

    # ── Detection summary banner ──
    if total_detected == 0:
        st.markdown("""
        <div class="det-banner none">
            <div class="icon">✅</div>
            <div>
                <div class="title" style="color:#86EFAC;">لا توجد شروخ مكتشفة</div>
                <div class="sub">السطح يبدو في حالة جيدة — لم يتم رصد أي أضرار مرئية</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sev   = analysis.get("overall_severity", "HIGH")
        color = SEVERITY_COLORS.get(sev, "#F97316")
        models_used = analysis.get("_detection_info", {}).get("models_used", 1)
        dual_count  = sum(1 for c in cracks if c.get("dual_confirmed"))
        st.markdown(f"""
        <div class="det-banner found">
            <div class="icon">🔴</div>
            <div style="flex:1;">
                <div class="title" style="color:{color};">
                    تم اكتشاف {total_detected} شرخ/شق
                </div>
                <div class="sub">
                    كُشف عبر {models_used} نماذج YOLO متخصصة ·
                    {dual_count} شرخ مؤكد من أكثر من نموذج
                </div>
            </div>
            <div>{sev_badge(sev)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Key metrics ──
    overall_sev  = analysis.get("overall_severity",  "UNKNOWN")
    overall_conf = analysis.get("overall_confidence", 0)
    material     = analysis.get("material_type",      "غير محدد")
    surface_cond = analysis.get("surface_condition",  "")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="mcard">
            <div class="lbl">مستوى الخطورة</div>
            <div style="margin-top:0.4rem;">{sev_badge(overall_sev)}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        conf_color = "#EF4444" if overall_conf < 60 else "#EAB308" if overall_conf < 80 else "#22C55E"
        st.markdown(f"""<div class="mcard">
            <div class="lbl">نسبة الثقة</div>
            <div class="val" style="color:{conf_color};">{overall_conf}%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        sev_color = SEVERITY_COLORS.get(overall_sev, "#6B7280")
        st.markdown(f"""<div class="mcard">
            <div class="lbl">الشروخ المكتشفة</div>
            <div class="val" style="color:{sev_color};">{total_detected}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="mcard">
            <div class="lbl">نوع المادة</div>
            <div class="val" style="font-size:1rem;padding-top:0.5rem;">{material}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if analysis.get("summary"):
        st.markdown("#### 📋 الملخص الهندسي")
        st.markdown(f"""
        <div class="rec-card" style="border-left-color:#38BDF8;">
            <p style="font-size:1rem;line-height:2;color:#CBD5E1;margin:0;">{analysis['summary']}</p>
        </div>""", unsafe_allow_html=True)

    info_c1, info_c2 = st.columns(2)
    with info_c1:
        if surface_cond:
            st.markdown(f"🔲 **حالة السطح:** {surface_cond}")
    with info_c2:
        env = analysis.get("environmental_factors", "")
        if env:
            st.markdown(f"🌤 **العوامل البيئية:** {env}")

    # ── Crack details ──
    if cracks:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### 🔬 تفاصيل الشروخ المكتشفة")

        for crack in cracks:
            sev        = crack.get("severity", "UNKNOWN")
            sev_label  = SEVERITY_AR.get(sev, sev)
            crack_type = crack.get("type", "غير محدد")
            crack_id   = crack.get("id", "?")
            is_struct  = crack.get("is_structural", False)
            conf_val   = crack.get("confidence", 0)
            dual_conf  = crack.get("dual_confirmed", False)
            dual_tag   = " ✅" if dual_conf else ""

            with st.expander(
                f"الشرخ #{crack_id} — {crack_type} | {sev_label} | ثقة: {conf_val}%{dual_tag}",
                expanded=(sev in ["CRITICAL", "HIGH"])
            ):
                cl, cr = st.columns(2)
                with cl:
                    st.markdown(f"**النوع:** {crack_type}")
                    struct_lbl = "🔴 إنشائي" if is_struct else "🟢 سطحي"
                    st.markdown(f"**التصنيف:** <span class='badge {'badge-red' if is_struct else 'badge-green'}'>{struct_lbl}</span>", unsafe_allow_html=True)
                    if dual_conf:
                        st.markdown("<span class='badge badge-green'>✅ مؤكد من نماذج متعددة</span>", unsafe_allow_html=True)
                    st.markdown(f"**العرض التقديري:** {crack.get('estimated_width_mm','غير محدد')} مم")
                    st.markdown(f"**الطول التقديري:** {crack.get('estimated_length_cm','غير محدد')} سم")
                with cr:
                    st.markdown(f"**تقييم العمق:** {crack.get('depth_assessment','غير محدد')}")
                    st.markdown(f"**نسبة الثقة:** {conf_val}%")
                    risk = crack.get("progression_risk","غير محدد")
                    risk_cls = "badge-red" if "عالي" in str(risk) else "badge-yellow" if "متوسط" in str(risk) else "badge-green"
                    st.markdown(f"**خطر التفاقم:** <span class='badge {risk_cls}'>{risk}</span>", unsafe_allow_html=True)
                    st.markdown(sev_badge(sev), unsafe_allow_html=True)

                if crack.get("description"):
                    st.markdown(f"**📝 الوصف:** {crack['description']}")
                if crack.get("cause_analysis"):
                    st.markdown(f"**🔎 تحليل السبب:** {crack['cause_analysis']}")
                if crack.get("immediate_action"):
                    st.warning(f"**⚡ الإجراء الفوري:** {crack['immediate_action']}")

    # ── Recommendations ──
    if analysis.get("recommendations"):
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### 💡 التوصيات الهندسية")
        for rec in analysis["recommendations"]:
            priority = rec.get("priority", 0)
            icon = "🔴" if priority == 1 else "🟡" if priority == 2 else "🟢"
            st.markdown(f"""
            <div class="rec-card">
                <p style="font-weight:700;color:#38BDF8;margin-bottom:0.4rem;">
                    {icon} الأولوية {priority}: {rec.get('action','')}
                </p>
                <p style="color:#475569;font-size:0.85rem;margin:0 0 0.4rem;">
                    ⏱ {rec.get('timeline','—')} &nbsp;|&nbsp; 💰 {rec.get('estimated_cost_level','—')}
                </p>
                <p style="color:#94A3B8;margin:0;">{rec.get('details','')}</p>
            </div>""", unsafe_allow_html=True)

    if analysis.get("monitoring_plan"):
        st.info(f"📅 **خطة المراقبة:** {analysis['monitoring_plan']}")
    if analysis.get("professional_consultation_required"):
        st.warning("⚠️ يُنصح بالتشاور مع مهندس إنشائي متخصص للفحص الميداني الدقيق.")

    # ── Save ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    sv1, sv2 = st.columns([3, 1])
    with sv1:
        save_notes = st.text_input("ملاحظات إضافية (اختياري)", placeholder="أضف أي ملاحظات...")
    with sv2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 حفظ في السجل", type="primary", use_container_width=True):
            os.makedirs("data/uploads", exist_ok=True)
            os.makedirs("data/results", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in uploaded_file.name)
            img_path    = f"data/uploads/{ts}_{safe}"
            result_path = f"data/results/{ts}_result.jpg"
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
            st.success(f"✅ تم الحفظ بنجاح! (رقم السجل: {record_id})")


#  History Page
# ─────────────────────────────────────────────

def render_history_page():
    st.markdown("""
    <div class="page-header">
        <div class="tag">📂 الأرشيف</div>
        <h1>سجل التحليلات</h1>
        <p>عرض وإدارة جميع التحليلات السابقة المحفوظة</p>
    </div>
    """, unsafe_allow_html=True)

    records = get_all_records()
    if not records:
        st.info("لا توجد تحليلات محفوظة بعد. قم بتحليل صورة أولاً ثم احفظها.")
        return

    st.markdown(f"**إجمالي التحليلات:** {len(records)}")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    for record in records:
        with st.container():
            c_img, c_info, c_act = st.columns([1, 2.5, 1])
            with c_img:
                displayed = False
                for pk in ["result_image_path", "image_path"]:
                    p = record.get(pk)
                    if p and os.path.exists(p):
                        st.image(p, use_container_width=True)
                        displayed = True
                        break
                if not displayed:
                    st.markdown("🖼 لا توجد صورة")
            with c_info:
                sev = record.get("overall_severity", "UNKNOWN")
                st.markdown(f"**📁 الملف:** {record['image_filename']}")
                st.markdown(f"**📅 التاريخ:** {record['created_at'][:19].replace('T',' ')}")
                st.markdown(f"**🔢 عدد الشروخ:** {record['num_cracks']}")
                st.markdown(sev_badge(sev), unsafe_allow_html=True)
                st.markdown(f"**📊 نسبة الثقة:** {record.get('overall_confidence',0)}%")
                if record.get("notes"):
                    st.markdown(f"📌 {record['notes']}")
            with c_act:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 عرض", key=f"v_{record['id']}", use_container_width=True):
                    st.session_state["view_record_id"] = record["id"]
                    st.session_state["show_detail"] = True
                if st.button("🗑 حذف", key=f"d_{record['id']}", use_container_width=True):
                    delete_record(record["id"])
                    st.session_state.pop("view_record_id", None)
                    st.session_state.pop("show_detail", None)
                    st.rerun()
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if st.session_state.get("show_detail") and "view_record_id" in st.session_state:
        record = get_record(st.session_state["view_record_id"])
        if record:
            st.markdown(f"### 🔬 تفاصيل التحليل — #{record['id']}")
            rc1, rc2 = st.columns(2)
            with rc1:
                if record.get("image_path") and os.path.exists(record["image_path"]):
                    st.image(record["image_path"], use_container_width=True)
            with rc2:
                if record.get("result_image_path") and os.path.exists(record["result_image_path"]):
                    st.image(record["result_image_path"], use_container_width=True)
            if record.get("analysis_json"):
                analysis = json.loads(record["analysis_json"])
                if analysis.get("summary"):
                    st.markdown(f"""<div class="rec-card" style="border-left-color:#38BDF8;">
                        <p style="line-height:1.9;color:#CBD5E1;">{analysis['summary']}</p>
                    </div>""", unsafe_allow_html=True)
                if analysis.get("cracks"):
                    for crack in analysis["cracks"]:
                        with st.expander(f"الشرخ #{crack.get('id')} — {crack.get('type','')}"):
                            st.json(crack)
            if st.button("✖ إغلاق"):
                st.session_state.pop("view_record_id", None)
                st.session_state.pop("show_detail", None)
                st.rerun()


# ───────────────────────────────────────
#  Dashboard Page
# ─────────────────────────────────────────────...................

def render_dashboard_page():
    st.markdown("""
    <div class="page-header">
        <div class="tag">📊 لوحة التحكم</div>
        <h1>لوحة المعلومات والإحصائيات</h1>
        <p>نظرة شاملة على جميع التحليلات مع مؤشرات الأداء والتوصيات الذكية</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_statistics()
    if stats["total"] == 0:
        st.info("لا توجد بيانات بعد. قم بتحليل بعض الصور أولاً.")
        return

    critical = stats['by_severity'].get('CRITICAL', 0)
    high     = stats['by_severity'].get('HIGH',     0)
    medium   = stats['by_severity'].get('MEDIUM',   0)
    low      = stats['by_severity'].get('LOW',      0)
    alert_count = critical + high

    # ── KPI Row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("إجمالي التحليلات", stats['total'],          "#38BDF8"),
        ("متوسط الشروخ",     stats['avg_cracks'],     "#A78BFA"),
        ("متوسط الثقة",      f"{stats['avg_confidence']}%", "#34D399"),
        ("حالات حرجة/عالية", alert_count,             "#EF4444" if alert_count > 0 else "#22C55E"),
        ("منخفضة الخطورة",   low,                    "#22C55E"),
    ]
    for col, (lbl, val, color) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""<div class="mcard">
                <div class="lbl">{lbl}</div>
                <div class="val" style="color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Charts Row 1 ──
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### توزيع مستويات الخطورة")
        sev_data = {"⛔ خطير":critical,"🔴 عالي":high,"🟡 متوسط":medium,"🟢 منخفض":low}
        sev_data = {k:v for k,v in sev_data.items() if v>0}
        if sev_data:
            fig = go.Figure(data=[go.Pie(
                labels=list(sev_data.keys()),
                values=list(sev_data.values()),
                marker_colors=["#EF4444","#F97316","#EAB308","#22C55E"],
                hole=0.5,
                textinfo="label+percent+value",
                textfont=dict(size=12)
            )])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"), height=300,
                margin=dict(t=10,b=10,l=10,r=10), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with cc2:
        st.markdown("#### توزيع أنواع المواد")
        mat_data = {(k if k else "غير محدد"):v for k,v in stats.get("by_material",{}).items()}
        if mat_data:
            mat_df = pd.DataFrame(mat_data.items(), columns=["المادة","العدد"])
            fig2 = px.bar(mat_df, x="المادة", y="العدد",
                          color_discrete_sequence=["#38BDF8"], text="العدد")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"), height=300,
                margin=dict(t=10,b=10,l=10,r=10),
                xaxis=dict(gridcolor="#1E3A5F"), yaxis=dict(gridcolor="#1E3A5F")
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    cc3, cc4 = st.columns(2)

    with cc3:
        st.markdown("#### مؤشر الخطورة التراكمي")
        sev_values = {"CRITICAL":4,"HIGH":3,"MEDIUM":2,"LOW":1,"UNKNOWN":0}
        if stats.get("recent"):
            gauge_val = sum(sev_values.get(r.get("overall_severity",""),0) for r in stats["recent"])
            gauge_max = len(stats["recent"]) * 4
            gauge_pct = (gauge_val / gauge_max * 100) if gauge_max > 0 else 0
            gauge_color = "#EF4444" if gauge_pct > 66 else "#EAB308" if gauge_pct > 33 else "#22C55E"
            fig3 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(gauge_pct, 1),
                number={"suffix":"%","font":{"color":"#CBD5E1","size":28}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#334155"},
                    "bar":{"color":gauge_color},
                    "bgcolor":"#0D1F3C",
                    "bordercolor":"#1E3A5F",
                    "steps":[
                        {"range":[0,33],"color":"rgba(34,197,94,0.1)"},
                        {"range":[33,66],"color":"rgba(234,179,8,0.1)"},
                        {"range":[66,100],"color":"rgba(239,68,68,0.1)"},
                    ],
                    "threshold":{"line":{"color":"#F1F5F9","width":2},"thickness":0.75,"value":gauge_pct}
                }
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1"),
                height=280, margin=dict(t=20,b=20,l=30,r=30)
            )
            st.plotly_chart(fig3, use_container_width=True)

    with cc4:
        st.markdown("#### توزيع الشروخ (الأعلى/الأدنى)")
        if stats.get("recent"):
            names = [r.get("image_filename","")[:15] for r in stats["recent"]]
            vals  = [r.get("num_cracks",0) for r in stats["recent"]]
            colors_bar = ["#EF4444" if v == max(vals) else "#38BDF8" for v in vals]
            fig4 = go.Figure(go.Bar(
                x=names, y=vals,
                marker_color=colors_bar,
                text=vals, textposition="outside"
            ))
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"), height=280,
                margin=dict(t=10,b=10,l=10,r=10),
                xaxis=dict(gridcolor="#1E3A5F"), yaxis=dict(gridcolor="#1E3A5F")
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── Timeline ──
    if stats.get("recent") and len(stats["recent"]) > 1:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### تطور الحالات عبر الزمن")
        sev_values = {"CRITICAL":4,"HIGH":3,"MEDIUM":2,"LOW":1,"UNKNOWN":0}
        tl = []
        for r in reversed(stats["recent"]):
            tl.append({
                "التاريخ": r["created_at"][:10],
                "عدد الشروخ": r.get("num_cracks",0),
                "مستوى الخطورة": sev_values.get(r.get("overall_severity",""),0)
            })
        tl_df = pd.DataFrame(tl)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=tl_df["التاريخ"], y=tl_df["عدد الشروخ"],
            mode="lines+markers+text", name="عدد الشروخ",
            line=dict(color="#38BDF8",width=2),
            marker=dict(size=8,color="#38BDF8"),
            text=tl_df["عدد الشروخ"], textposition="top center"
        ))
        fig5.add_trace(go.Scatter(
            x=tl_df["التاريخ"], y=tl_df["مستوى الخطورة"],
            mode="lines+markers", name="مستوى الخطورة",
            line=dict(color="#EF4444",width=2,dash="dot"),
            marker=dict(size=8,color="#EF4444"), yaxis="y2"
        ))
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"), height=300,
            margin=dict(t=20,b=20,l=20,r=20),
            xaxis=dict(gridcolor="#1E3A5F"),
            yaxis=dict(title="عدد الشروخ",gridcolor="#1E3A5F"),
            yaxis2=dict(title="الخطورة",overlaying="y",side="right",gridcolor="#1E3A5F"),
            legend=dict(orientation="h",y=1.1)
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── AI Recommendations ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🤖 التوصيات الشاملة بالذكاء الاصطناعي")
    if st.button("🔄 توليد تقرير وتوصيات شاملة", type="primary", use_container_width=True):
        with st.spinner("جاري تحليل جميع السجلات..."):
            records = get_all_records()
            sev_name = {"CRITICAL":"خطير جداً","HIGH":"عالي","MEDIUM":"متوسط","LOW":"منخفض"}
            lines = [
                f"• {r['image_filename']}: {r['num_cracks']} شروخ | "
                f"{sev_name.get(r.get('overall_severity',''),'غير محدد')} | "
                f"ثقة {r.get('overall_confidence',0)}% | {r.get('material_type','غير محدد')}"
                for r in records
            ]
            summary_txt = "\n".join(lines)
            summary_txt += f"\n\nإجمالي: {stats['total']} | حرجة: {critical} | عالية: {high} | متوسطة: {medium} | منخفضة: {low}"
            recs = generate_dashboard_recommendations(summary_txt)

        if recs.get("overall_assessment"):
            st.markdown(f"""<div class="rec-card" style="border-left-color:#38BDF8;">
                <p style="font-weight:700;color:#38BDF8;margin-bottom:0.5rem;">📋 التقييم الشامل</p>
                <p style="line-height:2;color:#CBD5E1;">{recs['overall_assessment']}</p>
            </div>""", unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            if recs.get("priority_actions"):
                st.markdown("**⚡ الإجراءات ذات الأولوية:**")
                for a in recs["priority_actions"]:
                    st.markdown(f"- {a}")
            if recs.get("maintenance_schedule"):
                st.info(f"📅 **جدول الصيانة:** {recs['maintenance_schedule']}")
            if recs.get("budget_estimate"):
                st.markdown(f"💰 **تقدير الميزانية:** {recs['budget_estimate']}")
        with rc2:
            if recs.get("risk_areas"):
                st.markdown("**⚠️ المناطق الأكثر خطورة:**")
                for area in recs["risk_areas"]:
                    st.warning(area)
            if recs.get("preventive_measures"):
                st.markdown("**🛡 الإجراءات الوقائية:**")
                for m in recs["preventive_measures"]:
                    st.markdown(f"- {m}")


# .......................................
#  Main
# ........................................

def main():
    page = render_sidebar()
    if   page == "🔬 تحليل صورة جديدة":  render_analysis_page()
    elif page == "📂 سجل التحليلات":      render_history_page()
    elif page == "📊 لوحة المعلومات":     render_dashboard_page()


if __name__ == "__main__":
    main()
