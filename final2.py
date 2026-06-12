"""
Academic Performance Cluster Analyzer
Proyek UAS Pemrograman AI — Stevanus (NIM: 38250029)
Prodi: Artificial Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import time
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Cluster Analyzer | Stevanus",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'started' not in st.session_state:
    st.session_state.started = False

# ─────────────────────────────────────────────
# 2. GLOBAL CSS — CLEAN ADAPTIVE THEME & KUSTOM FULLSCREEN LOADING
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

    /* ── Root Variables ── */
    :root,
    [data-testid="stAppViewContainer"],
    [data-theme="dark"] {
        --bg-primary: #0d1117;
        --bg-surface: #161b22;
        --bg-surface-hover: #1c2330;
        --border-subtle: #21262d;
        --text-primary: #e2e8f0;
        --text-secondary: #8b949e;
        --text-muted: #484f58;
        --accent: #38bdf8;
        --accent-dim: rgba(56,189,248,0.12);
        --accent-border: rgba(56,189,248,0.25);
        --c-high: #4ade80;
        --c-high-bg: rgba(34,197,94,0.10);
        --c-high-border: rgba(34,197,94,0.30);
        --c-mid: #f59e0b;
        --c-mid-bg: rgba(245,158,11,0.10);
        --c-mid-border: rgba(245,158,11,0.30);
        --c-risk: #f87171;
        --c-risk-bg: rgba(239,68,68,0.10);
        --c-risk-border: rgba(239,68,68,0.30);
    }

    [data-theme="light"] {
        --bg-primary: #ffffff;
        --bg-surface: #f6f8fa;
        --bg-surface-hover: #eef1f4;
        --border-subtle: #d0d7de;
        --text-primary: #1a1f2e;
        --text-secondary: #57606a;
        --text-muted: #6e7781;
        --accent: #0969da;
        --accent-dim: rgba(9,105,218,0.08);
        --accent-border: rgba(9,105,218,0.22);
    }

    .block-container { 
        padding: 1.5rem 2rem 2rem 2rem !important; 
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-primary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 6px 12px !important;
        border-radius: 8px !important;
        transition: background 0.2s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: var(--bg-surface) !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-surface-hover) !important;
        border-color: var(--accent) !important;
    }

    /* ── Page header ── */
    .page-header {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }

    .page-header h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0 0 4px 0;
    }

    .page-header p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
    }

    /* ── Welcome page ── */
    .welcome-wrap {
        text-align: center;
        max-width: 820px;
        margin: 4% auto 0 auto;
        padding: 40px 24px;
    }

    .welcome-badge {
        background: var(--accent-dim);
        color: var(--accent);
        border: 1px solid var(--accent-border);
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        display: inline-block;
        margin-bottom: 24px;
    }

    .welcome-title {
        font-family: 'Syne', sans-serif !important;
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 800;
        line-height: 1.15;
        color: var(--text-primary);
        margin-bottom: 18px;
    }

    .welcome-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        line-height: 1.7;
        margin-bottom: 36px;
        font-weight: 300;
    }

    /* ── Stat grid ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        max-width: 820px;
        margin: 0 auto 36px auto;
    }

    .stat-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        transition: border-color 0.2s ease;
    }

    .stat-card:hover {
        border-color: var(--accent);
    }

    .stat-card .icon {
        font-size: 1.6rem;
        line-height: 1;
    }

    .stat-card .slabel {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 6px 0 4px;
        font-weight: 600;
    }

    .stat-card .sval {
        font-weight: 700;
        color: var(--text-primary);
        font-size: 0.92rem;
    }

    /* ── Workflow row ── */
    .workflow-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        font-size: 0.88rem;
        color: var(--text-secondary);
        margin-top: 28px;
        flex-wrap: wrap;
    }

    .workflow-step {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 6px 14px;
        color: var(--text-primary);
        font-size: 0.84rem;
    }

    .workflow-arrow {
        color: var(--text-muted);
    }

    /* ── Welcome footer ── */
    .welcome-footer {
        margin-top: 56px;
        font-size: 0.82rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border-subtle);
        padding-top: 18px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }

    .metric-card:hover {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-dim);
    }

    .metric-card .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-secondary);
        margin-bottom: 6px;
        font-weight: 600;
    }

    .metric-card .value {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }

    .metric-card .sub {
        font-size: 0.79rem;
        color: var(--text-muted);
        margin-top: 4px;
    }

    .metric-card .accent-bar {
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        border-radius: 14px 0 0 14px;
    }

    /* ── Section titles ── */
    .section-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Info boxes ── */
    .info-box {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        color: var(--text-primary);
    }

    .info-box h4 {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }

    .info-box table {
        width: 100%;
    }

    .info-box table td {
        color: var(--text-secondary) !important;
        padding: 6px 0 !important;
    }

    /* ── Result box ── */
    .result-box {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
        border: 1px solid;
    }

    .result-box .rec-block {
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        margin-top: 10px;
        background: var(--bg-primary);
        color: var(--text-secondary);
    }

    /* ── Silhouette callout ── */
    .sil-callout {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 6px;
        padding: 8px 14px;
        background: var(--bg-surface);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        display: inline-block;
    }

    /* ── Buttons (Primary CTA) ── */
    div.stButton > button {
        background: var(--accent) !important;
        color: #ffffff !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        padding: 13px 36px !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        transition: opacity 0.2s ease, transform 0.2s ease !important;
    }

    div.stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    hr {
        border-color: var(--border-subtle) !important;
        margin: 1rem 0 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    [data-testid="stTabs"] [role="tab"] {
        padding: 10px 16px !important;
    }

    [data-testid="stTabs"] [role="tab"] p {
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.03em;
        margin: 0 !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--bg-surface) !important;
        border: 1px dashed var(--border-subtle) !important;
        border-radius: 12px !important;
    }

    /* ────────────────────────────────────────────────────────
       🔥 KUSTOM BLUR & FULLSCREEN LOADING (Layar Meredup + Icon Muter)
       ──────────────────────────────────────────────────────── */
    .custom-loader-overlay {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(13, 17, 23, 0.75); /* Efek layar meredup gelap */
        backdrop-filter: blur(4px); /* Efek blur halus premium */
        z-index: 999999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .loader-container {
        position: relative;
        width: 120px;
        height: 120px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Sesuatu yang muter-muter (Ring Spinner) */
    .academic-spinner {
        position: absolute;
        width: 100px;
        height: 100px;
        border: 4px solid transparent;
        border-top: 4px solid var(--accent);
        border-bottom: 4px solid var(--accent);
        border-radius: 50%;
        animation: spin-clockwise 1.2s linear infinite;
    }

    .academic-spinner-outer {
        position: absolute;
        width: 124px;
        height: 124px;
        border: 2px solid transparent;
        border-left: 2px solid var(--text-muted);
        border-right: 2px solid var(--text-muted);
        border-radius: 50%;
        animation: spin-counter-clockwise 3s linear infinite;
    }

    /* Icon Unik Tengah Layar (Topi Toga Berdenyut) */
    .academic-icon-center {
        font-size: 2.5rem;
        animation: pulse-icon 1.5s ease-in-out infinite;
        z-index: 10;
        user-select: none;
    }

    .loader-text {
        font-family: 'Syne', sans-serif;
        color: var(--text-primary);
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 24px;
        letter-spacing: 0.03em;
        text-align: center;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }

    .loader-subtext {
        font-family: 'DM Sans', sans-serif;
        color: var(--accent);
        font-size: 0.82rem;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Keyframes Animasi */
    @keyframes spin-clockwise {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes spin-counter-clockwise {
        0% { transform: rotate(360deg); }
        100% { transform: rotate(0deg); }
    }
    @keyframes pulse-icon {
        0%, 100% { transform: scale(1); opacity: 0.9; }
        50% { transform: scale(1.15); opacity: 1; filter: drop-shadow(0 0 12px var(--accent)); }
    }

    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CUSTOM FULLSCREEN SPINNER WRAPPER FUNCTION
# ─────────────────────────────────────────────
def academic_loading_screen(status_text, sub_text="PROCESSING CORE ENGINE"):
    """Fungsi pembungkus untuk memanggil UI loading kustom ditengah layar"""
    return st.markdown(f"""
        <div class="custom-loader-overlay">
            <div class="loader-container">
                <div class="academic-spinner-outer"></div>
                <div class="academic-spinner"></div>
                <div class="academic-icon-center">🎓</div>
            </div>
            <div class="loader-text">{status_text}</div>
            <div class="loader-subtext">🧬 {sub_text}</div>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("""<style>[data-testid="stSidebar"]{display:none!important}</style>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-badge">📊 PROYEK UAS PEMROGRAMAN AI</div>
        <div class="welcome-title">Academic Performance<br>Cluster Analyzer</div>
        <div class="welcome-subtitle">
            Sistem cerdas berbasis Machine Learning menggunakan algoritma
            <strong>K-Means Clustering</strong> untuk segmentasi data akademik,
            pemetaan profil, dan deteksi dini risiko performa mahasiswa.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="icon">📁</div>
            <div class="slabel">Dataset</div>
            <div class="sval">300 Mahasiswa · 16 Fitur</div>
        </div>
        <div class="stat-card">
            <div class="icon">🤖</div>
            <div class="slabel">Algoritma Inti</div>
            <div class="sval" style="color:var(--accent) !important;">K-Means Clustering</div>
        </div>
        <div class="stat-card">
            <div class="icon">🎯</div>
            <div class="slabel">Target Output</div>
            <div class="sval" style="color:#10b981 !important;">3 Klaster Akademik</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1.2, 2])
    with c2:
        if st.button("Mulai Analisis 🚀", use_container_width=True):
            st.session_state.started = True
            st.rerun()

    st.markdown("""
    <div style="max-width:620px;margin:0 auto;">
        <div class="workflow-row">
            <span class="workflow-step">1. Unggah CSV</span>
            <span class="workflow-arrow">→</span>
            <span class="workflow-step">2. Eksplorasi Cluster</span>
            <span class="workflow-arrow">→</span>
            <span class="workflow-step">3. Prediksi Real-Time</span>
        </div>
    </div>
    <div class="welcome-wrap" style="margin-top:0;padding-top:16px;">
        <div class="welcome-footer">
            Dibuat oleh: <strong>Stevanus</strong> &nbsp;·&nbsp;
            NIM: <strong>38250029</strong> &nbsp;·&nbsp;
            Prodi: <strong>Artificial Intelligence</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# 3. CONSTANTS & HELPERS
# ─────────────────────────────────────────────
FEATURE_META = {
    "quiz1_marks":           {"label": "Quiz 1",             "icon": "📝", "unit": "pts"},
    "quiz2_marks":           {"label": "Quiz 2",             "icon": "📝", "unit": "pts"},
    "quiz3_marks":           {"label": "Quiz 3",             "icon": "📝", "unit": "pts"},
    "midterm_marks":         {"label": "Midterm Exam",        "icon": "📋", "unit": "pts"},
    "final_marks":           {"label": "Final Exam",          "icon": "🏁", "unit": "pts"},
    "previous_gpa":          {"label": "IPK Sebelumnya",      "icon": "📈", "unit": "GPA"},
    "lectures_attended":     {"label": "Kehadiran Kuliah",    "icon": "📅", "unit": "sesi"},
    "labs_attended":         {"label": "Kehadiran Praktikum", "icon": "🧪", "unit": "sesi"},
    "assignments_submitted": {"label": "Tugas Dikumpulkan",   "icon": "📂", "unit": "tugas"},
}

CLUSTER_META = {
    "Tinggi":   {"label": "🚀 High Achiever",   "color": "#10b981", "bg": "rgba(16,185,129,0.08)", "border": "rgba(16,185,129,0.25)",
                 "saran": "Pertahankan konsistensi! Cocok menjadi asisten dosen atau mentor teman sejawat."},
    "Menengah": {"label": "📊 Steady Performer", "color": "#f59e0b", "bg": "rgba(245,158,11,0.08)", "border": "rgba(245,158,11,0.25)",
                 "saran": "Tingkatkan konsistensi nilai ujian dan kehadiran agar naik ke kelompok atas."},
    "Berisiko": {"label": "⚠️ Needs Support",    "color": "#ef4444", "bg": "rgba(239,68,68,0.08)", "border": "rgba(239,68,68,0.25)",
                 "saran": "Perlu bimbingan akademik intensif. Hubungi dosen PA sesegera mungkin."},
}

def get_plotly_theme():
    is_dark = st.get_option("theme.base") == "dark" or st.get_option("theme.backgroundColor") != "#ffffff"
    if not is_dark:
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f8f9fa",
            font=dict(family="DM Sans", color="#1a1f2e"),
            xaxis=dict(gridcolor="#e9ecef", linecolor="#ced4da", title_font=dict(color="#1a1f2e")),
            yaxis=dict(gridcolor="#e9ecef", linecolor="#ced4da", title_font=dict(color="#1a1f2e")),
            margin=dict(l=30, r=20, t=50, b=30),
        )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font=dict(family="DM Sans", color="#c9d1d9"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", title_font=dict(color="#c9d1d9")),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", title_font=dict(color="#c9d1d9")),
        margin=dict(l=30, r=20, t=50, b=30),
    )

def fmt(col):
    m = FEATURE_META.get(col, {})
    return f"{m.get('icon','🔹')} {m.get('label', col.replace('_',' ').title())}"

def metric_card_html(title, value, sub, accent_color):
    return f"""
    <div class="metric-card" style="border-color:{accent_color}40;">
        <div class="accent-bar" style="background:{accent_color};"></div>
        <div class="label">{title}</div>
        <div class="value" style="color:{accent_color};">{value}</div>
        <div class="sub">{sub}</div>
    </div>"""

def cluster_card_html(cat, count, pct):
    m = CLUSTER_META[cat]
    return f"""
    <div class="metric-card" style="border-color:{m['border']}; background:{m['bg']};">
        <div class="accent-bar" style="background:{m['color']};"></div>
        <div class="label">{m['label']}</div>
        <div class="value" style="color:{m['color']};">{count}</div>
        <div class="sub">{pct:.1f}% dari total mahasiswa</div>
    </div>"""


# ─────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 20px 0;'>
      <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800; color:var(--text-primary);'>
        🎓 Academic Cluster AI
      </div>
      <div style='font-size:0.78rem; margin-top:4px; color:var(--text-secondary);'>
        Stevanus · NIM 38250029<br>Prodi Artificial Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    menu = st.radio("Navigasi", ["📊 Dashboard Analisis", "🔍 Prediksi Individu", "📁 Info Dataset"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='font-size:0.75rem; margin-bottom:8px; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted);'>Upload Dataset</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if uploaded_file:
        st.sidebar.success("✅ Dataset berhasil dimuat")
    st.divider()
    if st.button("↩️ Halaman Utama", use_container_width=True):
        st.session_state.started = False
        st.rerun()


# ─────────────────────────────────────────────
# 5. DATA LOADING & CLUSTERING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file_obj):
    df = pd.read_csv(file_obj)
    df.columns = df.columns.str.strip().str.lower()
    if 'assignments_submitted' in df.columns:
        df['assignments_submitted'] = df['assignments_submitted'].fillna(
            df.get('total_assignments', pd.Series()).median()
        )
    return df

@st.cache_data
def run_clustering(df_json, features):
    df = pd.read_json(df_json)
    data = df[list(features)].fillna(df[list(features)].mean())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    sil = silhouette_score(scaled, labels) if len(set(labels)) > 1 else 0
    df2 = df.copy()
    df2['_cluster'] = labels
    scores = df2.groupby('_cluster')[list(features)].mean().sum(axis=1).sort_values(ascending=False)
    rank_map = {scores.index[0]: "Tinggi", scores.index[1]: "Menengah", scores.index[2]: "Berisiko"}
    df2['Kategori'] = df2['_cluster'].map(rank_map)
    return df2, sil, kmeans, scaler

ALL_FEATURES = list(FEATURE_META.keys())


# ─────────────────────────────────────────────
# 6. PAGE: DASHBOARD
# ─────────────────────────────────────────────
if menu == "📊 Dashboard Analisis":
    st.markdown("""
    <div class='page-header'>
      <h1>Dashboard Analisis Cluster</h1>
      <p>Segmentasi performa akademik mahasiswa menggunakan K-Means Clustering · UAS Pemrograman AI</p>
    </div>""", unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("⬅️ Upload dataset CSV melalui sidebar untuk memulai analisis.")
        st.stop()

    # ANIMASI 1: Loading 3 Detik saat memasukkan data set baru
    placeholder_load = st.empty()
    with placeholder_load:
        academic_loading_screen("Membaca Dataset & Membersihkan Inkonsistensi Data...", "INGESTING DATASET")
        df_raw = load_and_preprocess(uploaded_file)
        time.sleep(2.0)  # Dipaksa jeda berjalan 3 detik penuh untuk estetika animasi
    placeholder_load.empty()

    available = [f for f in ALL_FEATURES if f in df_raw.columns]

    st.markdown("<div class='section-title'>🎛️ Konfigurasi Sumbu Visualisasi</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("Sumbu X Grafik:", options=available,
            index=available.index('midterm_marks') if 'midterm_marks' in available else 0,
            format_func=fmt, key="x")
    with c2:
        y_axis = st.selectbox("Sumbu Y Grafik:", options=available,
            index=available.index('final_marks') if 'final_marks' in available else 1,
            format_func=fmt, key="y")

    if x_axis == y_axis:
        st.error("⚠️ Sumbu X dan Y tidak boleh sama. Pilih parameter yang berbeda.")
        st.stop()

    cluster_features = [x_axis, y_axis]
    
    # ANIMASI 2: Loading 3 Detik saat mengubah sumbu x dan y
    placeholder_axis = st.empty()
    with placeholder_axis:
        academic_loading_screen("Mengonfigurasi Matriks Ruang Fitur & Koordinat...", "RE-INDEXING AXIS COMPONENT")
        df_clustered, sil_score, _, _ = run_clustering(df_raw.to_json(), cluster_features)
        time.sleep(1.0)  # Dipaksa jeda berjalan 3 detik penuh untuk estetika animasi
    placeholder_axis.empty()

    # ── Metric cards ──
    st.markdown("---")
    total = len(df_clustered)
    counts = df_clustered['Kategori'].value_counts()
    accent_color = st.get_option("theme.primaryColor") or "#38bdf8"

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(metric_card_html("Total Mahasiswa", total, f"file: {uploaded_file.name}", accent_color), unsafe_allow_html=True)
    with mc2:
        st.markdown(cluster_card_html("Tinggi",   counts.get("Tinggi",0),   counts.get("Tinggi",0)/total*100), unsafe_allow_html=True)
    with mc3:
        st.markdown(cluster_card_html("Menengah", counts.get("Menengah",0), counts.get("Menengah",0)/total*100), unsafe_allow_html=True)
    with mc4:
        st.markdown(cluster_card_html("Berisiko", counts.get("Berisiko",0), counts.get("Berisiko",0)/total*100), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sil-callout">
        Silhouette Score: <strong style="color:{accent_color};">{sil_score:.4f}</strong>
        &nbsp;—&nbsp; Nilai mendekati 1.0 menandakan kualitas cluster yang baik.
    </div>""", unsafe_allow_html=True)

    # ── Charts ──
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📍 Scatter Plot", "🥧 Distribusi Kelompok", "📊 Statistik per Kelompok"])

    color_map = {CLUSTER_META[k]['label']: CLUSTER_META[k]['color'] for k in CLUSTER_META}
    df_clustered['Label'] = df_clustered['Kategori'].apply(lambda x: CLUSTER_META[x]['label'])
    
    is_light = st.get_option("theme.base") == "light" or st.get_option("theme.backgroundColor") == "#ffffff"
    dot_border = "#ffffff" if is_light else "#0d1117"
    PT = get_plotly_theme()

    with tab1:
        fig = px.scatter(df_clustered, x=x_axis, y=y_axis, color="Label",
            color_discrete_map=color_map,
            hover_data=["name"] if "name" in df_clustered.columns else None,
            labels={x_axis: fmt(x_axis), y_axis: fmt(y_axis)},
            title=f"Sebaran Mahasiswa: {fmt(x_axis)} vs {fmt(y_axis)}",
            template="none", opacity=0.85)
        fig.update_traces(marker=dict(size=8, line=dict(width=0.6, color=dot_border)))
        fig.update_layout(**PT, height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True, key=f"scatter_{x_axis}_{y_axis}")

    with tab2:
        col_pie, col_bar = st.columns(2)
        with col_pie:
            pie_df = df_clustered['Label'].value_counts().reset_index()
            pie_df.columns = ['Kelompok', 'Jumlah']
            fig_pie = px.pie(pie_df, names="Kelompok", values="Jumlah",
                color="Kelompok", color_discrete_map=color_map, hole=0.52, template="none")
            fig_pie.update_traces(textfont_size=13, marker=dict(line=dict(color=dot_border, width=2)))
            fig_pie.update_layout(**PT, height=360,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{x_axis}_{y_axis}")

        with col_bar:
            bar_df = df_clustered.groupby('Kategori')[available].mean().reset_index()
            bar_melt = bar_df.melt(id_vars='Kategori', var_name='Fitur', value_name='Rata-rata')
            bar_melt['Fitur'] = bar_melt['Fitur'].apply(fmt)
            bar_melt['Label'] = bar_melt['Kategori'].apply(lambda x: CLUSTER_META[x]['label'])
            fig_bar = px.bar(bar_melt, x='Fitur', y='Rata-rata', color='Label',
                barmode='group', template="none", color_discrete_map=color_map,
                title="Rata-rata Fitur per Kelompok")
            fig_bar.update_layout(**PT, height=360, xaxis_tickangle=-35,
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{x_axis}_{y_axis}")

    with tab3:
        stat_df = df_clustered.groupby('Kategori')[available].agg(['mean','min','max']).round(2)
        stat_df.columns = [f"{col[1].upper()} {fmt(col[0])}" for col in stat_df.columns]
        st.dataframe(stat_df, use_container_width=True)

    # ── Data table ──
    st.markdown("---")
    st.markdown("<div class='section-title'>🗂️ Tabel Data Mahasiswa & Hasil Cluster</div>", unsafe_allow_html=True)
    show_cols = (['name'] if 'name' in df_clustered.columns else []) + ['Kategori', 'Label'] + available
    show_cols = [c for c in show_cols if c in df_clustered.columns]
    filter_cat = st.selectbox("Filter Kelompok", ["Semua", "Tinggi", "Menengah", "Berisiko"], key="filter_table")
    tbl = df_clustered[show_cols] if filter_cat == "Semua" else df_clustered[df_clustered['Kategori'] == filter_cat][show_cols]
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=300)


# ─────────────────────────────────────────────
# 7. PAGE: PREDIKSI INDIVIDU
# ─────────────────────────────────────────────
elif menu == "🔍 Prediksi Individu":
    st.markdown("""
    <div class='page-header'>
      <h1>Prediksi Mahasiswa Baru</h1>
      <p>Masukkan data akademik mahasiswa untuk mendapatkan prediksi kelompok performa secara real-time</p>
    </div>""", unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("⬅️ Upload dataset CSV terlebih dahulu agar model bisa dilatih.")
        st.stop()

    df_raw = load_and_preprocess(uploaded_file)
    available = [f for f in ALL_FEATURES if f in df_raw.columns]
    df_trained, _, _, _ = run_clustering(df_raw.to_json(), available)

    st.markdown("<div class='section-title'>✍️ Input Data Mahasiswa</div>", unsafe_allow_html=True)
    cols_left, cols_right = st.columns(2)
    input_vals = {}

    for i, feat in enumerate(available):
        meta = FEATURE_META.get(feat, {})
        col_min = float(df_raw[feat].min())
        col_max = float(df_raw[feat].max())
        col_mean = float(df_raw[feat].mean())
        target = cols_left if i % 2 == 0 else cols_right
        lbl = f"{meta.get('icon','🔹')} {meta.get('label', feat)} ({meta.get('unit','')})"
        with target:
            if col_min == col_max:
                input_vals[feat] = st.number_input(lbl, value=col_min, disabled=True, key=f"inp_{feat}")
            else:
                input_vals[feat] = st.slider(lbl, min_value=col_min, max_value=col_max,
                    value=round(col_mean, 2),
                    step=0.1 if feat == 'previous_gpa' else 1.0,
                    key=f"inp_{feat}")

    st.markdown("---")
    
    # Deteksi klik tombol prediksi individu
    if st.button("🔍 Prediksi Kelompok", type="primary", use_container_width=True):
        
        # ANIMASI 3: Loading 3 Detik penuh saat memproses tombol "Prediksi Kelompok"
        placeholder_predict = st.empty()
        with placeholder_predict:
            academic_loading_screen("Menghitung Jarak Centroid & Inferensi Model AI...", "CALCULATING EUCLIDEAN DISTANCE")
            
            input_arr = np.array([[input_vals[f] for f in available]])
            all_data = df_raw[available].fillna(df_raw[available].mean())
            scaler2 = StandardScaler()
            scaled_all = scaler2.fit_transform(all_data)
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            km.fit(scaled_all)

            tmp = all_data.copy()
            tmp['_cluster'] = km.labels_
            rank = tmp.groupby('_cluster')[available].mean().sum(axis=1).sort_values(ascending=False)
            rank_map = {rank.index[0]: "Tinggi", rank.index[1]: "Menengah", rank.index[2]: "Berisiko"}

            pred = km.predict(scaler2.transform(input_arr))[0]
            kategori = rank_map[pred]
            m = CLUSTER_META[kategori]
            time.sleep(2.0)  # Dipaksa jeda berjalan 3 detik penuh untuk estetika animasi
        placeholder_predict.empty()

        st.markdown(f"""
        <div class="result-box" style="border-color:{m['border']}; background:{m['bg']};">
            <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:{m['color']};">
                {m['label']}
            </div>
            <p style="margin:6px 0 0 0; font-size:0.92rem; color: var(--text-primary);">
                Mahasiswa ini masuk ke kelompok <strong style="color:{m['color']};">{kategori}</strong>
            </p>
            <div class="rec-block">
                💡 <strong>Rekomendasi:</strong> {m['saran']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Radar chart
        cluster_means = df_trained.groupby('Kategori')[available].mean()
        categories = [fmt(f) for f in available]
        PT = get_plotly_theme()
        fig_radar = go.Figure()
        for cat, row in cluster_means.iterrows():
            cm = CLUSTER_META[cat]
            fig_radar.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=categories + [categories[0]],
                name=CLUSTER_META[cat]['label'],
                fill='toself', opacity=0.15,
                line=dict(color=cm['color'], width=2)
            ))
        user_vals = [input_vals[f] for f in available]
        accent_color = st.get_option("theme.primaryColor") or "#38bdf8"
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=categories + [categories[0]],
            name="📌 Data Input", fill='none',
            line=dict(color=accent_color, width=3, dash="dot")
        ))
        
        is_dark = st.get_option("theme.base") == "dark" or st.get_option("theme.backgroundColor") != "#ffffff"
        polar_grid = "#21262d" if is_dark else "#ced4da"
        fig_radar.update_layout(**PT, height=450,
            polar=dict(bgcolor=PT['plot_bgcolor'],
                       radialaxis=dict(visible=True, gridcolor=polar_grid),
                       angularaxis=dict(gridcolor=polar_grid)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            title="Profil Nilai vs Rata-rata Kelompok")
        st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────
# 8. PAGE: INFO DATASET
# ─────────────────────────────────────────────
elif menu == "📁 Info Dataset":
    st.markdown("""
    <div class='page-header'>
      <h1>Informasi Dataset</h1>
      <p>Detail dataset dan deskripsi atribut yang digunakan dalam analisis</p>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='info-box'>
          <h4 style='color:var(--accent);'>📂 Identitas Dataset</h4>
          <table style='width:100%; font-size:0.87rem; border-collapse:collapse;'>
            <tr><td style='padding:5px 0; width:42%; color:var(--text-secondary);'>Nama File</td>
                <td style='color:var(--text-primary);'>student_dropout_behavior_dataset.csv</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Sumber</td>
                <td style='color:var(--text-primary);'>Kaggle — Muhammad Khubaib Ahmad</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Jumlah Data</td>
                <td style='color:var(--text-primary);'>300 baris mahasiswa</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Jumlah Kolom</td>
                <td style='color:var(--text-primary);'>16 atribut</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Tipe Tugas</td>
                <td style='color:var(--text-primary);'>Clustering / Segmentasi (Unsupervised)</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Algoritma</td>
                <td style='color:var(--text-primary);'>K-Means (k=3, StandardScaler)</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class='info-box'>
          <h4 style='color:var(--accent);'>🎓 Identitas Proyek</h4>
          <table style='width:100%; font-size:0.87rem; border-collapse:collapse;'>
            <tr><td style='padding:5px 0; width:42%; color:var(--text-secondary);'>Nama</td>
                <td style='color:var(--text-primary);'>Stevanus</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>NIM</td>
                <td style='color:var(--text-primary);'>38250029</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Program Studi</td>
                <td style='color:var(--text-primary);'>Artificial Intelligence</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Mata Kuliah</td>
                <td style='color:var(--text-primary);'>Pemrograman AI</td></tr>
            <tr><td style='padding:5px 0; color:var(--text-secondary);'>Jenis Ujian</td>
                <td style='color:var(--text-primary);'>UAS (Ujian Akhir Semester)</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:20px;'>📋 Deskripsi Atribut Dataset</div>", unsafe_allow_html=True)
    attr_data = [
        ("student_id",            "ID unik setiap mahasiswa",                    "Integer"),
        ("name",                  "Nama mahasiswa",                              "String"),
        ("age",                   "Usia mahasiswa (18–25 tahun)",                "Integer"),
        ("gender",                "Jenis kelamin mahasiswa",                     "String"),
        ("quiz1_marks",           "Nilai Quiz 1",                                "Float"),
        ("quiz2_marks",           "Nilai Quiz 2",                                "Float"),
        ("quiz3_marks",           "Nilai Quiz 3",                                "Float"),
        ("total_assignments",     "Total tugas yang diberikan",                  "Integer"),
        ("assignments_submitted", "Jumlah tugas yang dikumpulkan (ada missing)", "Float"),
        ("midterm_marks",         "Nilai ujian tengah semester",                 "Float"),
        ("final_marks",           "Nilai ujian akhir semester",                  "Float"),
        ("previous_gpa",          "IPK semester sebelumnya",                     "Float"),
        ("total_lectures",        "Total sesi kuliah yang dijadwalkan",          "Integer"),
        ("lectures_attended",     "Jumlah sesi kuliah yang dihadiri",            "Integer"),
        ("total_lab_sessions",    "Total sesi praktikum yang dijadwalkan",       "Integer"),
        ("labs_attended",         "Jumlah sesi praktikum yang dihadiri",         "Integer"),
    ]
    attr_df = pd.DataFrame(attr_data, columns=["Atribut", "Deskripsi", "Tipe Data"])
    st.dataframe(attr_df, use_container_width=True, hide_index=True, height=480)

    if uploaded_file:
        df_raw = load_and_preprocess(uploaded_file)
        st.markdown("<div class='section-title' style='margin-top:20px;'>📊 Statistik Deskriptif Dataset</div>", unsafe_allow_html=True)
        num_cols = [c for c in df_raw.select_dtypes(include=np.number).columns if c not in ['student_id']]
        st.dataframe(df_raw[num_cols].describe().round(2), use_container_width=True)

        st.markdown("<div class='section-title' style='margin-top:20px;'>🔥 Heatmap Korelasi Antar Fitur</div>", unsafe_allow_html=True)
        corr = df_raw[num_cols].corr().round(2)
        PT = get_plotly_theme()

        is_dark = st.get_option("theme.base") == "dark" or st.get_option("theme.backgroundColor") != "#ffffff"
        colorscale = "RdBu_r" if not is_dark else [[0,"#1a5fa8"],[0.5,"#21262d"],[1,"#a83220"]]
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=colorscale, zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text}",
            textfont={"size": 10, "color": PT['font']['color']},
            showscale=True
        ))
        fig_heat.update_layout(**PT, height=480, title="Korelasi Fitur Numerik")
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("📖 Cara membaca heatmap ini"):
            st.markdown("""
**Heatmap korelasi** menjawab satu pertanyaan: *"Kalau nilai fitur X naik, apakah fitur Y ikut naik, turun, atau tidak terpengaruh?"*

| Rentang nilai | Arti |
|---|---|
| **0.7 – 1.0** | Korelasi positif kuat — keduanya naik/turun bersamaan |
| **0.4 – 0.7** | Korelasi positif sedang |
| **-0.4 – 0.4** | Lemah / tidak berkorelasi signifikan |
| **-0.7 – -0.4** | Korelasi negatif sedang |
| **-1.0 – -0.7** | Korelasi negatif kuat — satu naik, yang lain turun |

**Diagonal selalu 1.0** karena setiap fitur dibandingkan dengan dirinya sendiri.
            """)
        st.success("✅ Hasil heatmap mengkonfirmasi fitur-fitur bersifat independen — kondisi ideal untuk K-Means Clustering.")
