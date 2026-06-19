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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
# 2. GLOBAL CSS 
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

    /* ── Interpretasi Box ── */
    .interpretation-box {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
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
       BLUR & FULLSCREEN LOADING 
       ──────────────────────────────────────────────────────── */
    .custom-loader-overlay {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(13, 17, 23, 0.75);
        backdrop-filter: blur(4px);
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
# WELCOME SCREEN & CORE LOGIC ROUTER
# ─────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("""<style>[data-testid="stSidebar"]{display:none!important}</style>""", unsafe_allow_html=True)
    welcome_container = st.empty()
    

    with welcome_container.container():
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
                welcome_container.markdown("""
                    <style>
                    @keyframes spin3D {
                        0% { transform: rotateY(0deg) rotateX(0deg); }
                        100% { transform: rotateY(360deg) rotateX(360deg); }
                    }
                    @keyframes fadeInOverlay {
                        from { opacity: 0; backdrop-filter: blur(0px); }
                        to { opacity: 1; backdrop-filter: blur(4px); }
                    }
                    .custom-loader-overlay {
                        animation: fadeInOverlay 0.4s ease forwards;
                    }
                    .cube-3d-container {
                        perspective: 1000px;
                        width: 150px;
                        height: 150px;
                        margin: 40px auto;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    .cube-3d {
                        width: 120px;
                        height: 120px;
                        position: relative;
                        transform-style: preserve-3d;
                        animation: spin3D 6s infinite linear;
                    }
                    .cube-face {
                        position: absolute;
                        width: 120px;
                        height: 120px;
                        background: rgba(56, 189, 248, 0.18);
                        border: 2px solid #38bdf8;
                        color: #e2e8f0;
                        font-family: 'Syne', sans-serif;
                        font-size: 0.85rem;
                        font-weight: bold;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        text-align: center;
                        border-radius: 8px;
                        box-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
                        backdrop-filter: blur(2px);
                    }
                    .front  { transform: translateZ(60px); }
                    .back   { transform: rotateY(180deg) translateZ(60px); }
                    .left   { transform: rotateY(-90deg) translateZ(60px); }
                    .right  { transform: rotateY(90deg) translateZ(60px); }
                    .top    { transform: rotateX(90deg) translateZ(60px); }
                    .bottom { transform: rotateX(-90deg) translateZ(60px); }
                    </style>

                    <div class="custom-loader-overlay">
                        <div class="cube-3d-container">
                            <div class="cube-3d">
                                <div class="cube-face front"><span>📝</span><br>KUIS</div>
                                <div class="cube-face back"><span>📊</span><br>UJIAN</div>
                                <div class="cube-face left"><span>📅</span><br>ABSEN</div>
                                <div class="cube-face right"><span>📂</span><br>TUGAS</div>
                                <div class="cube-face top"><span>📈</span><br>IPK</div>
                                <div class="cube-face bottom"><span>🤖</span><br>AI ENGINE</div>
                            </div>
                        </div>
                        <div class="loader-text" style="margin-top:40px;">Memuat Modul Komponen AI & Struktur Interface...</div>
                        <div class="loader-subtext">🧬 LOADING INTERFACE MODULES</div>
                    </div>
                """, unsafe_allow_html=True)
                
                time.sleep(5) 
                welcome_container.empty()
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
    "Tinggi":   {"label": "🚀 Performa Tinggi",   "color": "#10b981", "bg": "rgba(16,185,129,0.08)", "border": "rgba(16,185,129,0.25)",
                 "saran": "Pertahankan konsistensi! Prospektif menjadi asisten laboratorium atau peer-mentor."},
    "Menengah": {"label": "📊 Performa Stabil", "color": "#f59e0b", "bg": "rgba(245,158,11,0.08)", "border": "rgba(245,158,11,0.25)",
                 "saran": "Optimalkan konsistensi ujian komparatif dan submisi tugas agar menembus klaster utama."},
    "Berisiko": {"label": "⚠️ Perlu Bimbingan",    "color": "#ef4444", "bg": "rgba(239,68,68,0.08)", "border": "rgba(239,68,68,0.25)",
                 "saran": "Perlu bimbingan intensif reguler bersama Penasihat Akademik (PA) sesegera mungkin."},
    "Override_Berisiko": {"label": "🛑 Tidak Layak", "color": "#ef4444", "bg": "rgba(239,68,68,0.15)", "border": "#ef4444",
                 "saran": "OTOMATIS TIDAK MEMENUHI SYARAT KELAYAKAN UJIAN KAMPUS. Angka akumulasi parameter berada di zona batas kritis (0 - 3)."}
}

def get_plotly_theme():
    is_dark_global = st.get_option("theme.base") == "dark" or st.get_option("theme.backgroundColor") != "#ffffff"
    if not is_dark_global:
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
    m = CLUSTER_META.get(cat, CLUSTER_META["Berisiko"])
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
        Stevanus - NIM 38250029<br>Prodi Artificial Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    menu = st.radio("Navigasi", ["📊 Dashboard Analisis", "🔍 Prediksi Individu", "📁 Info Dataset"],
                    label_visibility="collapsed")
    st.divider()
    
    # ── CONFIG BOBOT PERSENTASE ──
    st.markdown("<div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted);'>Kustomisasi Bobot AI</div>", unsafe_allow_html=True)
    with st.expander("⚙️ Atur Bobot", expanded=False):
        w_attendance = st.slider("Bobot Absensi & Tugas", 0, 100, 25, 5)
        w_exams = st.slider("Bobot Ujian (UTS/UAS)", 0, 100, 35, 5)
        w_quizzes = st.slider("Bobot Kuis", 0, 100, 20, 5)
        w_gpa = st.slider("Bobot IPK Sebelumnya", 0, 100, 20, 5)
        
        total_percentage = w_attendance + w_exams + w_quizzes + w_gpa
        is_weight_valid = (total_percentage == 100)
        
        if is_weight_valid:
            st.markdown(f"<div style='color:#10b981; font-weight:bold; font-size:0.85rem; text-align:center;'>✅ Total Bobot: {total_percentage}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#ef4444; font-weight:bold; font-size:0.85rem; text-align:center;'>❌ Total Bobot: {total_percentage}%<br>(Wajib Berjumlah 100%)</div>", unsafe_allow_html=True)
    
    #   BLACKLIST SYARAT KAMPUS
    st.divider()
    st.markdown("<div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted);'>Aturan Akademik Kampus</div>", unsafe_allow_html=True)
    with st.expander("🛑 Aturan Wajib Kampus", expanded=False):
        is_override_enabled = st.checkbox("Aktifkan", value=False)
        override_feature = st.selectbox("Parameter:", ["lectures_attended", "labs_attended", "assignments_submitted"], format_func=fmt)
        max_violation_limit = st.number_input("Batas Wajib (<=):", min_value=0, max_value=20, value=3, step=1, help="Kondisi tidak layak jika nilai kehadiran berada di angka kurang dari atau sama dengan nilai yang ditentukan.")

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
# 5. DATA LOADING & CLUSTERING FUNCTIONS
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

def run_clustering_with_weights(df_json, features, w_att, w_ex, w_qz, w_gp, override_active, ov_feat, ov_limit):
    df = pd.read_json(df_json)
    data = df[list(features)].fillna(df[list(features)].mean())
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled, columns=features)
    
    f_att = w_att / 100.0
    f_ex = w_ex / 100.0
    f_qz = w_qz / 100.0
    f_gp = w_gp / 100.0
    
    for col in features:
        if 'attended' in col or 'submitted' in col:
            scaled_df[col] = scaled_df[col] * f_att
        elif 'marks' in col and ('midterm' in col or 'final' in col):
            scaled_df[col] = scaled_df[col] * f_ex
        elif 'quiz' in col:
            scaled_df[col] = scaled_df[col] * f_qz
        elif 'gpa' in col:
            scaled_df[col] = scaled_df[col] * f_gp
            
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_df.values)
    
    df2 = df.copy()
    df2['_cluster'] = labels
    scores = df2.groupby('_cluster')[list(features)].mean().sum(axis=1).sort_values(ascending=False)
    rank_map = {scores.index[0]: "Tinggi", scores.index[1]: "Menengah", scores.index[2]: "Berisiko"}
    df2['Kategori'] = df2['_cluster'].map(rank_map)
    
    # Blacklist Override Logic
    if override_active:
        for idx, row in df2.iterrows():
            if ov_feat in df2.columns:
                val = row[ov_feat]
                if val <= ov_limit: 
                    df2.at[idx, 'Kategori'] = "Override_Berisiko"
                
    sil = silhouette_score(scaled_df.values, labels) if len(set(labels)) > 1 else 0
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

    if not is_weight_valid:
        st.warning("⚠️ Analisis Dashboard dikunci. Silakan sesuaikan Bobot Fitur di sidebar agar berjumlah pas 100%.")
        st.stop()

# Metric cards 
    st.markdown("---")
    total = len(df_clustered)
    counts = df_clustered['Kategori'].value_counts()
    accent_color = st.get_option("theme.primaryColor") or "#38bdf8"

    has_override = "Override_Berisiko" in counts and counts["Override_Berisiko"] > 0

    if has_override:
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.markdown(metric_card_html("Total Mahasiswa", total, f"file: {uploaded_file.name}", accent_color), unsafe_allow_html=True)
        with mc2:
            st.markdown(cluster_card_html("Tinggi", counts.get("Tinggi", 0), counts.get("Tinggi", 0)/total*100), unsafe_allow_html=True)
        with mc3:
            st.markdown(cluster_card_html("Menengah", counts.get("Menengah", 0), counts.get("Menengah", 0)/total*100), unsafe_allow_html=True)
        with mc4:
            st.markdown(cluster_card_html("Berisiko", counts.get("Berisiko", 0), counts.get("Berisiko", 0)/total*100), unsafe_allow_html=True)
        with mc5:
            st.markdown(cluster_card_html("Override_Berisiko", counts.get("Override_Berisiko", 0), counts.get("Override_Berisiko", 0)/total*100), unsafe_allow_html=True)
    else:
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(metric_card_html("Total Mahasiswa", total, f"file: {uploaded_file.name}", accent_color), unsafe_allow_html=True)
        with mc2:
            st.markdown(cluster_card_html("Tinggi", counts.get("Tinggi", 0), counts.get("Tinggi", 0)/total*100), unsafe_allow_html=True)
        with mc3:
            st.markdown(cluster_card_html("Menengah", counts.get("Menengah", 0), counts.get("Menengah", 0)/total*100), unsafe_allow_html=True)
        with mc4:
            st.markdown(cluster_card_html("Berisiko", counts.get("Berisiko", 0), counts.get("Berisiko", 0)/total*100), unsafe_allow_html=True)

    if counts.get("Override_Berisiko", 0) > 0:
        st.error(f"🛑 **Sistem Mendeteksi:** Sebanyak **{counts.get('Override_Berisiko')} Mahasiswa** otomatis masuk status Tidak Layak karena angka berada di bawah (0 - {int(max_violation_limit)})!")

    # Charts & Tab Layout 
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📍 Visualisasi Data", 
        "🥧 Proporsi & Karakteristik", 
        "📊 Detail Deskriptif Nilai", 
        "📈 Pengujian Cluster",
        "🧠 Signifikansi Fitur (XAI)"
    ])

    df_clustered['Label'] = df_clustered['Kategori'].apply(lambda x: CLUSTER_META.get(x, CLUSTER_META["Berisiko"])['label'])
    color_map = {CLUSTER_META[k]['label']: CLUSTER_META[k]['color'] for k in CLUSTER_META}
    
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

        st.markdown("---")
        st.markdown("<div class='section-title'>🗂️ Tabel Data Mahasiswa & Hasil Cluster</div>", unsafe_allow_html=True)
        show_cols = (['name'] if 'name' in df_clustered.columns else []) + ['Kategori', 'Label'] + available
        show_cols = [c for c in show_cols if c in df_clustered.columns]
        
        # ── MAP FILTER UNTUK MENYESUAIKAN LABEL DISPLAY DAN KATEGORI INTERNAL DATA ──
        filter_options = {
            "Semua": "Semua",
            "Tinggi": "Tinggi",
            "Stabil": "Menengah",
            "Perlu Bimbingan": "Berisiko",
            "Tidak Layak": "Override_Berisiko"
        }
        
        filter_cat = st.selectbox("Filter Kelompok", list(filter_options.keys()), key="filter_table")
        internal_cat = filter_options[filter_cat]
        
        tbl = df_clustered[show_cols] if filter_cat == "Semua" else df_clustered[df_clustered['Kategori'] == internal_cat][show_cols]
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=300)

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
            bar_df = df_clustered.groupby('Label')[available].mean().reset_index()
            bar_melt = bar_df.melt(id_vars='Label', var_name='Fitur', value_name='Rata-rata')
            bar_melt['Fitur'] = bar_melt['Fitur'].apply(fmt)
            
            fig_bar = px.bar(bar_melt, x='Fitur', y='Rata-rata', color='Label',
                barmode='group', template="none", color_discrete_map=color_map,
                title="Rata-rata Fitur per Kelompok (Up to Date)")
            fig_bar.update_layout(**PT, height=360, xaxis_tickangle=-35,
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{x_axis}_{y_axis}")

    with tab3:
        stat_df = df_clustered.groupby('Kategori')[available].agg(['mean','min','max']).round(2)
        stat_df.columns = [f"{col[1].upper()} {fmt(col[0])}" for col in stat_df.columns]
        st.dataframe(stat_df, use_container_width=True)

    with tab4:
        st.markdown("<div class='section-title'>📉 Penentuan Centroid via Elbow & Silhouette Score</div>", unsafe_allow_html=True)
        data_ev = df_raw[available].fillna(df_raw[available].mean())
        scaled_ev = StandardScaler().fit_transform(data_ev)
        
        inertias = []
        silhouettes = []
        k_list = list(range(2, 8))
        for k_val in k_list:
            km_test = KMeans(n_clusters=k_val, random_state=42, n_init=5).fit(scaled_ev)
            inertias.append(km_test.inertia_)
            silhouettes.append(silhouette_score(scaled_ev, km_test.labels_))
            
        c_el1, c_el2 = st.columns(2)
        with c_el1:
            fig_el = go.Figure(data=go.Scatter(x=k_list, y=inertias, mode='lines+markers', line=dict(color=accent_color, width=3)))
            fig_el.add_vline(x=3, line_dash="dash", line_color="#ef4444", annotation_text="K Pilihan = 3")
            fig_el.update_layout(**PT, title="Grafik Elbow", height=320, xaxis_title="Jumlah Cluster (k)", yaxis_title="Inertia / WCSS")
            st.plotly_chart(fig_el, use_container_width=True)
        with c_el2:
            fig_sil = go.Figure(data=go.Scatter(x=k_list, y=silhouettes, mode='lines+markers', line=dict(color="#10b981", width=3)))
            fig_sil.add_vline(x=3, line_dash="dash", line_color="#ef4444", annotation_text="K Tertinggi = 3")
            fig_sil.update_layout(**PT, title="Grafik Silhouette Score", height=320, xaxis_title="Jumlah Cluster (k)", yaxis_title="Silhouette Score")
            st.plotly_chart(fig_sil, use_container_width=True)

    with tab5:
        st.markdown("<div class='section-title'>🧠 Explainable AI (XAI): Fitur Paling Berpengaruh secara Global</div>", unsafe_allow_html=True)
        data_xai = df_raw[available].fillna(df_raw[available].mean())
        scaled_xai = StandardScaler().fit_transform(data_xai)
        
        km_xai = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_xai = km_xai.fit_predict(scaled_xai)
        
        rf_proxy = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_proxy.fit(scaled_xai, labels_xai)
        result_xai = permutation_importance(rf_proxy, scaled_xai, labels_xai, n_repeats=5, random_state=42)
        
        importance_df = pd.DataFrame({
            'Fitur': [fmt(f) for f in available],
            'Bobot Pengaruh': result_xai.importances_mean
        }).sort_values(ascending=True, by='Bobot Pengaruh')
        
        fig_xai = px.bar(importance_df, x='Bobot Pengaruh', y='Fitur', orientation='h', template='none')
        
        is_dark_global = st.get_option("theme.base") == "dark" or st.get_option("theme.backgroundColor") != "#ffffff"
        colorscale_fixed = "RdBu_r" if not is_dark_global else [[0,"#1a5fa8"],[0.5,"#21262d"],[1,"#a83220"]]
        
        fig_xai.update_traces(marker_color='#38bdf8', marker_line_color=dot_border, marker_line_width=0.5)
        fig_xai.update_layout(**PT, height=400, title="Tingkat Sensitivitas Matriks terhadap Penentuan Klaster Global",
                              xaxis_title="Skor Signifikansi Vektor", yaxis_title="")
        st.plotly_chart(fig_xai, use_container_width=True)


# ─────────────────────────────────────────────
# 7. PAGE: PREDIKSI INDIVIDU
# ─────────────────────────────────────────────
elif menu == "🔍 Prediksi Individu":
    st.markdown("""
    <div class='page-header'>
      <h1>Prediksi Mahasiswa Baru & Perbandingan Mahasiswa</h1>
      <p>Masukkan data akademik mahasiswa untuk mendapatkan prediksi kelompok performa secara real-time</p>
    </div>""", unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("⬅️ Upload dataset CSV terlebih dahulu agar model bisa dilatih.")
        st.stop()

    if not is_weight_valid:
        st.warning("⚠️ Prediksi Real-Time dikunci. Silakan sesuaikan Bobot Fitur di sidebar agar berjumlah pas 100%.")
        st.stop()

    df_raw = load_and_preprocess(uploaded_file)
    available = [f for f in ALL_FEATURES if f in df_raw.columns]
    
    df_trained, _, _, _ = run_clustering_with_weights(
        df_raw.to_json(), available, w_attendance, w_exams, w_quizzes, w_gpa,
        is_override_enabled, override_feature, max_violation_limit
    )

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    is_compare_mode = st.toggle("👥 Atifkan Mode Perbandingan Mahasiswa", value=False)

    if not is_compare_mode:
        st.markdown("<div class='section-title'>✍️ Data Mahasiswa</div>", unsafe_allow_html=True)
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
    else:
        c_std1, c_std2 = st.columns(2)
        input_vals = {}
        input_vals_b = {}
        
        with c_std1:
            st.markdown("<div class='section-title' style='color:#38bdf8;'>👤 Data Mahasiswa A</div>", unsafe_allow_html=True)
            for feat in available:
                meta = FEATURE_META.get(feat, {})
                col_min = float(df_raw[feat].min())
                col_max = float(df_raw[feat].max())
                col_mean = float(df_raw[feat].mean())
                lbl = f"{meta.get('icon','🔹')} {meta.get('label', feat)} A"
                if col_min == col_max:
                    input_vals[feat] = st.number_input(lbl, value=col_min, disabled=True, key=f"compA_{feat}")
                else:
                    input_vals[feat] = st.slider(lbl, min_value=col_min, max_value=col_max, value=round(col_mean, 2), step=0.1 if feat == 'previous_gpa' else 1.0, key=f"compA_{feat}")
                    
        with c_std2:
            st.markdown("<div class='section-title' style='color:#a855f7;'>👤 Data Mahasiswa B</div>", unsafe_allow_html=True)
            for feat in available:
                meta = FEATURE_META.get(feat, {})
                col_min = float(df_raw[feat].min())
                col_max = float(df_raw[feat].max())
                col_mean = float(df_raw[feat].mean())
                lbl = f"{meta.get('icon','🔹')} {meta.get('label', feat)} B"
                if col_min == col_max:
                    input_vals_b[feat] = st.number_input(lbl, value=col_min, disabled=True, key=f"compB_{feat}")
                else:
                    input_vals_b[feat] = st.slider(lbl, min_value=col_min, max_value=col_max, value=round(col_mean, 2), step=0.1 if feat == 'previous_gpa' else 1.0, key=f"compB_{feat}")

    st.markdown("---")
    
    if st.button("🔍 Prediksi Kelompok", type="primary", use_container_width=True):
        placeholder_predict = st.empty()
        with placeholder_predict:
            academic_loading_screen("Menghitung Kedekatan Vektor & Memeriksa Syarat Kelayakan Akademik...", "PROSES KLASIFIKASI & DETEKSI DINI AI")
            
            all_data = df_raw[available].fillna(df_raw[available].mean())
            
            f_att = w_attendance / 100.0
            f_ex = w_exams / 100.0
            f_qz = w_quizzes / 100.0
            f_gp = w_gpa / 100.0
            
            scaled_all_df = pd.DataFrame(StandardScaler().fit_transform(all_data), columns=available)
            for col in available:
                if 'attended' in col or 'submitted' in col: scaled_all_df[col] *= f_att
                elif 'marks' in col and ('midterm' in col or 'final' in col): scaled_all_df[col] *= f_ex
                elif 'quiz' in col: scaled_all_df[col] *= f_qz
                elif 'gpa' in col: scaled_all_df[col] *= f_gp
                
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            km.fit(scaled_all_df.values)

            tmp = all_data.copy()
            tmp['_cluster'] = km.labels_
            rank = tmp.groupby('_cluster')[available].mean().sum(axis=1).sort_values(ascending=False)
            rank_map = {rank.index[0]: "Tinggi", rank.index[1]: "Menengah", rank.index[2]: "Berisiko"}

            input_df = pd.DataFrame([input_vals])
            scaler_global = StandardScaler().fit(all_data)
            scaled_input = scaler_global.transform(input_df)
            scaled_input_df = pd.DataFrame(scaled_input, columns=available)
            for col in available:
                if 'attended' in col or 'submitted' in col: scaled_input_df[col] *= f_att
                elif 'marks' in col and ('midterm' in col or 'final' in col): scaled_input_df[col] *= f_ex
                elif 'quiz' in col: scaled_input_df[col] *= f_qz
                elif 'gpa' in col: scaled_input_df[col] *= f_gp

            pred = km.predict(scaled_input_df.values)[0]
            kategori = rank_map[pred]
            
            hit_blacklist_a = False
            if is_override_enabled and (override_feature in input_vals):
                if input_vals[override_feature] <= max_violation_limit:
                    hit_blacklist_a = True
            
            if hit_blacklist_a: kategori = "Override_Berisiko"
            m = CLUSTER_META.get(kategori, CLUSTER_META["Berisiko"])
            
            if is_compare_mode:
                input_df_b = pd.DataFrame([input_vals_b])
                scaled_input_b = scaler_global.transform(input_df_b)
                scaled_input_df_b = pd.DataFrame(scaled_input_b, columns=available)
                for col in available:
                    if 'attended' in col or 'submitted' in col: scaled_input_df_b[col] *= f_att
                    elif 'marks' in col and ('midterm' in col or 'final' in col): scaled_input_df_b[col] *= f_ex
                    elif 'quiz' in col: scaled_input_df_b[col] *= f_qz
                    elif 'gpa' in col: scaled_input_df_b[col] *= f_gp
                    
                pred_b = km.predict(scaled_input_df_b.values)[0]
                kategori_b = rank_map[pred_b]
                
                hit_blacklist_b = False
                if is_override_enabled and (override_feature in input_vals_b):
                    if input_vals_b[override_feature] <= max_violation_limit:
                        hit_blacklist_b = True
                
                if hit_blacklist_b: kategori_b = "Override_Berisiko"
                m_b = CLUSTER_META.get(kategori_b, CLUSTER_META["Berisiko"])
                
            time.sleep(2.0)
        placeholder_predict.empty()

        # ─────────────────────────────────────────────
        # SUB-FUNGSI: SMART IMPROVEMENT ADVICE
        # ─────────────────────────────────────────────
        def get_improvement_advice(current_cat, input_features_vals, df_clustered_data, available_cols):
            if "Override" in current_cat:
                advice_html = f"<div style='margin-top:12px; padding:12px; background:rgba(239,68,68,0.1); border-radius:8px; border-left:4px solid #ef4444;'>"
                advice_html += f"<div style='font-size:0.85rem; font-weight:700; color:#f87171; margin-bottom:4px;'>❌ Pemulihan Status Kunci:</div>"
                advice_html += f"<p style='margin:0; font-size:0.8rem; color:var(--text-secondary);'>Sistem mendeteksi nilai parameter berada di rentang kritis bawah (0 - {int(max_violation_limit)}). Naikkan parameter kelayakan absensi/tugas Anda terlebih dahulu.</p></div>"
                return advice_html, "STATUS: Blacklist Aturan Wajib Kampus.\n"
                
            rank_order = ["Berisiko", "Menengah", "Tinggi"]
            try:
                curr_idx = rank_order.index(current_cat)
            except ValueError: return None, ""
            
            if curr_idx >= 2: return None, "Mahasiswa sudah berada di kelompok capaian tertinggi."
            
            target_cat = rank_order[curr_idx + 1]
            target_means = df_clustered_data[df_clustered_data['Kategori'] == target_cat][available_cols].mean()
            
            current_vals = pd.Series(input_features_vals)
            diff = target_means - current_vals
            improvements = diff.nlargest(3)
            
            advice_html = f"<div style='margin-top:12px; padding:12px; background:var(--bg-primary); border-radius:8px; border-left:4px solid var(--accent);'>"
            advice_html += f"<div style='font-size:0.85rem; font-weight:700; color:var(--text-primary); margin-bottom:6px;'>🚀 Target Strategis Naik ke Klaster {target_cat}:</div>"
            advice_html += "<ul style='margin:0; padding-left:18px; font-size:0.8rem; color:var(--text-secondary);'>"
            
            text_report = f"Target Strategis Naik ke Klaster {target_cat}:\n"
            for feat, gap in improvements.items():
                if gap > 0:
                    lbl = FEATURE_META.get(feat, {}).get('label', feat)
                    advice_html += f"<li>Tambahkan nilai <strong>{lbl}</strong> minimal <strong>+{gap:.1f} poin</strong> lagi.</li>"
                    text_report += f"- Tambahkan nilai {lbl} minimal +{gap:.1f} poin lagi.\n"
            
            advice_html += "</ul></div>"
            return advice_html, text_report

        # Kotak Hasil Prediksi Klaster Utama + Saran Peningkatan
        advice_text_a = ""
        advice_text_b = ""
        if not is_compare_mode:
            improvement, advice_text_a = get_improvement_advice(kategori, input_vals, df_trained, available)
            st.markdown(f"""
            <div class="result-box" style="border-color:{m['border']}; background:{m['bg']};">
                <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:{m['color']};">
                    {m['label']}
                </div>
                <p style="margin:6px 0 0 0; font-size:0.92rem; color: var(--text-primary);">
                    Mahasiswa ini masuk ke kelompok <strong style="color:{m['color']};">{kategori.replace('Override_','')}</strong>
                </p>
                <div class="rec-block">
                    💡 <strong>Rekomendasi Utama:</strong> {m['saran']}
                </div>
                {improvement if improvement else ""}
            </div>
            """, unsafe_allow_html=True)
        else:
            improvement_a, advice_text_a = get_improvement_advice(kategori, input_vals, df_trained, available)
            improvement_b, advice_text_b = get_improvement_advice(kategori_b, input_vals_b, df_trained, available)
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.markdown(f"""
                <div class="result-box" style="border-color:{m['border']}; background:{m['bg']};">
                    <div style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700; color:{m['color']};">👤 Mahasiswa A: {m['label']}</div>
                    <div class="rec-block">💡 <strong>Rekomendasi:</strong> {m['saran']}</div>
                    {improvement_a if improvement_a else ""}
                </div>""", unsafe_allow_html=True)
            with c_res2:
                st.markdown(f"""
                <div class="result-box" style="border-color:{m_b['border']}; background:{m_b['bg']};">
                    <div style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700; color:{m_b['color']};">👤 Mahasiswa B: {m_b['label']}</div>
                    <div class="rec-block">💡 <strong>Rekomendasi:</strong> {m_b['saran']}</div>
                    {improvement_b if improvement_b else ""}
                </div>""", unsafe_allow_html=True)

        # DOWNLOAD LAPORAN HASIL PREDIKSI (.TXT) 
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        
        report_data = "========================================================\n"
        report_data += "      LAPORAN HASIL PREDIKSI AKADEMIK CLUSTER AI        \n"
        report_data += "========================================================\n\n"
        report_data += f"Konfigurasi Bobot Akademik: Absen={w_attendance}% | Ujian={w_exams}% | Kuis={w_quizzes}% | IPK={w_gpa}%\n"
        if is_override_enabled:
            report_data += f"Aturan Blacklist: AKTIF (Maks Batas Kritis Fitur [{override_feature}] <= {max_violation_limit})\n\n"
        else:
            report_data += "Aturan Blacklist: NON-AKTIF\n\n"
            
        if not is_compare_mode:
            report_data += f"Hasil Prediksi: {kategori}\n"
            report_data += f"Rekomendasi: {m['saran']}\n\n"
            report_data += f"{advice_text_a}\n"
            report_data += "Data Input Atribut:\n"
            for f in available: report_data += f"- {f}: {input_vals[f]}\n"
        else:
            report_data += f"Hasil Prediksi Mahasiswa A: {kategori}\n"
            report_data += f"Saran A: {m['saran']}\n"
            report_data += f"{advice_text_a}\n"
            report_data += "--------------------------------------------------------\n"
            report_data += f"Hasil Prediksi Mahasiswa B: {kategori_b}\n"
            report_data += f"Saran B: {m_b['saran']}\n"
            report_data += f"{advice_text_b}\n\n"
            report_data += "Data Input Mahasiswa A & B:\n"
            for f in available: report_data += f"- {f}: A={input_vals[f]} | B={input_vals_b[f]}\n"
            
        st.download_button(
            label="📥 Unduh Rangkuman Laporan Hasil Prediksi (.txt)",
            data=report_data,
            file_name=f"Laporan_Prediksi_Akademik_{int(time.time())}.txt",
            mime="text/plain"
        )

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        
        # Tampilan Radar Chart Komparasi
        cluster_means = df_trained.groupby('Kategori')[available].mean()
        categories = [fmt(f) for f in available]
        PT = get_plotly_theme()
        fig_radar = go.Figure()
        
        for cat, row in cluster_means.iterrows():
            if "Override" in cat: continue 
            cm = CLUSTER_META[cat]
            fig_radar.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=categories + [categories[0]],
                name=CLUSTER_META[cat]['label'],
                fill='toself', opacity=0.06,
                line=dict(color=cm['color'], width=1.5)
            ))
            
        user_vals = [input_vals[f] for f in available]
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]], theta=categories + [categories[0]],
            name="📌 Vektor Mahasiswa A" if is_compare_mode else "📌 Vektor Mahasiswa",
            fill='none', line=dict(color="#38bdf8", width=4)
        ))
        
        if is_compare_mode:
            user_vals_b = [input_vals_b[f] for f in available]
            fig_radar.add_trace(go.Scatterpolar(
                r=user_vals_b + [user_vals_b[0]], theta=categories + [categories[0]],
                name="📌 Vektor Mahasiswa B", fill='none', line=dict(color="#a855f7", width=4)
            ))
        
        is_dark_local = st.get_option("theme.base") == "dark" or st.get_option("theme.backgroundColor") != "#ffffff"
        polar_grid = "#21262d" if is_dark_local else "#ced4da"
        
        fig_radar.update_layout(**PT, height=500,
            polar=dict(bgcolor=PT['plot_bgcolor'], radialaxis=dict(visible=True, gridcolor=polar_grid), angularaxis=dict(gridcolor=polar_grid)),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            title="Profil Geometris Kedekatan Titik Vektor Terhadap Centroid Klaster")
        st.plotly_chart(fig_radar, use_container_width=True)

        # BLOK INTERPRETASI HASIL PREDIKSI 
        dim_length = len(available)
        with st.container():
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            if not is_compare_mode:
                st.markdown(f"### 📌 Hasil Analisis Titik Koordinat Mahasiswa:")
                if "Override" in kategori:
                    st.markdown(f"1. **Pencegahan Aturan Kampus:** Vektor input berada dalam zona kritis bawah (0 - {int(max_violation_limit)}). Walaupun koordinat matematika model K-Means melempar data ke klaster atas, status di-override secara mutlak menjadi **Berisiko/Blacklist**.")
                else:
                    st.markdown(f"1. **Pemetaan Spasial:** Secara geometris dalam ruang dimensi $R^{dim_length}$, model K-Means mendapat jarak terkecil (*Euclidean Distance*) dari data input condong mengarah ke area centroid kelompok **{kategori}**.")
                st.markdown(f"2. **Karakteristik Dominan:** Berdasarkan bentuk jaring *Radar Chart*, profil mahasiswa ini memiliki kecocokan pola batas atas/bawah pada atribut nilai ujian dan absensi yang menyerupai rata-rata perilaku kelompok **{kategori.replace('Override_','')}**.")
                st.markdown(f"3. **Evaluasi Academic:** Mahasiswa disarankan mengikuti rencana aksi taktis pada peta jalan peningkatan (*improvement advice*) untuk meretas gap antar kelompok klaster.")
            else:
                st.markdown(f"### 👥 Analisis Komparatif Lintas Objek (Siswa A vs Siswa B):")
                st.markdown(f"1. **Dilema Centroid:** Mahasiswa A berhasil diklasifikasikan ke dalam kelompok **{kategori.replace('Override_','')}**, sedangkan Mahasiswa B dialokasikan pada kelompok **{kategori_b.replace('Override_','')}**. Hal ini membuktikan adanya varians koordinat yang signifikan di antara keduanya.")
                st.markdown("2. **Divergensi Jaring Kembar:**")
                st.markdown(f"* Garis <span style='color:#38bdf8; font-weight:bold;'>Biru</span> memperlihatkan pola sebaran Mahasiswa A.", unsafe_allow_html=True)
                st.markdown(f"* Garis <span style='color:#a855f7; font-weight:bold;'>Ungu</span> memperlihatkan pola sebaran Mahasiswa B yang berbeda, mengonfirmasi mengapa sistem memisahkan mereka ke dalam klaster yang berlainan.", unsafe_allow_html=True)
                st.markdown(f"3. **Kesimpulan Statistika:** Meskipun beberapa nilai komponen slider mungkin terlihat mirip, perbedaan sekecil apa pun pada fitur utama setelah melalui tahap *StandardScaler* akan memperlebar jarak euklidian antar objek di ruang multidimensi.")
            st.markdown('</div>', unsafe_allow_html=True)


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
        ("labs_attended",         "Jumlah sesi praktikum yang dihadiri",            "Integer"),
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

        is_dark_global = st.get_option("theme.base") == "dark" or st.get_option("theme.backgroundColor") != "#ffffff"
        colorscale = "RdBu_r" if not is_dark_global else [[0,"#1a5fa8"],[0.5,"#21262d"],[1,"#a83220"]]
        
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
