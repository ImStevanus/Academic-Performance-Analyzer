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

# Initialize session state for welcome screen
if 'started' not in st.session_state:
    st.session_state.started = False

# ─────────────────────────────────────────────
# 2. GLOBAL CSS — Dark academic + Welcome Page UI/UX
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Remove Streamlit default padding */
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 6px 12px; border-radius: 8px; transition: background .2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #161b22; }

/* ─── Page header band ─── */
.page-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.page-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e2e8f0, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 4px 0;
}
.page-header p {
    color: #8b949e;
    font-size: 0.9rem;
    margin: 0;
}

/* ─── Metric cards ─── */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color .25s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-card .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #8b949e;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 4px;
}
.metric-card .accent-bar {
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 14px 0 0 14px;
}

/* ─── Section titles ─── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ─── Cluster badge pills ─── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-high   { background: rgba(34,197,94,.15);  color: #4ade80; border: 1px solid rgba(34,197,94,.3); }
.badge-mid    { background: rgba(234,179,8,.15);  color: #facc15; border: 1px solid rgba(234,179,8,.3); }
.badge-risk   { background: rgba(239,68,68,.15);  color: #f87171; border: 1px solid rgba(239,68,68,.3); }

/* ─── Info boxes ─── */
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.info-box h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0 0 8px 0;
}

/* ─── Selectboxes & sliders ─── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] > div {
    border-radius: 10px !important;
}

/* ─── Upload area ─── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* ─── Tab bar ─── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: .04em;
}

/* ─── Divider ─── */
hr { border-color: #21262d !important; }

/* ─── DataFrame ─── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ─── Plotly chart background patch ─── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ─── Alert/warning ─── */
.stAlert { border-radius: 10px !important; }

/* ─── Welcome Page Custom CSS ─── */
.welcome-container {
    text-align: center;
    max-width: 800px;
    margin: 6% auto 2% auto;
    padding: 40px;
    background: radial-gradient(circle at top, rgba(56,189,248,0.05) 0%, transparent 80%);
}
.welcome-badge {
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    padding: 6px 16px;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    display: inline-block;
    margin-bottom: 24px;
    border: 1px solid rgba(56, 189, 248, 0.2);
}
.welcome-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}
.welcome-subtitle {
    color: #8b949e;
    font-size: 1.15rem;
    line-height: 1.6;
    margin-bottom: 40px;
    font-weight: 300;
}
.welcome-footer {
    margin-top: 80px;
    font-size: 0.82rem;
    color: #484f58;
    border-top: 1px solid #21262d;
    padding-top: 20px;
}
/* Style button center welcome screen */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 14px 36px !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 26px rgba(56, 189, 248, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# WELCOME INTERACTIVE SCREEN (UPDATED UI/UX)
# ─────────────────────────────────────────────
if not st.session_state.started:
    # Sembunyikan sidebar di halaman sambutan agar fokus ke tengah
    st.markdown("""<style>[data-testid="stSidebar"] { display: none !important; }</style>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="welcome-container" style="margin-top: 3%;">
        <div class="welcome-badge">📊 PROYEK UAS PEMROGRAMAN AI</div>
        <div class="welcome-title">Academic Performance<br>Cluster Analyzer</div>
        <div class="welcome-subtitle">
            Sistem cerdas berbasis Machine Learning menggunakan algoritma <b>K-Means Clustering</b> 
            untuk segmentasi data akademik, pemetaan profil, dan deteksi dini risiko performa mahasiswa secara presisi.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 🌟 TAMBAHAN 1: Quick Stats Box (Bento Grid Style)
    st.markdown("""
    <div style="max-width: 850px; margin: 0 auto 40px auto;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
            <div style="background: #161b22; border: 1px solid #21262d; padding: 16px; border-radius: 12px; text-align: center;">
                <div style="font-size: 1.5rem;">📁</div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em;">Dataset Batasan</div>
                <div style="font-weight: 700; color: #e2e8f0; margin-top: 2px;">300 Mahasiswa / 16 Fitur</div>
            </div>
            <div style="background: #161b22; border: 1px solid #21262d; padding: 16px; border-radius: 12px; text-align: center;">
                <div style="font-size: 1.5rem;">🤖</div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em;">Algoritma Inti</div>
                <div style="font-weight: 700; color: #38bdf8; margin-top: 2px;">K-Means Clustering</div>
            </div>
            <div style="background: #161b22; border: 1px solid #21262d; padding: 16px; border-radius: 12px; text-align: center;">
                <div style="font-size: 1.5rem;">🎯</div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em;">Target Output</div>
                <div style="font-weight: 700; color: #4ade80; margin-top: 2px;">3 Klaster Akademik</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tombol Mulai di Tengah Layar
    c_left, c_mid, c_right = st.columns([2, 1.2, 2])
    with c_mid:
        if st.button("Mulai Analisis Sistem 🚀", use_container_width=True):
            st.session_state.started = True
            st.rerun()
            
    # 🌟 TAMBAHAN 2: Workflow singkat di bawah tombol
    st.markdown("""
    <div style="max-width: 600px; margin: 40px auto 0 auto; text-align: center;">
        <p style="font-size: 0.82rem; color: #8b949e; margin-bottom: 12px; letter-spacing: 0.05em; text-transform: uppercase;">Alur Kerja Aplikasi</p>
        <div style="display: flex; justify-content: center; align-items: center; gap: 12px; color: #484f58; font-size: 0.88rem;">
            <span style="color: #c9d1d9;">1. Unggah CSV</span>
            <span>➔</span>
            <span style="color: #c9d1d9;">2. Analisis Klaster Eksploratif</span>
            <span>➔</span>
            <span style="color: #c9d1d9;">3. Prediksi Real-Time</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
            
    st.markdown(f"""
    <div class="welcome-container" style="margin-top: 0px; padding-top: 20px;">
        <div class="welcome-footer">
            Dibuat oleh: <b>Stevanus</b> &nbsp;·&nbsp; NIM: <b>38250029</b> &nbsp;·&nbsp; Prodi: <b>Artificial Intelligence</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# 3. CONSTANTS & HELPERS
# ─────────────────────────────────────────────

FEATURE_META = {
    "quiz1_marks":          {"label": "Quiz 1",             "icon": "📝", "unit": "pts"},
    "quiz2_marks":          {"label": "Quiz 2",             "icon": "📝", "unit": "pts"},
    "quiz3_marks":          {"label": "Quiz 3",             "icon": "📝", "unit": "pts"},
    "midterm_marks":        {"label": "Midterm Exam",        "icon": "📉", "unit": "pts"},
    "final_marks":          {"label": "Final Exam",          "icon": "📊", "unit": "pts"},
    "previous_gpa":         {"label": "IPK Sebelumnya",      "icon": "📈", "unit": "GPA"},
    "lectures_attended":    {"label": "Kehadiran Kuliah",    "icon": "📅", "unit": "sesi"},
    "labs_attended":        {"label": "Kehadiran Praktikum", "icon": "🧪", "unit": "sesi"},
    "assignments_submitted":{"label": "Tugas Dikumpulkan",   "icon": "📂", "unit": "tugas"},
}

CLUSTER_META = {
    "Tinggi":   {"label": "🚀 High Achiever",   "color": "#4ade80", "bg": "rgba(34,197,94,.08)",  "border": "rgba(34,197,94,.3)",  "badge": "badge-high",
                 "saran": "Pertahankan konsistensi! Cocok menjadi asisten dosen atau mentor teman sejawat."},
    "Menengah": {"label": "📊 Steady Performer", "color": "#facc15", "bg": "rgba(234,179,8,.08)",  "border": "rgba(234,179,8,.3)",  "badge": "badge-mid",
                 "saran": "Tingkatkan konsistensi nilai ujian dan kehadiran agar naik ke kelompok atas."},
    "Berisiko": {"label": "⚠️ Needs Support",    "color": "#f87171", "bg": "rgba(239,68,68,.08)",  "border": "rgba(239,68,68,.3)",  "badge": "badge-risk",
                 "saran": "Perlu bimbingan akademik intensif. Hubungi dosen PA sesegera mungkin."},
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(family="DM Sans", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=20, r=20, t=40, b=20),
)

def fmt(col):
    m = FEATURE_META.get(col, {})
    return f"{m.get('icon','🔹')} {m.get('label', col.replace('_',' ').title())}"

def cluster_card(cat, count, pct):
    m = CLUSTER_META[cat]
    return f"""
    <div class="metric-card" style="border-color:{m['border']}; background:{m['bg']};">
        <div class="accent-bar" style="background:{m['color']};"></div>
        <div class="label">{m['label']}</div>
        <div class="value" style="color:{m['color']};">{count}</div>
        <div class="sub">{pct:.1f}% dari total mahasiswa</div>
    </div>"""


# ─────────────────────────────────────────────
# 4. SIDEBAR (Hanya muncul jika sudah klik Mulai)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 20px 0;'>
      <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                  background:linear-gradient(90deg,#e2e8f0,#38bdf8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🎓 Academic<br>Cluster AI
      </div>
      <div style='font-size:0.78rem;color:#8b949e;margin-top:4px;'>
        Stevanus · NIM 38250029<br>Prodi Artificial Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    menu = st.radio(
        "Navigasi",
        ["📊 Dashboard Analisis", "🔍 Prediksi Individu", "📁 Info Dataset"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("<div style='font-size:0.78rem;color:#8b949e;margin-bottom:8px;'>UPLOAD DATASET</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if uploaded_file:
        st.success("✅ Dataset berhasil dimuat")
        
    st.divider()
    if st.button("↩️ Halaman Utama", use_container_width=True):
        st.session_state.started = False
        st.rerun()


# ─────────────────────────────────────────────
# 5. LOAD + PREPROCESS DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file_obj):
    df = pd.read_csv(file_obj)
    df.columns = df.columns.str.strip().str.lower()
    if 'assignments_submitted' in df.columns:
        df['assignments_submitted'] = df['assignments_submitted'].fillna(df.get('total_assignments', pd.Series()).median())
    return df

@st.cache_data
def run_clustering(df_json, features):
    df = pd.read_json(df_json)
    data = df[features].fillna(df[features].mean())
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


# ─────────────────────────────────────────────
# 6. MAIN FEATURE LIST
# ─────────────────────────────────────────────
ALL_FEATURES = list(FEATURE_META.keys())


# ─────────────────────────────────────────────
# 7. PAGE: DASHBOARD
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

    df_raw = load_and_preprocess(uploaded_file)
    available = [f for f in ALL_FEATURES if f in df_raw.columns]

    st.markdown("<div class='section-title'>🎛️ Konfigurasi Sumbu Visualisasi & Clustering</div>", unsafe_allow_html=True)
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
        st.error("⚠️ Sumbu X dan Y tidak boleh sama. Pilih parameter berbeda.")
        st.stop()

    cluster_features = [x_axis, y_axis]
    df_clustered, sil_score, _, _ = run_clustering(df_raw.to_json(), cluster_features)

    st.markdown("---")
    total = len(df_clustered)
    counts = df_clustered['Kategori'].value_counts()

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class='metric-card'>
            <div class='accent-bar' style='background:#38bdf8;'></div>
            <div class='label'>Total Mahasiswa</div>
            <div class='value'>{total}</div>
            <div class='sub'>dataset: {uploaded_file.name}</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(cluster_card("Tinggi",   counts.get("Tinggi",0),   counts.get("Tinggi",0)/total*100), unsafe_allow_html=True)
    with mc3:
        st.markdown(cluster_card("Menengah", counts.get("Menengah",0), counts.get("Menengah",0)/total*100), unsafe_allow_html=True)
    with mc4:
        st.markdown(cluster_card("Berisiko", counts.get("Berisiko",0), counts.get("Berisiko",0)/total*100), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:6px;font-size:0.8rem;color:#8b949e;'>Silhouette Score (kualitas cluster): <b style='color:#38bdf8;'>{:.4f}</b> — Nilai mendekati 1.0 menandakan cluster yang sangat baik.</div>".format(sil_score), unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📍 Scatter Plot", "🥧 Distribusi Kelompok (Live Update)", "📊 Statistik per Kelompok"])

    color_map = {CLUSTER_META[k]['label']: CLUSTER_META[k]['color'] for k in CLUSTER_META}
    df_clustered['Label'] = df_clustered['Kategori'].apply(lambda x: CLUSTER_META[x]['label'])

    with tab1:
        fig_scatter = px.scatter(
            df_clustered, x=x_axis, y=y_axis,
            color="Label",
            color_discrete_map=color_map,
            hover_data=["name"] if "name" in df_clustered.columns else None,
            labels={x_axis: fmt(x_axis), y_axis: fmt(y_axis)},
            title=f"Sebaran Mahasiswa: {fmt(x_axis)} vs {fmt(y_axis)}",
            template="none",
            opacity=0.82,
        )
        fig_scatter.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#0d1117")))
        fig_scatter.update_layout(**PLOTLY_THEME, height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
        st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{x_axis}_{y_axis}")

    with tab2:
        col_pie, col_bar = st.columns(2)
        with col_pie:
            pie_df = df_clustered['Label'].value_counts().reset_index()
            pie_df.columns = ['Kelompok', 'Jumlah']
            fig_pie = px.pie(pie_df, names="Kelompok", values="Jumlah",
                color="Kelompok", color_discrete_map=color_map,
                hole=0.52, template="none")
            fig_pie.update_traces(textfont_size=13, marker=dict(line=dict(color="#0d1117", width=2)))
            fig_pie.update_layout(**PLOTLY_THEME, height=360,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{x_axis}_{y_axis}")

        with col_bar:
            bar_df = df_clustered.groupby('Kategori')[available].mean().reset_index()
            bar_melt = bar_df.melt(id_vars='Kategori', var_name='Fitur', value_name='Rata-rata')
            bar_melt['Fitur'] = bar_melt['Fitur'].apply(fmt)
            bar_melt['Label'] = bar_melt['Kategori'].apply(lambda x: CLUSTER_META[x]['label'])
            fig_bar = px.bar(bar_melt, x='Fitur', y='Rata-rata', color='Label',
                barmode='group', template="none",
                color_discrete_map=color_map,
                title="Rata-rata Fitur per Kelompok")
            fig_bar.update_layout(**PLOTLY_THEME, height=360,
                xaxis_tickangle=-35,
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{x_axis}_{y_axis}")

    with tab3:
        stat_df = df_clustered.groupby('Kategori')[available].agg(['mean','min','max']).round(2)
        stat_df.columns = [f"{col[1].upper()} {fmt(col[0])}" for col in stat_df.columns]
        st.dataframe(stat_df, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>🗂️ Tabel Data Mahasiswa & Hasil Cluster</div>", unsafe_allow_html=True)
    show_cols = (['name'] if 'name' in df_clustered.columns else []) + ['Kategori', 'Label'] + available
    show_cols = [c for c in show_cols if c in df_clustered.columns]
    filter_cat = st.selectbox("Filter Kelompok", ["Semua", "Tinggi", "Menengah", "Berisiko"], key="filter_table")
    tbl = df_clustered[show_cols] if filter_cat == "Semua" else df_clustered[df_clustered['Kategori'] == filter_cat][show_cols]
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=300)


# ─────────────────────────────────────────────
# 8. PAGE: PREDIKSI INDIVIDU
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

    df_trained, _, _, scaler_fit = run_clustering(df_raw.to_json(), available)

    st.markdown("<div class='section-title'>✍️ Input Data Mahasiswa</div>", unsafe_allow_html=True)

    cols_left, cols_right = st.columns(2)
    input_vals = {}
    for i, feat in enumerate(available):
        meta = FEATURE_META.get(feat, {})
        col_min = float(df_raw[feat].min())
        col_max = float(df_raw[feat].max())
        col_mean = float(df_raw[feat].mean())
        target_col = cols_left if i % 2 == 0 else cols_right
        
        with target_col:
            if col_min == col_max:
                input_vals[feat] = st.number_input(
                    f"{meta.get('icon','🔹')} {meta.get('label', feat)} ({meta.get('unit','')})",
                    value=col_min,
                    disabled=True,
                    help="Fitur ini bernilai konstan/seragam di seluruh baris dataset.",
                    key=f"input_{feat}"
                )
            else:
                input_vals[feat] = st.slider(
                    f"{meta.get('icon','🔹')} {meta.get('label', feat)} ({meta.get('unit','')})",
                    min_value=col_min, max_value=col_max, value=round(col_mean, 2),
                    step=0.1 if feat == 'previous_gpa' else 1.0,
                    key=f"input_{feat}"
                )

    st.markdown("---")
    if st.button("🔍 Prediksi Kelompok", type="primary", use_container_width=True):
        input_arr = np.array([[input_vals[f] for f in available]])
        scaled_input = scaler_fit.transform(input_arr)

        all_features_data = df_raw[available].fillna(df_raw[available].mean())
        scaler2 = StandardScaler()
        scaled_all = scaler2.fit_transform(all_features_data)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(scaled_all)

        scores = pd.DataFrame(all_features_data)
        scores['_cluster'] = km.labels_
        rank = scores.groupby('_cluster')[available].mean().sum(axis=1).sort_values(ascending=False)
        rank_map = {rank.index[0]: "Tinggi", rank.index[1]: "Menengah", rank.index[2]: "Berisiko"}

        scaled_new = scaler2.transform(input_arr)
        pred_cluster = km.predict(scaled_new)[0]
        kategori = rank_map[pred_cluster]
        meta = CLUSTER_META[kategori]

        st.markdown(f"""
        <div class='info-box' style='border-color:{meta["border"]};background:{meta["bg"]};margin-top:16px;'>
            <h4 style='color:{meta["color"]};font-size:1.3rem;'>{meta["label"]}</h4>
            <p style='color:#c9d1d9;margin:4px 0 10px 0;font-size:0.92rem;'>
                Mahasiswa ini masuk ke kelompok <b style='color:{meta["color"]};'>{kategori}</b>
            </p>
            <div style='background:rgba(0,0,0,.2);border-radius:8px;padding:12px 16px;font-size:0.88rem;color:#c9d1d9;'>
                💡 <b>Rekomendasi:</b> {meta["saran"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        cluster_means = df_trained.groupby('Kategori')[available].mean()
        categories = [fmt(f) for f in available]
        fig_radar = go.Figure()
        for cat, row in cluster_means.iterrows():
            cm = CLUSTER_META[cat]
            fig_radar.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=categories + [categories[0]],
                name=cm['label'], fill='toself', opacity=0.25,
                line=dict(color=cm['color'], width=2)
            ))
        user_vals = [input_vals[f] for f in available]
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=categories + [categories[0]],
            name="📌 Data Anda", fill='toself', opacity=0.55,
            line=dict(color="#38bdf8", width=3, dash="dot")
        ))
        fig_radar.update_layout(
            **PLOTLY_THEME,
            polar=dict(bgcolor="rgba(22,27,34,0.6)",
                       radialaxis=dict(visible=True, gridcolor="#21262d"),
                       angularaxis=dict(gridcolor="#21262d")),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=420, title="Profil Nilai vs Rata-rata Kelompok"
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────
# 9. PAGE: INFO DATASET
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
          <h4 style='color:#38bdf8;'>📂 Identitas Dataset</h4>
          <table style='width:100%;font-size:0.87rem;color:#c9d1d9;border-collapse:collapse;'>
            <tr><td style='padding:4px 0;color:#8b949e;width:40%;'>Nama File</td><td>student_dropout_behavior_dataset.csv</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Sumber</td><td>Kaggle — Muhammad Khubaib Ahmad</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Jumlah Data</td><td>300 baris mahasiswa</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Jumlah Kolom</td><td>16 atribut</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Tipe Tugas</td><td>Clustering / Segmentasi (Unsupervised)</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Algoritma</td><td>K-Means (k=3, StandardScaler)</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class='info-box'>
          <h4 style='color:#38bdf8;'>🎓 Identitas Proyek</h4>
          <table style='width:100%;font-size:0.87rem;color:#c9d1d9;border-collapse:collapse;'>
            <tr><td style='padding:4px 0;color:#8b949e;width:40%;'>Nama</td><td>Stevanus</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>NIM</td><td>38250029</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Program Studi</td><td>Artificial Intelligence</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Mata Kuliah</td><td>Pemrograman AI</td></tr>
            <tr><td style='padding:4px 0;color:#8b949e;'>Jenis Ujian</td><td>UAS (Ujian Akhir Semester)</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:20px;'>📋 Deskripsi Atribut Dataset</div>", unsafe_allow_html=True)
    attr_data = [
        ("student_id",           "ID unik setiap mahasiswa",                    "Integer"),
        ("name",                 "Nama mahasiswa",                              "String"),
        ("age",                  "Usia mahasiswa (18–25 tahun)",                "Integer"),
        ("gender",               "Jenis kelamin mahasiswa",                     "String"),
        ("quiz1_marks",          "Nilai Quiz 1",                                "Float"),
        ("quiz2_marks",          "Nilai Quiz 2",                                "Float"),
        ("quiz3_marks",          "Nilai Quiz 3",                                "Float"),
        ("total_assignments",    "Total tugas yang diberikan",                  "Integer"),
        ("assignments_submitted","Jumlah tugas yang dikumpulkan (ada missing)", "Float"),
        ("midterm_marks",        "Nilai ujian tengah semester",                 "Float"),
        ("final_marks",          "Nilai ujian akhir semester",                  "Float"),
        ("previous_gpa",         "IPK semester sebelumnya",                     "Float"),
        ("total_lectures",       "Total sesi kuliah yang dijadwalkan",          "Integer"),
        ("lectures_attended",    "Jumlah sesi kuliah yang dihadiri",            "Integer"),
        ("total_lab_sessions",   "Total sesi praktikum yang dijadwalkan",       "Integer"),
        ("labs_attended",        "Jumlah sesi praktikum yang dihadiri",         "Integer"),
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
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="Blues", zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text}",
            textfont={"size": 11}, showscale=True
        ))
        fig_heat.update_layout(**PLOTLY_THEME, height=460, title="Korelasi Fitur Numerik")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.success("Hasil heatmap menunjukkan korelasi antar fitur yang sangat rendah, mengindikasikan setiap fitur bersifat independen. Ini justru menguntungkan K-Means karena clustering tidak bias terhadap satu fitur dominan.")
