import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Dynamic Academic Analyzer | Stevanus",
    page_icon="🎓",
    layout="wide"
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1); 
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px; border-radius: 12px; color: inherit;
    }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# 3. FUNGSI LOAD MODEL STATIC (Untuk Prediksi Individu)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model_static, scaler_static = load_assets()

# 4. FUNGSI INFORMASI KATEGORI
def get_info(cat):
    info = {
        "Tinggi": {"label": "🚀 High Achiever", "color": "#2ecc71", "saran": "Pertahankan! Cocok jadi asisten dosen."},
        "Menengah": {"label": "📊 Steady / Average", "color": "#f1c40f", "saran": "Tingkatkan konsistensi nilai ujian."},
        "Berisiko": {"label": "⚠️ Underperformer", "color": "#e74c3c", "saran": "Butuh bimbingan akademik intensif."}
    }
    return info.get(cat, {"label": "Unknown", "color": "#7f8c8d", "saran": "-"})

# 5. FUNGSI FORMAT NAMA PARAMETER BER-EMOJI
def format_parameter_name(col_name):
    mapping = {
        "quiz1_marks": "📝 Nilai Quiz 1",
        "quiz2_marks": "📝 Nilai Quiz 2",
        "quiz3_marks": "📝 Nilai Quiz 3",
        "midterm_marks": "📉 Nilai Midterm Exam",
        "final_marks": "📊 Nilai Final Exam",
        "previous_gpa": "📈 IPK Sebelumnya (GPA)",
        "lectures_attended": "📅 Kehadiran Kuliah (Lectures)",
        "labs_attended": "🧪 Kehadiran Praktikum (Labs)"
    }
    return mapping.get(col_name, f"🔍 {col_name.replace('_', ' ').title()}")

# 6. SIDEBAR
with st.sidebar:
    st.title("🎓 Smart Campus AI")
    st.write(f"User: **Stevanus**")
    st.divider()
    menu = st.radio("Navigasi Utama", ["🏠 Dashboard Analisis Dinamis", "🔍 Prediksi Individu"])
    st.divider()
    uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")

# 7. LOGIKA UTAMA
if menu == "🏠 Dashboard Analisis Dinamis":
    st.title("📊 Dashboard Analisis Dinamis")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        
        all_features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']
        available_features = [f for f in all_features if f in df.columns]
        
        # --- DROP-DOWN SELEKSI SUMBU VISUALISASI ---
        st.subheader("🎛️ Pengaturan Sumbu Visualisasi")
        st.write("Silakan pilih parameter Sumbu X dan Sumbu Y. Sistem akan langsung mengelompokkan data (Clustering) berdasarkan 2 sumbu terpilih ini.")
        
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            x_axis = st.selectbox(
                "Sumbu X Grafik:",
                options=available_features,
                index=available_features.index('midterm_marks') if 'midterm_marks' in available_features else 0,
                format_func=format_parameter_name,
                key="sb_x_ultimate"
            )
        with col_select2:
            y_axis = st.selectbox(
                "Sumbu Y Grafik:",
                options=available_features,
                index=available_features.index('final_marks') if 'final_marks' in available_features else 1,
                format_func=format_parameter_name,
                key="sb_y_ultimate"
            )

        # Mencegah error jika dosen memilih kolom X dan Y yang sama persis
        if x_axis == y_axis:
            st.error("⚠️ Sumbu X dan Sumbu Y tidak boleh memilih parameter yang sama. Silakan bedakan pilihannya.")
        else:
            # --- 🔥 PERUBAHAN KRUSIAL: CLUSTERING HANYA BERDASARKAN SUMBU X & Y TERPILIH 🔥 ---
            selected_pair = [x_axis, y_axis]
            data_to_cluster = df[selected_pair].fillna(df[selected_pair].mean())
            
            scaler_dynamic = StandardScaler()
            scaled_dynamic = scaler_dynamic.fit_transform(data_to_cluster)
            
            kmeans_dynamic = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Cluster_Dynamic'] = kmeans_dynamic.fit_predict(scaled_dynamic)

            # Sorting cluster berdasarkan akumulasi nilai dari 2 sumbu aktif agar warnanya konsisten (Hijau = Tertinggi, Merah = Terendah)
            cluster_scores = df.groupby('Cluster_Dynamic')[selected_pair].mean().sum(axis=1).sort_values(ascending=False)
            rank_map = {cluster_scores.index[0]: "Tinggi", cluster_scores.index[1]: "Menengah", cluster_scores.index[2]: "Berisiko"}
            df['Kategori'] = df['Cluster_Dynamic'].map(rank_map)
            df['Label'] = df['Kategori'].apply(lambda x: get_info(x)['label'])

            # --- PANEL RINGKASAN METRIK ---
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Mahasiswa", len(df))
            m2.metric("Sumbu X (Clustering)", x_axis.split('_')[0].title())
            m3.metric("Sumbu Y (Clustering)", y_axis.split('_')[0].title())

            # --- PANEL VISUALISASI ---
            col_left, col_right = st.columns([6, 4])
            
            with col_left:
                st.markdown("##### 📍 Sebaran Kelompok Berdasarkan Sumbu Pilihan")
                fig = px.scatter(df, x=x_axis, y=y_axis, color="Label", 
                                 template="none", 
                                 labels={x_axis: format_parameter_name(x_axis), y_axis: format_parameter_name(y_axis)},
                                 color_discrete_map={
                                     get_info("Tinggi")['label']: "#2ecc71",
                                     get_info("Menengah")['label']: "#f1c40f",
                                     get_info("Berisiko")['label']: "#e74c3c"
                                 })
                st.plotly_chart(fig, use_container_width=True, key=f"scatter_{x_axis}_{y_axis}")

            with col_right:
                st.markdown("##### 🥧 Proporsi Kelompok Mahasiswa (Live Update)")
                
                # Menghitung ulang rangkuman persentase khusus dari data cluster Sumbu X & Y di atas
                pie_data = df['Label'].value_counts().reset_index()
                pie_data.columns = ['Kelompok', 'Jumlah']
                
                fig_pie = px.pie(pie_data, names="Kelompok
