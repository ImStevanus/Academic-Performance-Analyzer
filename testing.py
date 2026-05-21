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
        
        # --- PROSES DYNAMIC CLUSTERING OTOMATIS ---
        data_to_cluster = df[available_features].fillna(df[available_features].mean())
        scaler_dynamic = StandardScaler()
        scaled_dynamic = scaler_dynamic.fit_transform(data_to_cluster)
        
        kmeans_dynamic = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster_Dynamic'] = kmeans_dynamic.fit_predict(scaled_dynamic)

        # Sorting cluster berdasarkan performa
        cluster_scores = df.groupby('Cluster_Dynamic')[available_features].mean().sum(axis=1).sort_values(ascending=False)
        rank_map = {cluster_scores.index[0]: "Tinggi", cluster_scores.index[1]: "Menengah", cluster_scores.index[2]: "Berisiko"}
        df['Kategori'] = df['Cluster_Dynamic'].map(rank_map)
        df['Label'] = df['Kategori'].apply(lambda x: get_info(x)['label'])

        # --- DROP-DOWN SELEKSI SUMBU GRAFIK ---
        st.subheader("🎛️ Pengaturan Sumbu Visualisasi")
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

        # --- PANEL RINGKASAN METRIK ---
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Mahasiswa", len(df))
        m2.metric("Sumbu X Terpilih", x_axis.split('_')[0].title())
        m3.metric("Sumbu Y Terpilih", y_axis.split('_')[0].title())

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
            # Menggunakan key dinamis agar scatter plot refresh otomatis
            st.plotly_chart(fig, use_container_width=True, key=f"scatter_{x_axis}_{y_axis}")

        with col_right:
            st.markdown("##### 🥧 Proporsi Kelompok Mahasiswa (Live Update)")
            
            # 🔥 TRIK UTAMA: Isolasi total rangkuman data ke DataFrame baru yang berdiri sendiri
            pie_data = df['Label'].value_counts().reset_index()
            pie_data.columns = ['Kelompok', 'Jumlah']
            
            fig_pie = px.pie(pie_data, names="Kelompok", values="Jumlah", hole=0.4, template="none",
                             color="Kelompok", color_discrete_map={
                                 get_info("Tinggi")['label']: "#2ecc71",
                                 get_info("Menengah")['label']: "#f1c40f",
                                 get_info("Berisiko")['label']: "#e74c3c"
                             })
            
            # 🔥 KUNCI DINAMIS: Setiap kali sumbu diubah, key pie chart ikut berganti nama.
            # Ini memaksa Streamlit menghancurkan gambar lama dan merender ulang proporsinya.
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{x_axis}_{y_axis}")

        st.subheader("📋 Data Detail Lengkap")
        st.dataframe(df[available_features + ['Label']], use_container_width=True)
        
    else:
        st.info("Silakan unggah dataset CSV untuk memulai analisis dinamis.")

else:
    st.title("🔍 Prediksi Individu")
    if model_static is None:
        st.error("Model .pkl tidak ditemukan untuk prediksi individu.")
    else:
        with st.form("input_form"):
            c1, c2 = st.columns(2)
            with c1:
                q1 = st.slider("Quiz 1", 0.0, 10.0, 0.0); q2 = st.slider("Quiz 2", 0.0, 10.0, 0.0); q3 = st.slider("Quiz 3", 0.0, 10.0, 0.0)
                mid = st.number_input("Midterm Marks", 0, 30, 0)
            with c2:
                fin = st.number_input("Final Marks", 0, 50, 0)
                gpa = st.slider("GPA", 0.0, 4.0, 0.0)
                lec = st.number_input("Lectures", 0, 12, 0); lab = st.number_input("Labs", 0, 6, 0)
            submit = st.form_submit_button("Prediksi")

        if submit:
            if mid == 0 and fin == 0:
                res_cat = "Berisiko"
            else:
                input_data = pd.DataFrame([[q1,q2,q3,mid,fin,gpa,lec,lab]], 
                                          columns=['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended'])
                scaled = scaler_static.transform(input_data)
                cluster_id = model_static.predict(scaled)[0]
                res_cat = "Berisiko" if cluster_id == 2 else ("Menengah" if cluster_id == 1 else "Tinggi")
            
            info = get_info(res_cat)
            st.markdown(f"""
                <div style="background-color:{info['color']}; padding:30px; border-radius:15px; text-align:center; color:white;">
                    <h1 style="color:white; margin:0;">{info['label']}</h1>
                    <p style="font-size:1.2em; margin-top:10px;">{info['saran']}</p>
                </div>
            """, unsafe_allow_html=True)

st.divider()
st.caption("UAS Pemrograman AI - Stevanus - Built with Fixed Dynamic Key Render Engine")
