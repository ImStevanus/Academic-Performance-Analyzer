import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Academic Performance Analyzer | Stevanus",
    page_icon="🎓",
    layout="wide"
)

# 2. CUSTOM CSS (SUPPORT DARK & LIGHT MODE)
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1); 
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px;
        border-radius: 12px;
        color: inherit;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .stSlider, .stNumberInput {
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. FUNGSI LOAD ASSETS
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

# Fungsi dinamis untuk menentukan arti cluster berdasarkan profil data
def get_dynamic_cluster_info(cluster_num, mapping_dict=None):
    # Mapping default jika dataset belum di-upload di menu utama
    if mapping_dict is None:
        mapping_dict = {0: "Tinggi", 1: "Menengah", 2: "Berisiko"}
        
    status = mapping_dict.get(cluster_num, "Menengah")
    
    if status == "Tinggi":
        return {
            "label": "🚀 High Achiever (Performa Tinggi)",
            "desc": "Mahasiswa dengan metrik akademis keseluruhan di atas rata-rata kelompok dan kehadiran sangat konsisten.",
            "saran": "Pertahankan ritme belajar. Sangat direkomendasikan menjadi asisten dosen atau mentor sebaya.",
            "color": "#2ecc71"
        }
    elif status == "Menengah":
        return {
            "label": "📊 Steady / Average (Performa Menengah)",
            "desc": "Mahasiswa menunjukkan performa stabil pada tingkat rata-rata, namun memiliki ruang untuk peningkatan pada aspek nilai ujian atau lab.",
            "saran": "Fokus meningkatkan nilai ujian utama dan tugas harian untuk mengamankan nilai akhir.",
            "color": "#f1c40f"
        }
    else:
        return {
            "label": "⚠️ Underperformer (Performa Berisiko)",
            "desc": "Nilai ujian cenderung rendah atau tingkat kehadiran di bawah batas minimum kelompok. Berisiko tinggi mengalami kendala kelulusan.",
            "saran": "Segera jadwalkan sesi bimbingan konseling akademik untuk pemulihan nilai.",
            "color": "#e74c3c"
        }

model, scaler = load_assets()

# 4. SIDEBAR
with st.sidebar:
    st.title("🎓 Smart Campus AI")
    st.markdown(f"User: **Stevanus**\n\nTema: **Academic Clustering**")
    st.divider()
    menu = st.radio("Navigasi Utama", ["🏠 Dashboard Analisis", "🔍 Prediksi Individu"])
    st.divider()
    uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")

# Initialize session state untuk menyimpan mapping cluster secara global
if 'cluster_mapping' not in st.session_state:
    st.session_state['cluster_mapping'] = {0: "Tinggi", 1: "Menengah", 2: "Berisiko"}

# 5. LOGIKA UTAMA
if model is None or scaler is None:
    st.error("❌ File model (.pkl) tidak ditemukan. Pastikan sudah menjalankan training di Colab!")
else:
    features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']

    # --- MENU 1: DASHBOARD ANALISIS ---
    if menu == "🏠 Dashboard Analisis":
        st.title("📊 Dashboard Analisis Performa Mahasiswa")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing & Clustering
            df_clean = df[features].fillna(df[features].mean())
            scaled_data = scaler.transform(df_clean)
            df['Cluster'] = model.predict(scaled_data)
            
            # FITUR KRITIKAL: Mapping Otomatis Berdasarkan Rata-Rata Nilri Riil (Final Marks + Midterm Marks)
            cluster_means = df.groupby('Cluster')[['final_marks', 'midterm_marks']].mean().sum(axis=1)
            sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
            
            # Buat mapping dinamis
            dynamic_mapping = {}
            if len(sorted_clusters) >= 3:
                dynamic_mapping[sorted_clusters[0]] = "Tinggi"
                dynamic_mapping[sorted_clusters[1]] = "Menengah"
                dynamic_mapping[sorted_clusters[2]] = "Berisiko"
            else:
                for idx, c_id in enumerate(sorted_clusters):
                    dynamic_mapping[c_id] = "Tinggi" if idx == 0 else "Berisiko"
                    
            st.session_state['cluster_mapping'] = dynamic_mapping
            
            # BAGIAN METRIK
            st.subheader("📌 Ringkasan Data Saat Ini")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Mahasiswa", len(df))
            m2.metric("Rata-rata Final", f"{df['final_marks'].mean():.1f}")
            m3.metric("Rata-rata IPK", f"{df['previous_gpa'].mean():.2f}")
            m4.metric("Kategori Cluster", len(df['Cluster'].unique()))
            
            st.divider()

            # VISUALISASI
            col_left, col_right = st.columns([6, 4])
            
            with col_left:
                st.subheader("📍 Sebaran Cluster (Midterm vs Final)")
                fig = px.scatter(df, x="midterm_marks", y="final_marks", color="Cluster", 
                                 template="none", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.subheader("🥧 Proporsi Kelompok")
                df['Cluster_Name'] = df['Cluster'].apply(lambda x: get_dynamic_cluster_info(x, st.session_state['cluster_mapping'])['label'])
                fig_pie = px.pie(df, names="Cluster_Name", hole=0.4, template="none")
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("📋 Eksplorasi Data Lengkap")
            cl_filter = st.multiselect("Filter Cluster:", options=sorted(df['Cluster'].unique()), default=df['Cluster'].unique())
            st.dataframe(df[df['Cluster'].isin(cl_filter)], use_container_width=True)
            
        else:
            st.info("Silakan unggah dataset 'student_dropout_behavior_dataset.csv' pada sidebar untuk mengaktifkan kalkulasi klaster dinamis.")

    # --- MENU 2: PREDIKSI INDIVIDU ---
    else:
        st.title("🔍 Prediksi Kategori Mahasiswa")
        st.write("Masukkan parameter akademis mahasiswa di bawah ini:")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                q1 = st.slider("Quiz 1", 0.0, 10.0, 7.5)
                q2 = st.slider("Quiz 2", 0.0, 10.0, 7.5)
                q3 = st.slider("Quiz 4", 0.0, 10.0, 7.5)
                mid = st.number_input("Midterm Marks (0-100)", 0, 100, 70)
            with c2:
                fin = st.number_input("Final Marks (0-100)", 0, 100, 75)
                gpa = st.slider("Previous GPA (0.0-4.0)", 0.0, 4.0, 3.2)
                lec = st.number_input("Lectures Attended", 0, 12, 10)
                lab = st.number_input("Labs Attended", 0, 6, 5)
            
            submit = st.form_submit_button("🚀 Analisis Performa")

        if submit:
            input_df = pd.DataFrame([[q1, q2, q3, mid, fin, gpa, lec, lab]], columns=features)
            scaled_input = scaler.transform(input_df)
            res = model.predict(scaled_input)[0]
            
            # Memanggil info klaster menggunakan mapping dinamis hasil kalkulasi data utama
            info = get_dynamic_cluster_info(res, st.session_state['cluster_mapping'])
            
            # Kartu Hasil yang Mencolok
            st.markdown(f"""
                <div style="background-color:{info['color']}; padding:30px; border-radius:15px; text-align:center; color:white;">
                    <h1 style="color:white; margin:0;">{info['label']}</h1>
                    <p style="font-size:1.2em; margin:15px 0;">{info['desc']}</p>
                    <div style="background-color:rgba(0,0,0,0.15); padding:15px; border-radius:10px;">
                        <b>💡 Rekomendasi:</b> {info['saran']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# 7. FOOTER
st.divider()
st.caption(f"UAS Pemrograman AI - Stevanus - {len(features)} Fitur Teranalisis")
