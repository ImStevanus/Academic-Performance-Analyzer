import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Academic Performance Analyzer | Stevanus",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS untuk UI yang lebih bersih
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. FUNGSI PEMBANTU
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

def get_cluster_info(cluster_num):
    info = {
        0: {
            "label": "🚀 High Achiever (Performa Tinggi)",
            "desc": "Mahasiswa dengan nilai kuis/ujian di atas rata-rata dan kehadiran sangat konsisten.",
            "saran": "Pertahankan ritme belajar. Cocok menjadi asisten dosen atau mentor sebaya.",
            "color": "#2ecc71" # Hijau
        },
        1: {
            "label": "📊 Steady / Average (Performa Menengah)",
            "desc": "Mahasiswa menunjukkan performa stabil namun memiliki potensi untuk ditingkatkan lagi.",
            "saran": "Fokus pada area yang fluktuatif (misal: nilai lab) untuk mendongkrak IPK.",
            "color": "#f1c40f" # Kuning
        },
        2: {
            "label": "⚠️ Underperformer / At-Risk (Performa Berisiko)",
            "desc": "Nilai cenderung rendah dan kehadiran di bawah 60%. Berisiko tinggi mengalami kegagalan studi.",
            "saran": "Perlu bimbingan akademik intensif dan pemantauan kehadiran berkala.",
            "color": "#e74c3c" # Merah
        }
    }
    return info.get(cluster_num, {"label": "Unknown", "desc": "-", "saran": "-", "color": "#7f8c8d"})

# 3. LOAD MODEL
model, scaler = load_assets()

# 4. SIDEBAR
with st.sidebar:
    st.title("🎓 Smart Campus AI")
    st.write(f"Oleh: **Stevanus**")
    st.divider()
    menu = st.radio("Navigasi Utama", ["🏠 Dashboard Analisis", "🔍 Prediksi Individu"])
    st.divider()
    uploaded_file = st.file_uploader("Upload 'student_dropout_behavior_dataset.csv'", type="csv")

# 5. LOGIKA UTAMA
if model is None or scaler is None:
    st.error("❌ Model atau Scaler (.pkl) tidak ditemukan. Jalankan file notebook di Colab terlebih dahulu!")
else:
    features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']

    # --- MENU 1: DASHBOARD ANALISIS ---
    if menu == "🏠 Dashboard Analisis":
        st.title("📊 Dashboard Analisis Performa Mahasiswa")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing & Prediction
            df_clean = df[features].fillna(df[features].mean())
            scaled_data = scaler.transform(df_clean)
            df['Cluster'] = model.predict(scaled_data)
            
            # Metrik Ringkasan
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Mahasiswa", len(df))
            c2.metric("Rata-rata Final", f"{df['final_marks'].mean():.1f}")
            c3.metric("Rata-rata IPK", f"{df['previous_gpa'].mean():.2f}")
            c4.metric("Kategori Cluster", len(df['Cluster'].unique()))
            
            st.divider()

            # Visualisasi
            col_left, col_right = st.columns([6, 4])
            
            with col_left:
                st.subheader("📍 Sebaran Cluster (Midterm vs Final)")
                fig = px.scatter(df, x="midterm_marks", y="final_marks", color="Cluster", 
                                 color_continuous_scale="Viridis", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.subheader("🥧 Distribusi Per Kategori")
                # Mengganti angka cluster dengan label teks untuk pie chart
                df['Cluster_Label'] = df['Cluster'].apply(lambda x: get_cluster_info(x)['label'])
                fig_pie = px.pie(df, names="Cluster_Label", hole=0.4, 
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Tabel Data dengan Filter
            st.subheader("📋 Eksplorasi Data Mahasiswa")
            cl_filter = st.multiselect("Filter Berdasarkan Cluster:", options=sorted(df['Cluster'].unique()), default=df['Cluster'].unique())
            st.dataframe(df[df['Cluster'].isin(cl_filter)], use_container_width=True)
            
        else:
            st.info("Silakan unggah dataset di sidebar untuk melihat analisis menyeluruh.")

    # --- MENU 2: PREDIKSI INDIVIDU ---
    else:
        st.title("🔍 Prediksi Kategori Mahasiswa Baru")
        st.write("Masukkan data di bawah untuk melihat interpretasi performa.")

        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                q1 = st.slider("Quiz 1", 0.0, 10.0, 7.0)
                q2 = st.slider("Quiz 2", 0.0, 10.0, 7.0)
                q3 = st.slider("Quiz 3", 0.0, 10.0, 7.0)
                mid = st.number_input("Midterm Marks", 0, 100, 75)
            with col2:
                fin = st.number_input("Final Marks", 0, 100, 75)
                gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0)
                lec = st.number_input("Lectures Attended", 0, 12, 10)
                lab = st.number_input("Labs Attended", 0, 6, 5)
            
            btn = st.form_submit_button("Analisis Performa")

        if btn:
            # Predict
            input_df = pd.DataFrame([[q1, q2, q3, mid, fin, gpa, lec, lab]], columns=features)
            scaled_input = scaler.transform(input_df)
            cluster_id = model.predict(scaled_input)[0]
            
            # Get Info
            info = get_cluster_info(cluster_id)
            
            # Display Card
            st.markdown(f"""
                <div style="background-color:{info['color']}; padding:30px; border-radius:15px; text-align:center;">
                    <h1 style="color:white; margin-bottom:0;">{info['label']}</h1>
                    <p style="color:white; font-size:1.2em; margin-top:10px;">{info['desc']}</p>
                    <div style="background-color:rgba(255,255,255,0.2); padding:15px; border-radius:10px; margin-top:20px;">
                        <p style="color:white; margin:0;"><b>💡 Rekomendasi:</b> {info['saran']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# 6. FOOTER
st.divider()
st.caption(f"© 2024 UAS Pemrograman AI - Stevanus (Cluster 15)")
