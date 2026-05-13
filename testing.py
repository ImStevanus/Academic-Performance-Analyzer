import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Academic Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD MODEL & SCALER
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# 3. SIDEBAR NAVIGATION
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)
    st.title("Proyek UAS AI")
    st.info(f"Nama: **Stevanus**\n\nTema: Academic Clustering")
    menu = st.radio("Menu Utama", ["🏠 Beranda & Analisis", "🔍 Prediksi Individu"])
    st.divider()
    uploaded_file = st.file_uploader("Upload Data Mahasiswa (CSV)", type="csv")

# 4. LOGIK UTAMA
if model is None:
    st.error("⚠️ File model (kmeans_model.pkl) atau scaler tidak ditemukan! Pastikan sudah menjalankan training di Colab.")
else:
    if menu == "🏠 Beranda & Analisis":
        st.title("📊 Dashboard Analisis Performa Akademik")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']
            
            # Preprocessing & Clustering
            df_clean = df[features].fillna(df[features].mean())
            scaled_data = scaler.transform(df_clean)
            df['Cluster'] = model.predict(scaled_data)
            
            # FITUR BARU: METRIKS RINGKASAN
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Total Mahasiswa", len(df))
            col_m2.metric("Rata-rata Final", f"{df['final_marks'].mean():.1f}")
            col_m3.metric("Rata-rata IPK", f"{df['previous_gpa'].mean():.2f}")
            col_m4.metric("Jumlah Cluster", len(df['Cluster'].unique()))

            st.divider()

            # VISUALISASI INTERAKTIF DENGAN PLOTLY
            c1, c2 = st.columns([6, 4])
            
            with c1:
                st.subheader("📍 Sebaran Klaster (Midterm vs Final)")
                fig_scatter = px.scatter(df, x="midterm_marks", y="final_marks", color="Cluster",
                                         hover_data=['name'] if 'name' in df.columns else None,
                                         template="plotly_white", color_continuous_scale="Viridis")
                st.plotly_chart(fig_scatter, use_container_width=True)

            with c2:
                st.subheader("🥧 Komposisi Mahasiswa")
                fig_pie = px.pie(df, names="Cluster", hole=0.4, template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)

            # FITUR BARU: TABEL DATA DENGAN FILTER
            st.subheader("📋 Daftar Mahasiswa Berdasarkan Klaster")
            selected_cluster = st.multiselect("Filter Klaster:", options=sorted(df['Cluster'].unique()), default=df['Cluster'].unique())
            filtered_df = df[df['Cluster'].isin(selected_cluster)]
            st.dataframe(filtered_df, use_container_width=True)
            
        else:
            st.warning("Silakan unggah file 'student_dropout_behavior_dataset.csv' untuk melihat dashboard.")
            st.image("https://i.imgur.com/8M2Y9T8.png", caption="Contoh Dashboard Analisis")

    elif menu == "🔍 Prediksi Individu":
        st.title("🔍 Prediksi Kategori Mahasiswa")
        st.write("Gunakan form di bawah untuk menganalisis kategori performa mahasiswa secara instan.")
        
        with st.container():
            col_a, col_b = st.columns(2)
            with col_a:
                q1 = st.slider("Nilai Quiz 1", 0.0, 10.0, 7.5)
                q2 = st.slider("Nilai Quiz 2", 0.0, 10.0, 7.5)
                mid = st.number_input("Nilai Midterm", 0.0, 100.0, 70.0)
                lec = st.number_input("Jam Kehadiran Kuliah", 0, 12, 10)
            with col_b:
                q3 = st.slider("Nilai Quiz 3", 0.0, 10.0, 7.5)
                gpa = st.slider("IPK Sebelumnya", 0.0, 4.0, 3.2)
                fin = st.number_input("Nilai Final Exam", 0.0, 100.0, 75.0)
                lab = st.number_input("Jam Kehadiran Lab", 0, 6, 5)

            if st.button("🚀 Jalankan Analisis"):
                input_raw = pd.DataFrame([[q1, q2, q3, mid, fin, gpa, lec, lab]], 
                                       columns=['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended'])
                
                scaled_input = scaler.transform(input_raw)
                cluster_res = model.predict(scaled_input)[0]
                
                # Desain Card Hasil
                st.divider()
                if cluster_res == 0:
                    st.success("### 🌟 Hasil: CLUSTER 0 - PERFORMA TINGGI")
                    st.write("Mahasiswa menunjukkan dedikasi tinggi dengan nilai stabil dan kehadiran maksimal.")
                elif cluster_res == 1:
                    st.warning("### 📊 Hasil: CLUSTER 1 - PERFORMA MENENGAH")
                    st.write("Mahasiswa memiliki potensi namun perlu meningkatkan konsistensi pada tugas/quiz.")
                else:
                    st.error("### ⚠️ Hasil: CLUSTER 2 - PERFORMA BERISIKO")
                    st.write("Mahasiswa memerlukan bimbingan khusus karena nilai atau kehadiran di bawah rata-rata.")

# 5. FOOTER
st.sidebar.markdown("---")
st.sidebar.caption("UAS Pemrograman AI © 2024 - Stevanus")
