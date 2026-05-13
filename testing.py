import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Dashboard Analisis Mahasiswa - Stevanus", layout="wide")

# 2. FUNGSI LOAD MODEL & DATA
@st.cache_resource
def load_assets():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ Model/Scaler tidak ditemukan. Pastikan sudah menjalankan training di Colab.")

# 3. SIDEBAR & NAVIGASI
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu:", ["Analisis Dataset", "Prediksi Mahasiswa Baru"])

# 4. MENU 1: ANALISIS DATASET
if menu == "Analisis Dataset":
    st.title("📊 Analisis Data Mahasiswa")
    
    uploaded_file = st.sidebar.file_uploader("Unggah Dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Preprocessing sederhana untuk visualisasi
        features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']
        df_clean = df[features].fillna(df[features].mean())
        
        # Jalankan Clustering pada dataset yang diunggah
        scaled_data = scaler.transform(df_clean)
        df['Cluster'] = model.predict(scaled_data)
        
        # Statistik Deskriptif
        st.subheader("Ringkasan Statistik per Cluster")
        st.write(df.groupby('Cluster')[features].mean())
        
        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Sebaran Cluster (Midterm vs Final)")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='midterm_marks', y='final_marks', hue='Cluster', palette='viridis', ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.write("### Distribusi IPK per Cluster")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Cluster', y='previous_gpa', palette='magma', ax=ax)
            st.pyplot(fig)
            
        st.write("### Data Mahasiswa dengan Label Cluster")
        st.dataframe(df)
    else:
        st.info("Silakan unggah file 'student_dropout_behavior_dataset.csv' di sidebar untuk memulai analisis.")

# 5. MENU 2: PREDIKSI MAHASISWA BARU
else:
    st.title("🔍 Prediksi Cluster Mahasiswa")
    st.write("Masukkan data mahasiswa secara manual untuk mengetahui kategori performanya.")
    
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        q1 = c1.number_input("Quiz 1", 0.0, 10.0, 7.0)
        q2 = c2.number_input("Quiz 2", 0.0, 10.0, 7.0)
        q3 = c3.number_input("Quiz 3", 0.0, 10.0, 7.0)
        
        mid = c1.number_input("Midterm", 0.0, 100.0, 75.0)
        fin = c2.number_input("Final Exam", 0.0, 100.0, 75.0)
        gpa = c3.number_input("Prev GPA", 0.0, 4.0, 3.0)
        
        lec = c1.number_input("Lectures Attended", 0, 12, 10)
        lab = c2.number_input("Labs Attended", 0, 6, 5)
        
        submitted = st.form_submit_button("Analisis Performa")
        
    if submitted:
        input_data = pd.DataFrame([[q1, q2, q3, mid, fin, gpa, lec, lab]], columns=features)
        scaled_input = scaler.transform(input_data)
        res = model.predict(scaled_input)[0]
        
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"] # Hijau, Kuning, Merah
        st.markdown(f"""
            <div style="background-color:{colors[res]}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white;">Mahasiswa Terdeteksi di Cluster {res}</h2>
            </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.write("Oleh: **Stevanus**")
