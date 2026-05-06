import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman
st.set_page_config(page_title="Shopper Segment Analyzer", layout="wide")

st.title("🛍️ Online Shoppers Segment Analyzer")
st.write("Sistem Cerdas untuk mengelompokkan perilaku pelanggan secara otomatis.")

# 1. DATA INPUT
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV data pengunjung", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.write(df.head())

    # Pilih fitur untuk clustering (Misal: Revenue, Administrative, ExitRates)
    features = st.multiselect("Pilih fitur untuk analisis:", df.columns, default=df.columns[:3])

    if len(features) >= 2:
        # 2. MODEL AI & PROSES PREDIKSI
        X = df[features]
        
        # Scaling data agar model bekerja maksimal
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        num_clusters = st.sidebar.slider("Jumlah Segmen (Cluster):", 2, 5, 3)
        
        model = KMeans(n_clusters=num_clusters, random_state=42)
        df['Segment'] = model.fit_predict(X_scaled)

        # 3. HASIL ANALISIS
        st.subheader("Hasil Segmentasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Data dengan Label Cluster:")
            st.write(df.head())
        
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Segment'], palette='viridis', ax=ax)
            plt.title("Visualisasi Segmen Pelanggan")
            st.pyplot(fig)
            
        st.success(f"Analisis Selesai! Ditemukan {num_clusters} kelompok pelanggan.")
else:
    st.info("Silakan unggah file CSV di sidebar untuk memulai.")
