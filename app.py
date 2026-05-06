import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Shopper AI Analyzer", layout="wide")

# Gaya Visual Pro
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .explanation-box { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h3 { color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI BACKEND ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def preprocess_data(df):
    df_clean = df.copy()
    le = LabelEncoder()
    for col in ['Month', 'VisitorType', 'Weekend', 'Revenue']:
        df_clean[col] = le.fit_transform(df_clean[col])
    return df_clean

# --- 3. SIDEBAR ---
st.sidebar.title("🎮 Kontrol Sistem")
uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])

# --- 4. MAIN CONTENT ---
st.title("🛍️ Online Shoppers Segment Analyzer")
st.markdown("Sistem Cerdas Analisis Perilaku Belanja untuk Strategi Pemasaran Berbasis Data.")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    df_numeric = preprocess_data(data)

    tab1, tab2, tab3 = st.tabs(["📊 1. Eksplorasi Data", "🤖 2. Logika Mesin AI", "💡 3. Insight Strategis"])

    # --- TAB 1: EKSPLORASI & EDUKASI ---
    with tab1:
        st.subheader("📍 Memahami Variabel Input")
        st.markdown("""
        <div class='explanation-box'>
        <b>Apa yang kita analisis?</b><br>
        Sistem ini membandingkan interaksi pengunjung. Fitur utama yang dianalisis adalah:
        <ul>
            <li><b>PageValues:</b> Seberapa bernilai halaman yang dikunjungi terhadap potensi pembelian.</li>
            <li><b>Bounce Rates:</b> Persentase pengunjung yang 'Mental' (langsung keluar) dari situs.</li>
            <li><b>Product Duration:</b> Menunjukkan intensitas minat pengunjung terhadap katalog produk.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Perbandingan Konversi (Revenue)**")
            st.plotly_chart(px.pie(data, names='Revenue', hole=0.4), use_container_width=True)
        with c2:
            st.write("**Sebaran Page Value terhadap Exit Rates**")
            st.plotly_chart(px.scatter(data, x='PageValues', y='ExitRates', color='Revenue', opacity=0.5), use_container_width=True)

    # --- TAB 2: LOGIKA MESIN AI ---
    with tab2:
        st.subheader("⚙️ Bagaimana AI Mengelompokkan Data?")
        st.markdown("""
        <div class='explanation-box'>
        <b>Proses Logika K-Means:</b><br>
        1. <b>Scaling:</b> AI menyamakan skala data (misal: durasi ribuan detik vs bounce rate 0.01) agar adil.<br>
        2. <b>Elbow Method:</b> Mencari jumlah kelompok (K) terbaik agar tidak terlalu banyak atau sedikit.<br>
        3. <b>Iterasi:</b> Mesin menghitung jarak terdekat antar data untuk menentukan siapa yang memiliki 'sifat' serupa.
        </div>
        """, unsafe_allow_html=True)

        features = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
        selected = st.multiselect("Pilih Fitur yang Akan Dibandingkan AI:", features, default=['ProductRelated_Duration', 'BounceRates', 'PageValues'])

        if len(selected) >= 2:
            X = df_numeric[selected]
            X_scaled = StandardScaler().fit_transform(X)

            # Elbow Plot
            distortions = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in range(1, 11)]
            st.write("**Visualisasi Optimalisasi (Elbow Method)**")
            st.plotly_chart(px.line(x=list(range(1, 11)), y=distortions, markers=True), use_container_width=True)

            k_val = st.slider("Atur Jumlah Cluster (K):", 2, 6, 3)
            data['Cluster'] = KMeans(n_clusters=k_val, n_init=10, random_state=42).fit_predict(X_scaled)
            
            st.write("**Visualisasi Hasil Logika Mesin**")
            st.plotly_chart(px.scatter(data, x=selected[0], y=selected[1], color='Cluster', symbol='Revenue'), use_container_width=True)
        else:
            st.warning("Pilih minimal 2 fitur.")

    # --- TAB 3: INSIGHT STRATEGIS ---
    with tab3:
        if 'Cluster' in data.columns:
            st.subheader("💡 Apa Makna Angka Ini Bagi Bisnis?")
            st.markdown("""
            <div class='explanation-box'>
            <b>Hasil Perbandingan Antar Segmen:</b><br>
            Tabel di bawah membandingkan nilai rata-rata tiap kelompok. Kita mencari 
            <b>karakter dominan</b> (angka tertinggi) untuk menentukan strategi promosi yang berbeda tiap segmen.
            </div>
            """, unsafe_allow_html=True)

            stats = data.groupby('Cluster')[selected].mean()
            st.dataframe(stats.style.highlight_max(axis=0, color='#d4edda').highlight_min(axis=0, color='#f8d7da'), use_container_width=True)

            st.markdown("### 📊 Radar Perbandingan Kekuatan")
            fig_radar = go.Figure()
            for i in range(k_val):
                fig_radar.add_trace(go.Scatterpolar(r=stats.iloc[i].values, theta=selected, fill='toself', name=f'Segmen {i}'))
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("---")
            st.subheader("📝 Rekomendasi Tindakan")
            
            # Cari cluster terbaik & terburuk secara dinamis
            best_id = stats['PageValues'].idxmax()
            worst_id = stats['BounceRates'].idxmax()

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.success(f"**Segmen {best_id}: Potensi Cuan Tinggi**")
                st.write("Segmen ini memiliki Page Value tertinggi. Strategi: Berikan diskon waktu terbatas (Flash Sale) untuk mendorong penyelesaian transaksi segera.")
            with col_res2:
                st.error(f"**Segmen {worst_id}: Risiko Tinggi**")
                st.write("Segmen ini sering mental (Bounce Rate tinggi). Strategi: Perbaiki konten halaman landing atau tawarkan bantuan Live Chat otomatis.")
        else:
            st.info("Selesaikan langkah di Tab 2.")

else:
    st.info("Silakan unggah dataset di sidebar untuk memulai.")
