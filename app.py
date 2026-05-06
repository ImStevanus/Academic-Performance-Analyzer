import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Shopper Segment AI Pro", layout="wide")

# Custom CSS untuk gaya profesional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .feature-box { background-color: #e9ecef; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI BACKEND ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def preprocess_data(df):
    df_clean = df.copy()
    le = LabelEncoder()
    # Mengonversi kolom kategori untuk keperluan analisis internal
    for col in ['Month', 'VisitorType', 'Weekend', 'Revenue']:
        df_clean[col] = le.fit_transform(df_clean[col])
    return df_clean

# --- 3. SIDEBAR ---
st.sidebar.title("🎮 Panel Kontrol AI")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
st.sidebar.info("Gunakan dataset 'Online Shoppers Intention' untuk hasil terbaik.")

# --- 4. MAIN CONTENT ---
st.title("🛍️ Online Shoppers Segment Analyzer")
st.markdown("Sistem Cerdas berbasis *Machine Learning* untuk segmentasi perilaku pelanggan e-commerce.")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    df_numeric = preprocess_data(data)

    tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi & Edukasi", "🤖 Mesin Prediksi AI", "💡 Insight Perbandingan"])

    # --- TAB 1: EKSPLORASI & EDUKASI ---
    with tab1:
        st.subheader("🔍 Memahami Variabel yang Dibandingkan")
        st.markdown("""
        Sebelum melakukan segmentasi, sistem membandingkan perilaku pengunjung berdasarkan metrik utama. 
        Berikut adalah penjelasan fitur yang digunakan dalam perbandingan:
        """)
        
        col_ed1, col_ed2 = st.columns(2)
        with col_ed1:
            st.markdown("""
            <div class='feature-box'>
            <b>1. PageValues vs Revenue</b><br>
            Membandingkan nilai halaman terhadap hasil transaksi. <i>PageValues</i> adalah prediktor terkuat untuk mengetahui apakah seseorang akan membeli.
            </div>
            <div class='feature-box'>
            <b>2. Bounce Rates vs Exit Rates</b><br>
            Membandingkan seberapa cepat pengunjung meninggalkan situs. Ini membantu AI mendeteksi pengunjung yang 'hanya mampir'.
            </div>
            """, unsafe_allow_html=True)
        
        with col_ed2:
            st.markdown("""
            <div class='feature-box'>
            <b>3. Product Related Duration</b><br>
            Waktu yang dihabiskan di halaman produk. Semakin lama durasinya, semakin tinggi minat belanja mereka.
            </div>
            <div class='feature-box'>
            <b>4. Visitor Type</b><br>
            Membandingkan perilaku antara pengunjung baru dan pelanggan lama (Returning Visitor).
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        
        # Visualisasi Distribusi
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Distribusi Konversi Transaksi**")
            fig_pie = px.pie(data, names='Revenue', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.write("**Hubungan Page Value & Niat Beli**")
            fig_box = px.box(data, x='Revenue', y='PageValues', color='Revenue')
            st.plotly_chart(fig_box, use_container_width=True)

    # --- TAB 2: MESIN PREDIKSI AI ---
    with tab2:
        st.subheader("⚙️ Proses Training K-Means")
        
        # Fitur untuk dibandingan dalam Clustering
        features = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
        selected = st.multiselect("Pilih Fitur untuk Dibandingkan oleh AI:", features, default=['ProductRelated_Duration', 'BounceRates', 'PageValues'])

        if len(selected) >= 2:
            X = df_numeric[selected]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Optimasi K
            distortions = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(X_scaled)
                distortions.append(km.inertia_)
            
            st.write("**Grafik Elbow (Mencari K Optimal)**")
            st.plotly_chart(px.line(x=list(range(1, 11)), y=distortions, markers=True), use_container_width=True)

            k_val = st.slider("Tentukan Jumlah Kelompok:", 2, 6, 3)
            model = KMeans(n_clusters=k_val, n_init=10, random_state=42)
            data['Cluster'] = model.fit_predict(X_scaled)

            st.success("Analisis Cluster Selesai!")
            
            st.write("**Hasil Perbandingan Visual antar Cluster**")
            fig_cluster = px.scatter(data, x=selected[0], y=selected[1], color='Cluster', 
                                   hover_data=['VisitorType'], size='PageValues', opacity=0.6)
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("Pilih minimal 2 fitur.")

    # --- TAB 3: INSIGHT PERBANDINGAN ---
    with tab3:
        if 'Cluster' in data.columns:
            st.subheader("💡 Perbandingan Karakteristik Antar Segmen")
            
            # Analisis Statistik
            stats = data.groupby('Cluster')[selected].mean()
            
            st.write("**Tabel Perbandingan Rata-rata Skor:**")
            st.dataframe(stats.style.highlight_max(axis=0, color='#d4edda').highlight_min(axis=0, color='#f8d7da'), use_container_width=True)
            
            st.markdown("""
            **Cara Membaca Perbandingan:**
            - Kolom **Hijau**: Nilai paling unggul di kelompok tersebut.
            - Kolom **Merah**: Nilai terendah (titik lemah kelompok).
            """)

            st.divider()
            
            # Narasi Perbandingan Cerdas
            st.subheader("📝 Ringkasan Strategis")
            
            # Cari cluster terbaik & terburuk
            best_cluster = stats['PageValues'].idxmax()
            worst_cluster = stats['BounceRates'].idxmax()

            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.success(f"✅ **Pemenang (Cluster {best_cluster})**")
                st.write(f"Cluster ini memiliki perbandingan Page Value tertinggi ({stats.loc[best_cluster, 'PageValues']:.2f}). Mereka adalah target iklan utama.")
            
            with c_res2:
                st.error(f"⚠️ **Risiko (Cluster {worst_cluster})**")
                st.write(f"Cluster ini memiliki Bounce Rate tertinggi. Sistem membandingkan bahwa mereka tidak tertarik dengan UI/UX saat ini.")

            # Radar Chart untuk perbandingan visual (Advanced)
            st.write("**Visualisasi Radar (Perbandingan Kekuatan Segmen)**")
            fig_radar = go.Figure()
            for i in range(k_val):
                fig_radar.add_trace(go.Scatterpolar(
                    r=stats.iloc[i].values,
                    theta=selected,
                    fill='toself',
                    name=f'Segmen {i}'
                ))
            st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.info("Lakukan training di Tab 2 terlebih dahulu.")

else:
    st.image("https://images.unsplash.com/photo-1551288049-bbbda536339a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_container_width=True)
    st.info("Silakan upload dataset untuk memulai perbandingan cerdas.")
