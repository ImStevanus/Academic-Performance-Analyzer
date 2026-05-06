import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Shopper Segment AI", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .stMetric { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI BACKEND ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def preprocess_data(df):
    df_clean = df.copy()
    le = LabelEncoder()
    # Mengonversi kolom kategori agar bisa dibaca model AI jika diperlukan
    for col in ['Month', 'VisitorType', 'Weekend', 'Revenue']:
        df_clean[col] = le.fit_transform(df_clean[col])
    return df_clean

# --- 3. SIDEBAR ---
st.sidebar.title("🎮 Menu Kontrol")
st.sidebar.markdown("Unggah dataset dan atur parameter model di sini.")
uploaded_file = st.sidebar.file_uploader("Upload 'online_shoppers_intention.csv'", type=["csv"])

# --- 4. MAIN CONTENT ---
st.title("🛍️ Online Shoppers Segment Analyzer")
st.markdown("""
Sistem ini menggunakan **Artificial Intelligence (Unsupervised Learning)** untuk memahami pola perilaku pengunjung website e-commerce secara otomatis.
""")

if uploaded_file is not None:
    # Load Data
    data = load_data(uploaded_file)
    df_numeric = preprocess_data(data)

    # Tabs untuk navigasi
    tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data", "🤖 Training Model AI", "💡 Insight Strategis"])

    # --- TAB 1: EKSPLORASI DATA ---
    with tab1:
        st.subheader("Ringkasan Statistik")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sampel", f"{data.shape[0]}")
        col2.metric("Rata-rata PageValue", f"{data['PageValues'].mean():.2f}")
        col3.metric("Bounce Rate Rata-rata", f"{data['BounceRates'].mean():.3f}")
        col4.metric("Konversi (Revenue)", f"{(data['Revenue'].sum()/len(data)*100):.1f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Distribusi Niat Beli (Revenue)**")
            fig_pie = px.pie(data, names='Revenue', hole=0.4, color_discrete_sequence=['#ef553b', '#636efa'])
            st.plotly_chart(fig_pie, use_container_width=True)
            st.info("**Penjelasan:** Sebagian besar pengunjung (False) tidak melakukan transaksi. AI akan membantu mencari segmen kecil yang berpotensi beli (True).")

        with c2:
            st.write("**Bounce Rates vs Exit Rates**")
            fig_scatter = px.scatter(data, x='BounceRates', y='ExitRates', color='Revenue', opacity=0.4)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("**Penjelasan:** Area kanan atas menunjukkan 'zona kebocoran' di mana pengunjung cepat meninggalkan situs tanpa interaksi.")

    # --- TAB 2: TRAINING MODEL AI ---
    with tab2:
        st.subheader("Konfigurasi Mesin K-Means")
        
        # Fitur yang digunakan untuk Clustering
        features = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
        selected = st.multiselect("Pilih Fitur Analisis:", features, default=['ProductRelated_Duration', 'BounceRates', 'PageValues'])

        if len(selected) >= 2:
            # Preprocessing & Scaling
            X = df_numeric[selected]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method (Visualisasi Optimasi)
            distortions = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(X_scaled)
                distortions.append(km.inertia_)
            
            st.write("**Optimasi Jumlah Cluster (Elbow Method)**")
            fig_elbow = px.line(x=list(range(1, 11)), y=distortions, markers=True)
            fig_elbow.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Inertia (Error)")
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption("Pilih jumlah cluster di mana grafik mulai melandai (membentuk 'siku').")

            # Input K
            k_val = st.slider("Tentukan Jumlah Kelompok (Cluster):", 2, 6, 3)
            
            # Model Fit
            model = KMeans(n_clusters=k_val, n_init=10, random_state=42)
            data['Cluster'] = model.fit_predict(X_scaled)

            st.success(f"AI berhasil mengelompokkan pengunjung menjadi {k_val} segmen!")
            
            # Plot Hasil Cluster
            st.write("**Visualisasi Hasil Segmentasi**")
            fig_res = px.scatter(data, x=selected[0], y=selected[1], color='Cluster', 
                               hover_data=['VisitorType', 'Revenue'], symbol='Revenue')
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.warning("Silakan pilih minimal 2 fitur untuk memulai proses AI.")

    # --- TAB 3: INSIGHT STRATEGIS ---
    with tab3:
        if 'Cluster' in data.columns:
            st.subheader("Interpretasi Segmen & Strategi Bisnis")
            
            # Tabel Rata-rata per Cluster
            stats = data.groupby('Cluster')[selected].mean()
            st.write("**Karakteristik Tiap Kelompok:**")
            st.dataframe(stats.style.highlight_max(axis=0, color='#d4edda'))

            # Narasi Cerdas Otomatis
            st.markdown("### 💡 Rekomendasi Marketing")
            cols = st.columns(k_val)
            
            for i in range(k_val):
                with cols[i]:
                    st.markdown(f"#### Segmen {i}")
                    c_data = stats.iloc[i]
                    
                    # Logika Penentuan Profil Berdasarkan Data
                    if c_data['PageValues'] > data['PageValues'].mean() * 1.2:
                        st.success("💎 **VIP Buyer**")
                        st.write("Pengunjung ini sangat berharga. Berikan promo 'Free Shipping' untuk memastikan checkout.")
                    elif c_data['BounceRates'] > data['BounceRates'].mean():
                        st.error("🚪 **Lost Traffic**")
                        st.write("Pengunjung yang langsung pergi. Perlu optimasi kecepatan loading atau desain landing page.")
                    elif c_data['ProductRelated_Duration'] > data['ProductRelated_Duration'].mean():
                        st.warning("🕵️ **Researcher**")
                        st.write("Mereka lama melihat produk tapi ragu. Tampilkan ulasan produk (review) atau testimoni.")
                    else:
                        st.info("🚶 **Casual Visitor**")
                        st.write("Pengunjung umum. Pertahankan dengan konten menarik agar mereka kembali lagi.")
        else:
            st.info("Selesaikan langkah di tab 'Training Model AI' untuk melihat hasil analisis.")

else:
    # Tampilan jika file belum diunggah
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80")
    st.warning("Menunggu dataset... Silakan unggah file CSV di sidebar untuk mengaktifkan sistem.")
