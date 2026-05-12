import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Segmentasi Produk App", layout="wide")

st.title("Laporan Proyek: Segmentasi Produk (Clustering)")
st.write("Aplikasi ini mengelompokkan produk berdasarkan kategori dan tag menggunakan algoritma K-Means.")

# --- LOAD DATA ---
@st.cache_data # Cache agar data tidak di-load ulang setiap interaksi
def load_data():
    df = pd.read_csv('data_set.csv')
    # Preprocessing: Isi nilai kosong
    tag_cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    df[tag_cols] = df[tag_cols].fillna('')
    # Gabungkan fitur
    df['combined_features'] = df[tag_cols].apply(lambda x: ' '.join(x), axis=1)
    return df

try:
    df = load_data()
    
    # --- SIDEBAR: KONTROL MODEL ---
    st.sidebar.header("Pengaturan Model")
    k_value = st.sidebar.slider("Pilih Jumlah Cluster (k)", min_value=2, max_value=20, value=10)
    
    # --- PROSES CLUSTERING ---
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)
    
    # --- LAYOUT UTAMA ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Visualisasi Cluster (PCA)")
        pca = PCA(n_components=2)
        coords = pca.fit_transform(tfidf_matrix.toarray())
        df['pca_1'] = coords[:, 0]
        df['pca_2'] = coords[:, 1]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='pca_1', y='pca_2', hue='Cluster', data=df, palette='viridis', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Filter Data berdasarkan Cluster")
        selected_cluster = st.selectbox("Pilih Cluster untuk ditampilkan:", sorted(df['Cluster'].unique()))
        filtered_df = df[df['Cluster'] == selected_cluster]
        st.write(f"Menampilkan {len(filtered_df)} produk dalam Cluster {selected_cluster}")
        st.dataframe(filtered_df[['Product Name', 'Category', 'tag1', 'tag2', 'tag3']], use_container_width=True)

    # --- TABEL DATA LENGKAP ---
    st.divider()
    st.subheader("Seluruh Data Tersegmentasi")
    st.dataframe(df.drop(columns=['combined_features', 'pca_1', 'pca_2']), use_container_width=True)

    # --- DOWNLOAD HASIL ---
    csv = df.drop(columns=['combined_features', 'pca_1', 'pca_2']).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data Segmentasi (CSV)",
        data=csv,
        file_name='segmented_products_output.csv',
        mime='text/csv',
    )

except FileNotFoundError:
    st.error("File 'data_set.csv' tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
