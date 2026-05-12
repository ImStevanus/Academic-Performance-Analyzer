import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Product Segmenter Pro",
    page_icon="📊",
    layout="wide"
)

# --- LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_process():
    # Membaca data
    df = pd.read_csv('data_set.csv')
    
    # Menghapus kolom duplikat jika ada di file CSV asli
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Daftar kolom fitur
    cols_to_combine = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    
    # Pastikan kolom-kolom di atas ada di dataframe
    existing_cols = [c for c in cols_to_combine if c in df.columns]
    
    # Isi nilai kosong dan gabungkan
    df[existing_cols] = df[existing_cols].fillna('')
    df['combined_features'] = df[existing_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    return df, existing_cols

try:
    df_raw, feature_cols = load_and_process()
    df = df_raw.copy()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Kontrol Panel")
        k_value = st.slider("Jumlah Cluster (k)", 2, 20, 10)
        st.divider()
        st.info("Tips: Jika cluster terlihat terlalu menumpuk, coba naikkan nilai K.")

    # --- ENGINE CLUSTERING ---
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)

    # --- HEADER ---
    st.title("📊 Product Segmentation Dashboard")

    # --- TABS UI ---
    tab1, tab2, tab3 = st.tabs(["📈 Analisis Cluster", "🔍 Pencarian & Filter", "📄 Data Master"])

    with tab1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Produk", len(df))
        m2.metric("Jumlah Cluster", k_value)
        m3.metric("Rata-rata Produk/Cluster", int(len(df)/k_value))

        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            st.subheader("Visualisasi Ruang Produk (PCA 2D)")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(tfidf_matrix.toarray())
            df['pca_1'] = coords[:, 0]
            df['pca_2'] = coords[:, 1]
            
            fig_pca = px.scatter(
                df, x='pca_1', y='pca_2', color='Cluster',
                hover_data=['Product Name', 'Category'],
                template="plotly_white",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_right:
            st.subheader("Distribusi Cluster")
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', color='Count')
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Eksplorasi Detail Cluster")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            sel_cluster = st.selectbox("Pilih Cluster:", sorted(df['Cluster'].unique()))
            cluster_data = df[df['Cluster'] == sel_cluster]
            
            st.write("**Top Keywords dalam Cluster:**")
            words = " ".join(cluster_data['combined_features']).split()
            # Filter kata pendek
            top_words = pd.Series([w for w in words if len(w) > 3]).value_counts().head(10)
            st.write(top_words)

        with c2:
            st.write(f"Daftar Produk di Cluster {sel_cluster}")
            # Menghindari error duplikat kolom dengan memilih kolom secara aman
            cols_show = ['Product Name', 'Category', 'Cluster']
            st.dataframe(cluster_data[cols_show], use_container_width=True)

        st.divider()
        st.subheader("🔎 Cari Produk")
        search_query = st.text_input("Masukkan nama produk:")
        if search_query:
            # Filter pencarian
            results = df[df['Product Name'].str.contains(search_query, case=False, na=False)]
            if not results.empty:
                st.success(f"Ditemukan {len(results)} produk.")
                # Tampilkan hasil tanpa duplikasi kolom
                st.dataframe(results[['Product Name', 'Category', 'Cluster']], use_container_width=True)
            else:
                st.warning("Produk tidak ditemukan.")

    with tab3:
        st.subheader("Seluruh Data Tersegmentasi")
        # Hapus kolom pembantu sebelum ditampilkan ke user
        df_display = df.drop(columns=['combined_features', 'pca_1', 'pca_2'], errors='ignore')
        st.dataframe(df_display, use_container_width=True)
        
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report (CSV)", data=csv, file_name='report_segmentasi.csv', mime='text/csv')

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
