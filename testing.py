import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_gsheets import GSheetsConnection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Smart Category Assistant Pro", page_icon="🤖", layout="wide")

# URL Google Sheets Stevanus
SQL_URL = "https://docs.google.com/spreadsheets/d/1EzAuFcdhr77yDHsO2jwGvYt7wmYBbfAB41LAyOd7nFc/edit?usp=sharing"

# --- KONEKSI DATABASE ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data_from_gsheets():
    # Mengambil data terbaru (ttl=0 memastikan data tidak basi)
    df_raw = conn.read(spreadsheet=SQL_URL, ttl="0")
    
    # Buat copy agar tidak merusak data asli saat manipulasi
    df = df_raw.copy()
    
    # Bersihkan kolom duplikat jika ada
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Pastikan kolom utama tersedia
    cols_needed = ['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    for col in cols_needed:
        if col not in df.columns:
            df[col] = ""
    
    # Memastikan semua data bertipe string dan menangani nilai NaN
    for col in cols_needed:
        df[col] = df[col].astype(str).replace(['nan', 'None', 'None'], '')

    # FIX ERROR: Cara penggabungan fitur yang lebih stabil
    # Menggunakan list comprehension untuk memastikan hasilnya adalah Series tunggal
    feature_cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    df['combined_features'] = [
        ' '.join(filter(None, row)) 
        for row in df[feature_cols].values
    ]
    
    return df

# --- ENGINE CLUSTERING ---
@st.cache_resource
def train_model(df, k):
    if len(df) < k: k = max(1, len(df))
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit transform pada kolom teks tunggal
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)
    return df, vectorizer, model, tfidf_matrix

# --- MAIN INTERFACE ---
try:
    df_raw_loaded = load_data_from_gsheets()

    with st.sidebar:
        st.title("🤖 Cloud Settings")
        if 'k_val' not in st.session_state: 
            st.session_state['k_val'] = 10
        
        if not df_raw_loaded.empty and len(df_raw_loaded) > 2:
            st.subheader("Optimasi Cluster")
            if st.button("🔍 Cari K Ideal (Silhouette)"):
                with st.spinner("Menganalisis pola data di Cloud..."):
                    best_k, max_score = 2, -1
                    limit_k = min(15, len(df_raw_loaded) - 1)
                    
                    vec_t = TfidfVectorizer(stop_words='english')
                    mtx_t = vec_t.fit_transform(df_raw_loaded['combined_features'])
                    
                    for kt in range(2, limit_k + 1):
                        km = KMeans(n_clusters=kt, random_state=42, n_init=5)
                        labels = km.fit_predict(mtx_t)
                        score = silhouette_score(mtx_t, labels)
                        if score > max_score:
                            max_score = score
                            best_k = kt
                    
                    st.session_state['k_val'] = best_k
                    st.success(f"Saran K Terbaik: {best_k}")

        k_val = st.slider("Jumlah Kelompok (K)", 2, 20, st.session_state['k_val'])
        
        st.divider()
        if st.button("🔄 Paksa Sinkronisasi Cloud"):
            st.cache_resource.clear()
            st.rerun()

    if df_raw_loaded.empty:
        st.warning("Database Google Sheets Kosong! Pastikan Header sudah benar.")
    else:
        # Jalankan Clustering
        df, vec, model, matrix = train_model(df_raw_loaded, k_val)

        st.title("🚀 Smart Category Assistant (Cloud Mode)")
        st.markdown("Terhubung ke: `Product_Database` di Google Sheets")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analitik", "➕ Tambah Produk", "🔍 Eksplorasi Segmen", "📁 Database Live"])

        with tab1:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Produk", len(df))
            m2.metric("Kelompok AI", k_val)
            m3.metric("Status Server", "Online")
            
            # Visualisasi PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(matrix.toarray())
            df['x'], df['y'] = coords[:, 0], coords[:, 1]
            fig = px.scatter(df, x='x', y='y', color='Cluster', 
                           hover_data=['Product Name', 'Category'], 
                           template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("➕ Tambah Produk Baru ke Cloud")
            
            with st.form("gsheets_form", clear_on_submit=True):
                f_name = st.text_input("Nama Produk")
                f_cat = st.text_input("Kategori Utama")
                f_tag = st.text_area("Tag/Deskripsi (pisahkan dengan spasi)")
                submitted = st.form_submit_button("Simpan Permanen ke Google Sheets")
                
            if submitted and f_name:
                # Siapkan baris baru
                tags_list = (f_tag.split() + [""] * 5)[:5]
                new_row = pd.DataFrame([[f_name, f_cat] + tags_list], 
                                        columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
                
                # Bersihkan data lama dari kolom kalkulasi AI sebelum upload
                cols_to_keep = ['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
                clean_old_df = df[cols_to_keep]
                updated_df = pd.concat([clean_old_df, new_row], ignore_index=True)
                
                # Update Google Sheets
                conn.update(spreadsheet=SQL_URL, data=updated_df)
                st.success(f"✅ Produk '{f_name}' berhasil disimpan!")
                
                # Hapus cache dan refresh
                st.cache_resource.clear()
                st.rerun()

        with tab3:
            sel_c = st.selectbox("Pilih Cluster:", sorted(df['Cluster'].unique()))
            st.write(f"Daftar Produk di Cluster {sel_c}:")
            st.dataframe(df[df['Cluster'] == sel_c][['Product Name', 'Category', 'tag1', 'tag2']], use_container_width=True)

        with tab4:
            st.subheader("📁 Tampilan Data Google Sheets")
            # Menampilkan data asli (tanpa kolom koordinat PCA)
            st.dataframe(df_raw_loaded.drop(columns=['combined_features'], errors='ignore'), use_container_width=True)

except Exception as e:
    st.error(f"Terjadi Kendala: {e}")
    st.info("Saran: Periksa apakah baris pertama Google Sheets berisi header: Product Name, Category, tag1, tag2, tag3, tag4, tag5")
