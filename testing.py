import streamlit as st
import pandas as pd
import plotly.express as px
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Category Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- FUNGSI LOAD DATA ---
def get_data():
    file_path = 'data_set.csv'
    if not os.path.exists(file_path):
        df_empty = pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
        df_empty.to_csv(file_path, index=False)
        return df_empty
    
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except:
        return pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])

    df = df.loc[:, ~df.columns.duplicated()]
    cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    for col in cols:
        if col not in df.columns:
            df[col] = ""
            
    df[cols] = df[cols].fillna('')
    df['combined_features'] = df[cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

# --- FUNGSI TRAINING MODEL ---
@st.cache_resource
def train_model(df, k):
    if len(df) < k:
        k = max(1, len(df))
        
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)
    
    return df, vectorizer, model, tfidf_matrix

# --- MAIN APP ---
try:
    df_raw = get_data()

    # Sidebar: Pengaturan & Auto-Saran Cluster
    with st.sidebar:
        st.title("🤖 Assistant Settings")
        
        if not df_raw.empty and len(df_raw) > 2:
            st.subheader("Optimasi Cluster")
            if st.button("🔍 Cari K Ideal (Silhouette)"):
                with st.spinner("Menghitung skor kecocokan..."):
                    best_k = 2
                    max_score = -1
                    # Test dari k=2 sampai k=15
                    limit_k = min(15, len(df_raw) - 1)
                    
                    vectorizer_test = TfidfVectorizer(stop_words='english')
                    matrix_test = vectorizer_test.fit_transform(df_raw['combined_features'])
                    
                    for k_test in range(2, limit_k + 1):
                        km = KMeans(n_clusters=k_test, random_state=42, n_init=5)
                        labels = km.fit_predict(matrix_test)
                        score = silhouette_score(matrix_test, labels)
                        if score > max_score:
                            max_score = score
                            best_k = k_test
                    st.success(f"Saran K terbaik: {best_k} (Skor: {max_score:.2f})")
                    st.session_state['k_val'] = best_k
        
        # Inisialisasi k_val di session state jika belum ada
        if 'k_val' not in st.session_state:
            st.session_state['k_val'] = 10
            
        k_val = st.slider("Jumlah Kelompok (K)", 2, 20, st.session_state['k_val'])
        
        st.divider()
        if st.button("🔄 Latih Ulang Model"):
            st.cache_resource.clear()
            st.rerun()

    # Jalankan Engine
    if df_raw.empty:
        st.warning("Database kosong. Tambahkan data di tab 'Input Produk Baru'.")
        df, vec, model, matrix = None, None, None, None
    else:
        df, vec, model, matrix = train_model(df_raw, k_val)

    st.title("🤖 Smart Category Assistant")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Analitik", "🔮 Input Produk Baru", "🔍 Eksplorasi Segmen", "📁 Database Master"
    ])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        if df is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Produk", len(df))
            m2.metric("Kelompok Terbentuk", k_val)
            m3.metric("Status Data", "Live")

            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("Peta Kedekatan Produk (PCA)")
                pca = PCA(n_components=2)
                coords = pca.fit_transform(matrix.toarray())
                df['x'], df['y'] = coords[:, 0], coords[:, 1]
                fig = px.scatter(df, x='x', y='y', color='Cluster', 
                               hover_data=['Product Name', 'Category'],
                               template="plotly_white", height=500)
                st.plotly_chart(fig, use_container_width=True)
            with col_right:
                st.subheader("Populasi Per Kelompok")
                counts = df['Cluster'].value_counts().reset_index()
                counts.columns = ['Cluster', 'Jumlah']
                st.plotly_chart(px.bar(counts, x='Cluster', y='Jumlah', color='Cluster'), use_container_width=True)

    # --- TAB 2: INPUT PRODUK & PREDIKSI ---
    with tab2:
        st.subheader("🔮 Input & Analisis Instan")
        with st.form("new_product_form", clear_on_submit=True):
            f_name = st.text_input("Nama Produk")
            f_cat = st.text_input("Kategori Utama")
            f_tags = st.text_area("Masukkan Tag / Deskripsi")
            submitted = st.form_submit_button("Simpan & Lihat Cluster")
            
        if submitted:
            if f_name and f_cat:
                # 1. Simpan ke CSV
                clean_name = f_name.replace(',', ' ')
                clean_cat = f_cat.replace(',', ' ')
                clean_tags = f_tags.replace(',', ' ').split()
                new_row = [clean_name, clean_cat] + (clean_tags + [""] * 5)[:5]
                
                with open('data_set.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    writer.writerow(new_row)

                st.success(f"✅ Produk '{f_name}' berhasil disimpan!")

                # 2. Prediksi & Tampilkan Tabel Item Sejenis
                if vec is not None:
                    input_text = f"{clean_cat} {' '.join(clean_tags)}"
                    input_vec = vec.transform([input_text])
                    prediction = model.predict(input_vec)[0]
                    
                    st.markdown(f"### 🎯 Hasil Analisis: Produk ini masuk ke **Cluster {prediction}**")
                    
                    # Tampilkan item lain di cluster yang sama
                    similar_items = df[df['Cluster'] == prediction][['Product Name', 'Category', 'tag1', 'tag2']].head(10)
                    st.write(f"**Produk serupa di Cluster {prediction}:**")
                    st.table(similar_items)
                
                st.info("💡 Jangan lupa klik 'Latih Ulang Model' di sidebar agar peta koordinat diperbarui.")
            else:
                st.error("Nama Produk dan Kategori wajib diisi!")

    # --- TAB 3: DETAIL SEGMEN ---
    with tab3:
        if df is not None:
            sel_cluster = st.selectbox("Pilih nomor cluster:", sorted(df['Cluster'].unique()))
            cluster_filtered = df[df['Cluster'] == sel_cluster]
            
            c_stats, c_table = st.columns([1, 2])
            with c_stats:
                st.write(f"**Top Kata Kunci Cluster {sel_cluster}**")
                all_words = " ".join(cluster_filtered['combined_features']).split()
                top_words = pd.Series([w for w in all_words if len(w) > 3]).value_counts().head(10)
                st.plotly_chart(px.bar(top_words, orientation='h'), use_container_width=True)
            with c_table:
                st.write(f"**Seluruh Produk di Cluster {sel_cluster}**")
                st.dataframe(cluster_filtered[['Product Name', 'Category', 'tag1', 'tag2']], use_container_width=True)

    # --- TAB 4: MASTER DATA ---
    with tab4:
        if not df_raw.empty:
            final_display = df.drop(columns=['combined_features', 'x', 'y'], errors='ignore') if df is not None else df_raw
            st.dataframe(final_display, use_container_width=True)
            csv_data = final_display.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Laporan CSV", csv_data, "inventory_report.csv", "text/csv")

except Exception as e:
    st.error(f"Terjadi kendala: {e}")
