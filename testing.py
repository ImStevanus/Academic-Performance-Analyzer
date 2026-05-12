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
        # Membuat file baru jika belum ada
        df_empty = pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
        df_empty.to_csv(file_path, index=False)
        return df_empty
    
    try:
        # Membaca data dengan proteksi baris rusak
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except:
        return pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])

    # Pembersihan Data
    df = df.loc[:, ~df.columns.duplicated()]
    cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    for col in cols:
        if col not in df.columns:
            df[col] = ""
            
    df[cols] = df[cols].fillna('')
    # Menggabungkan fitur teks untuk AI
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
    # Memuat data terbaru dari CSV
    df_raw = get_data()

    # Sidebar: Pengaturan & Silhouette Score
    with st.sidebar:
        st.title("🤖 Assistant Settings")
        
        # Inisialisasi k_val di session state jika belum ada
        if 'k_val' not in st.session_state:
            st.session_state['k_val'] = 10

        if not df_raw.empty and len(df_raw) > 2:
            st.subheader("Optimasi Otomatis")
            if st.button("🔍 Cari K Ideal (Silhouette)"):
                with st.spinner("Menganalisis kepadatan data..."):
                    best_k = 2
                    max_score = -1
                    # Cek hingga maksimal 15 cluster
                    limit_k = min(15, len(df_raw) - 1)
                    
                    vec_test = TfidfVectorizer(stop_words='english')
                    mtx_test = vec_test.fit_transform(df_raw['combined_features'])
                    
                    for k_test in range(2, limit_k + 1):
                        km = KMeans(n_clusters=k_test, random_state=42, n_init=5)
                        labels = km.fit_predict(mtx_test)
                        score = silhouette_score(mtx_test, labels)
                        if score > max_score:
                            max_score = score
                            best_k = k_test
                    
                    st.session_state['k_val'] = best_k
                    st.success(f"Saran K terbaik: {best_k}")
        
        # Slider menggunakan nilai dari session state
        k_val = st.slider("Jumlah Kelompok (K)", 2, 20, st.session_state['k_val'])
        
        st.divider()
        if st.button("🔄 Segarkan Database", help="Klik ini jika data baru belum muncul"):
            st.cache_resource.clear()
            st.rerun()

    # Menjalankan Engine Clustering
    if df_raw.empty:
        st.warning("Database kosong. Silakan tambahkan produk di tab 'Input Produk Baru'.")
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
            m2.metric("Kelompok Aktif", k_val)
            m3.metric("Sinkronisasi", "Terhubung")

            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.subheader("Peta Kedekatan Produk (PCA)")
                pca = PCA(n_components=2)
                coords = pca.fit_transform(matrix.toarray())
                df['x'], df['y'] = coords[:, 0], coords[:, 1]
                fig = px.scatter(df, x='x', y='y', color='Cluster', 
                               hover_data=['Product Name', 'Category'],
                               template="plotly_white", height=500)
                st.plotly_chart(fig, use_container_width=True)
            with c_right:
                st.subheader("Distribusi Item")
                counts = df['Cluster'].value_counts().reset_index()
                counts.columns = ['Cluster', 'Jumlah']
                st.plotly_chart(px.bar(counts, x='Cluster', y='Jumlah', color='Cluster'), use_container_width=True)

    # --- TAB 2: INPUT PRODUK (OTOMATIS REFRESH) ---
    with tab2:
        st.subheader("🔮 Input & Prediksi Real-time")
        
        # Simpan hasil prediksi sementara di session state agar tidak hilang saat rerun
        if 'last_prediction' in st.session_state:
            res = st.session_state['last_prediction']
            st.success(f"✅ Produk Terakhir: '**{res['name']}**' masuk ke **Cluster {res['cluster']}**")
            st.write(f"Berikut adalah produk serupa di Cluster {res['cluster']}:")
            # Filter data terbaru untuk mencari item sejenis
            similar = df[df['Cluster'] == res['cluster']][['Product Name', 'Category', 'tag1']].head(10)
            st.table(similar)
            st.divider()

        with st.form("input_form", clear_on_submit=True):
            f_name = st.text_input("Nama Produk Baru")
            f_cat = st.text_input("Kategori Utama")
            f_tags = st.text_area("Tag / Deskripsi Produk")
            submitted = st.form_submit_button("Simpan & Sinkronisasi")
            
        if submitted:
            if f_name and f_cat:
                # 1. Bersihkan & Simpan ke CSV
                c_name = f_name.replace(',', ' ')
                c_cat = f_cat.replace(',', ' ')
                c_tags = f_tags.replace(',', ' ').split()
                new_row = [c_name, c_cat] + (c_tags + [""] * 5)[:5]
                
                with open('data_set.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    writer.writerow(new_row)

                # 2. Prediksi Cluster (Sebelum Rerun)
                if vec is not None:
                    txt = f"{c_cat} {' '.join(c_tags)}"
                    p_cluster = model.predict(vec.transform([txt]))[0]
                    st.session_state['last_prediction'] = {'name': f_name, 'cluster': p_cluster}

                # 3. Paksa Aplikasi Baca Ulang Data
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Nama Produk dan Kategori wajib diisi!")

    # --- TAB 3: DETAIL SEGMEN ---
    with tab3:
        if df is not None:
            sel_c = st.selectbox("Pilih nomor cluster:", sorted(df['Cluster'].unique()))
            sub_df = df[df['Cluster'] == sel_c]
            
            w1, w2 = st.columns([1, 2])
            with w1:
                st.write(f"**Karakteristik Cluster {sel_c}**")
                all_w = " ".join(sub_df['combined_features']).split()
                top_w = pd.Series([w for w in all_w if len(w) > 3]).value_counts().head(10)
                st.plotly_chart(px.bar(top_w, orientation='h', labels={'index':'Kata'}), use_container_width=True)
            with w2:
                st.write(f"**Anggota Cluster {sel_c}**")
                st.dataframe(sub_df[['Product Name', 'Category', 'tag1', 'tag2']], use_container_width=True)

    # --- TAB 4: DATABASE MASTER ---
    with tab4:
        st.subheader("📁 Database Inventory Lengkap")
        if not df_raw.empty:
            # Tampilkan data dengan kolom cluster terbaru
            disp = df.drop(columns=['combined_features', 'x', 'y'], errors='ignore') if df is not None else df_raw
            st.dataframe(disp, use_container_width=True)
            
            # Tombol Download
            csv_out = disp.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Ekspor Database (CSV)", csv_out, "master_inventory.csv", "text/csv")
        else:
            st.info("Database kosong.")

except Exception as e:
    st.error(f"Sistem Error: {e}")
