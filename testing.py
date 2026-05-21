import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Dynamic Academic Analyzer | Stevanus",
    page_icon="🎓",
    layout="wide"
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1); 
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px; border-radius: 12px; color: inherit;
    }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# 3. FUNGSI LOAD MODEL STATIC (Untuk Prediksi Individu)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model_static, scaler_static = load_assets()

# 4. FUNGSI INFORMASI KATEGORI
def get_info(cat):
    info = {
        "Tinggi": {"label": "🚀 High Achiever", "color": "#2ecc71", "saran": "Pertahankan! Cocok jadi asisten dosen."},
        "Menengah": {"label": "📊 Steady / Average", "color": "#f1c40f", "saran": "Tingkatkan konsistensi nilai ujian."},
        "Berisiko": {"label": "⚠️ Underperformer", "color": "#e74c3c", "saran": "Butuh bimbingan akademik intensif."}
    }
    return info.get(cat, {"label": "Unknown", "color": "#7f8c8d", "saran": "-"})

# 5. SIDEBAR
with st.sidebar:
    st.title("🎓 Smart Campus AI")
    st.write(f"User: **Stevanus**")
    st.divider()
    menu = st.radio("Navigasi Utama", ["🏠 Dashboard Analisis Dinamis", "🔍 Prediksi Individu"])
    st.divider()
    uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")

# 6. LOGIKA UTAMA
if menu == "🏠 Dashboard Analisis Dinamis":
    st.title("📊 Dashboard Analisis Dinamis")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        
        st.subheader("⚙️ Konfigurasi Re-Clustering")
        st.write("Silakan pilih parameter. Sistem akan menghitung ulang klaster secara real-time:")
        
        all_features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']
        available_features = [f for f in all_features if f in df.columns]
        
        # Menggunakan on_change untuk memicu st.rerun() secara otomatis saat pilihan berubah
        selected_features = st.multiselect(
            "Pilih Parameter untuk Analisis Kelompok:",
            options=available_features,
            default=['midterm_marks', 'final_marks', 'lectures_attended'] if len(available_features) >= 3 else available_features[:2],
            format_func=lambda x: x.replace('_', ' ').title(),
            key="active_features_widget",
            on_change=lambda: st.rerun() if hasattr(st, "rerun") else None
        )

        if len(selected_features) < 2:
            st.warning("⚠️ Pilih minimal 2 parameter agar sistem bisa melakukan kalkulasi ulang (Re-Clustering).")
        else:
            # --- PROSES DYNAMIC CLUSTERING ---
            data_to_cluster = df[selected_features].fillna(df[selected_features].mean())
            
            scaler_dynamic = StandardScaler()
            scaled_dynamic = scaler_dynamic.fit_transform(data_to_cluster)
            
            kmeans_dynamic = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['cluster_dynamic'] = kmeans_dynamic.fit_predict(scaled_dynamic)

            # Hitung ranking performa berdasarkan akumulasi nilai rata-rata dari parameter terpilih
            cluster_scores = df.groupby('cluster_dynamic')[selected_features].mean().sum(axis=1).sort_values(ascending=False)
            
            rank_map = {
                cluster_scores.index[0]: "Tinggi", 
                cluster_scores.index[1]: "Menengah", 
                cluster_scores.index[2]: "Berisiko"
            }
            
            df['kategori_live'] = df['cluster_dynamic'].map(rank_map)
            df['label_live'] = df['kategori_live'].apply(lambda x: get_info(x)['label'])

            # --- PANEL RINGKASAN METRIK ---
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Mahasiswa", len(df))
            if 'final_marks' in df.columns:
                m2.metric("Rata-rata Final Exam", f"{df['final_marks'].mean():.1f}")
            else:
                m2.metric("Rata-rata Final Exam", "N/A")
            m3.metric("Status Komputasi", "🔄 Hard Refreshed Live")

            # --- PANEL VISUALISASI LIVE GRAPH ---
            col_left, col_right = st.columns([6, 4])
            
            with col_left:
                st.markdown("##### 📍 Sebaran Kelompok Berdasarkan Pilihan")
                x_axis = st.selectbox("Sumbu X Grafik:", selected_features, index=0, key="sb_x_ultimate")
                y_axis = st.selectbox("Sumbu Y Grafik:", selected_features, index=1 if len(selected_features) > 1 else 0, key="sb_y_ultimate")
                
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, color="label_live", 
                    template="none", 
                    color_discrete_map={
                        get_info("Tinggi")['label']: "#2ecc71",
                        get_info("Menengah")['label']: "#f1c40f",
                        get_info("Berisiko")['label']: "#e74c3c"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("##### 🥧 Proporsi Kelompok Mahasiswa (Up-to-Date)")
                
                # ISOLASI TOTAL: Membuat DataFrame baru yang independen khusus untuk pie chart
                pie_data = pd.DataFrame({
                    'Kelompok': df['label_live'].value_counts().index,
                    'Jumlah': df['label_live'].value_counts().values
                })
                
                fig_pie = px.pie(
                    pie_data, names="Kelompok", values="Jumlah",
                    hole=0.4, template="none",
                    color="Kelompok", 
                    color_discrete_map={
                        get_info("Tinggi")['label']: "#2ecc71",
                        get_info("Menengah")['label']: "#f1c40f",
                        get_info("Berisiko")['label']: "#e74c3c"
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("📋 Eksplorasi Data Detail")
            st.dataframe(df[selected_features + ['label_live']], use_container_width=True)
    else:
        st.info("Silakan unggah dataset CSV pada sidebar untuk memulai analisis dinamis.")

else:
    st.title("🔍 Prediksi Individu")
    if model_static is None:
        st.error("Model .pkl tidak ditemukan untuk prediksi individu.")
    else:
        with st.form("input_form"):
            c1, c2 = st.columns(2)
            with c1:
                q1 = st.slider("Quiz 1", 0.0, 10.0, 0.0)
                q2 = st.slider("Quiz 2", 0.0, 10.0, 0.0)
                q3 = st.slider("Quiz 3", 0.0, 10.0, 0.0)
                mid = st.number_input("Midterm Marks", 0, 30, 0)
            with c2:
                fin = st.number_input("Final Marks", 0, 50, 0)
                gpa = st.slider("GPA", 0.0, 4.0, 0.0)
                lec = st.number_input("Lectures", 0, 12, 0)
                lab = st.number_input("Labs", 0, 6, 0)
            submit = st.form_submit_button("Prediksi")

        if submit:
            if mid == 0 and fin == 0:
                res_cat = "Berisiko"
            else:
                input_data = pd.DataFrame(
                    [[q1, q2, q3, mid, fin, gpa, lec, lab]], 
                    columns=['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']
                )
                scaled = scaler_static.transform(input_data)
                cluster_id = model_static.predict(scaled)[0]
                res_cat = "Berisiko" if cluster_id == 2 else ("Menengah" if cluster_id == 1 else "Tinggi")
            
            info = get_info(res_cat)
            st.markdown(f"""
                <div style="background-color:{info['color']}; padding:30px; border-radius:15px; text-align:center; color:white;">
                    <h1 style="color:white; margin:0;">{info['label']}</h1>
                    <p style="font-size:1.2em; margin-top:10px;">{info['saran']}</p>
                </div>
            """, unsafe_allow_html=True)

st.divider()
st.caption("UAS Pemrograman AI - Stevanus - Built with Ultimate Anti-Cache Engine")
