# 🎓 Academic Performance Cluster Analyzer
> **Proyek Ujian Akhir Semester (UAS) - Pemrograman Kecerdasan Buatan**
> Ditujukan untuk memenuhi tugas individu mata kuliah Pemrograman AI (AIB02) - Universitas Bunda Mulia.

---

## 👤 Identitas Mahasiswa
* **Nama:** Stevanus
* **NIM:** 38250029
* **Program Studi:** Artificial Intelligence

---

## 📌 Deskripsi Proyek
**Academic Performance Cluster Analyzer** adalah sebuah sistem cerdas berbasis *Machine Learning* yang menggunakan algoritma **K-Means Clustering** untuk melakukan segmentasi, pemetaan profil, dan deteksi dini risiko performa akademik mahasiswa. 

Aplikasi ini dibangun menggunakan **Streamlit** sebagai antarmuka interaktif dan **Plotly** untuk visualisasi data spasial multidimensi secara *real-time*. Sistem ini membantu dosen, kaprodi, dan penasihat akademik dalam mengambil keputusan intervensi yang objektif berdasarkan data rekam jejak perilaku belajar mahasiswa.

---

## 🚀 Fitur Unggulan & Komponen AI
Aplikasi ini tidak hanya melakukan clustering standar, melainkan dilengkapi dengan beberapa fitur tingkat lanjut:

1. **Dynamic AI Weighting Customization:** Pengguna dapat mengatur bobot persentase pengaruh komponen akademik (Absensi/Tugas, Nilai Ujian, Nilai Kuis, dan IPK Sebelumnya) secara dinamis sebelum proses clustering dilakukan.
2. **Aturan Wajib Kampus (Blacklist Override):** Integrasi logika regulasi institusi. Jika mahasiswa berada di bawah ambang batas parameter kritis tertentu (misal: kehadiran kuliah $\le 3$ sesi), statusnya akan di-*override* secara mutlak menjadi **🛑 Tidak Layak**.
3. **Explainable AI (XAI):** Menampilkan tingkat sensitivitas matriks fitur global menggunakan algoritma *Permutation Importance* untuk transparansi penentuan klaster.
4. **Smart Improvement Advice Engine:** Analisis kesenjangan (*gap analysis*) yang menghitung selisih koordinat vektor mahasiswa saat ini dengan rata-rata klaster di atasnya, lalu memberikan saran taktis tertulis (misal: *"Tambahkan nilai Quiz minimal +5.3 poin"*).
5. **Dual Mode Analytics:** Mendukung prediksi untuk data satu mahasiswa baru (*Single Mode*) maupun perbandingan spasial langsung antara dua mahasiswa (*Comparison Mode*) lewat grafik *Radar Chart* interaktif.
6. **Downloadable Report:** Fitur unduh otomatis rangkuman hasil prediksi dalam format `.txt`.

---

## 📁 Struktur Dataset
Dataset yang digunakan berasal dari Kaggle (*student_dropout_behavior_dataset.csv* oleh Muhammad Khubaib Ahmad) berisi 300 baris sampel dengan 16 fitur pendukung, antara lain:
* **Matriks Nilai:** `quiz1_marks`, `quiz2_marks`, `quiz3_marks`, `midterm_marks`, `final_marks`, `previous_gpa`.
* **Matriks Kehadiran & Tugas:** `lectures_attended`, `labs_attended`, `assignments_submitted`.

---
