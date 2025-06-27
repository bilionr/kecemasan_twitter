# Deteksi Kecemasan dari Teks Twitter Menggunakan Machine Learning

Proyek ini adalah implementasi dari penelitian untuk mengklasifikasikan potensi kecemasan (*anxiety*) pada pengguna Twitter berdasarkan analisis konten teks dan pola perilaku mereka. Skrip ini menggunakan pendekatan *Natural Language Processing* (NLP) dan *Machine Learning* untuk membangun model prediktif, melakukan analisis profil laten, dan mengevaluasi performa model.

## Tim Peneliti (Kelompok 9)
- **Ruli Hendrawan Saputra** (187221068)
- **Dzakwan Fiodora Syafi’i** (187221079)
- **Hiekam Nursal Muhammad** (187221090)
- **Fardhan Erfandyar** (187221029)
- **Zhillan Ahil Arrafi Siswadhi** (187221064)
- **Billy Alexander Sugiyanto** (187221036)
  
## Fitur Utama
- **Pra-pemrosesan Teks Bahasa Indonesia**: Membersihkan dan menstandarisasi teks tweet, termasuk *case folding*, tokenisasi, dan penghapusan *stopword* dengan kamus Sastrawi yang diperluas.
- **Ekstraksi Fitur Komprehensif**: Mengekstrak fitur perilaku (waktu posting, panjang teks) dan fitur sentimen (berdasarkan leksikon InSet).
- **Analisis Profil Laten (LPA)**: Menggunakan *Gaussian Mixture Models* (GMM) untuk mengidentifikasi sub-profil pengguna berdasarkan karakteristik perilaku dan sentimen mereka.
- **Pelatihan & Evaluasi Model**: Melatih dan membandingkan tiga model klasifikasi yang berbeda: Random Forest, Naïve Bayes, dan Regresi LASSO (Logistic Regression dengan regularisasi L1).
- **Hyperparameter Tuning**: Menggunakan `GridSearchCV` untuk menemukan parameter terbaik untuk setiap model.
- **Penyeimbangan Data**: Menangani ketidakseimbangan kelas dalam data latih menggunakan `RandomUnderSampler`.
- **Visualisasi Hasil**: Menghasilkan plot untuk perbandingan akurasi model, kurva ROC AUC, pentingnya fitur (*feature importance*), dan karakteristik profil laten.
- **Prediksi pada Data Uji**: Mengaplikasikan model terbaik pada dataset uji yang tidak berlabel untuk menghasilkan prediksi kondisi.

## Struktur File
Berikut adalah penjelasan singkat mengenai file-file utama dalam repositori ini:

```
.
├── main.py                     # Skrip utama untuk menjalankan seluruh pipeline analisis
├── final_training_dataset.csv  # Dataset untuk melatih model (harus berisi kolom 'full_text' dan 'label')
├── final_testing_dataset.csv   # Dataset untuk diuji oleh model (tidak perlu label)
├── positive.csv                # Bagian dari leksikon sentimen InSet (kata-kata positif)
└── negative.csv                # Bagian dari leksikon sentimen InSet (kata-kata negatif)
```

## Metodologi
Alur kerja yang diimplementasikan dalam skrip `main.py` adalah sebagai berikut:
1.  **Muat Data**: Memuat data latih, data uji, dan leksikon sentimen InSet dari file-file CSV.
2.  **Pra-pemrosesan Teks**: Setiap tweet dibersihkan melalui beberapa tahap:
    -   Mengubah teks menjadi huruf kecil (*case folding*).
    -   Menghapus URL, hashtag, dan mention.
    -   Menghapus tanda baca dan karakter non-alfanumerik.
    -   Memecah teks menjadi token (*tokenization*).
    -   Menghapus *stopwords* (kata umum) dalam bahasa Indonesia.
3.  **Ekstraksi Fitur**: Fitur-fitur berikut diekstrak dari setiap tweet:
    -   **Fitur Perilaku**: `TextLength`, `HourOfDay`, `DayOfWeek`, `IsWeekend`.
    -   **Fitur Sentimen**: `inset_pos` (jumlah kata positif), `inset_neg` (jumlah kata negatif), dan `inset_score` (skor sentimen ternormalisasi).
4.  **Analisis Profil Laten (LPA)**: Mengelompokkan pengguna dalam data latih ke dalam 4 profil berbeda (High Sentiment, Low Sentiment, Self-Distancing, Neutral) berdasarkan fitur perilaku dan sentimen.
5.  **Vektorisasi Teks**: Teks yang sudah bersih diubah menjadi representasi numerik menggunakan **TF-IDF Vectorizer**.
6.  **Penyeimbangan Data**: Kelas dalam data latih diseimbangkan menggunakan `RandomUnderSampler` untuk mencegah bias pada model.
7.  **Pelatihan Model**: Tiga model berbeda (Random Forest, Naïve Bayes, LASSO) dilatih pada data latih yang sudah diproses dan diseimbangkan.
8.  **Evaluasi Model**: Performa setiap model dievaluasi pada data validasi menggunakan metrik Akurasi, Presisi, Recall, F1-Score, dan kurva ROC AUC.
9.  **Prediksi**: Model terbaik (Random Forest) digunakan untuk memprediksi kondisi pada `final_testing_dataset.csv`. Hasil prediksi kemudian diagregasi per akun.

## Instalasi dan Setup
Untuk menjalankan proyek ini, Anda perlu menyiapkan lingkungan Python dan menginstal dependensi yang diperlukan.

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/bilionr/nama-repositori-anda.git](https://github.com/bilionr/nama-repositori-anda.git)
    cd nama-repositori-anda
    ```

2.  **Buat dan Aktifkan Virtual Environment** (Sangat Direkomendasikan)
    ```bash
    # Membuat venv
    python -m venv venv

    # Mengaktifkan di Windows (PowerShell)
    .\venv\Scripts\activate

    # Mengaktifkan di macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal Dependensi**
    Buat file bernama `requirements.txt` dan salin teks di bawah ini ke dalamnya.
    ```txt
    pandas
    numpy
    scikit-learn
    nltk
    Sastrawi
    tabulate
    matplotlib
    seaborn
    scikit-learn-contrib-imblearn
    ```
    Kemudian, jalankan perintah berikut di terminal Anda:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh Resource NLTK**
    Skrip akan secara otomatis mengunduh `punkt` dari NLTK saat pertama kali dijalankan.

## Cara Menjalankan
Pastikan semua file CSV yang diperlukan (`final_training_dataset.csv`, `final_testing_dataset.csv`, `positive.csv`, `negative.csv`) berada di direktori yang sama dengan `main.py`.

Jalankan skrip utama dari terminal:
```bash
python main.py
```

## Output
Skrip akan menghasilkan beberapa file output, di antaranya:
-   **`labeled_tweets.csv`**: File hasil prediksi pada data uji, berisi teks tweet beserta prediksi kondisinya.
-   **`preprocessed_tweets.csv`**: Data uji beserta teks yang sudah dibersihkan.
-   **`lpa_results.png`**: Plot visualisasi hasil dari Analisis Profil Laten.
-   **`model_roc_curves.png`**: Plot perbandingan kurva ROC AUC dari semua model.
-   **`model_accuracy_comparison.png`**: Plot perbandingan akurasi antar model.
-   **`behavioral_sentiment_feature_importance.png`**: Plot yang menunjukkan fitur perilaku dan sentimen mana yang paling berpengaruh pada model Random Forest.

