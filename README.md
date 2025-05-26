# Proyek Prediksi Diabetes dengan Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--learn-0.24%2B-orange?logo=scikit-learn" alt="Scikit-learn Version">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-red?logo=jupyter" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

Deteksi dini penyakit kronis seperti diabetes sangat penting untuk meningkatkan kualitas hidup pasien. Proyek ini bertujuan untuk mengembangkan model *machine learning* yang mampu memprediksi apakah seseorang menderita diabetes berdasarkan data pemeriksaan medis. Kami menggunakan algoritma **Decision Tree** dan **Random Forest** serta membandingkan performanya untuk menemukan model terbaik.

---

## Daftar Isi

* [Latar Belakang](#latar-belakang)
* [Tujuan Proyek](#tujuan-proyek)
* [Dataset](#dataset)
* [Struktur Proyek](#struktur-proyek)
* [Metodologi](#metodologi)
* [Hasil dan Analisis](#hasil-dan-analisis)
* [Kesimpulan](#kesimpulan)
* [Saran Pengembangan](#saran-pengembangan)
* [Cara Menjalankan Proyek](#cara-menjalankan-proyek)
* [Lisensi](#lisensi)
* [Referensi](#referensi)

---

## Latar Belakang

Diabetes adalah penyakit kronis yang memengaruhi jutaan orang di seluruh dunia, dengan angka yang terus meningkat setiap tahun. Gejala yang sering tidak tampak pada tahap awal menjadi tantangan dalam diagnosis dini. Dengan memanfaatkan kekuatan *machine learning*, kita dapat membangun sistem prediksi otomatis yang akurat untuk membantu deteksi awal diabetes, sehingga memungkinkan intervensi medis yang lebih cepat dan efektif.

---

## Tujuan Proyek

1.  Mengembangkan model klasifikasi menggunakan algoritma *Decision Tree* dan *Random Forest* untuk mendeteksi diabetes.
2.  Melakukan *hyperparameter tuning* pada kedua model untuk optimasi performa.
3.  Membandingkan performa model berdasarkan metrik **Akurasi, Precision, Recall, dan F1-Score** untuk mengidentifikasi model terbaik.

---

## Dataset

Dataset yang digunakan adalah **Pima Indians Diabetes Database**, tersedia secara publik melalui Kaggle. Dataset ini terdiri dari 768 sampel data dengan 8 fitur input dan 1 label output (`Outcome`: 0 = Tidak diabetes, 1 = Diabetes).

**Fitur-fitur:**
* `Pregnancies`: Jumlah kehamilan
* `Glucose`: Kadar glukosa darah
* `BloodPressure`: Tekanan darah diastolik (mm Hg)
* `SkinThickness`: Ketebalan lipatan kulit triceps (mm)
* `Insulin`: Level insulin serum (mu U/ml)
* `BMI`: Indeks massa tubuh (kg/m²)
* `DiabetesPedigreeFunction`: Riwayat keluarga diabetes
* `Age`: Usia (tahun)
* `Outcome`: Target (0 = Tidak diabetes, 1 = Diabetes)

---

## Struktur Proyek
.
├── README.md
├── diabetes.csv          # Dataset yang digunakan
└── diabetes_prediction.ipynb # Notebook Jupyter berisi seluruh kode dan analisis

---

## Metodologi

Proyek ini mengikuti tahapan standar dalam alur kerja *machine learning*:

1.  **Data Understanding**: Memahami karakteristik dataset, termasuk distribusi kelas dan nilai-nilai yang tidak valid (misalnya, nilai 0 pada fitur `Glucose`, `BMI`, dll.).
2.  **Data Preparation**:
    * Mengganti nilai 0 yang tidak valid pada fitur numerik dengan nilai **median** masing-masing fitur.
    * **Normalisasi data** menggunakan `MinMaxScaler` untuk menyetarakan skala fitur.
    * Memisahkan data menjadi *training set* (80%) dan *testing set* (20%) menggunakan `train_test_split`.
3.  **Modeling**:
    * Membangun model menggunakan **Decision Tree Classifier** dan **Random Forest Classifier**.
    * Melakukan **Hyperparameter Tuning** pada kedua model menggunakan `GridSearchCV` untuk mencari kombinasi *hyperparameter* terbaik (misalnya `max_depth`, `n_estimators`, `min_samples_split`).
4.  **Evaluation**:
    * Mengevaluasi performa model terbaik dari kedua algoritma pada *testing set*.
    * Metrik yang digunakan: **Akurasi, Precision, Recall, dan F1-Score**.

---

## Hasil dan Analisis

Setelah melakukan *hyperparameter tuning* dan evaluasi, berikut adalah perbandingan performa kedua model:

| Model         | Akurasi | Precision | Recall | F1 Score |
| :------------ | :------ | :-------- | :----- | :------- |
| Decision Tree | 0.7078  | 0.5676    | 0.7636 | 0.6512   |
| Random Forest | 0.7403  | 0.6316    | 0.6545 | 0.6429   |

**Analisis:**
* **Akurasi dan Precision:** **Random Forest** menunjukkan akurasi (0.7403) dan precision (0.6316) yang lebih tinggi, mengindikasikan kemampuannya yang lebih baik dalam membuat prediksi benar secara keseluruhan dan mengurangi *false positives*.
* **Recall:** **Decision Tree** unggul signifikan dalam *recall* (0.7636) dibandingkan Random Forest (0.6545). Ini berarti Decision Tree lebih efektif dalam mendeteksi sebagian besar kasus diabetes yang sebenarnya (meminimalkan *false negatives*).
* **F1 Score:** Decision Tree memiliki F1 Score yang sedikit lebih tinggi (0.6512) dibandingkan Random Forest (0.6429), yang menunjukkan keseimbangan *precision* dan *recall* yang lebih baik dalam skenario ini, terutama didorong oleh *recall* yang kuat.

**Kesimpulan dari Hasil:**
Pemilihan model terbaik sangat bergantung pada prioritas aplikasi. Jika tujuan utama adalah **meminimalkan *false negatives* (deteksi dini semua kasus diabetes)**, maka **Decision Tree** adalah pilihan yang lebih kuat karena *recall*-nya yang superior. Namun, jika fokusnya adalah **akurasi dan *precision* secara keseluruhan** (meminimalkan *false positives* dan *false negatives* secara seimbang), maka **Random Forest** lebih direkomendasikan.

---

## Kesimpulan

Proyek ini berhasil membangun dan mengevaluasi model klasifikasi untuk deteksi diabetes. **Decision Tree menunjukkan kapabilitas yang kuat dalam mendeteksi kasus diabetes yang sebenarnya (tinggi *recall*)**, yang sangat penting dalam konteks medis. Sementara itu, **Random Forest memberikan akurasi dan *precision* yang lebih tinggi secara keseluruhan**. Pemilihan model akhir akan bergantung pada kebutuhan spesifik dan dampak kesalahan prediksi di lingkungan klinis.

---

## Saran Pengembangan

Untuk meningkatkan proyek ini lebih lanjut, beberapa area yang dapat dieksplorasi meliputi:

* **Penanganan *Class Imbalance* Lanjutan**: Menerapkan teknik seperti **SMOTE** atau *oversampling/undersampling* untuk mengatasi ketidakseimbangan kelas pada dataset.
* **Eksplorasi Algoritma Lain**: Mencoba algoritma klasifikasi lain seperti Support Vector Machine (SVM), Logistic Regression, atau Gradient Boosting (XGBoost, LightGBM) untuk perbandingan performa lebih luas.
* **Validasi Eksternal**: Menguji model pada dataset diabetes lain yang independen untuk memastikan generalisasi dan *robustness* model.
* **Interpretasi Model**: Menggunakan teknik seperti **SHAP values** atau **LIME** untuk memahami fitur mana yang paling berpengaruh dalam prediksi diabetes, yang dapat memberikan wawasan klinis berharga.
* **Optimasi *Threshold***: Menyesuaikan *threshold* klasifikasi untuk menyeimbangkan *precision* dan *recall* sesuai kebutuhan aplikasi.

---

## Cara Menjalankan Proyek

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/dionp3/Machine-Learning-Terapan.git](https://github.com/dionp3/Machine-Learning-Terapan.git)
    cd Machine-Learning-Terapan
    ```
2.  **Instal dependensi yang diperlukan:**
    ```bash
    pip install pandas numpy scikit-learn jupyter
    ```
3.  **Unduh Dataset:**
    Unduh file `diabetes.csv` dari [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) dan letakkan di folder root proyek ini.
4.  **Buka Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Buka file `diabetes_prediction.ipynb` dan jalankan sel-selnya secara berurutan.

---

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.

---

## Referensi

* [1] World Health Organization. (2023). Diabetes. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/diabetes
* Pima Indians Diabetes Database: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
