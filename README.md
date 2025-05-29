# Proyek Prediksi Diabetes dengan Machine Learning

Deteksi dini penyakit kronis seperti diabetes sangat penting untuk meningkatkan kualitas hidup pasien. Proyek ini bertujuan mengembangkan model *machine learning* yang mampu memprediksi apakah seseorang menderita diabetes berdasarkan data pemeriksaan medis. Kami menggunakan algoritma **Decision Tree** dan **Random Forest** serta membandingkan performanya untuk menemukan model terbaik yang sesuai dengan kebutuhan di domain kesehatan.

-----

## Daftar Isi

  * [Domain Proyek](https://www.google.com/search?q=%23domain-proyek)
  * [Business Understanding](https://www.google.com/search?q=%23business-understanding)
  * [Data Understanding](https://www.google.com/search?q=%23data-understanding)
  * [Data Preparation](https://www.google.com/search?q=%23data-preparation)
  * [Modeling](https://www.google.com/search?q=%23modeling)
  * [Evaluation](https://www.google.com/search?q=%23evaluation)
  * [Kesimpulan](https://www.google.com/search?q=%23kesimpulan)
  * [Saran Pengembangan](https://www.google.com/search?q=%23saran-pengembangan)
  * [Cara Menjalankan Proyek](https://www.google.com/search?q=%23cara-menjalankan-proyek)
  * [Lisensi](https://www.google.com/search?q=%23lisensi)
  * [Referensi](https://www.google.com/search?q=%23referensi)

-----

## Domain Proyek

**Kesehatan** merupakan sektor krusial dalam kehidupan manusia. Deteksi dini terhadap penyakit kronis seperti **diabetes** sangat diperlukan untuk meningkatkan kualitas hidup pasien. Menurut laporan WHO tahun 2023, lebih dari 422 juta orang di dunia hidup dengan diabetes, dan angka ini terus meningkat setiap tahunnya [1]. Gejala yang sering kali tidak tampak signifikan pada tahap awal menjadi tantangan utama dalam mendeteksi diabetes. Oleh karena itu, pendekatan berbasis data melalui *machine learning* dapat membantu diagnosis awal diabetes secara otomatis dan akurat.

-----

## Business Understanding

### Problem Statements

1.  Bagaimana cara memprediksi apakah seseorang menderita diabetes berdasarkan data pemeriksaan medis?
2.  Bagaimana performa beberapa algoritma klasifikasi populer seperti *Decision Tree* dan *Random Forest* dalam mendeteksi diabetes, dan mana yang paling sesuai dengan kebutuhan domain kesehatan?

### Goals

1.  Mengembangkan model klasifikasi untuk mendeteksi diabetes menggunakan *dataset* medis.
2.  Membandingkan performa dua algoritma klasifikasi (*Decision Tree* dan *Random Forest*) untuk menentukan model terbaik berdasarkan metrik evaluasi, dengan mempertimbangkan prioritas dalam domain kesehatan.

### Solution Statements

1.  Menggunakan algoritma *Decision Tree* dan *Random Forest* untuk membangun model prediksi.
2.  Melakukan *hyperparameter tuning* pada masing-masing model untuk meningkatkan performa.
3.  Menggunakan metrik **akurasi, *precision, recall*, dan *F1-score*** untuk mengevaluasi model, dengan fokus pada *recall* karena konsekuensi tinggi dari *false negatives* dalam deteksi penyakit.

-----

## Data Understanding

Dataset yang digunakan adalah **Pima Indians Diabetes Database**, tersedia secara publik melalui Kaggle: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Dataset ini memiliki 768 data sampel dengan 8 fitur *input* dan 1 label *output*.

### Variabel-Variabel

Berikut adalah deskripsi variabel yang terdapat dalam *dataset*:

  * `Pregnancies`: Jumlah kehamilan
  * `Glucose`: Kadar glukosa darah
  * `BloodPressure`: Tekanan darah diastolik (mm Hg)
  * `SkinThickness`: Ketebalan lipatan kulit triceps (mm)
  * `Insulin`: Level insulin serum (mu U/ml)
  * `BMI`: Indeks massa tubuh (kg/m²)
  * `DiabetesPedigreeFunction`: Riwayat keluarga diabetes
  * `Age`: Usia (tahun)
  * `Outcome`: Label target (0 = Tidak diabetes, 1 = Diabetes)

### Kondisi Data dan *Exploratory Data Analysis* (EDA)

Setelah memuat dataset, kami melakukan eksplorasi awal untuk memahami karakteristik data.

**Distribusi *Outcome***:
Data menunjukkan adanya ketidakseimbangan kelas (*class imbalance*) pada variabel target `Outcome`. Sekitar 65% sampel adalah non-diabetes dan 35% adalah diabetes. Ini akan memengaruhi pemilihan metrik evaluasi.

**Nilai 0 yang Tidak Valid**:
Beberapa fitur numerik seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` mengandung nilai 0 yang secara medis tidak mungkin. Nilai-nilai ini diperlakukan sebagai *missing values* dan akan ditangani pada tahap *Data Preparation*.

**Distribusi Fitur Berdasarkan *Outcome***:
Analisis distribusi fitur-fitur penting berdasarkan label `Outcome` memberikan wawasan tentang relevansi fitur. Terlihat bahwa fitur seperti `Glucose`, `BMI`, dan `Age` menunjukkan perbedaan distribusi yang signifikan antara kedua kelompok, mengindikasikan potensi prediktifnya.

**Korelasi Antar Fitur dan dengan *Outcome***:
Matriks korelasi membantu memahami hubungan linier antar fitur dan seberapa kuat hubungan fitur dengan variabel target `Outcome`. Fitur `Glucose` menunjukkan korelasi positif tertinggi dengan `Outcome`, menggarisbawahi pentingnya kadar glukosa dalam diagnosis diabetes.

-----

## Data Preparation

Langkah-langkah *data preparation* bertujuan membersihkan, mentransformasi, dan membagi data sehingga siap digunakan untuk pelatihan model.

1.  **Pemisahan Fitur (X) dan Target (y)**:
    Kami memisahkan dataset menjadi variabel fitur (X) yang akan digunakan untuk prediksi, dan variabel target (y) yaitu `Outcome`.

2.  **Mengatasi Nilai 0 yang Tidak Valid**:
    Nilai 0 pada fitur `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` diganti dengan nilai **median** dari masing-masing fitur. Pemilihan median lebih *robust* terhadap *outlier* dibandingkan *mean* untuk mengisi *missing values*.

3.  **Normalisasi Data**:
    **Normalisasi data** menggunakan `MinMaxScaler` diterapkan untuk menyetarakan skala fitur. Hal ini penting untuk beberapa algoritma yang sensitif terhadap skala fitur dan membantu proses pembelajaran model menjadi lebih stabil.

4.  **Memisahkan Data *Training* dan *Testing***:
    Data dibagi menjadi *training set* (80%) dan *testing set* (20%) menggunakan `train_test_split`. Penggunaan `random_state=42` memastikan reproduktibilitas hasil pembagian data, sehingga hasil yang diperoleh konsisten setiap kali kode dijalankan.

-----

## Modeling

Tahapan ini membahas model *machine learning* yang digunakan untuk menyelesaikan permasalahan prediksi diabetes, termasuk cara kerja algoritma dan hasil *hyperparameter tuning*.

### Algoritma 1: *Decision Tree*

*Decision Tree* adalah algoritma *machine learning* non-parametrik yang kuat dan mudah diinterpretasi, digunakan untuk tugas klasifikasi dan regresi. Cara kerjanya menyerupai serangkaian aturan `if-else` yang membentuk struktur pohon. Model ini belajar dengan membagi data menjadi subset-subset yang lebih kecil berdasarkan fitur-fitur *input*, memilih fitur dan *threshold* terbaik pada setiap "node" untuk memisahkan kelas *outcome*. Prediksi untuk data baru dilakukan dengan menelusuri pohon dari "akar" hingga "daun".

**Kelebihan *Decision Tree*:**

  * **Mudah Diinterpretasi**: Strukturnya yang seperti pohon membuatnya mudah dipahami dan divisualisasikan, yang sangat berharga di domain kesehatan untuk menjelaskan keputusan model.
  * **Tidak Membutuhkan Normalisasi/Skala**: Tidak sensitif terhadap penskalaan fitur.
  * **Mampu Menangani Data Kategorikal dan Numerik**: Dapat bekerja dengan berbagai jenis data.

**Kekurangan *Decision Tree*:**

  * **Cenderung *Overfitting***: Terutama pada pohon yang dalam, mudah beradaptasi terlalu spesifik pada data pelatihan dan gagal menggeneralisasi ke data baru.
  * **Tidak Robust Terhadap Perubahan Kecil**: Sedikit perubahan pada data dapat menghasilkan pohon yang sangat berbeda.
  * **Bias Terhadap Kelas Dominan**: Jika ada ketidakseimbangan kelas, model cenderung bias ke kelas mayoritas.

**Proses Pemodelan dan *Hyperparameter Tuning***

Untuk mendapatkan performa terbaik dari *Decision Tree*, kami melakukan *hyperparameter tuning* menggunakan **`GridSearchCV`**. Proses ini secara sistematis mencari kombinasi *hyperparameter* yang optimal dengan mengevaluasi model pada berbagai kombinasi parameter melalui validasi silang (cross-validation) pada *training set*.

*Hyperparameter* yang di-*tuning*:

  * `max_depth`: Kedalaman maksimum pohon. Ini mengontrol kompleksitas model dan membantu mencegah *overfitting*.
  * `min_samples_split`: Jumlah minimum sampel yang dibutuhkan untuk membagi sebuah *node*.

**Parameter Terbaik untuk *Decision Tree***:
Berdasarkan hasil `GridSearchCV`, parameter terbaik yang didapatkan untuk *Decision Tree* adalah **`max_depth=4`** dan **`min_samples_split=2`**. Model akhir *Decision Tree* dibangun menggunakan parameter ini.

### Algoritma 2: *Random Forest*

*Random Forest* adalah algoritma *ensemble learning* yang membangun banyak *Decision Tree* secara independen dan paralel. Ide dasarnya adalah bahwa banyak pohon yang "agak benar" dan beragam akan secara kolektif menghasilkan prediksi yang lebih akurat dan stabil dibandingkan satu pohon keputusan tunggal. Setiap pohon dalam *forest* dilatih pada subset data pelatihan yang di-*bootstrap* (dengan penggantian) dan hanya menggunakan subset fitur acak pada setiap *node*. Ketika melakukan prediksi, *Random Forest* mengumpulkan *voting* dari semua pohon (untuk klasifikasi) atau merata-ratakan prediksi (untuk regresi) untuk menghasilkan *outcome* akhir. Proses ini membantu mengurangi masalah *overfitting* yang sering terjadi pada *Decision Tree* tunggal dan meningkatkan generalisasi model.

**Kelebihan *Random Forest*:**

  * **Akurasi Tinggi**: Umumnya memberikan akurasi yang sangat baik dan *robust* terhadap *noise* pada data.
  * **Mengatasi *Overfitting***: Karena merupakan algoritma *ensemble*, sangat efektif dalam mengurangi *overfitting* dibandingkan *Decision Tree* tunggal.
  * **Dapat Menangani Banyak Fitur**: Mampu bekerja dengan dataset yang memiliki banyak fitur dan mengidentifikasi fitur penting.
  * **Kurang Sensitif Terhadap *Outlier***: Lebih *robust* terhadap *outlier* dan *missing values*.

**Kekurangan *Random Forest*:**

  * **Kurang Dapat Diinterpretasi**: Dibandingkan *Decision Tree* tunggal, *Random Forest* lebih sulit untuk diinterpretasi ("black box") karena melibatkan agregasi dari banyak pohon.
  * **Membutuhkan Sumber Daya Komputasi Lebih Besar**: Pelatihan banyak pohon membutuhkan lebih banyak waktu dan memori, terutama pada dataset yang sangat besar.

**Proses Pemodelan dan *Hyperparameter Tuning***

Sama seperti *Decision Tree*, kami juga melakukan *hyperparameter tuning* untuk *Random Forest* menggunakan `GridSearchCV` guna menemukan konfigurasi parameter terbaik yang mengoptimalkan performa model pada *training set*.

*Hyperparameter* yang di-*tuning*:

  * `n_estimators`: Jumlah pohon (*Decision Tree*) dalam *forest*. Semakin banyak pohon, semakin stabil prediksinya, namun komputasi juga akan meningkat.
  * `max_depth`: Kedalaman maksimum setiap pohon dalam *forest*. Membatasi pertumbuhan pohon individu untuk mengontrol kompleksitas.

**Parameter Terbaik untuk *Random Forest***:
Berdasarkan hasil `GridSearchCV`, parameter terbaik yang didapatkan untuk *Random Forest* adalah **`n_estimators=100`** dan **`max_depth=None`**. Model akhir *Random Forest* dibangun menggunakan parameter ini.

-----

## Evaluation

Pada bagian ini, kami akan menjelaskan metrik evaluasi yang digunakan dan memaparkan hasil proyek berdasarkan metrik-metrik tersebut, serta menentukan model terbaik sebagai solusi.

### Metrik Evaluasi yang Digunakan

Dalam proyek klasifikasi deteksi diabetes ini, kami menggunakan beberapa metrik evaluasi untuk mendapatkan gambaran yang komprehensif mengenai performa model. Pemilihan metrik ini relevan dengan konteks domain kesehatan, di mana identifikasi kasus positif (diabetes) dan negatif (non-diabetes) memiliki bobot konsekuensi yang berbeda:

  * **Akurasi (Accuracy)**:
    Mengukur proporsi total prediksi yang benar (baik positif maupun negatif). Ini adalah metrik yang intuitif, namun bisa menyesatkan pada dataset dengan *class imbalance*.
    Formula: $Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}$
  * **Precision**:
    Mengukur proporsi prediksi positif yang sebenarnya positif. Penting ketika biaya *false positives* (salah mendiagnosis orang sehat sebagai penderita diabetes) sangat tinggi, seperti kecemasan yang tidak perlu atau tes lanjutan.
    Formula: $Precision = \\frac{TP}{TP + FP}$
  * **Recall (Sensitivity)**:
    Mengukur proporsi kasus positif sebenarnya yang berhasil dideteksi oleh model. Sangat krusial ketika biaya *false negatives* (melewatkan diagnosis diabetes pada penderita sebenarnya) sangat tinggi, karena dapat menyebabkan keterlambatan penanganan dan komplikasi serius.
    Formula: $Recall = \\frac{TP}{TP + FN}$
  * **F1 Score**:
    Merupakan rata-rata harmonik dari *precision* dan *recall*. Metrik ini sangat berguna ketika ada ketidakseimbangan kelas dan kita ingin keseimbangan yang baik antara *precision* dan *recall*.
    Formula: $F1 Score = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$

*Dimana:*

  * *TP (True Positive)*: Jumlah kasus diabetes yang diprediksi benar sebagai diabetes.
  * *TN (True Negative)*: Jumlah kasus non-diabetes yang diprediksi benar sebagai non-diabetes.
  * *FP (False Positive)*: Jumlah kasus non-diabetes yang salah diprediksi sebagai diabetes.
  * *FN (False Negative)*: Jumlah kasus diabetes yang salah diprediksi sebagai non-diabetes.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Setelah melatih dan menyetel kedua model, kami mengevaluasi performa mereka pada *testing set* yang belum pernah dilihat model sebelumnya.

| Model         | Akurasi | Precision | Recall | F1 Score |
| :------------ | :------ | :-------- | :----- | :------- |
| Decision Tree | 0.7078  | 0.5676    | 0.7636 | 0.6512   |
| Random Forest | 0.7403  | 0.6316    | 0.6545 | 0.6429   |

**Analisis Hasil Proyek dan Pemilihan Model Terbaik:**

  * **Akurasi**: **Random Forest** menunjukkan akurasi yang lebih tinggi (0.7403) dibandingkan *Decision Tree* (0.7078). Ini menandakan *Random Forest* secara keseluruhan membuat prediksi yang benar lebih sering.
  * **Precision**: **Random Forest** memiliki *precision* yang lebih baik (0.6316) daripada *Decision Tree* (0.5676). Ini berarti *Random Forest* lebih jarang memberikan *false positives* (salah mendiagnosis orang sehat sebagai penderita diabetes).
  * **Recall**: **Decision Tree** unggul secara signifikan dalam *recall* (0.7636) dibandingkan *Random Forest* (0.6545). Ini adalah metrik yang sangat kritis dalam deteksi penyakit; *recall* yang tinggi berarti *Decision Tree* lebih baik dalam mengidentifikasi sebagian besar individu yang benar-benar menderita diabetes (meminimalkan *false negatives*). Melewatkan diagnosis diabetes (*false negative*) dapat berakibat fatal bagi pasien karena keterlambatan penanganan.
  * **F1 Score**: *Decision Tree* memiliki *F1 Score* yang sedikit lebih tinggi (0.6512) dibandingkan *Random Forest* (0.6429). Ini menunjukkan bahwa *Decision Tree* mencapai keseimbangan yang sedikit lebih baik antara *precision* dan *recall* dalam skenario ini, didorong oleh *recall* yang kuat.

**Pemilihan Model Terbaik sebagai Solusi**: Mengingat bahwa dalam domain kesehatan, **konsekuensi dari *false negatives* (melewatkan diagnosis diabetes) seringkali lebih serius daripada *false positives*** (salah mendiagnosis, yang dapat dikoreksi dengan pemeriksaan lebih lanjut), **Decision Tree dengan *recall* yang lebih tinggi (`0.7636`) dipertimbangkan sebagai model yang lebih diutamakan sebagai solusi awal** untuk tujuan skrining yang sensitif.

-----

## Kesimpulan

Proyek ini berhasil membangun dan mengevaluasi model klasifikasi untuk deteksi diabetes menggunakan *dataset* Pima Indians Diabetes. Setelah melalui tahapan *data preparation* yang komprehensif, algoritma **Decision Tree** dan **Random Forest** dilatih dan dioptimalkan melalui *hyperparameter tuning*.

**Decision Tree**, dengan parameter terbaik `max_depth=4` dan `min_samples_split=2`, menunjukkan *recall* yang lebih tinggi (`0.7636`), menjadikannya model yang sangat baik dalam mendeteksi sebagian besar kasus diabetes yang sebenarnya. Ini adalah keunggulan kritis dalam domain kesehatan untuk meminimalkan *false negatives*.

**Random Forest**, dengan parameter terbaik `n_estimators=100` dan `max_depth=None`, memberikan akurasi (`0.7403`) dan *precision* (`0.6316`) yang lebih tinggi secara keseluruhan. Model ini lebih *robust* dan unggul dalam mengurangi *false positives*.

**Pemilihan model akhir sangat bergantung pada kebutuhan spesifik dan dampak kesalahan prediksi di lingkungan klinis.** Untuk prioritas **sensitivitas tinggi pada skrining awal (meminimalkan *false negatives*)**, *Decision Tree* adalah pilihan yang tepat. Untuk kebutuhan **akurasi dan *precision* yang lebih tinggi secara keseluruhan** dengan *robustness* model, *Random Forest* bisa menjadi alternatif yang kuat.

-----

## Saran Pengembangan

Untuk meningkatkan proyek ini lebih lanjut dan memperkuat aplikasinya di dunia nyata, beberapa area yang dapat dieksplorasi meliputi:

1.  **Penanganan *Class Imbalance* Lanjutan**: Implementasi teknik seperti **SMOTE** (*Synthetic Minority Over-sampling Technique*) atau *oversampling/undersampling* lainnya untuk meningkatkan performa model, terutama *recall* dari *Random Forest*.
2.  **Eksplorasi Algoritma Lain**: Mencoba algoritma klasifikasi lain seperti *Support Vector Machine* (SVM), *Logistic Regression*, atau *Gradient Boosting Machines* (seperti XGBoost, LightGBM) untuk perbandingan performa yang lebih luas.
3.  **Validasi Eksternal**: Menguji model pada *dataset* diabetes lain yang independen dari sumber atau populasi yang berbeda untuk memastikan generalisasi dan *robustness* model.
4.  **Interpretasi Model**: Menggunakan teknik interpretasi model seperti **SHAP values** atau **LIME** untuk memahami fitur mana yang paling berpengaruh dalam prediksi diabetes.
5.  **Optimasi *Threshold***: Menyesuaikan *threshold* klasifikasi untuk menyeimbangkan *precision* dan *recall* sesuai kebutuhan aplikasi.

-----

## Cara Menjalankan Proyek

1.  **Clone repositori ini:**
    ```bash
    git clone https://github.com/dionp3/Machine-Learning-Terapan.git
    cd Machine-Learning-Terapan
    ```
2.  **Instal dependensi yang diperlukan:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Unduh Dataset:**
    Unduh file `diabetes.csv` dari [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) dan letakkan di folder root proyek ini.
4.  **Buka Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Buka file `diabetes_prediction.ipynb` dan jalankan sel-selnya secara berurutan.

-----

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.

-----

## Referensi

  * [1] World Health Organization. (2023). Diabetes. [Online]. Available: [https://www.who.int/news-room/fact-sheets/detail/diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
