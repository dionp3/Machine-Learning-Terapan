# -*- coding: utf-8 -*-
"""Submission 1_Machine Learning Terapan.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/184frywl5YuFqFlLx-phG1kn27eEXnKbO

# Laporan Proyek Machine Learning - Dion Prayoga

## Domain Proyek

Kesehatan merupakan sektor penting dalam kehidupan manusia. Deteksi dini terhadap penyakit kronis seperti diabetes sangat diperlukan untuk meningkatkan kualitas hidup pasien. Berdasarkan laporan WHO tahun 2023, lebih dari 422 juta orang di dunia hidup dengan diabetes, dan angka ini terus meningkat setiap tahunnya [1]. Tantangan utama dalam mendeteksi diabetes adalah gejala yang sering kali tidak tampak secara signifikan pada tahap awal. Oleh karena itu, pendekatan berbasis data melalui machine learning dapat digunakan untuk membantu diagnosis awal diabetes secara otomatis dan akurat.

[1] World Health Organization. (2023). Diabetes. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/diabetes

##Business Understanding

###Problem Statements

1. Bagaimana cara memprediksi apakah seseorang menderita diabetes berdasarkan data pemeriksaan medis?

2. Bagaimana performa beberapa algoritma klasifikasi populer seperti Decision Tree dan Random Forest dalam mendeteksi diabetes?

###Goals

1. Mengembangkan model klasifikasi untuk mendeteksi diabetes menggunakan dataset medis.

2. Membandingkan performa dua algoritma klasifikasi (Decision Tree dan Random Forest) untuk menentukan model terbaik berdasarkan metrik evaluasi.

###Solution Statements

1. Menggunakan algoritma Decision Tree dan Random Forest untuk membangun model prediksi.

2. Melakukan hyperparameter tuning pada masing-masing model untuk meningkatkan performa.

3. Menggunakan metrik akurasi, precision, recall, dan F1-score untuk mengevaluasi model.

##Data Understanding

Dataset yang digunakan adalah Pima Indians Diabetes Database yang tersedia secara publik melalui Kaggle:

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database.

Dataset ini memiliki 768 data sampel dengan 8 fitur input dan 1 label output.

###Variabel-Variabel

Berikut adalah deskripsi variabel yang terdapat dalam dataset:

- Pregnancies: Jumlah kehamilan
- Glucose: Kadar glukosa darah
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit triceps (mm)
- Insulin: Level insulin serum (mu U/ml)
- BMI: Indeks massa tubuh (kg/m²)
- DiabetesPedigreeFunction: Riwayat keluarga diabetes
- Age: Usia (tahun)
- Outcome: Label target (0 = Tidak diabetes, 1 = Diabetes)

###Exploratory Data Analysis (EDA) Singkat

- Distribusi Outcome: Terdapat 500 sampel dengan Outcome 0 (tidak diabetes) dan 268 sampel dengan Outcome 1 (diabetes). Ini menunjukkan adanya ketidakseimbangan kelas (class imbalance) yang perlu diperhatikan dalam evaluasi model.
- Nilai 0 yang Tidak Valid: Beberapa fitur seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI memiliki nilai 0 yang secara medis tidak mungkin (misalnya, BMI = 0). Nilai-nilai ini akan dianggap sebagai missing values dan ditangani pada tahap Data Preparation.
"""

import pandas as pd
import numpy as np

# Memuat dataset
df = pd.read_csv('diabetes.csv') # Pastikan nama file sesuai

# Tampilkan 5 baris pertama
print(df.head())

# Melihat tipe data dan jumlah non-null tiap kolom
df.info()

# Melihat semua nama kolom (variabel) dalam dataset
print("\nNama-nama kolom:")
print(df.columns)

"""##Data Preparation

Langkah-langkah data preparation bertujuan untuk membersihkan dan mentransformasi data sehingga siap digunakan untuk pelatihan model.

###Mengatasi Nilai 0 yang Tidak Valid

Nilai 0 pada fitur-fitur seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI akan diganti dengan nilai median dari masing-masing fitur. Pemilihan median lebih robust terhadap outlier dibandingkan mean.
"""

# Mengganti nilai 0 yang tidak valid dengan NaN
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, np.nan)

# Mengisi nilai NaN dengan median masing-masing kolom
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Memisahkan fitur (X) dan target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

"""###Normalisasi Data

Normalisasi data menggunakan MinMaxScaler akan diterapkan untuk menyetarakan skala fitur. Hal ini penting untuk algoritma yang sensitif terhadap skala fitur, meskipun untuk Decision Tree dan Random Forest dampaknya tidak terlalu signifikan. Namun, ini praktik yang baik untuk menjaga konsistensi dan adaptasi jika model lain digunakan.
"""

from sklearn.preprocessing import MinMaxScaler

# Normalisasi data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # Opsional: mengembalikan ke DataFrame untuk memudahkan inspeksi

"""###Memisahkan Data Training dan Testing

Data akan dibagi menjadi training set dan testing set dengan rasio 80:20 menggunakan train_test_split. Penggunaan random_state akan memastikan reproduktibilitas hasil pembagian data.
"""

from sklearn.model_selection import train_test_split

# Memisahkan data menjadi training dan testing set (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Ukuran X_train: {X_train.shape}")
print(f"Ukuran X_test: {X_test.shape}")
print(f"Ukuran y_train: {y_train.shape}")
print(f"Ukuran y_test: {y_test.shape}")

"""##Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan prediksi diabetes. Kami akan menjelaskan tahapan pemodelan, cara kerja algoritma, serta parameter yang digunakan pada proses pemodelan.

###Algoritma 1: Decision Tree

Decision Tree adalah algoritma machine learning non-parametrik yang kuat dan mudah diinterpretasi, digunakan untuk tugas klasifikasi dan regresi. Cara kerjanya menyerupai pohon keputusan (if-else) yang kita buat secara manual. Model ini belajar dengan membagi data menjadi subset-subset yang lebih kecil berdasarkan fitur-fitur input. Pada setiap "node" dalam pohon, algoritma memilih fitur dan threshold yang paling baik memisahkan data ke dalam kelas outcome yang berbeda. Proses pembagian ini berlanjut secara rekursif hingga kriteria berhenti terpenuhi (misalnya, mencapai kedalaman maksimum atau jumlah sampel minimum pada sebuah node). Prediksi untuk data baru dilakukan dengan menelusuri pohon dari "akar" hingga "daun" berdasarkan nilai fitur data tersebut, dan outcome dari daun yang dicapai adalah prediksinya.

####Kelebihan Decision Tree:

- Mudah Diinterpretasi: Strukturnya yang seperti pohon membuatnya mudah dipahami dan divisualisasikan.
- Tidak Membutuhkan Normalisasi/Skala: Tidak sensitif terhadap penskalaan fitur.
- Mampu Menangani Data Kategorikal dan Numerik: Dapat bekerja dengan berbagai jenis data.

####Kekurangan Decision Tree:

- Cenderung Overfitting: Terutama pada pohon yang dalam, mudah beradaptasi terlalu spesifik pada data pelatihan.
- Tidak Robust Terhadap Perubahan Kecil: Sedikit perubahan pada data dapat menghasilkan pohon yang sangat berbeda.
- Bias Terhadap Kelas Dominan: Jika ada ketidakseimbangan kelas, cenderung bias ke kelas mayoritas.

####Proses Pemodelan dan Hyperparameter Tuning

Untuk mendapatkan performa terbaik dari Decision Tree, kami melakukan hyperparameter tuning menggunakan GridSearchCV. Proses ini mencari kombinasi hyperparameter yang optimal dengan mengevaluasi model pada berbagai kombinasi parameter melalui validasi silang (cross-validation).

####Hyperparameter yang di-tuning:

- max_depth: Kedalaman maksimum pohon. Ini mengontrol kompleksitas model dan membantu mencegah overfitting dengan membatasi seberapa jauh pohon dapat tumbuh.
- min_samples_split: Jumlah minimum sampel yang dibutuhkan untuk membagi sebuah node. Jika sebuah node memiliki sampel kurang dari nilai ini, ia tidak akan dibagi lebih lanjut.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Definisi model Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)

# Grid hyperparameter untuk Decision Tree
param_grid_dt = {
    'max_depth': [4, 6, 8, 10, None], # None berarti tidak ada batasan kedalaman
    'min_samples_split': [2, 5, 10]
}

# Melakukan GridSearchCV
grid_search_dt = GridSearchCV(dt_classifier, param_grid_dt, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search_dt.fit(X_train, y_train)

# Model Decision Tree terbaik
best_dt_model = grid_search_dt.best_estimator_
print(f"Best hyperparameters for Decision Tree: {grid_search_dt.best_params_}")

"""####Parameter Terbaik untuk Decision Tree:
Berdasarkan hasil GridSearchCV, parameter terbaik yang didapatkan untuk Decision Tree adalah max_depth=4 dan min_samples_split=2. Nilai-nilai ini akan digunakan oleh best_dt_model untuk membuat prediksi.

###Algoritma 2: Random Forest

Random Forest adalah algoritma ensemble learning yang membangun banyak Decision Tree (disebut juga "pohon keputusan") secara independen dan paralel. Ide dasarnya adalah bahwa banyak pohon yang "agak benar" dan beragam akan secara kolektif menghasilkan prediksi yang lebih akurat dan stabil dibandingkan satu pohon keputusan tunggal. Setiap pohon dalam forest dilatih pada subset data pelatihan yang di-bootstrap (dengan penggantian) dan hanya menggunakan subset fitur acak pada setiap node. Ketika melakukan prediksi, Random Forest mengumpulkan voting dari semua pohon (untuk klasifikasi) atau merata-ratakan prediksi (untuk regresi) untuk menghasilkan outcome akhir. Proses ini membantu mengurangi masalah overfitting yang sering terjadi pada Decision Tree tunggal dan meningkatkan generalisasi model.

####Kelebihan Random Forest:

- Akurasi Tinggi: Umumnya memberikan akurasi yang sangat baik dan robust.
- Mengatasi Overfitting: Karena merupakan algoritma ensemble, sangat efektif dalam mengurangi overfitting dibandingkan Decision Tree tunggal.
- Dapat Menangani Banyak Fitur: Mampu bekerja dengan dataset yang memiliki banyak fitur.
- Kurang Sensitif Terhadap Outlier: Lebih robust terhadap outlier dan missing values.

####Kekurangan Random Forest:

- Kurang Dapat Diinterpretasi: Dibandingkan Decision Tree tunggal, Random Forest lebih sulit untuk diinterpretasi karena melibatkan banyak pohon.
- Membutuhkan Sumber Daya Komputasi Lebih Besar: Pelatihan banyak pohon membutuhkan lebih banyak waktu dan memori.

####Proses Pemodelan dan Hyperparameter Tuning

Sama seperti Decision Tree, kami juga melakukan hyperparameter tuning untuk Random Forest menggunakan GridSearchCV guna menemukan konfigurasi parameter terbaik yang mengoptimalkan performa model.

####Hyperparameter yang di-tuning:

- n_estimators: Jumlah pohon (Decision Tree) dalam forest. Semakin banyak pohon, semakin stabil prediksinya, namun komputasi juga akan meningkat.
- max_depth: Kedalaman maksimum setiap pohon dalam forest. Membatasi pertumbuhan pohon individu untuk mengontrol kompleksitas.
"""

from sklearn.ensemble import RandomForestClassifier

# Definisi model Random Forest
rf_classifier = RandomForestClassifier(random_state=42)

# Grid hyperparameter untuk Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [4, 6, 8, 10, None]
}

# Melakukan GridSearchCV
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train, y_train)

# Model Random Forest terbaik
best_rf_model = grid_search_rf.best_estimator_
print(f"Best hyperparameters for Random Forest: {grid_search_rf.best_params_}")

"""####Parameter Terbaik untuk Random Forest:

Berdasarkan hasil GridSearchCV, parameter terbaik yang didapatkan untuk Random Forest adalah n_estimators=100 dan max_depth=None. Nilai-nilai ini akan digunakan oleh best_rf_model untuk membuat prediksi.

##Evaluation

Pada bagian ini, kami akan menyebutkan metrik evaluasi yang digunakan dan menjelaskan hasil proyek berdasarkan metrik-metrik tersebut.

###Metrik evaluasi yang Digunakan

Dalam proyek klasifikasi deteksi diabetes ini, kami menggunakan beberapa metrik evaluasi untuk mendapatkan gambaran yang komprehensif mengenai performa model, mengingat pentingnya identifikasi kasus positif dan negatif:

- Akurasi (Accuracy): Mengukur proporsi total prediksi yang benar (baik positif maupun negatif). Ini adalah metrik yang intuitif dan sering digunakan, namun bisa menyesatkan pada dataset dengan class imbalance. Formula: Accuracy= TP+TN/TP+TN+FP+FN

- Precision: Mengukur proporsi prediksi positif yang sebenarnya positif. Ini penting ketika biaya false positives (salah mendiagnosis orang sehat sebagai penderita diabetes) sangat tinggi. Formula: Precision=
TP+FP
TP

- Recall (Sensitivity): Mengukur proporsi kasus positif sebenarnya yang berhasil dideteksi oleh model. Ini sangat krusial ketika biaya false negatives (melewatkan diagnosis diabetes pada penderita sebenarnya) sangat tinggi, seperti dalam aplikasi medis. Formula: Recall= TP/TP+FN

- F1 Score: Merupakan rata-rata harmonik dari precision dan recall. Metrik ini sangat berguna ketika ada ketidakseimbangan kelas dan kita ingin keseimbangan antara precision dan recall, bukan hanya salah satunya. Formula: F1Score=2× (Precision×Recall/Precision+Recall)

Dimana:

- TP (True Positive): Jumlah kasus diabetes yang diprediksi benar sebagai diabetes.
- TN (True Negative): Jumlah kasus non-diabetes yang diprediksi benar sebagai non-diabetes.
- FP (False Positive): Jumlah kasus non-diabetes yang salah diprediksi sebagai diabetes.
- FN (False Negative): Jumlah kasus diabetes yang salah diprediksi sebagai non-diabetes.

###Hasil Proyek Berdasarkan Metrik Evaluasi

Setelah melatih dan menyetel kedua model, kami mengevaluasi performa mereka pada testing set yang belum pernah dilihat model sebelumnya. Berikut adalah hasil evaluasinya:
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # <--- BARIS INI YANG PENTING DITAMBAHKAN

# Prediksi menggunakan model Decision Tree terbaik
y_pred_dt = best_dt_model.predict(X_test)

# Evaluasi Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("--- Evaluasi Decision Tree ---")
print(f"Akurasi: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1 Score: {f1_dt:.4f}")
print("-" * 30)

# Prediksi menggunakan model Random Forest terbaik
y_pred_rf = best_rf_model.predict(X_test)

# Evaluasi Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("--- Evaluasi Random Forest ---")
print(f"Akurasi: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print("-" * 30)

"""###Analisis Hasil:

Dari tabel hasil evaluasi, kita dapat mengamati perbedaan performa yang menarik antara Decision Tree dan Random Forest:

- Akurasi: Random Forest menunjukkan akurasi yang lebih tinggi (0.7403) dibandingkan Decision Tree (0.7078). Ini berarti Random Forest secara keseluruhan membuat prediksi benar lebih banyak.
- Precision: Random Forest memiliki precision yang lebih baik (0.6316) daripada Decision Tree (0.5676). Ini mengindikasikan bahwa ketika Random Forest memprediksi seseorang menderita diabetes, probabilitas prediksinya benar lebih tinggi. Ini penting untuk meminimalkan false positives, di mana seseorang yang sehat salah didiagnosis diabetes.
- Recall: Decision Tree menunjukkan recall yang significantly lebih tinggi (0.7636) dibandingkan Random Forest (0.6545). Ini adalah temuan krusial: Decision Tree lebih baik dalam mengidentifikasi sebagian besar kasus diabetes yang sebenarnya. Dalam konteks deteksi penyakit seperti diabetes, recall yang tinggi sangat penting untuk meminimalkan false negatives (kasus diabetes yang tidak terdeteksi), karena melewatkan diagnosis dapat memiliki konsekuensi kesehatan yang serius.
- F1 Score: Decision Tree sedikit lebih unggul dalam F1 Score (0.6512) dibandingkan Random Forest (0.6429). Meskipun selisihnya tipis, F1 Score yang lebih tinggi pada Decision Tree menunjukkan keseimbangan yang sedikit lebih baik antara precision dan recall dalam kasus ini, terutama didorong oleh recall yang sangat tinggi.

####Kesimpulan dari Analisis:

Pilihan model terbaik sangat bergantung pada prioritas kasus penggunaan. Jika tujuan utamanya adalah untuk meminimalkan false negatives (memastikan semua kasus diabetes terdeteksi, bahkan jika ada beberapa false positives), maka Decision Tree mungkin menjadi pilihan yang lebih unggul karena recall-nya yang tinggi. Ini relevan dalam skenario skrining awal di mana deteksi dini lebih diutamakan.

Namun, jika tujuannya adalah untuk memiliki akurasi dan precision yang lebih baik secara keseluruhan (meminimalkan false positives dan false negatives secara seimbang, dengan sedikit fokus pada precision), maka Random Forest akan lebih cocok. Random Forest umumnya lebih robust dan kurang rentan terhadap overfitting dibandingkan Decision Tree tunggal, yang bisa menjadi keuntungan dalam skenario dunia nyata.

Dalam proyek ini, dengan mempertimbangkan pentingnya mendeteksi kasus positif (diabetes), Decision Tree menunjukkan keunggulan dalam hal recall, yang sangat berharga dalam konteks medis.

##Kesimpulan

Proyek ini berhasil mengembangkan model klasifikasi untuk deteksi diabetes menggunakan dataset Pima Indians Diabetes. Setelah melalui tahapan data preparation yang meliputi penanganan nilai missing dan normalisasi, dua algoritma klasifikasi, Decision Tree dan Random Forest, dilatih dan dievaluasi.

Hyperparameter tuning dengan GridSearchCV memainkan peran penting dalam mengoptimalkan performa kedua model. Dari hasil evaluasi, Decision Tree menunjukkan recall yang lebih tinggi, menjadikannya pilihan yang kuat jika prioritas utama adalah mendeteksi sebanyak mungkin kasus diabetes (meminimalkan false negatives). Di sisi lain, Random Forest menawarkan akurasi dan precision yang lebih tinggi, yang bermanfaat jika fokusnya adalah memastikan kebenaran prediksi positif dan mengurangi false positives. Pemilihan model akhir akan bergantung pada prioritas klinis dan dampak dari false positives dan false negatives dalam aplikasi praktis.

##Saran untuk Pengembangan Lebih Lanjut

1. Penanganan Class Imbalance Lanjutan: Meskipun F1-score sudah digunakan sebagai metrik yang robust, teknik penanganan class imbalance seperti SMOTE (Synthetic Minority Over-sampling Technique), ADASYN, atau undersampling dapat dieksplorasi lebih lanjut untuk melihat apakah performa model, terutama recall dari Random Forest, dapat ditingkatkan.
2. Eksplorasi Algoritma Lain: Mencoba algoritma klasifikasi lain seperti Support Vector Machine (SVM), Logistic Regression, Gradient Boosting Machines (seperti XGBoost, LightGBM), atau bahkan Neural Networks untuk membandingkan performa lebih lanjut dan melihat potensi peningkatan.
3. Validasi Eksternal: Menguji model pada dataset diabetes lain yang independen dari sumber atau populasi yang berbeda dapat memberikan validasi yang lebih kuat terhadap generalisasi dan robustness model.
4. Interpretasi Model: Menggunakan teknik interpretasi model seperti SHAP values atau LIME akan sangat bermanfaat untuk memahami fitur mana yang paling berpengaruh dalam prediksi diabetes oleh model. Wawasan ini tidak hanya meningkatkan kepercayaan pada model tetapi juga dapat memberikan temuan medis yang berharga bagi para profesional kesehatan.
5. Optimasi Threshold: Karena perbedaan recall dan precision yang signifikan, melakukan optimasi threshold klasifikasi (misalnya, pada probabilitas prediksi) dapat membantu menyesuaikan model untuk kebutuhan spesifik, misalnya, untuk mendapatkan recall yang lebih tinggi tanpa mengorbankan precision secara drastis, atau sebaliknya.
"""