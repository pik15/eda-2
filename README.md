# 🧬 Analisis Data: Prediksi PCOS (Polycystic Ovary Syndrome)

Proyek ini bertujuan untuk melakukan **analisis data eksploratif (EDA)** dan **pembangunan model machine learning** untuk memprediksi kemungkinan seseorang mengalami PCOS (Polycystic Ovary Syndrome), berdasarkan data klinis.

## 📂 Dataset
- Sumber: [Kaggle - PCOS Dataset (Rotterdam)](https://www.kaggle.com/datasets/lucass0s0/polycystic-ovary-syndrome-pcos)
- Fitur yang digunakan:
  - `Age`
  - `BMI`
  - `Testosterone_Level(ng/dL)`
  - `Antral_Follicle_Count`
- Target: `PCOS_Diagnosis`

## 🧪 Teknologi yang Digunakan
- Python
- Pandas, Matplotlib, Seaborn
- Scikit-learn (RandomForestClassifier)
- Joblib (untuk menyimpan model)

## 📊 Hasil EDA
- Distribusi usia dan BMI divisualisasikan dengan histogram dan boxplot.
- Korelasi antar variabel divisualisasikan dengan heatmap.

## 🧠 Model Machine Learning
- Model: **Random Forest Classifier**
- Akurasi dan evaluasi dilakukan menggunakan confusion matrix dan classification report.

## 📦 Output
- File model: `model_random_forest_pcos.pkl`
- Script Python: `analisis_data.py`
- Laporan PDF: `laporan_pcos.pdf`

## 🚀 Cara Menjalankan
1. Pastikan dependencies terinstall:
   ```bash
   pip install -r requirements.txt
