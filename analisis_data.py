import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

# 1. Load data
df = pd.read_csv('pcos_rotterdam_balanceado.csv', encoding='ISO-8859-1')
df.columns = df.columns.str.strip()

print("🔹 5 Baris Pertama:")
print(df.head())

# 2. Info umum
print("\n🔹 Info Dataset:")
print(df.info())

print("\n🔹 Deskripsi Statistik:")
print(df.describe())

print("\n🔹 Cek Missing Values:")
print(df.isnull().sum())

print("\n🔹 Cek Duplikat:")
print(df.duplicated().sum())

# Hapus duplikat
df = df.drop_duplicates()

# 3. Visualisasi
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], kde=True)
plt.title('Distribusi Umur')
plt.xlabel('Age')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['BMI'])
plt.title('Boxplot BMI')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antar Variabel')
plt.tight_layout()
plt.show()

# 4. Modeling
fitur_penting = ['Age', 'BMI', 'Testosterone_Level(ng/dL)', 'Antral_Follicle_Count']
X = df[fitur_penting]
y = df['PCOS_Diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)

print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Simpan model
dump(model, 'model_random_forest_pcos.pkl')
print("\n✅ Model berhasil disimpan sebagai 'model_random_forest_pcos.pkl'")
