# =========================================================
# IMPLEMENTASI NAÏVE BAYES
# KLASIFIKASI PENYAKIT MIGRAIN
# Dataset: migraine_data.csv
# =========================================================

# 1. Import Library
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 2. Load Dataset
# =========================================================
df = pd.read_csv("migraine_data.csv")

print("Jumlah data dan atribut:", df.shape)
print("\nContoh data:")
print(df.head())


# =========================================================
# 3. Preprocessing Data
# (Menghapus missing value sesuai jurnal)
# =========================================================
print("\nJumlah missing value tiap kolom:")
print(df.isnull().sum())

df = df.dropna()

print("\nJumlah data setelah cleaning:", df.shape)


# =========================================================
# 4. Pisahkan Fitur (X) dan Label (y)
# =========================================================
X = df.drop("Type", axis=1)
y = df["Type"]

print("\nJumlah fitur:", X.shape[1])
print("Jumlah label:", y.nunique())


# =========================================================
# 5. Split Data (70% Train - 30% Test)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y
)

print("\nJumlah data training:", X_train.shape)
print("Jumlah data testing :", X_test.shape)


# =========================================================
# 6. Training Model Gaussian Naïve Bayes
# =========================================================
model = GaussianNB()
model.fit(X_train, y_train)


# =========================================================
# 7. Evaluasi Model - Training
# =========================================================
y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)

print("\n================ HASIL TRAINING ================")
print("Akurasi Training:", train_accuracy)
print("\nClassification Report (Training):")
print(classification_report(y_train, y_train_pred))


# =========================================================
# 8. Evaluasi Model - Testing
# =========================================================
y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n================ HASIL TESTING ================")
print("Akurasi Testing:", test_accuracy)
print("\nClassification Report (Testing):")
print(classification_report(y_test, y_test_pred))


# =========================================================
# 9. Confusion Matrix - Testing
# =========================================================
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Data Testing")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# =========================================================
# 10. Simpan Hasil Prediksi (Opsional)
# =========================================================
hasil_prediksi = X_test.copy()
hasil_prediksi["Actual"] = y_test.values
hasil_prediksi["Predicted"] = y_test_pred

hasil_prediksi.to_csv("hasil_prediksi_migrain.csv", index=False)

print("\nFile hasil_prediksi_migrain.csv berhasil disimpan")


# =========================================================
# SELESAI
# =========================================================
