# Tugas-Besar-PCD_Tiar-Firman-Syah
# Card Image Classification using MobileNetV3

Project ini merupakan implementasi **klasifikasi citra kartu permainan** dengan mengombinasikan teknik **pengolahan citra digital** dan metode **deep learning** berbasis **MobileNetV3**. Sistem dikembangkan dalam bentuk **aplikasi web** menggunakan Flask, di mana pengguna dapat mengunggah gambar kartu dan memperoleh hasil prediksi kelas secara langsung.

---

## Deskripsi Singkat

Klasifikasi citra kartu permainan merupakan permasalahan multikelas yang menantang karena banyak kartu memiliki karakteristik visual yang mirip. Untuk mengatasi hal tersebut, project ini menerapkan beberapa teknik prapemrosesan citra untuk meningkatkan kualitas input sebelum dilakukan klasifikasi menggunakan model deep learning.

Model MobileNetV3 dipilih karena memiliki performa yang baik dengan kompleksitas komputasi yang rendah. Proses ekstraksi fitur dilakukan secara otomatis oleh lapisan konvolusional model, sementara fine-tuning dilakukan pada lapisan klasifikasi akhir.

---

## Metode yang Digunakan

### 1. Prapemrosesan Citra
Tahapan prapemrosesan meliputi:
- Gaussian Filter (reduksi noise)
- Median Filter (menghilangkan noise impulsif)
- CLAHE (peningkatan kontras lokal)
- Contrast Stretching (peningkatan kontras global)

### 2. Ekstraksi Fitur
Ekstraksi fitur dilakukan secara otomatis oleh **Convolutional Neural Network (CNN)** MobileNetV3 melalui lapisan konvolusional.

### 3. Klasifikasi
- Model: **MobileNetV3**
- Pendekatan: Transfer Learning dan Fine-Tuning ringan pada lapisan klasifikasi
- Jumlah kelas: **53 kelas kartu**

---

## Hasil Evaluasi

Model dievaluasi menggunakan data uji dengan metrik sebagai berikut:
- **Accuracy:** 89.06%
- **Precision (macro avg):** 0.8970
- **Recall (macro avg):** 0.8906
- **F1-score (macro avg):** 0.8834

Evaluasi juga dilengkapi dengan confusion matrix untuk menganalisis kesalahan klasifikasi antar kelas.

---

## Implementasi Aplikasi

Aplikasi dikembangkan menggunakan **Flask** dan menyediakan fitur:
- Upload gambar kartu
- Visualisasi hasil preprocessing
- Prediksi kelas kartu
- Confidence score dan Top-3 prediction

---

---

## Link Dataset
https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

