# ShopeeSense: Analisis Sentimen Ulasan Aplikasi Shopee (Google Play Store)
### Dataset Ulasan Bahasa Indonesia dengan Preprocessing Kata Gaul & Informal

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-2%2C000%20Ulasan-orange)](data/raw/)
[![Status](https://img.shields.io/badge/Status-Final%20Submission-brightgreen)](.)

---

## Deskripsi Dataset

Dataset ini berisi **2.000 ulasan pengguna** aplikasi Shopee dari Google Play Store dalam **Bahasa Indonesia**, lengkap dengan proses preprocessing yang menangani ragam bahasa informal, singkatan, dan ekspresi gaul khas ulasan e-commerce.

| Atribut | Nilai |
|---|---|
| **Jumlah Ulasan** | 2.000 baris |
| **Kolom Utama** | `review`, `rating`, `tanggal`, `label` |
| **Bahasa** | Bahasa Indonesia (informal & formal) |
| **Sumber** | Google Play Store (scraping manual) |
| **Label Sentimen** | Positif, Netral, Negatif |

### Distribusi Label

| Label | Jumlah | Persentase |
|---|---|---|
| ✅ Positif | 1.404 | 70,2% |
| ⚠️ Negatif | 503 | 25,1% |
| 🔵 Netral | 93 | 4,7% |

---

## Struktur Folder

```text
 ShopeeSense/
├── data/
│   ├── raw/
│   │   ├── shopee_reviews_raw.csv          ← Dataset mentah (belum diproses)
│   │   └── shopee_reviews_2000.csv         ← Dataset mentah 2000 baris
│   └── processed/
│       └── shopee_reviews_processed.csv    ← Dataset bersih (siap ML/DL)
│
├── scripts/
│   └── preprocessing.py                    ← Pipeline pembersihan lengkap
│
├── dictionary/
│   └── kamus_gaul_shopee.json              ← Kamus 85+ kata gaul & singkatan
│
├── assets/
│   └── figures/
│       ├── wordcloud_sebelum_sesudah.png   ← Visualisasi Word Cloud
│       ├── distribusi_sentimen_kata.png    ← Grafik distribusi label
│       └── diagram_metodologi.png          ← Diagram alur metodologi
│
├── docs/
│   ├── jurnal_artikel.md                   ← Draf jurnal ilmiah
│   └── ringkasan_penelitian.txt            ← Ringkasan singkat
│
├── notebooks/
│   ├── analisis_sentimen_shopee.ipynb      ← Notebook ML klasik (Naive Bayes, dll)
│   └── next_executed.ipynb                 ← Notebook Deep Learning (BiLSTM)
│
└── README.md                               ← Dokumentasi ini
```

---

## Tahapan Preprocessing

Pipeline preprocessing dirancang berlapis untuk memaksimalkan kualitas teks:

```
Raw Text
   │
   ▼
┌─────────────────────────────────┐
│  1. Case Folding                │  → Semua huruf kecil
│     "BAGUS Banget!" → "bagus banget!"
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  2. Text Cleaning               │  → Hapus URL, emoji, angka, simbol
│     "https://t.co/abc 😍🔥" → ""
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  3. Normalisasi Kata Gaul       │  → Menggunakan kamus_gaul_shopee.json
│     "ga bgt ngeselin" → "tidak banget menjengkelkan"
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  4. Pelabelan Sentimen          │  → Berdasarkan rating bintang
│     Rating 4-5 → Positif       │
│     Rating 3   → Netral        │
│     Rating 1-2 → Negatif       │
└────────────────┬────────────────┘
                 │
                 ▼
            Clean Dataset
```

### Mengapa Kamus Kata Gaul?

Ulasan pengguna e-commerce mengandung banyak **ekspresi informal** yang tidak tertangani oleh kamus standar:
- **Singkatan**: `gak` → `tidak`, `bgt` → `banget`, `udh` → `sudah`
- **Ekspresi gaul**: `mantul` → `mantap betul`, `zonk` → `mengecewakan`
- **Istilah e-commerce**: `cod`, `free ongkir`, `cashback`, `flash sale`

---

## Cara Menjalankan

### 1. Clone Repositori

```bash
git clone https://github.com/[USERNAME]/ShopeeSense.git
cd ShopeeSense
```

### 2. Install Dependensi

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn
```

### 3. Jalankan Preprocessing

```bash
python scripts/preprocessing.py
```

Output yang dihasilkan:
- `data/processed/shopee_reviews_processed_v2.csv` — Dataset bersih
- `assets/figures/wordcloud_sebelum_sesudah.png` — Word Cloud perbandingan
- `assets/figures/distribusi_sentimen_kata.png` — Grafik distribusi

### 4. Jalankan Notebook Utama

```bash
jupyter notebook notebooks/analisis_sentimen_shopee.ipynb
```

---

## Visualisasi

### Word Cloud: Sebelum vs Sesudah Preprocessing

> Menunjukkan efektivitas pembersihan teks dari noise menuju kata-kata bermakna.

*(Lihat `assets/figures/wordcloud_sebelum_sesudah.png` setelah menjalankan skrip)*

### Hasil Model Machine Learning

| Model | Accuracy | F1-Score |
|---|---|---|
| **Naive Bayes** ⭐ | **80.30%** | **81.24%** |
| BiLSTM (Deep Learning) | 80.81% | 80.19% |
| Logistic Regression | 79.04% | 79.93% |
| SVM (LinearSVC) | 78.79% | 78.74% |
| SVM (Tuned) | 77.27% | 77.68% |

**Model Terbaik: Naive Bayes** — dipilih karena performa tertinggi pada metrik Accuracy dan F1-Score Macro.

---

## Kontribusi Kamus Kata Gaul

File [`dictionary/kamus_gaul_shopee.json`](dictionary/kamus_gaul_shopee.json) berisi **85+ entri** yang dikategorikan dalam:

| Kategori | Jumlah Entri |
|---|---|
| Singkatan Umum | 40+ |
| Ekspresi Sentimen Positif | 20 |
| Ekspresi Sentimen Negatif | 25 |
| Istilah Belanja Online | 15 |

Kamus ini dapat digunakan sebagai **baseline normalization** untuk proyek NLP bahasa Indonesia lainnya.

---

## Nilai Guna Dataset

Dataset yang dihasilkan dapat digunakan untuk:

1. **Sentiment Analysis** — Melatih model klasifikasi sentimen teks Bahasa Indonesia
2. **Text Classification** — Eksperimen dengan BERT, IndoBERT, atau model transformer lain
3. **Chatbot** — Basis pengetahuan respons ulasan pelanggan e-commerce
4. **Studi Linguistik** — Analisis evolusi bahasa informal digital Indonesia
5. **Benchmarking NLP** — Dataset pembanding untuk model NLP Bahasa Indonesia

---

## Informasi Peneliti

| | |
|---|---|
| **Nama** | [Akhmad Zamri Ardani] |
| **NIM** | [202310370311406] |
| **Program Studi** | [Teknik Informatika] |
| **Institusi** | [Universitas Muhammadiyah Malang] |
| **Mata Kuliah** | Data, Informasi dan Pengetahuan (DIP) |
| **Tahun** | 2026 |

---

## Lisensi

Dataset dan kode dalam repositori ini dilisensikan di bawah [MIT License](LICENSE).
Dataset hanya untuk keperluan akademis dan penelitian.

---

## Referensi

1. Sastrawi — Indonesian Stemmer Library
2. NLTK — Natural Language Toolkit
3. scikit-learn — Machine Learning in Python
4. Tala, F.Z. (2003). *A Study of Stemming Effects on Information Retrieval in Bahasa Indonesia*. ILLC, Universiteit van Amsterdam.

---

<p align="center">
  <i>Dibuat dengan ❤️ untuk keperluan akademis | 2026</i>
</p>
