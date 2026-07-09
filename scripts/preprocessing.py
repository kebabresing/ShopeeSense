"""
=============================================================================
Preprocessing Ulasan Shopee - Analisis Sentimen Bahasa Indonesia
=============================================================================
Deskripsi:
    Skrip ini melakukan preprocessing teks terhadap dataset ulasan pengguna
    aplikasi Shopee dari Google Play Store. Tujuannya adalah menghasilkan
    dataset yang bersih dan siap digunakan untuk pelatihan model Machine
    Learning (Sentiment Analysis).

Tahapan Preprocessing:
    1. Pembersihan Teks (Text Cleaning)
    2. Normalisasi Kata Gaul (Slang Normalization)
    3. Case Folding (Konversi ke huruf kecil)
    4. Tokenisasi
    5. Penghapusan Stopword (Bahasa Indonesia)
    6. Stemming (Sastrawi)
    7. Pelabelan Sentimen berdasarkan Rating

Kebutuhan Library:
    pip install pandas nltk Sastrawi wordcloud matplotlib seaborn

Penulis  : [Nama Mahasiswa]
NIM      : [NIM]
Mata Kuliah: Dasar-Dasar Ilmu Pengetahuan (DIP)
Institusi: [Nama Universitas]
Tanggal  : 2026-06-22
=============================================================================
"""

import os
import re
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk server/script
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR      = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DICTIONARY_DIR    = BASE_DIR / "dictionary"
ASSETS_DIR        = BASE_DIR / "assets"

# Pastikan folder output ada
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# MUAT KAMUS KATA GAUL
# ─────────────────────────────────────────────
def load_slang_dictionary(dict_path: Path) -> dict:
    """
    Memuat kamus kata gaul dari file JSON.

    Kamus ini mencakup singkatan umum, ekspresi sentimen,
    dan istilah khas belanja online dalam Bahasa Indonesia.
    Penggunaan kamus khusus lebih efektif dibanding kamus generik
    karena domain ulasan Shopee memiliki kosakata yang spesifik.

    Args:
        dict_path (Path): Lokasi file JSON kamus.

    Returns:
        dict: Mapping dari kata gaul ke bentuk baku.
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Gabungkan semua kategori kamus menjadi satu flat dict
    flat_dict = {}
    for category, entries in data['kamus'].items():
        flat_dict.update(entries)

    print(f"[OK] Kamus dimuat: {len(flat_dict)} entri dari '{dict_path.name}'")
    return flat_dict


# ─────────────────────────────────────────────
# FUNGSI PEMBERSIHAN TEKS
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Membersihkan teks mentah dari noise.

    Proses:
    - Menghapus URL (http/https)
    - Menghapus mention (@username) dan hashtag (#tag)
    - Menghapus karakter non-alfanumerik & non-spasi
    - Menghapus angka (nomor tidak bermakna semantik untuk sentimen)
    - Menghapus spasi berlebih
    - Konversi ke huruf kecil (case folding)

    Args:
        text (str): Teks asli dari kolom ulasan.

    Returns:
        str: Teks yang sudah dibersihkan.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Case folding
    text = text.lower()

    # Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Hapus mention & hashtag
    text = re.sub(r'@\w+|#\w+', '', text)

    # Hapus emoji & karakter unicode khusus
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Hapus karakter non-huruf (kecuali spasi)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Normalisasi whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_slang(text: str, slang_dict: dict) -> str:
    """
    Mengganti kata gaul/informal dengan bentuk baku.

    Strategi: word-by-word replacement menggunakan kamus.
    Pendekatan ini dipilih karena:
    - Lebih cepat dari regex berbasis aturan untuk korpus besar
    - Mudah di-update cukup dengan menambah entri ke kamus JSON
    - Reproducible dan dapat diaudit

    Args:
        text (str): Teks yang sudah melalui clean_text().
        slang_dict (dict): Kamus kata gaul -> kata baku.

    Returns:
        str: Teks dengan kata gaul sudah dinormalisasi.
    """
    tokens = text.split()
    normalized = [slang_dict.get(token, token) for token in tokens]
    return ' '.join(normalized)


def label_sentiment(rating: int) -> str:
    """
    Melabeli sentimen berdasarkan nilai rating bintang.

    Skema pelabelan:
    - Rating 4–5 -> Positif  (pengguna puas)
    - Rating 3   -> Netral   (pengalaman biasa)
    - Rating 1–2 -> Negatif  (pengguna tidak puas)

    Skema ini lebih interpretatif daripada sekadar mengandalkan teks,
    karena rating adalah ground truth eksplisit dari pengguna.

    Args:
        rating (int): Nilai rating dari 1 hingga 5.

    Returns:
        str: Label sentimen ('Positif', 'Netral', 'Negatif').
    """
    if rating >= 4:
        return 'Positif'
    elif rating == 3:
        return 'Netral'
    else:
        return 'Negatif'


# ─────────────────────────────────────────────
# PIPELINE PREPROCESSING UTAMA
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(raw_path: Path, dict_path: Path) -> pd.DataFrame:
    """
    Menjalankan pipeline preprocessing lengkap.

    Urutan tahapan:
    1. Muat dataset mentah
    2. Muat kamus kata gaul
    3. Labeli sentimen dari rating
    4. Bersihkan teks (clean_text)
    5. Normalisasi kata gaul (normalize_slang)
    6. Tokenisasi sederhana (split by space)
    7. Hapus baris kosong setelah preprocessing

    Args:
        raw_path (Path): Path ke file CSV mentah.
        dict_path (Path): Path ke file JSON kamus kata gaul.

    Returns:
        pd.DataFrame: DataFrame dengan kolom teks hasil preprocessing.
    """
    print("\n" + "="*60)
    print("  PIPELINE PREPROCESSING ULASAN SHOPEE")
    print("="*60)

    # ── Step 1: Muat data ──
    print(f"\n[1/6] Memuat dataset: {raw_path.name}")
    df = pd.read_csv(raw_path, encoding='utf-8')
    print(f"      Jumlah baris awal : {len(df):,}")
    print(f"      Kolom             : {list(df.columns)}")

    # Pastikan kolom yang dibutuhkan ada
    required_cols = {'review', 'rating'}
    available = set(df.columns.str.lower())
    if not required_cols.issubset(available):
        # Coba deteksi otomatis
        df.columns = df.columns.str.lower().str.strip()

    # ── Step 2: Muat kamus ──
    print(f"\n[2/6] Memuat kamus kata gaul")
    slang_dict = load_slang_dictionary(dict_path)

    # ── Step 3: Pelabelan sentimen ──
    print(f"\n[3/6] Pelabelan sentimen berdasarkan rating")
    if 'label' not in df.columns:
        df['label'] = df['rating'].apply(label_sentiment)
    dist = df['label'].value_counts()
    for lbl, cnt in dist.items():
        pct = cnt / len(df) * 100
        print(f"      {lbl:10s}: {cnt:5,} ({pct:.1f}%)")

    # ── Step 4: Simpan teks asli untuk perbandingan ──
    df['review_asli'] = df['review'].copy()

    # ── Step 5: Pembersihan teks ──
    print(f"\n[4/6] Membersihkan teks (case folding, remove noise)")
    df['review_clean'] = df['review'].apply(clean_text)

    # ── Step 6: Normalisasi kata gaul ──
    print(f"\n[5/6] Normalisasi kata gaul menggunakan kamus")
    df['review_normalized'] = df['review_clean'].apply(
        lambda t: normalize_slang(t, slang_dict)
    )

    # ── Step 7: Hapus baris kosong ──
    print(f"\n[6/6] Menghapus baris dengan teks kosong setelah preprocessing")
    before = len(df)
    df = df[df['review_normalized'].str.strip().str.len() > 0].copy()
    df = df.reset_index(drop=True)
    removed = before - len(df)
    print(f"      Dihapus  : {removed} baris")
    print(f"      Tersisa  : {len(df):,} baris")

    # ── Kolom final ──
    final_cols = ['review_asli', 'review_clean', 'review_normalized', 'rating', 'label']
    if 'tanggal' in df.columns:
        final_cols.append('tanggal')
    df_final = df[[c for c in final_cols if c in df.columns]]

    print(f"\n{'='*60}")
    print(f"  PREPROCESSING SELESAI")
    print(f"  Total baris final: {len(df_final):,}")
    print(f"{'='*60}\n")

    return df_final


# ─────────────────────────────────────────────
# VISUALISASI: WORD CLOUD SEBELUM VS SESUDAH
# ─────────────────────────────────────────────
def generate_wordcloud_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Membuat visualisasi Word Cloud sebelum dan sesudah preprocessing.

    Ini merupakan bukti visual yang menunjukkan efektivitas preprocessing:
    - Sebelum: banyak noise, kata gaul, angka, karakter khusus
    - Sesudah: kata-kata bermakna dan bersih yang mencerminkan topik

    Args:
        df (pd.DataFrame): DataFrame hasil preprocessing.
        output_dir (Path): Folder penyimpanan gambar.
    """
    try:
        from wordcloud import WordCloud
        print("[INFO] Membuat Word Cloud...")

        stopwords_id = {
            'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
            'untuk', 'pada', 'adalah', 'tidak', 'ada', 'saya', 'kami',
            'kita', 'mereka', 'juga', 'sudah', 'akan', 'lebih', 'atau',
            'bisa', 'jika', 'tapi', 'karena', 'sangat', 'dalam', 'saja',
            'sudah', 'belum', 'memang', 'kalau', 'sama', 'terus', 'harus',
            'mau', 'lagi', 'dulu', 'masuk', 'banget', 'sekali', 'lain',
            'setelah', 'sebelum', 'nya', 'pun', 'kami', 'anda', 'bagus'
        }

        text_before = ' '.join(df['review_asli'].dropna().astype(str))
        text_after  = ' '.join(df['review_normalized'].dropna().astype(str))

        wc_params = dict(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=150,
            stopwords=stopwords_id,
            collocations=False
        )

        wc_before = WordCloud(**wc_params).generate(text_before)
        wc_after  = WordCloud(**{**wc_params, 'colormap': 'plasma'}).generate(text_after)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.patch.set_facecolor('#1a1a2e')

        for ax, wc, title, color in zip(
            axes,
            [wc_before, wc_after],
            ['SEBELUM Preprocessing\n(Teks Mentah)', 'SESUDAH Preprocessing\n(Teks Bersih)'],
            ['#e94560', '#16c79a']
        ):
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, color=color, fontsize=15, fontweight='bold', pad=15)
            ax.set_facecolor('#0f3460')

        plt.suptitle(
            'Analisis Sentimen Ulasan Shopee\nPerbandingan Word Cloud Sebelum vs Sesudah Preprocessing',
            color='white', fontsize=17, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        out_path = output_dir / "wordcloud_sebelum_sesudah.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"[OK] Word Cloud disimpan -> {out_path}")

    except ImportError:
        print("[SKIP] WordCloud tidak terinstall. Jalankan: pip install wordcloud")


def generate_distribution_plots(df: pd.DataFrame, output_dir: Path):
    """
    Membuat visualisasi distribusi label dan frekuensi kata.

    Args:
        df (pd.DataFrame): DataFrame hasil preprocessing.
        output_dir (Path): Folder penyimpanan gambar.
    """
    sns.set_theme(style='darkgrid', palette='muted')
    colors = {'Positif': '#16c79a', 'Netral': '#f5a623', 'Negatif': '#e94560'}

    # ── Plot 1: Distribusi Sentimen ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    label_counts = df['label'].value_counts()
    ax = axes[0]
    bars = ax.bar(
        label_counts.index,
        label_counts.values,
        color=[colors.get(l, '#888') for l in label_counts.index],
        edgecolor='none', width=0.5
    )
    for bar, val in zip(bars, label_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:,}', ha='center', va='bottom', color='white', fontweight='bold')
    ax.set_title('Distribusi Label Sentimen', color='white', fontsize=13, fontweight='bold')
    ax.set_facecolor('#0f3460')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.set_ylabel('Jumlah Ulasan', color='white')

    # ── Plot 2: Top 20 Kata ──
    all_words = ' '.join(df['review_normalized'].dropna()).split()
    stopwords_id = {'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
                    'untuk', 'pada', 'tidak', 'ada', 'saya', 'juga', 'sudah',
                    'akan', 'atau', 'bisa', 'tapi', 'sangat', 'dalam', 'saja',
                    'memang', 'kalau', 'sama', 'terus', 'mau', 'lagi', 'nya',
                    'anda', 'banget', 'sekali', 'lain', 'belum', 'karena'}
    filtered_words = [w for w in all_words if w not in stopwords_id and len(w) > 2]
    word_freq = Counter(filtered_words).most_common(20)
    words_list, freqs = zip(*word_freq)

    ax2 = axes[1]
    bars2 = ax2.barh(range(len(words_list)), freqs,
                     color=sns.color_palette('viridis', len(words_list)))
    ax2.set_yticks(range(len(words_list)))
    ax2.set_yticklabels(words_list, color='white')
    ax2.set_title('Top 20 Kata Terfrekuen (Setelah Preprocessing)',
                  color='white', fontsize=11, fontweight='bold')
    ax2.set_facecolor('#0f3460')
    ax2.tick_params(axis='x', colors='white')
    ax2.spines[:].set_visible(False)
    ax2.set_xlabel('Frekuensi', color='white')
    ax2.invert_yaxis()

    plt.tight_layout()
    out_path = output_dir / "distribusi_sentimen_kata.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[OK] Distribusi plot disimpan -> {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Path file
    raw_csv  = DATA_RAW_DIR / "shopee_reviews_raw.csv"
    dict_json = DICTIONARY_DIR / "kamus_gaul_shopee.json"
    out_csv  = DATA_PROCESSED_DIR / "shopee_reviews_processed_v2.csv"

    # Jalankan pipeline
    df_processed = run_preprocessing_pipeline(raw_csv, dict_json)

    # Simpan hasil
    df_processed.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[SAVED] Dataset tersimpan -> {out_csv}")

    # Tampilkan contoh
    print("\nContoh 5 baris pertama:")
    print(df_processed[['review_asli', 'review_normalized', 'label']].head())

    # Buat visualisasi
    print("\nMembuat visualisasi...")
    generate_wordcloud_comparison(df_processed, ASSETS_DIR)
    generate_distribution_plots(df_processed, ASSETS_DIR)

    print("\n[SELESAI] Semua output berhasil disimpan.")
    print(f"  • Dataset bersih : {out_csv}")
    print(f"  • Visualisasi    : {ASSETS_DIR}/")
