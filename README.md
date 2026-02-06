# Analisis Sentimen Program Makan Bergizi Gratis (MBG)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Analisis sentimen opini publik tentang Program Makan Bergizi Gratis (MBG) menggunakan data dari Twitter dan TikTok dengan pendekatan Machine Learning dan IndoBERT.

## 📋 Deskripsi Project

Project ini merupakan bagian dari skripsi yang menganalisis sentimen masyarakat Indonesia terhadap Program Makan Bergizi Gratis (MBG) dengan menggunakan:
- **Auto-labeling**: IndoBERT untuk pelabelan sentimen otomatis
- **Machine Learning**: Naive Bayes, SVM, dan Random Forest
- **Feature Engineering**: TF-IDF Vectorization
- **Dataset**: 6,211 posts dari Twitter dan TikTok

## 🎯 Tahapan Analisis

1. **Preprocessing Data** - Text cleaning, normalisasi slang, dan filtering
2. **Auto-Labeling** - Pelabelan sentimen otomatis menggunakan IndoBERT
3. **Stopword Removal & Stemming** - Preprocessing lanjutan bahasa Indonesia
4. **Feature Engineering** - Ekstraksi fitur menggunakan TF-IDF
5. **Model Training** - Training 3 model klasifikasi
6. **Evaluasi & Visualisasi** - Analisis performa dan insight sentimen

## 🚀 Cara Menjalankan

### Prerequisites

- Python 3.11 atau lebih baru
- pip (Python package manager)
- Git
- Minimal 8GB RAM (untuk IndoBERT)
- GPU (opsional, untuk training lebih cepat)

### Instalasi

1. **Clone repository ini:**
```bash
git clone https://github.com/galihmawardi-maker/sentiment-analysis-mbg.git
cd sentiment-analysis-mbg
```

2. **Buat virtual environment (sangat disarankan):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup Hugging Face Token:**
   - Buat akun di [Hugging Face](https://huggingface.co/)
   - Generate token di [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Ganti token di cell notebook atau login via CLI:
   ```bash
   huggingface-cli login
   ```

### Persiapan Dataset

1. **Buat struktur folder:**
```bash
mkdir -p dataset output models
```

2. **Siapkan dataset:**
   - File CSV harus bernama `dataset_raw_twitter_tiktok.csv`
   - Letakkan di folder `dataset/`
   - Format kolom wajib:
     - `post_id`, `user_handle`, `text`, `url`
     - `search_query`, `platform`, `timestamp`
     - `likes`, `retweets`, `replies`

### Menjalankan Notebook

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Atau gunakan Jupyter Lab:**
```bash
jupyter lab
```

3. **Buka file `analisis_sentimen.ipynb`**

4. **Jalankan cell secara berurutan** (Shift + Enter)

### Menjalankan di Google Colab

1. Upload notebook ke Google Drive
2. Buka dengan Google Colab
3. Ubah runtime type ke GPU (Runtime > Change runtime type > GPU)
4. Upload dataset ke folder Colab
5. Sesuaikan path di notebook:
   ```python
   BASE_DIR = Path("/content/drive/MyDrive/sentiment_mbg")
   ```

## 📊 Hasil yang Diharapkan

### Output Files

Setelah menjalankan notebook, akan dihasilkan file-file berikut:

**Di folder `dataset/`:**
- `dataset_preprocessed_minimal.csv` - Data setelah preprocessing dasar
- `dataset_labeled.csv` - Data dengan label sentimen dari IndoBERT
- `dataset_labeled_high_conf.csv` - Data high-confidence (≥0.6)
- `dataset_preprocessed_final.csv` - Data final untuk training

**Di folder `output/`:**
- `tfidf_vectorizer.pkl` - Model TF-IDF
- `train_test_split_info.pkl` - Info split data
- `top_500_words_before_labeling.csv` - Analisis kata terbanyak
- `top_words_negative.csv`, `top_words_positive.csv`, `top_words_neutral.csv`
- Visualisasi wordcloud dan grafik performa model

**Di folder `models/`:**
- `naive_bayes_model.pkl`
- `svm_model.pkl`
- `random_forest_model.pkl`

### Performa Model (Expected)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~72.6% | ~73.3% | ~72.6% | ~72.3% |
| SVM | ~70.4% | ~70.6% | ~70.4% | ~70.5% |
| Random Forest | ~69.8% | ~71.3% | ~69.8% | ~69.3% |

## 📦 Dependencies Utama

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **transformers** - IndoBERT (Hugging Face)
- **torch** - Deep learning framework
- **sastrawi** - Indonesian stemmer
- **matplotlib, seaborn** - Visualisasi
- **wordcloud** - Word cloud generation
- **tqdm** - Progress bar

## ⚠️ Troubleshooting

### Error: "CUDA out of memory"
**Solusi:**
```python
# Ubah device ke CPU di cell IndoBERT
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="mdhugol/indonesia-bert-sentiment-classification",
    device=-1  # Gunakan CPU
)
```

### Error: "File not found"
**Solusi:**
- Cek path folder dengan `print(BASE_DIR.exists())`
- Pastikan struktur folder sudah benar
- Gunakan path absolut jika perlu

### Proses labeling terlalu lama
**Solusi:**
- Gunakan GPU jika tersedia
- Reduce dataset untuk testing: `df = df.head(1000)`
- Gunakan batch processing jika memungkinkan

### Error instalasi dependencies
**Solusi:**
```bash
# Upgrade pip terlebih dahulu
pip install --upgrade pip

# Install satu per satu jika ada error
pip install pandas numpy scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

## 📝 Struktur Project

```
sentiment-analysis-mbg/
├── analisis_sentimen.ipynb    # Main notebook
├── README.md                   # Dokumentasi ini
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
├── dataset/                   # Folder dataset (tidak di-track git)
│   └── dataset_raw_twitter_tiktok.csv
├── output/                    # Folder output (tidak di-track git)
│   ├── visualisasi/
│   └── models/
└── models/                    # Folder models (tidak di-track git)
```

## 👤 Author

**Galih Mawardi**
- NIM: 1152100034
- Institut: Institut Teknologi Indonesia (ITI)
- Program Studi: Informatika
- Email: [your-email@example.com]
- GitHub: [@galihmawardi-maker](https://github.com/galihmawardi-maker)

## 📄 License

Project ini dibuat untuk keperluan akademik (Skripsi). 

## 🙏 Acknowledgments

- [IndoBERT](https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification) oleh mdhugol
- [Sastrawi](https://github.com/har07/PySastrawi) - Indonesian Stemmer
- Dosen pembimbing dan ITI

## 📞 Support

Jika ada pertanyaan atau kendala:
1. Buka issue di GitHub repository ini
2. Email ke [your-email@example.com]
3. Diskusi di WhatsApp group skripsi

---

**Note**: Dataset tidak disertakan dalam repository ini karena privasi dan ukuran file. Untuk mendapatkan dataset, silakan hubungi author.
