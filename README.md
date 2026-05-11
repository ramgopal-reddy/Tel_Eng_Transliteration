# 🔤 Bidirectional Telugu–English Transliteration System

> NLP (Seq2Seq LSTM with Attention) + Unsupervised Learning (K-Means · UMAP · Plotly)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Dataset](<https://img.shields.io/badge/Dataset-Dakshina%20(HuggingFace)-yellow?logo=huggingface>)](https://huggingface.co/datasets/ramgopal-reddy/Telugu_to_English_Transliteration_Dakshina)
[![Word Pairs](https://img.shields.io/badge/Word%20Pairs-50%2C927-green)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-91.2%25-brightgreen)]()

---

## 📖 Overview

This project implements a **bidirectional transliteration system** between Telugu script and English script — converting words phonetically across scripts without translating their meaning.

| Direction        | Example                  |
| ---------------- | ------------------------ |
| English → Telugu | `Computer` → `కంప్యూటర్` |
| Telugu → English | `హెల్ప్` → `help`        |

The system is built on two complementary pipelines:

1. **Supervised** — Seq2Seq Bidirectional LSTM with Attention + Beam Search decoding
2. **Unsupervised** — K-Means clustering, hierarchical clustering, PCA, UMAP, and interactive Plotly visualisation on the full 50,927-word corpus

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  TRANSLITERATION SYSTEM                  │
├────────────────────────┬─────────────────────────────────┤
│   SUPERVISED MODULE    │      UNSUPERVISED MODULE        │
│   (Seq2Seq LSTM +      │  (K-Means · PCA · UMAP ·        │
│    Attention)          │   Plotly Interactive)           │
├────────────────────────┴─────────────────────────────────┤
│                    INFERENCE ENGINE                      │
│         combine_testing.py — Smart Bidirectional         │
│    (Auto-detects language, routes to correct model)      │
└──────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
TransLiteration/
├── eng2tel-transliteration.ipynb   # Seq2Seq LSTM training notebook (66 KB)
├── unsupervied_colab.ipynb         # Clustering & UMAP notebook (2.53 MB)
├── combine_testing.py              # Bidirectional inference CLI (8.97 KB)
├── eng2tel_lstm_full.pt            # Trained English→Telugu checkpoint (8.97 MB)
├── tel2eng_lstm_full.pt            # Trained Telugu→English checkpoint (8.85 MB)
└── .venv/                          # Python virtual environment
```

---

## 📊 Dataset

**Source:** [Dakshina Dataset](https://huggingface.co/datasets/ramgopal-reddy/Telugu_to_English_Transliteration_Dakshina) — Google's benchmark for Indic transliteration

| Property                | Value                |
| ----------------------- | -------------------- |
| Total word pairs        | 50,927               |
| Training samples        | 45,835 (90%)         |
| Validation samples      | 5,092 (10%)          |
| Telugu vocabulary       | 66 unique characters |
| English vocabulary      | 30 unique characters |
| Max Telugu word length  | 20 characters        |
| Max English word length | 25 characters        |

---

## 🧠 Model Architecture

### Encoder — Bidirectional LSTM

- Character-level embedding (128-dim)
- Bidirectional LSTM captures both left-to-right and right-to-left context
- FC layers compress bidirectional hidden/cell states from 512 → 256 dimensions
- Dropout: 0.5

### Attention Mechanism

At each decoder step, alignment scores are computed over all encoder outputs and converted to a weighted context vector — allowing the model to focus on the most relevant source characters for each output step.

### Decoder — Unidirectional LSTM

- Embedding (128-dim) + context vector (512-dim) concatenated → 640-dim input
- LSTM hidden: 256 units
- **Rich output projection**: `[output ‖ context ‖ embedding]` → FC → vocab logits

### Model Summary

| Component                  | Value                          |
| -------------------------- | ------------------------------ |
| Encoder embedding          | 128-dim                        |
| Encoder LSTM hidden        | 256-dim (bidir = 512 combined) |
| Decoder embedding          | 128-dim                        |
| Decoder LSTM hidden        | 256-dim                        |
| Total trainable parameters | **2,241,346**                  |

---

## ⚙️ Training Configuration

| Hyperparameter          | Value                              |
| ----------------------- | ---------------------------------- |
| Optimizer               | Adam                               |
| Initial learning rate   | 1e-3                               |
| Weight decay (L2)       | 1e-4                               |
| LR scheduler            | ReduceLROnPlateau                  |
| Gradient clipping       | 1.0                                |
| Batch size              | 128                                |
| Max epochs              | 80 (early stopping)                |
| Early stopping patience | 8 epochs                           |
| Teacher forcing         | 1.0 → 0.5 (annealed)               |
| Loss function           | CrossEntropyLoss (padding ignored) |
| Decoding                | Beam Search (beam size = 5)        |

**Data augmentation** (training inputs only):

- 5% probability of adjacent character swap
- 3% probability of character drop (words longer than 3 chars)

---

## 📈 Training Progress

| Epoch | Teacher Forcing | Train Loss | Val Loss | LR      |
| ----- | --------------- | ---------- | -------- | ------- |
| 1     | 0.97            | 1.7862     | 2.0436   | 1e-3    |
| 10    | 0.75            | 0.2999     | 0.5479   | 1e-3    |
| 20    | 0.50            | 0.2880     | 0.3936   | 1e-3    |
| 40    | 0.50            | 0.1929     | 0.2992   | 5e-4    |
| 60    | 0.50            | 0.1347     | 0.2525   | 1.25e-4 |

Loss converged from **1.79 → ~0.13** (train) and **2.04 → ~0.25** (val).

---

## 🔍 Results

| Metric                    | Value                       |
| ------------------------- | --------------------------- |
| **Exact Match Accuracy**  | **91.2%**                   |
| Best training loss        | ~0.12                       |
| Best validation loss      | ~0.25                       |
| Decoding strategy         | Beam Search (beam size = 5) |
| Checkpoint size (Eng→Tel) | 8.97 MB                     |
| Checkpoint size (Tel→Eng) | 8.85 MB                     |

---

## 🔬 Unsupervised Analysis

The unsupervised notebook applies clustering and dimensionality reduction to reveal latent phonological structure in the Telugu lexicon — with no labels required.

### Pipeline

1. **Feature Engineering** — Character n-gram TF-IDF (bigrams to 4-grams, 5000 features)
2. **K-Means Clustering** — 5 clusters over the full 50,927 word corpus
3. **Hierarchical Clustering** — Ward's linkage dendrogram on a 50-word sample
4. **PCA** — 2D projection for cluster separation visualisation
5. **UMAP** — Nonlinear 2D embedding (cosine distance, preserves local + global structure)
6. **Interactive Plotly** — Hover over any of 50,927 points to see the English transliteration

### Discovered Clusters

| Cluster | Inferred Pattern                | Sample Words                   |
| ------- | ------------------------------- | ------------------------------ |
| 0       | Long compound verbs             | manninchhamani, gadipesharu    |
| 1       | Short abstract nouns            | maitri, netiiki                |
| 2       | Dative/instrumental suffixed    | chooparu, mukhyudu             |
| 3       | Long descriptive words          | dhurmaargapu, darshanamistaayi |
| 4       | Accusative suffixed + loanwords | tappunu, vehicle               |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision datasets scikit-learn scipy umap-learn plotly pandas numpy matplotlib
```

### Training — Supervised (Seq2Seq LSTM)

1. Open `eng2tel-transliteration.ipynb` in **Google Colab** or **Kaggle** (GPU runtime recommended)
2. Run all cells sequentially
3. Checkpoint is saved as `eng2tel_lstm_full.pt`

### Training — Unsupervised (Clustering & UMAP)

1. Open `unsupervied_colab.ipynb`
2. Run all cells to generate clusters, dendrogram, PCA, UMAP, and Plotly visualisations

### Inference — Bidirectional CLI

```bash
cd path/to/TransLiteration
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

python combine_testing.py
```

```
Input: India
Output: ఇండియా

Input: హెల్ప్
Output: help
```

Type `exit` to quit. Mixed-script sentences are also supported — each word is auto-detected and routed to the appropriate model.

---

## 🌐 Language Detection

The inference engine uses Unicode range checks to automatically detect script:

```python
def is_telugu(text):
    for ch in text:
        if '\u0C00' <= ch <= '\u0C7F':   # Telugu Unicode block
            return True
    return False
```

Mixed-script sentences are handled word-by-word, with each word routed to the correct model.

---

## 🛠️ Technology Stack

| Category                  | Tool                                                 |
| ------------------------- | ---------------------------------------------------- |
| Deep Learning             | PyTorch 2.10+cu128 (GPU) / 2.11 (CPU)                |
| Data Loading              | HuggingFace `datasets`                               |
| Clustering                | scikit-learn (KMeans, PCA, TF-IDF)                   |
| Hierarchical Clustering   | scipy (linkage, dendrogram)                          |
| Dimensionality Reduction  | umap-learn                                           |
| Interactive Visualisation | Plotly Express                                       |
| Static Visualisation      | Matplotlib                                           |
| Data Manipulation         | pandas, numpy                                        |
| Training Platform         | Google Colab + Kaggle Notebook (GPU) + Local Windows |

---

## ⚠️ Known Limitations

| Limitation                                        | Proposed Improvement                     |
| ------------------------------------------------- | ---------------------------------------- |
| LSTM may struggle with very long words            | Switch to Transformer (e.g. fairseq)     |
| Telugu font not rendered in Matplotlib dendrogram | Use Noto Sans Telugu font                |
| UMAP spectral init fell back to random            | Increase `n_neighbors` or use PCA init   |
| No OOV character handling beyond `<unk>`          | Character-level byte-level BPE           |
| Inference runs on CPU only                        | Add CUDA support in `combine_testing.py` |
| No web interface                                  | Build Flask/FastAPI + React frontend     |

---

## 📋 Citation / Dataset

```bibtex
@dataset{dakshina,
  title   = {Dakshina Dataset — Telugu to English Transliteration},
  author  = {ramgopal-reddy},
  year    = {2021},
  url     = {https://huggingface.co/datasets/ramgopal-reddy/Telugu_to_English_Transliteration_Dakshina}
}
```

---

## 📄 License

This project is released for academic and educational use. The Dakshina dataset is subject to [Google's original dataset terms](https://github.com/google-research/dakshina).

---

<p align="center">Built with PyTorch · Trained on Dakshina · Visualised with UMAP + Plotly</p>
