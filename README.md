# 🎭 Emotion Detection from Text
### Using Transformer-Based Deep Learning Models (DistilBERT)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CPU-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Dataset](https://img.shields.io/badge/Dataset-ISEAR-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project implements an end-to-end **Emotion Detection System** that classifies natural language text into **7 emotion categories**:

| Emotion | Example |
|---------|---------|
| 😄 Joy | "I just got promoted at work, I'm absolutely thrilled!" |
| 😢 Sadness | "I lost my dog today, he was my best friend for 10 years." |
| 😠 Anger | "I cannot believe they fired me without any warning." |
| 😨 Fear | "The doctor said the results could be serious. I'm terrified." |
| 🤢 Disgust | "The way they treated those animals was absolutely revolting." |
| 😳 Shame | "I said something hurtful and I'm so ashamed of myself." |
| 😔 Guilt | "I should have been there for her when she needed me most." |

The system uses **DistilBERT** fine-tuned on the **ISEAR dataset** and includes advanced features such as sarcasm detection, double negation handling, and a custom abbreviation-aware tokenizer. It is deployed as an interactive **Streamlit web application**.

---

## 🗂️ Project Structure

```
IndustrialProject/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── isear.csv                  ← Downloaded by data_collection.py
│   ├── 📁 processed/
│   │   ├── train.csv                  ← 70% training split
│   │   ├── val.csv                    ← 15% validation split
│   │   ├── test.csv                   ← 15% test split
│   │   └── label_map.json             ← Emotion → integer mapping
│   └── 📁 augmented/
│       └── train_augmented.csv        ← Augmented training data
│
├── 📁 models/
│   ├── best_model.pt                  ← Best trained model checkpoint
│   ├── best_model_v2.pt               ← Optional: second model for A/B testing
│   ├── training_curves.png            ← Loss & accuracy plots
│   └── results.json                   ← Final test set metrics
│
├── 📁 logs/
│   ├── predictions.csv                ← Prediction tracker log
│   ├── emotion_model_<timestamp>.log  ← Timestamped run logs
│   └── 📁 ab_testing/
│       ├── ab_results.csv             ← A/B test raw results
│       ├── ab_summary.json            ← A/B test summary & recommendation
│       └── ab_comparison.png          ← A/B comparison chart
│
├── 📁 docs/
│   ├── emotion_detection_report.docx  ← Full research paper
│   ├── test_design_scenarios_cases.docx ← Test design & test cases
│   └── test_report.docx               ← Test execution report
│
├── data_collection.py                 ← Step 1A: Load ISEAR dataset
├── data_preprocessing.py              ← Step 1B: Clean, normalize, encode, split
├── data_augmentation.py               ← Step 1C: EDA augmentation
├── step2_complete.py                  ← Step 2: Model training + inference pipeline
├── step4_testing.py                   ← Step 4A: Real-world testing + logging
├── step4_ab_testing.py                ← Step 4B: A/B model comparison
├── step4_app.py                       ← Step 4C: Streamlit web application
│
├── requirements.txt                   ← All dependencies with versions
└── README.md                          ← This file
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection.git
cd emotion-detection
```

### 2. Install PyTorch (CPU version)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install All Other Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Run the scripts in this order:

### Step 1 — Data Preparation
```bash
# 1A. Download and save ISEAR dataset
python data_collection.py

# 1B. Preprocess: clean, normalize, encode, balance, split
python data_preprocessing.py

# 1C. Augment training data (optional but recommended)
python data_augmentation.py
```

### Step 2 — Model Training & Inference
```bash
# Trains DistilBERT on ISEAR, saves best model, runs inference demo
python step2_complete.py
```
> ⏱️ **Expected training time: ~1.5–2 hours on CPU**

### Step 3 — Testing & Logging
```bash
# Run real-world test suite + generate logs and confusion matrix
python step4_testing.py
```

### Step 4 — Launch Web Application
```bash
streamlit run step4_app.py
```
> Opens automatically at **http://localhost:8501**

### Optional — A/B Testing
```bash
# Copy model to create a second version first
copy models\best_model.pt models\best_model_v2.pt

# Then run A/B comparison
python step4_ab_testing.py
```

---

## 🏗️ System Architecture

```
Raw Text Input
      │
      ▼
┌─────────────────────┐
│   Custom Tokenizer  │  ← Abbreviation expansion (omg→oh my god)
│  (EmotionTokenizer) │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│     DistilBERT      │  ← Pre-trained transformer base
│    (6 layers)       │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Classification     │  ← Dropout → Linear(768→256) → ReLU → Linear(256→7)
│      Head           │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Sarcasm Detector   │  ← Rule-based + HuggingFace model
│  Negation Handler   │  ← Double/single negation resolution
└─────────────────────┘
      │
      ▼
  Final Emotion
   Prediction
```

---

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| Base Model | `distilbert-base-uncased` |
| Dataset | ISEAR (7,503 samples, 7 emotions) |
| Classes | anger, disgust, fear, guilt, joy, sadness, shame |
| Max Sequence Length | 128 tokens |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |
| Scheduler | Linear warmup |
| Epochs | 5 (freeze base for epochs 1-2) |
| Batch Size | 16 |
| Expected Test Accuracy | ~85-90% |
| Expected Weighted F1 | ~0.85-0.90 |

---

## 🌐 Web Application Features

The Streamlit app (`step4_app.py`) includes 4 tabs:

| Tab | Description |
|-----|-------------|
| 🔍 Single Prediction | Real-time emotion detection with confidence bar chart |
| 📂 Batch Prediction | Upload CSV, predict all rows, download results |
| 📊 Prediction Log | Session log of all predictions with emotion distribution chart |
| ⚖️ A/B Comparison | Compare two model versions side by side |

---

## 🔬 Key Features

- ✅ **DistilBERT fine-tuning** with transfer learning on ISEAR
- ✅ **Custom tokenizer** with 25-entry abbreviation expansion dictionary
- ✅ **Two-layer sarcasm detection** (rule-based + HuggingFace model)
- ✅ **Double negation handling** ("not bad" → positive, "not happy" → flip)
- ✅ **Data augmentation** using EDA (synonym replacement, insertion, swap, deletion)
- ✅ **Class balancing** using RandomOverSampler
- ✅ **Prediction logging** with timestamps to CSV and .log files
- ✅ **A/B testing framework** for model version comparison
- ✅ **Streamlit web app** with batch prediction and CSV download

---

## 📊 Dataset

**ISEAR (International Survey on Emotion Antecedents and Reactions)**
- ~7,500 self-reported emotional situation descriptions
- Collected from ~3,000 students across 37 countries
- 7 emotion classes, approximately balanced (~1,070 samples per class)
- Labels are self-reported → high reliability ground truth
- Loaded automatically via direct CSV download from GitHub mirror

---

## 📁 Documents

| Document | Description |
|----------|-------------|
| `docs/emotion_detection_report.docx` | Full research paper (15 pages) |
| `docs/test_design_scenarios_cases.docx` | Test design, 25 scenarios, 35 test cases |
| `docs/test_report.docx` | Test execution report — 91.4% pass rate |

---


---

## 📋 Requirements

See `requirements.txt` for full list. Key dependencies:

```
torch (CPU)
transformers
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
nltk
contractions
matplotlib
seaborn
plotly
requests
datasets
```

---

## 🗺️ Logic Flow

```
data_collection.py
        ↓
   isear.csv saved
        ↓
data_preprocessing.py
        ↓
  train/val/test.csv + label_map.json
        ↓
data_augmentation.py (optional)
        ↓
  train_augmented.csv
        ↓
step2_complete.py
        ↓
  best_model.pt + training_curves.png + results.json
        ↓
step4_testing.py
        ↓
  predictions.csv + confusion_matrix.png + .log file
        ↓
step4_app.py (Streamlit)
        ↓
  http://localhost:8501
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [ISEAR Dataset](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/) — Scherer & Wallbott (1994)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [DistilBERT](https://arxiv.org/abs/1910.01108) — Sanh et al. (2019)
- [EDA: Easy Data Augmentation](https://arxiv.org/abs/1901.11196) — Wei & Zou (2019)
