
import re
import json
import os
import pandas as pd
import numpy as np
import nltk
import contractions
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from imblearn.over_sampling import RandomOverSampler

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


# ══════════════════════════════════════════════
# 1. TEXT CLEANING
# ══════════════════════════════════════════════
def clean_text(text: str) -> str:
    """
    Remove noise from raw text:
      - URLs, mentions, hashtags
      - HTML tags
      - Special characters & excessive whitespace
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # keep hashtag word
    text = re.sub(r"<.*?>", "", text)                     # remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s'!?.,]", " ", text)     # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


# ══════════════════════════════════════════════
# 2. TEXT NORMALIZATION
# ══════════════════════════════════════════════
SLANG_DICT = {
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "omg": "oh my god",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "smh": "shaking my head",
    "ngl": "not going to lie",
    "idk": "i don't know",
    "ikr": "i know right",
    "rn": "right now",
    "bc": "because",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
}

def normalize_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Normalize text:
      - Expand contractions (can't → cannot)
      - Lowercase
      - Expand slang abbreviations
      - Optionally remove stopwords (NOT recommended for transformers)
    """
    try:
        text = contractions.fix(text)
    except Exception:
        pass

    text = text.lower()

    # Expand slang
    words = text.split()
    words = [SLANG_DICT.get(w, w) for w in words]
    text  = " ".join(words)

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        # Always keep negation words — critical for emotion context
        negations = {"no", "not", "nor", "neither", "never", "none",
                     "nobody", "nothing", "nowhere", "hardly", "barely"}
        stop_words -= negations
        text = " ".join([w for w in text.split() if w not in stop_words])

    return text


# ══════════════════════════════════════════════
# 3. FULL PREPROCESSING PIPELINE
# ══════════════════════════════════════════════
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline on ISEAR DataFrame.
    Expects columns: text, label
    """
    print("[INFO] Starting preprocessing pipeline...")

    # Drop nulls and duplicates
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    print(f"[INFO] After dedup/dropna     : {len(df)} samples")

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Normalize text
    df["norm_text"] = df["clean_text"].apply(normalize_text)

    # Remove samples that became too short after cleaning
    df = df[df["norm_text"].str.strip().str.len() > 5]
    print(f"[INFO] After cleaning         : {len(df)} samples")
    print(f"[INFO] Unique emotions        : {sorted(df['label'].unique())}")

    return df


# ══════════════════════════════════════════════
# 4. LABEL ENCODING
# ══════════════════════════════════════════════
def encode_labels(df: pd.DataFrame):
    """
    Encode string emotion labels to integers.
    Returns: df with 'label_encoded' column, encoder object, label map dict
    """
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"].astype(str))

    label_map = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    print(f"\n[INFO] Label encoding map: {label_map}")

    return df, le, label_map


# ══════════════════════════════════════════════
# 5. HANDLE CLASS IMBALANCE
# ══════════════════════════════════════════════
def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample minority emotion classes using RandomOverSampler.
    ISEAR is fairly balanced (~1k per class) but this handles any skew.
    """
    print(f"\n[INFO] Class distribution BEFORE balancing:")
    print(df["label_encoded"].value_counts().sort_index())

    X = df[["norm_text"]].values
    y = df["label_encoded"].values

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    balanced_df = pd.DataFrame({
        "norm_text":     X_res.flatten(),
        "label_encoded": y_res
    })

    print(f"\n[INFO] Class distribution AFTER balancing:")
    print(balanced_df["label_encoded"].value_counts().sort_index())

    return balanced_df


# ══════════════════════════════════════════════
# 6. BERT TOKENIZATION
# ══════════════════════════════════════════════
def tokenize_for_bert(texts: list,
                      tokenizer_name: str = "bert-base-uncased",
                      max_length: int = 128) -> dict:
    """
    Tokenize texts using BERT tokenizer.
    Returns input_ids, attention_mask as numpy arrays.

    Args:
        texts          : List of preprocessed strings
        tokenizer_name : HuggingFace model name
        max_length     : Max token length (128 suits ISEAR sentence length)
    """
    print(f"\n[INFO] Tokenizing {len(texts)} samples with {tokenizer_name}...")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )

    print(f"[INFO] Tokenization complete. Shape: {encodings['input_ids'].shape}")
    return encodings


# ══════════════════════════════════════════════
# 7. TRAIN / VALIDATION / TEST SPLIT
# ══════════════════════════════════════════════
def split_dataset(df: pd.DataFrame,
                  test_size:  float = 0.15,
                  val_size:   float = 0.15):
    """
    Stratified split → Train (70%) | Validation (15%) | Test (15%)
    """
    X = df["norm_text"].tolist()
    y = df["label_encoded"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42, stratify=y
    )

    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=42, stratify=y_temp
    )

    print(f"\n[INFO] Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    # ── Load ISEAR raw data ───────────────────────────────────────────────────
    print("[INFO] Loading ISEAR raw data...")
    df = pd.read_csv("data/raw/isear.csv")
    # Expected columns from step1_data_collection.py: text, label, source
    df = df[["text", "label"]]

    # ── Preprocess ───────────────────────────────────────────────────────────
    df = preprocess_dataframe(df)

    # ── Encode labels ────────────────────────────────────────────────────────
    df, encoder, label_map = encode_labels(df)

    # ── Balance classes ──────────────────────────────────────────────────────
    balanced_df = balance_dataset(df)

    # ── Split ────────────────────────────────────────────────────────────────
    train_data, val_data, test_data = split_dataset(balanced_df)

    # ── Save processed splits ────────────────────────────────────────────────
    pd.DataFrame({"text": train_data[0], "label": train_data[1]}).to_csv(
        "data/processed/train.csv", index=False)
    pd.DataFrame({"text": val_data[0],   "label": val_data[1]}).to_csv(
        "data/processed/val.csv",   index=False)
    pd.DataFrame({"text": test_data[0],  "label": test_data[1]}).to_csv(
        "data/processed/test.csv",  index=False)

    # ── Save label map for reference in model training ───────────────────────
    with open("data/processed/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n{'='*50}")
    print(f"[DONE] Processed data saved to data/processed/")
    print(f"Label map: {label_map}")
    print(f"{'='*50}")
