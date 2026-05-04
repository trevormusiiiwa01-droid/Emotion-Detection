
import os
import pandas as pd
import requests


# ══════════════════════════════════════════════
# ISEAR Dataset
# ══════════════════════════════════════════════

# Emotion label map used in the raw ISEAR file
ISEAR_EMOTION_MAP = {
    1: "joy",
    2: "fear",
    3: "anger",
    4: "sadness",
    5: "disgust",
    6: "shame",
    7: "guilt"
}

def load_isear() -> pd.DataFrame:
    """
    Load the ISEAR dataset.

    Tries two methods in order:
      1. HuggingFace datasets library (easiest)
      2. Direct CSV download from GitHub mirror (fallback)

    Returns a clean DataFrame with columns: text, label, source
    """

    # ── Method 1: HuggingFace datasets ───────────────────────────────────────
    try:
        from datasets import load_dataset
        print("[INFO] Loading ISEAR from HuggingFace...")
        dataset = load_dataset("emotions", split="train")  # alternate HF mirror
        df = pd.DataFrame(dataset)
        df = df.rename(columns={"text": "text", "label": "label"})
        df["source"] = "isear"
        print(f"[INFO] Loaded {len(df)} samples via HuggingFace.")
        return df[["text", "label", "source"]]

    except Exception as e:
        print(f"[WARNING] HuggingFace load failed: {e}")
        print("[INFO] Falling back to direct CSV download...")

    # ── Method 2: Direct CSV download (GitHub mirror) ────────────────────────
    try:
        url = "https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv"
        print(f"[INFO] Downloading ISEAR from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(response.text), sep="|", on_bad_lines="skip")

        # ISEAR CSV has many columns — we need 'SIT' (situation text) and 'EMOT' (emotion id)
        df = df[["EMOT", "SIT"]].copy()
        df.columns = ["label", "text"]

        # Drop missing values
        df = df.dropna(subset=["text", "label"])

        # Map integer emotion codes to names
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int).map(ISEAR_EMOTION_MAP)
        df = df.dropna(subset=["label"])  # drop any unmapped codes

        df["source"] = "isear"
        df = df[["text", "label", "source"]].reset_index(drop=True)

        print(f"[INFO] Loaded {len(df)} samples via direct download.")
        return df

    except Exception as e:
        print(f"[ERROR] Direct download also failed: {e}")
        raise RuntimeError(
            "Could not load ISEAR dataset. Please download it manually from:\n"
            "https://www.iseardataset.com/ \n"
            "and place isear.csv in data/raw/"
        )


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    # Load ISEAR
    df = load_isear()

    # Final validation — confirm no multi-label samples exist
    print(f"\n[INFO] Verifying single-label integrity...")
    print(f"[INFO] Any nulls in label column: {df['label'].isnull().sum()}")
    print(f"[INFO] Unique emotions: {sorted(df['label'].unique())}")

    # Save
    df.to_csv("data/raw/isear.csv", index=False)

    print(f"\n{'='*50}")
    print(f"[DONE] Total samples collected : {len(df)}")
    print(f"       Saved to data/raw/isear.csv")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"{'='*50}")
