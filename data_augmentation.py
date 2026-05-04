
import random
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

random.seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════
# HELPER — Get synonyms from WordNet
# ══════════════════════════════════════════════
def get_synonyms(word: str) -> list:
    """Return list of synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


# ══════════════════════════════════════════════
# 1. SYNONYM REPLACEMENT
# ══════════════════════════════════════════════
def synonym_replacement(text: str, n: int = 2) -> str:
    """
    Replace n random non-stopwords with their synonyms.
    
    Args:
        text : Input sentence
        n    : Number of words to replace
    """
    words = text.split()
    new_words = words.copy()

    # Filter out short/stopword-like tokens
    replaceable = [i for i, w in enumerate(words)
                   if len(w) > 3 and get_synonyms(w)]

    random.shuffle(replaceable)
    replacements_made = 0

    for idx in replaceable:
        if replacements_made >= n:
            break
        syns = get_synonyms(words[idx])
        if syns:
            new_words[idx] = random.choice(syns)
            replacements_made += 1

    return " ".join(new_words)


# ══════════════════════════════════════════════
# 2. RANDOM INSERTION
# ══════════════════════════════════════════════
def random_insertion(text: str, n: int = 1) -> str:
    """
    Insert a random synonym of a random word n times.
    """
    words = text.split()
    new_words = words.copy()

    for _ in range(n):
        word = random.choice(words)
        syns = get_synonyms(word)
        if syns:
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, random.choice(syns))

    return " ".join(new_words)


# ══════════════════════════════════════════════
# 3. RANDOM SWAP
# ══════════════════════════════════════════════
def random_swap(text: str, n: int = 1) -> str:
    """
    Randomly swap two words in the sentence n times.
    """
    words = text.split()
    if len(words) < 2:
        return text

    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

    return " ".join(new_words)


# ══════════════════════════════════════════════
# 4. RANDOM DELETION
# ══════════════════════════════════════════════
def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly delete each word with probability p.
    Keep at least one word.
    
    Args:
        p : Probability of deleting each word (0.1 = 10%)
    """
    words = text.split()
    if len(words) == 1:
        return text

    new_words = [w for w in words if random.random() > p]

    # Ensure sentence isn't empty
    return " ".join(new_words) if new_words else random.choice(words)


# ══════════════════════════════════════════════
# 5. BACK-TRANSLATION (using HuggingFace Helsinki-NLP models)
# ══════════════════════════════════════════════
def back_translate(text: str, intermediate_lang: str = "fr") -> str:
    """
    Translate text to an intermediate language, then back to English.
    This paraphrases the sentence while preserving meaning.
    
    Supported: 'fr' (French), 'de' (German), 'es' (Spanish)
    
    NOTE: Downloads ~300MB model on first run. 
          Best run in batches on GPU.
    """
    from transformers import pipeline

    src_to_tgt = pipeline(
        "translation",
        model=f"Helsinki-NLP/opus-mt-en-{intermediate_lang}",
        device=-1  # CPU; use 0 for GPU
    )
    tgt_to_src = pipeline(
        "translation",
        model=f"Helsinki-NLP/opus-mt-{intermediate_lang}-en",
        device=-1
    )

    translated = src_to_tgt(text, max_length=256)[0]["translation_text"]
    back       = tgt_to_src(translated, max_length=256)[0]["translation_text"]
    return back


# ══════════════════════════════════════════════
# EDA — Easy Data Augmentation (combined)
# ══════════════════════════════════════════════
def eda_augment(text: str,
                alpha_sr: float = 0.1,
                alpha_ri: float = 0.1,
                alpha_rs: float = 0.1,
                p_rd: float = 0.1,
                num_aug: int = 4) -> list:
    """
    Apply all 4 EDA techniques and return augmented versions.
    
    Args:
        alpha_sr  : % words to synonym-replace
        alpha_ri  : % words to randomly insert
        alpha_rs  : % words to randomly swap
        p_rd      : Probability of random deletion
        num_aug   : Number of augmented sentences to return
    """
    words = text.split()
    n_sr = max(1, int(alpha_sr * len(words)))
    n_ri = max(1, int(alpha_ri * len(words)))
    n_rs = max(1, int(alpha_rs * len(words)))

    augmented = []
    ops = [
        lambda t: synonym_replacement(t, n_sr),
        lambda t: random_insertion(t, n_ri),
        lambda t: random_swap(t, n_rs),
        lambda t: random_deletion(t, p_rd),
    ]

    while len(augmented) < num_aug:
        op = random.choice(ops)
        augmented.append(op(text))

    return list(set(augmented))[:num_aug]  # deduplicate


# ══════════════════════════════════════════════
# AUGMENT FULL DATASET
# ══════════════════════════════════════════════
def augment_dataset(df: pd.DataFrame,
                    text_col: str = "text",
                    label_col: str = "label",
                    num_aug: int = 2,
                    target_per_class: int = None) -> pd.DataFrame:
    """
    Augment the full dataset.
    
    Args:
        df               : Preprocessed DataFrame
        num_aug          : Augmented copies per sample
        target_per_class : If set, augment until each class reaches this count
    """
    print(f"[INFO] Augmenting dataset (num_aug={num_aug})...")

    new_rows = []
    label_counts = df[label_col].value_counts()

    for _, row in df.iterrows():
        text  = row[text_col]
        label = row[label_col]

        # If target_per_class set, only augment minority classes
        if target_per_class:
            current_count = label_counts.get(label, 0)
            if current_count >= target_per_class:
                continue

        augmented_texts = eda_augment(text, num_aug=num_aug)
        for aug_text in augmented_texts:
            new_rows.append({text_col: aug_text, label_col: label})

    aug_df = pd.DataFrame(new_rows)
    result_df = pd.concat([df, aug_df], ignore_index=True)
    result_df = result_df.drop_duplicates(subset=[text_col])

    print(f"[INFO] Dataset size: {len(df)} → {len(result_df)} samples")
    print(f"[INFO] Label distribution after augmentation:")
    print(result_df[label_col].value_counts())

    return result_df


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs("data/augmented", exist_ok=True)

    # Load preprocessed training data
    train_df = pd.read_csv("data/processed/train.csv")
    train_df.columns = ["text", "label"]

    # Augment
    augmented_df = augment_dataset(
        train_df,
        text_col="text",
        label_col="label",
        num_aug=2
    )

    augmented_df.to_csv("data/augmented/train_augmented.csv", index=False)
    print("\n[DONE] Augmented training data saved to data/augmented/train_augmented.csv")

    # Quick test — EDA on a sample
    sample = "I feel so frustrated and angry about what happened today"
    print(f"\nOriginal : {sample}")
    for i, aug in enumerate(eda_augment(sample, num_aug=4), 1):
        print(f"Aug #{i}  : {aug}")
