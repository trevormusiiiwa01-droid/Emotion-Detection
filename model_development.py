

import os
import re
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (DistilBertModel, DistilBertTokenizer,
                          get_linear_schedule_with_warmup, pipeline)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# 1. CUSTOM TOKENIZER
# ══════════════════════════════════════════════
class EmotionTokenizer:
    """
    Wrapper around DistilBERT tokenizer.
    Expands abbreviations and slang before tokenization.
    """

    ABBREVIATION_MAP = {
        "lol"   : "laughing out loud",
        "lmao"  : "laughing my head off",
        "rofl"  : "rolling on the floor laughing",
        "omg"   : "oh my god",
        "omfg"  : "oh my god",
        "smh"   : "shaking my head",
        "imo"   : "in my opinion",
        "imho"  : "in my honest opinion",
        "tbh"   : "to be honest",
        "ngl"   : "not going to lie",
        "istg"  : "i swear to god",
        "wtf"   : "what the hell",
        "fml"   : "my life is ruined",
        "ikr"   : "i know right",
        "idk"   : "i do not know",
        "idgaf" : "i do not care",
        "rn"    : "right now",
        "atm"   : "at the moment",
        "bc"    : "because",
        "gonna" : "going to",
        "wanna" : "want to",
        "gotta" : "got to",
        "kinda" : "kind of",
        "sorta" : "sort of",
        "dunno" : "do not know",
    }

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 max_length: int = 128):
        self.tokenizer  = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def expand_abbreviations(self, text: str) -> str:
        words    = text.lower().split()
        expanded = [self.ABBREVIATION_MAP.get(w, w) for w in words]
        return " ".join(expanded)

    def encode(self, texts: list) -> dict:
        expanded  = [self.expand_abbreviations(t) for t in texts]
        encodings = self.tokenizer(
            expanded,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encodings

    def encode_single(self, text: str) -> dict:
        return self.encode([text])


# ══════════════════════════════════════════════
# 2. EMOTION CLASSIFIER MODEL
# ══════════════════════════════════════════════
class EmotionClassifier(nn.Module):
    """
    DistilBERT base + custom classification head.
    Architecture: DistilBERT → [CLS] → Dropout → Linear(768→256) → ReLU → Dropout → Linear(256→num_classes)
    """

    def __init__(self, num_classes: int,
                 model_name: str = "distilbert-base-uncased",
                 dropout: float = 0.3,
                 freeze_base: bool = False):
        super(EmotionClassifier, self).__init__()

        self.distilbert = DistilBertModel.from_pretrained(model_name)

        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        hidden_size     = self.distilbert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs    = self.distilbert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_output)


# ══════════════════════════════════════════════
# 3. NEGATION HANDLER
# ══════════════════════════════════════════════
class NegationHandler:
    """Detects single and double negations that affect emotion."""

    DOUBLE_NEGATION_PATTERNS = [
        r"\bnot\s+\w*\s*(bad|terrible|awful|horrible|unpleasant)\b",
        r"\bnot\s+un\w+\b",
        r"\bnever\s+\w*\s*(bad|wrong|unfair)\b",
        r"\bno\s+\w*\s*(problem|issue|complaint|worries)\b",
    ]
    SINGLE_NEGATION_MARKERS = [
        r"\bnot\s+(happy|joyful|glad|pleased|excited)\b",
        r"\bnever\s+(happy|felt good|enjoyed)\b",
        r"\bno\s+(happiness|joy|pleasure)\b",
    ]

    def resolve(self, text: str) -> dict:
        text_lower = text.lower()
        double = any(re.search(p, text_lower) for p in self.DOUBLE_NEGATION_PATTERNS)
        single = any(re.search(p, text_lower) for p in self.SINGLE_NEGATION_MARKERS)
        return {
            "has_double_negation": double,
            "has_single_negation": single,
            "flip_emotion"       : single and not double
        }


# ══════════════════════════════════════════════
# 4. SARCASM DETECTOR
# ══════════════════════════════════════════════
class SarcasmDetector:
    """
    Two-layer sarcasm detection:
      Layer 1 — Rule-based  : fast, catches obvious patterns
      Layer 2 — Model-based : HuggingFace classifier for subtle cases
    """

    SARCASM_PHRASES = [
        r"\boh great\b", r"\bjust great\b", r"\byeah right\b",
        r"\bsure.*because.*that makes sense\b", r"\bas if\b",
        r"\boh.*because.*always\b", r"\breally.*genius\b",
    ]
    POSITIVE_WORDS   = {"amazing", "fantastic", "wonderful", "great",
                        "brilliant", "perfect", "awesome", "lovely"}
    NEGATIVE_CONTEXTS = {"monday", "traffic", "rain", "fail", "broke",
                         "lost", "worst", "hate", "stuck", "late"}

    def __init__(self, use_model: bool = True):
        self.use_model        = use_model
        self.sarcasm_pipeline = None
        if use_model:
            try:
                print("[INFO] Loading sarcasm detection model...")
                self.sarcasm_pipeline = pipeline(
                    "text-classification",
                    model="helinivan/english-sarcasm-detector",
                    device=-1
                )
                print("[INFO] Sarcasm model loaded.")
            except Exception as e:
                print(f"[WARNING] Sarcasm model failed to load: {e}")
                print("[INFO] Using rule-based sarcasm detection only.")
                self.use_model = False

    def rule_based_check(self, text: str) -> tuple:
        text_lower = text.lower()
        words      = set(text_lower.split())
        for pattern in self.SARCASM_PHRASES:
            if re.search(pattern, text_lower):
                return True, 0.85
        if (words & self.POSITIVE_WORDS) and (words & self.NEGATIVE_CONTEXTS):
            return True, 0.75
        return False, 0.0

    def detect(self, text: str) -> dict:
        is_sarcastic, confidence = self.rule_based_check(text)
        if is_sarcastic:
            return {"is_sarcastic": True,  "confidence": confidence,
                    "method": "rule_based"}
        if self.use_model and self.sarcasm_pipeline:
            try:
                result       = self.sarcasm_pipeline(text[:512])[0]
                is_sarcastic = result["label"].upper() == "SARCASM"
                confidence   = result["score"]
                if is_sarcastic and confidence > 0.7:
                    return {"is_sarcastic": True, "confidence": confidence,
                            "method": "model"}
            except Exception:
                pass
        return {"is_sarcastic": False, "confidence": 1.0, "method": "none"}


# Emotion flip map for sarcasm/negation adjustment
EMOTION_FLIP_MAP = {
    "joy"    : "sadness", "sadness": "joy",
    "anger"  : "joy",     "fear"   : "joy",
    "disgust": "joy",     "shame"  : "joy",  "guilt": "joy"
}


# ══════════════════════════════════════════════
# 5. PYTORCH DATASET
# ══════════════════════════════════════════════
class EmotionDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer: EmotionTokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_single(self.texts[idx])
        return {
            "input_ids"     : encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ══════════════════════════════════════════════
# 6. TRAINING UTILITIES
# ══════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        # Progress every 20 batches
        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1}/{len(loader)} | "
                  f"Loss: {total_loss/(i+1):.4f} | "
                  f"Acc: {correct/total:.4f}", end="\r")

    print()
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            logits         = model(input_ids, attention_mask)
            loss           = criterion(logits, labels)
            total_loss    += loss.item()
            preds          = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train", marker="o")
    axes[0].plot(history["val_loss"],   label="Val",   marker="o")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history["train_acc"], label="Train", marker="o")
    axes[1].plot(history["val_acc"],   label="Val",   marker="o")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


# ══════════════════════════════════════════════
# 7. MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════
def train(config: dict):
    os.makedirs(config["model_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    with open(config["label_map"], "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    id_to_label = {v: k for k, v in label_map.items()}
    print(f"[INFO] Classes ({num_classes}): {list(label_map.keys())}")

    train_df = pd.read_csv(config["train_path"])
    val_df   = pd.read_csv(config["val_path"])
    test_df  = pd.read_csv(config["test_path"])

    tokenizer     = EmotionTokenizer()
    train_dataset = EmotionDataset(train_df["text"].tolist(),
                                   train_df["label"].tolist(), tokenizer)
    val_dataset   = EmotionDataset(val_df["text"].tolist(),
                                   val_df["label"].tolist(),   tokenizer)
    test_dataset  = EmotionDataset(test_df["text"].tolist(),
                                   test_df["label"].tolist(),  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"])
    test_loader  = DataLoader(test_dataset,  batch_size=config["batch_size"])

    model = EmotionClassifier(
        num_classes=num_classes,
        freeze_base=config.get("freeze_base_initially", False)
    ).to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=config["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * config["epochs"]
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("warmup_steps", total_steps // 10),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    history     = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_f1 = 0.0
    best_epoch  = 0

    print(f"\n[INFO] Starting training for {config['epochs']} epochs...\n")

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        # Unfreeze base after epoch 2
        if config.get("freeze_base_initially") and epoch == 3:
            print("[INFO] Unfreezing DistilBERT base layers...")
            for param in model.distilbert.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(),
                              lr=config["lr"] / 10, weight_decay=0.01)

        print(f"\nEpoch {epoch}/{config['epochs']}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device)

        elapsed = time.time() - start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Time: {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(),
                       os.path.join(config["model_dir"], "best_model.pt"))
            print(f"  ✓ Best model saved (Val F1: {best_val_f1:.4f})")

    print(f"\n[INFO] Best model: epoch {best_epoch}, Val F1: {best_val_f1:.4f}")

    # Final test evaluation
    print("\n[INFO] Evaluating on test set...")
    model.load_state_dict(torch.load(
        os.path.join(config["model_dir"], "best_model.pt"), map_location=device))
    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=[id_to_label[i] for i in range(num_classes)]
    ))

    plot_training_curves(history, os.path.join(config["model_dir"],
                                               "training_curves.png"))

    with open(os.path.join(config["model_dir"], "results.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_f1": best_val_f1,
                   "test_acc": test_acc, "test_f1": test_f1}, f, indent=2)

    print(f"\n[DONE] Training complete. Model saved to {config['model_dir']}")
    return model, tokenizer


# ══════════════════════════════════════════════
# 8. INFERENCE PIPELINE
# ══════════════════════════════════════════════
class EmotionInferencePipeline:
    """Full inference with sarcasm and negation handling."""

    def __init__(self, model_path: str, label_map_path: str,
                 use_sarcasm_model: bool = True):
        with open(label_map_path, "r") as f:
            self.label_map  = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = EmotionTokenizer()
        self.model     = EmotionClassifier(num_classes=self.num_classes)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.sarcasm_detector = SarcasmDetector(use_model=use_sarcasm_model)
        self.negation_handler = NegationHandler()
        print(f"[INFO] Inference pipeline ready on {self.device}")

    def predict(self, text: str) -> dict:
        encoding = self.tokenizer.encode_single(text)
        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device)
            )
        probs        = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id      = int(np.argmax(probs))
        raw_emotion  = self.id_to_label[pred_id]
        confidence   = float(probs[pred_id])
        sarcasm_info  = self.sarcasm_detector.detect(text)
        negation_info = self.negation_handler.resolve(text)
        final_emotion = raw_emotion
        adjusted      = False
        if sarcasm_info["is_sarcastic"] or negation_info["flip_emotion"]:
            final_emotion = EMOTION_FLIP_MAP.get(raw_emotion, raw_emotion)
            adjusted      = True
        return {
            "text"         : text,
            "raw_emotion"  : raw_emotion,
            "final_emotion": final_emotion,
            "confidence"   : round(confidence, 4),
            "adjusted"     : adjusted,
            "sarcasm_info" : sarcasm_info,
            "negation_info": negation_info,
            "all_scores"   : {self.id_to_label[i]: round(float(probs[i]), 4)
                              for i in range(self.num_classes)}
        }

    def print_result(self, result: dict):
        print(f"\n{'='*55}")
        print(f"Text          : {result['text']}")
        print(f"Raw Emotion   : {result['raw_emotion']}")
        print(f"Final Emotion : {result['final_emotion']}"
              + (" ← adjusted" if result["adjusted"] else ""))
        print(f"Confidence    : {result['confidence']:.4f}")
        print(f"All Scores    :")
        for emotion, score in sorted(result["all_scores"].items(),
                                     key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"  {emotion:<10} {score:.4f}  {bar}")
        print(f"{'='*55}")


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":

    # ── PHASE 1: Train ────────────────────────────────────────────────────────
    config = {
        "train_path"           : "data/processed/train.csv",
        "val_path"             : "data/processed/val.csv",
        "test_path"            : "data/processed/test.csv",
        "label_map"            : "data/processed/label_map.json",
        "model_dir"            : "models/",
        "epochs"               : 5,
        "batch_size"           : 16,
        "lr"                   : 2e-5,
        "warmup_steps"         : 100,
        "freeze_base_initially": True   # Speeds up CPU training
    }

    model, tokenizer = train(config)

    # ── PHASE 2: Inference demo ───────────────────────────────────────────────
    print("\n[INFO] Running inference demo...\n")
    inference = EmotionInferencePipeline(
        model_path        = "models/best_model.pt",
        label_map_path    = "data/processed/label_map.json",
        use_sarcasm_model = True
    )

    test_texts = [
        "I felt so happy and grateful when I got the news",
        "The accident left me feeling terrified and helpless",
        "Oh great, another traffic jam on Monday morning",
        "I was not unhappy with the result at all",
        "I cannot believe they did that, I am absolutely furious",
        "smh omg i literally cannot believe this rn",
    ]

    for text in test_texts:
        result = inference.predict(text)
        inference.print_result(result)
