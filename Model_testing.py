

import os
import re
import json
import logging
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns


# ══════════════════════════════════════════════
# EMOTION TOKENIZER
# ══════════════════════════════════════════════
class EmotionTokenizer:
    ABBREVIATION_MAP = {
        "lol": "laughing out loud", "lmao": "laughing my head off",
        "omg": "oh my god",         "omfg": "oh my god",
        "smh": "shaking my head",   "tbh": "to be honest",
        "ngl": "not going to lie",  "idk": "i do not know",
        "wtf": "what the hell",     "fml": "my life is ruined",
        "ikr": "i know right",      "rn": "right now",
        "bc": "because",            "gonna": "going to",
        "wanna": "want to",         "gotta": "got to",
        "kinda": "kind of",         "sorta": "sort of",
        "dunno": "do not know",     "imo": "in my opinion",
    }

    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        self.tokenizer  = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def expand_abbreviations(self, text):
        words = text.lower().split()
        return " ".join([self.ABBREVIATION_MAP.get(w, w) for w in words])

    def encode(self, texts):
        expanded = [self.expand_abbreviations(t) for t in texts]
        return self.tokenizer(expanded, padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    def encode_single(self, text):
        return self.encode([text])


# ══════════════════════════════════════════════
# EMOTION CLASSIFIER
# ══════════════════════════════════════════════
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, model_name="distilbert-base-uncased",
                 dropout=0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden_size     = self.distilbert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


# ══════════════════════════════════════════════
# NEGATION HANDLER
# ══════════════════════════════════════════════
class NegationHandler:
    DOUBLE_NEG = [
        r"\bnot\s+\w*\s*(bad|terrible|awful|horrible|unpleasant)\b",
        r"\bnot\s+un\w+\b", r"\bnever\s+\w*\s*(bad|wrong|unfair)\b",
        r"\bno\s+\w*\s*(problem|issue|complaint|worries)\b",
    ]
    SINGLE_NEG = [
        r"\bnot\s+(happy|joyful|glad|pleased|excited)\b",
        r"\bnever\s+(happy|felt good|enjoyed)\b",
        r"\bno\s+(happiness|joy|pleasure)\b",
    ]

    def resolve(self, text):
        t      = text.lower()
        double = any(re.search(p, t) for p in self.DOUBLE_NEG)
        single = any(re.search(p, t) for p in self.SINGLE_NEG)
        return {"has_double_negation": double, "has_single_negation": single,
                "flip_emotion": single and not double}


# ══════════════════════════════════════════════
# SARCASM DETECTOR (rule-based only for speed)
# ══════════════════════════════════════════════
class SarcasmDetector:
    PATTERNS  = [r"\boh great\b", r"\bjust great\b", r"\byeah right\b",
                 r"\bas if\b",    r"\breally.*genius\b"]
    POS_WORDS = {"amazing","fantastic","wonderful","great","brilliant","perfect"}
    NEG_CTX   = {"monday","traffic","rain","fail","broke","lost","worst","hate"}

    def detect(self, text):
        t     = text.lower()
        words = set(t.split())
        for p in self.PATTERNS:
            if re.search(p, t):
                return {"is_sarcastic": True, "confidence": 0.85,
                        "method": "rule_based"}
        if (words & self.POS_WORDS) and (words & self.NEG_CTX):
            return {"is_sarcastic": True, "confidence": 0.75,
                    "method": "rule_based"}
        return {"is_sarcastic": False, "confidence": 1.0, "method": "none"}


EMOTION_FLIP_MAP = {
    "joy": "sadness", "sadness": "joy", "anger": "joy",
    "fear": "joy",    "disgust": "joy", "shame": "joy", "guilt": "joy"
}


# ══════════════════════════════════════════════
# LOGGER SETUP
# ══════════════════════════════════════════════
def setup_logger(log_dir="logs/"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"emotion_model_{timestamp}.log")

    logger = logging.getLogger("EmotionModel")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # clear existing handlers

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(console)
    logger.addHandler(fh)
    logger.info(f"Logger initialized → {log_file}")
    return logger


# ══════════════════════════════════════════════
# PREDICTION TRACKER
# ══════════════════════════════════════════════
class PredictionTracker:
    def __init__(self, save_path="logs/predictions.csv"):
        self.save_path = save_path
        self.records   = []
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def log(self, text, true_label, predicted, confidence, adjusted):
        self.records.append({
            "timestamp" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text"      : text[:200],
            "true_label": true_label,
            "predicted" : predicted,
            "confidence": round(confidence, 4),
            "correct"   : true_label == predicted,
            "adjusted"  : adjusted
        })

    def save(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.save_path, index=False)
        print(f"[INFO] Predictions saved → {self.save_path}")
        return df


# ══════════════════════════════════════════════
# REAL-WORLD TEST CASES
# ══════════════════════════════════════════════
REAL_WORLD_TEST_CASES = [
    {"text": "I just got promoted at work, I'm absolutely thrilled!",           "label": "joy"},
    {"text": "My best friend surprised me with a birthday party, so happy!",    "label": "joy"},
    {"text": "omg i just got accepted to my dream university!!!",                "label": "joy"},
    {"text": "I lost my dog today, he was my best friend for 10 years.",         "label": "sadness"},
    {"text": "Nobody showed up to my birthday party. I feel so alone.",          "label": "sadness"},
    {"text": "I failed my exam after studying so hard. I give up.",              "label": "sadness"},
    {"text": "I can't believe they lied to me after everything I did for them.", "label": "anger"},
    {"text": "The company fired me without any warning or reason. Outrageous.",  "label": "anger"},
    {"text": "smh people have no respect for others nowadays",                   "label": "anger"},
    {"text": "I have to give a speech in front of 500 people tomorrow.",         "label": "fear"},
    {"text": "The doctor said the results could be serious. I'm terrified.",     "label": "fear"},
    {"text": "Walking alone at night in that area made me extremely anxious.",   "label": "fear"},
    {"text": "The way they treated those animals was absolutely revolting.",     "label": "disgust"},
    {"text": "I feel sick seeing how corrupt the system has become.",            "label": "disgust"},
    {"text": "I cheated on the test and I can't look my teacher in the eye.",    "label": "shame"},
    {"text": "I said something really hurtful and I'm so ashamed of myself.",    "label": "shame"},
    {"text": "I should have been there for her when she needed me most.",        "label": "guilt"},
    {"text": "I feel so guilty for not telling the truth when I had the chance.","label": "guilt"},
    {"text": "Oh fantastic, my flight got cancelled again. Just perfect.",       "label": "anger"},  # sarcasm
    {"text": "I was not unhappy with how the presentation went.",                "label": "joy"},    # double negation
    {"text": "ngl tbh i kinda feel empty inside rn",                            "label": "sadness"},# slang
]


# ══════════════════════════════════════════════
# MODEL TESTER
# ══════════════════════════════════════════════
class ModelTester:
    def __init__(self, model_path, label_map_path, logger=None):
        with open(label_map_path) as f:
            self.label_map = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger      = logger or logging.getLogger("EmotionModel")
        self.tracker     = PredictionTracker()

        self.tokenizer        = EmotionTokenizer()
        self.model            = EmotionClassifier(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.sarcasm_detector = SarcasmDetector()
        self.negation_handler = NegationHandler()
        self.logger.info(f"ModelTester ready on {self.device}")

    def predict(self, text):
        enc = self.tokenizer.encode_single(text)
        with torch.no_grad():
            logits = self.model(enc["input_ids"].to(self.device),
                                enc["attention_mask"].to(self.device))
        probs        = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id      = int(np.argmax(probs))
        raw_emotion  = self.id_to_label[pred_id]
        confidence   = float(probs[pred_id])
        sarcasm      = self.sarcasm_detector.detect(text)
        negation     = self.negation_handler.resolve(text)
        adjusted     = sarcasm["is_sarcastic"] or negation["flip_emotion"]
        final_emotion= EMOTION_FLIP_MAP.get(raw_emotion, raw_emotion) if adjusted else raw_emotion
        return {"final_emotion": final_emotion, "raw_emotion": raw_emotion,
                "confidence": round(confidence, 4), "adjusted": adjusted,
                "all_scores": {self.id_to_label[i]: round(float(probs[i]), 4)
                               for i in range(self.num_classes)}}

    def run_test_suite(self, test_cases=None):
        test_cases = test_cases or REAL_WORLD_TEST_CASES
        self.logger.info(f"Running test suite — {len(test_cases)} samples...")
        results = []
        for case in test_cases:
            text, true_label = case["text"], case["label"]
            result = self.predict(text)
            self.tracker.log(text, true_label, result["final_emotion"],
                             result["confidence"], result["adjusted"])
            status = "✓" if true_label == result["final_emotion"] else "✗"
            self.logger.debug(f"{status} True:{true_label:<10} "
                              f"Pred:{result['final_emotion']:<10} | {text[:60]}")
            results.append({"text": text, "true_label": true_label,
                            "predicted": result["final_emotion"],
                            "confidence": result["confidence"],
                            "correct": true_label == result["final_emotion"],
                            "adjusted": result["adjusted"]})
        self.tracker.save()
        return pd.DataFrame(results)

    def generate_report(self, results_df, save_dir="logs/"):
        os.makedirs(save_dir, exist_ok=True)
        true_labels = results_df["true_label"].tolist()
        pred_labels = results_df["predicted"].tolist()
        accuracy    = accuracy_score(true_labels, pred_labels)
        f1          = f1_score(true_labels, pred_labels, average="weighted",
                               zero_division=0)

        self.logger.info(f"\n{'='*55}\nTEST RESULTS\n{'='*55}")
        self.logger.info(f"Samples   : {len(results_df)}")
        self.logger.info(f"Accuracy  : {accuracy:.4f}")
        self.logger.info(f"F1 Score  : {f1:.4f}")
        self.logger.info(f"\n{classification_report(true_labels, pred_labels, zero_division=0)}")

        # Confusion matrix
        emotions = sorted(results_df["true_label"].unique())
        cm = confusion_matrix(true_labels, pred_labels, labels=emotions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=emotions, yticklabels=emotions)
        plt.title("Confusion Matrix — Real World Test")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        self.logger.info(f"Confusion matrix → {cm_path}")

        # Improvement suggestions
        for emotion in emotions:
            subset = results_df[results_df["true_label"] == emotion]
            if len(subset) > 0 and subset["correct"].mean() < 0.6:
                self.logger.warning(
                    f"⚠ '{emotion}' accuracy low ({subset['correct'].mean():.2f})"
                    f" — add more training samples.")
        return {"accuracy": accuracy, "f1": f1}


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    logger = setup_logger()
    tester = ModelTester(
        model_path     = "models/best_model.pt",
        label_map_path = "data/processed/label_map.json",
        logger         = logger
    )
    results_df = tester.run_test_suite()
    tester.generate_report(results_df)
