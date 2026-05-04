

import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# SHARED CLASSES (copied — no external imports)
# ══════════════════════════════════════════════
class EmotionTokenizer:
    ABBREVIATION_MAP = {
        "lol": "laughing out loud", "lmao": "laughing my head off",
        "omg": "oh my god",         "smh": "shaking my head",
        "tbh": "to be honest",      "ngl": "not going to lie",
        "idk": "i do not know",     "wtf": "what the hell",
        "rn": "right now",          "bc": "because",
        "gonna": "going to",        "wanna": "want to",
        "kinda": "kind of",         "sorta": "sort of",
    }

    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        self.tokenizer  = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def expand_abbreviations(self, text):
        words = text.lower().split()
        return " ".join([self.ABBREVIATION_MAP.get(w, w) for w in words])

    def encode_single(self, text):
        expanded = self.expand_abbreviations(text)
        return self.tokenizer([expanded], padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, model_name="distilbert-base-uncased", dropout=0.3):
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


class NegationHandler:
    DOUBLE_NEG = [r"\bnot\s+\w*\s*(bad|terrible|awful|horrible)\b",
                  r"\bnot\s+un\w+\b"]
    SINGLE_NEG = [r"\bnot\s+(happy|joyful|glad|pleased|excited)\b"]

    def resolve(self, text):
        t      = text.lower()
        double = any(re.search(p, t) for p in self.DOUBLE_NEG)
        single = any(re.search(p, t) for p in self.SINGLE_NEG)
        return {"flip_emotion": single and not double}


class SarcasmDetector:
    PATTERNS  = [r"\boh great\b", r"\bjust great\b", r"\byeah right\b", r"\bas if\b"]
    POS_WORDS = {"amazing", "fantastic", "wonderful", "great", "brilliant"}
    NEG_CTX   = {"monday", "traffic", "fail", "broke", "lost", "worst", "hate"}

    def detect(self, text):
        t, words = text.lower(), set(text.lower().split())
        for p in self.PATTERNS:
            if re.search(p, t):
                return {"is_sarcastic": True}
        if (words & self.POS_WORDS) and (words & self.NEG_CTX):
            return {"is_sarcastic": True}
        return {"is_sarcastic": False}


EMOTION_FLIP_MAP = {
    "joy": "sadness", "sadness": "joy", "anger": "joy",
    "fear": "joy",    "disgust": "joy", "shame": "joy", "guilt": "joy"
}

REAL_WORLD_TEST_CASES = [
    {"text": "I just got promoted at work, I'm absolutely thrilled!",           "label": "joy"},
    {"text": "My best friend surprised me with a birthday party!",              "label": "joy"},
    {"text": "omg i just got accepted to my dream university!!!",               "label": "joy"},
    {"text": "I lost my dog today, he was my best friend for 10 years.",        "label": "sadness"},
    {"text": "Nobody showed up to my birthday party. I feel so alone.",         "label": "sadness"},
    {"text": "I failed my exam after studying so hard. I give up.",             "label": "sadness"},
    {"text": "I can't believe they lied to me after everything I did.",         "label": "anger"},
    {"text": "The company fired me without any warning. Outrageous.",           "label": "anger"},
    {"text": "I have to give a speech in front of 500 people tomorrow.",        "label": "fear"},
    {"text": "The doctor said the results could be serious. I'm terrified.",    "label": "fear"},
    {"text": "The way they treated those animals was absolutely revolting.",    "label": "disgust"},
    {"text": "I cheated on the test and can't look my teacher in the eye.",     "label": "shame"},
    {"text": "I should have been there for her when she needed me most.",       "label": "guilt"},
    {"text": "Oh fantastic, my flight got cancelled again. Just perfect.",      "label": "anger"},
    {"text": "I was not unhappy with how the presentation went.",               "label": "joy"},
    {"text": "ngl tbh i kinda feel empty inside rn",                           "label": "sadness"},
]


# ══════════════════════════════════════════════
# MODEL WRAPPER
# ══════════════════════════════════════════════
class ModelWrapper:
    def __init__(self, model_path, label_map, name="Model"):
        self.name        = name
        self.id_to_label = {v: k for k, v in label_map.items()}
        self.num_classes = len(label_map)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer        = EmotionTokenizer()
        self.sarcasm_detector = SarcasmDetector()
        self.negation_handler = NegationHandler()
        self.model = EmotionClassifier(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] {name} loaded from {model_path}")

    def predict(self, text):
        enc = self.tokenizer.encode_single(text)
        with torch.no_grad():
            logits = self.model(enc["input_ids"].to(self.device),
                                enc["attention_mask"].to(self.device))
        probs        = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id      = int(np.argmax(probs))
        raw_emotion  = self.id_to_label[pred_id]
        confidence   = float(probs[pred_id])
        adjusted     = (self.sarcasm_detector.detect(text)["is_sarcastic"] or
                        self.negation_handler.resolve(text)["flip_emotion"])
        final_emotion= EMOTION_FLIP_MAP.get(raw_emotion, raw_emotion) if adjusted else raw_emotion
        return {"final_emotion": final_emotion, "confidence": round(confidence, 4),
                "adjusted": adjusted}


# ══════════════════════════════════════════════
# A/B TESTER
# ══════════════════════════════════════════════
class ABTester:
    def __init__(self, model_a, model_b, save_dir="logs/ab_testing/"):
        self.model_a  = model_a
        self.model_b  = model_b
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def run(self, test_cases=None):
        test_cases = test_cases or REAL_WORLD_TEST_CASES
        print(f"\n[INFO] Running A/B test — {len(test_cases)} samples")
        print(f"  Model A: {self.model_a.name} | Model B: {self.model_b.name}\n")

        records = []
        for case in test_cases:
            text, true_label = case["text"], case["label"]
            ra = self.model_a.predict(text)
            rb = self.model_b.predict(text)
            records.append({
                "text"           : text,
                "true_label"     : true_label,
                "pred_a"         : ra["final_emotion"],
                "pred_b"         : rb["final_emotion"],
                "conf_a"         : ra["confidence"],
                "conf_b"         : rb["confidence"],
                "correct_a"      : true_label == ra["final_emotion"],
                "correct_b"      : true_label == rb["final_emotion"],
                "agree"          : ra["final_emotion"] == rb["final_emotion"],
                "a_right_b_wrong": (true_label == ra["final_emotion"] and
                                    true_label != rb["final_emotion"]),
                "b_right_a_wrong": (true_label == rb["final_emotion"] and
                                    true_label != ra["final_emotion"]),
            })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.save_dir, "ab_results.csv"), index=False)
        return df

    def report(self, df):
        true  = df["true_label"].tolist()
        acc_a = accuracy_score(true, df["pred_a"].tolist())
        acc_b = accuracy_score(true, df["pred_b"].tolist())
        f1_a  = f1_score(true, df["pred_a"].tolist(), average="weighted", zero_division=0)
        f1_b  = f1_score(true, df["pred_b"].tolist(), average="weighted", zero_division=0)

        print(f"\n{'='*60}\nA/B TESTING RESULTS\n{'='*60}")
        print(f"{'Metric':<25} {self.model_a.name:<20} {self.model_b.name:<20}")
        print(f"{'-'*60}")
        print(f"{'Accuracy':<25} {acc_a:<20.4f} {acc_b:<20.4f}  "
              f"{'← B wins' if acc_b > acc_a else '← A wins' if acc_a > acc_b else 'TIE'}")
        print(f"{'Weighted F1':<25} {f1_a:<20.4f} {f1_b:<20.4f}  "
              f"{'← B wins' if f1_b > f1_a else '← A wins' if f1_a > f1_b else 'TIE'}")
        print(f"{'Avg Confidence':<25} {df['conf_a'].mean():<20.4f} {df['conf_b'].mean():<20.4f}")
        print(f"{'Agreement Rate':<25} {df['agree'].mean():<20.4f}")
        print(f"{'A right, B wrong':<25} {df['a_right_b_wrong'].sum()}")
        print(f"{'B right, A wrong':<25} {df['b_right_a_wrong'].sum()}")

        winner = ("B" if f1_b > f1_a else "A" if f1_a > f1_b else "TIE")
        rec    = (f"Deploy {self.model_b.name}" if winner == "B"
                  else f"Keep {self.model_a.name}" if winner == "A"
                  else "No improvement — keep Model A")
        print(f"\n✅ WINNER: Model {winner}")
        print(f"Recommendation: {rec}")

        # Per-class comparison
        emotions = sorted(df["true_label"].unique())
        print(f"\nPer-class F1:  {'Emotion':<12} {'Model A':<12} {'Model B':<12} Winner")
        print(f"  {'-'*45}")
        for e in emotions:
            mask = df["true_label"] == e
            fa   = f1_score(df[mask]["true_label"], df[mask]["pred_a"],
                            labels=[e], average="macro", zero_division=0)
            fb   = f1_score(df[mask]["true_label"], df[mask]["pred_b"],
                            labels=[e], average="macro", zero_division=0)
            w    = "B" if fb > fa else "A" if fa > fb else "-"
            print(f"  {e:<12} {fa:<12.4f} {fb:<12.4f} {w}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metrics  = ["Accuracy", "Weighted F1"]
        vals_a   = [acc_a, f1_a]
        vals_b   = [acc_b, f1_b]
        x, width = np.arange(2), 0.35
        axes[0].bar(x - width/2, vals_a, width, label=self.model_a.name, color="#4C72B0")
        axes[0].bar(x + width/2, vals_b, width, label=self.model_b.name, color="#DD8452")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Performance Comparison")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
            axes[0].text(i - width/2, va + 0.01, f"{va:.3f}", ha="center", fontsize=9)
            axes[0].text(i + width/2, vb + 0.01, f"{vb:.3f}", ha="center", fontsize=9)
        axes[1].hist(df["conf_a"], bins=15, alpha=0.6, label=self.model_a.name, color="#4C72B0")
        axes[1].hist(df["conf_b"], bins=15, alpha=0.6, label=self.model_b.name, color="#DD8452")
        axes[1].set_title("Confidence Distribution")
        axes[1].set_xlabel("Confidence")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        chart_path = os.path.join(self.save_dir, "ab_comparison.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"\n[INFO] Chart saved → {chart_path}")

        with open(os.path.join(self.save_dir, "ab_summary.json"), "w") as f:
            json.dump({"model_a": self.model_a.name, "model_b": self.model_b.name,
                       "accuracy_a": acc_a, "accuracy_b": acc_b,
                       "f1_a": f1_a, "f1_b": f1_b,
                       "agreement_rate": float(df["agree"].mean()),
                       "recommendation": rec}, f, indent=2)
        print(f"[INFO] Summary saved → {self.save_dir}ab_summary.json")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    with open("data/processed/label_map.json") as f:
        label_map = json.load(f)

    # NOTE: Copy your model to create a v2 for testing:
    #   cp models/best_model.pt models/best_model_v2.pt
    model_a = ModelWrapper("models/best_model.pt",    label_map, "Model A (v1)")
    model_b = ModelWrapper("models/best_model_v2.pt", label_map, "Model B (v2)")

    tester = ABTester(model_a, model_b)
    df     = tester.run()
    tester.report(df)
