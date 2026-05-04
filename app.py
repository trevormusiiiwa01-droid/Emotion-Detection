

import os
import re
import json
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import streamlit as st
import plotly.express as px


# ══════════════════════════════════════════════
# SHARED CLASSES
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

EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨",
    "disgust": "🤢", "shame": "😳", "guilt": "😔", "neutral": "😐"
}

EMOTION_COLOR = {
    "joy": "#FFD700",    "sadness": "#4169E1", "anger": "#FF4500",
    "fear": "#9400D3",   "disgust": "#228B22", "shame": "#FF69B4",
    "guilt": "#8B4513",  "neutral": "#808080"
}


# ══════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════
@st.cache_resource
def load_model(model_path, label_map_path):
    with open(label_map_path) as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer   = EmotionTokenizer()
    model       = EmotionClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return (model, tokenizer, SarcasmDetector(), NegationHandler(),
            label_map, id_to_label, device)


def predict(text, model, tokenizer, sarcasm_detector,
            negation_handler, id_to_label, device):
    enc = tokenizer.encode_single(text)
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device),
                       enc["attention_mask"].to(device))
    probs        = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_id      = int(np.argmax(probs))
    raw_emotion  = id_to_label[pred_id]
    confidence   = float(probs[pred_id])
    sarcasm      = sarcasm_detector.detect(text)["is_sarcastic"]
    negation     = negation_handler.resolve(text)["flip_emotion"]
    adjusted     = sarcasm or negation
    final_emotion= EMOTION_FLIP_MAP.get(raw_emotion, raw_emotion) if adjusted else raw_emotion
    return {
        "final_emotion": final_emotion, "raw_emotion": raw_emotion,
        "confidence": round(confidence, 4), "adjusted": adjusted,
        "sarcasm": sarcasm, "negation": negation,
        "all_scores": {id_to_label[i]: round(float(probs[i]), 4)
                       for i in range(len(id_to_label))}
    }


def log_prediction(text, result):
    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = []
    st.session_state.prediction_log.append({
        "timestamp" : datetime.datetime.now().strftime("%H:%M:%S"),
        "text"      : text[:80],
        "emotion"   : result["final_emotion"],
        "confidence": result["confidence"],
        "adjusted"  : "Yes" if result["adjusted"] else "No"
    })


# ══════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="wide")

# Sidebar
st.sidebar.title("🎭 Emotion Detector")
st.sidebar.markdown("---")
model_path     = st.sidebar.text_input("Model Path",
                                        value="models/best_model.pt")
label_map_path = st.sidebar.text_input("Label Map Path",
                                        value="data/processed/label_map.json")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** DistilBERT")
st.sidebar.markdown("**Dataset:** ISEAR")
st.sidebar.markdown("**Emotions:** joy, sadness, anger, fear, disgust, shame, guilt")

# Check model exists
if not os.path.exists(model_path):
    st.error(f"Model not found at `{model_path}`. Train the model first using `step2_complete.py`.")
    st.stop()

(model, tokenizer, sarcasm_detector, negation_handler,
 label_map, id_to_label, device) = load_model(model_path, label_map_path)

st.sidebar.success(f"✅ Model loaded ({device})")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction", "📂 Batch Prediction",
    "📊 Prediction Log",    "⚖️ A/B Comparison"
])

# ── TAB 1: Single Prediction ─────────────────────────────────────────────────
with tab1:
    st.title("🔍 Real-Time Emotion Detection")
    text_input  = st.text_area("Enter text:", height=120,
                                placeholder="Type anything...")
    predict_btn = st.button("Detect Emotion", type="primary")

    if predict_btn and text_input.strip():
        with st.spinner("Analyzing..."):
            result = predict(text_input, model, tokenizer,
                             sarcasm_detector, negation_handler,
                             id_to_label, device)
        log_prediction(text_input, result)

        emotion = result["final_emotion"]
        emoji   = EMOTION_EMOJI.get(emotion, "❓")
        color   = EMOTION_COLOR.get(emotion, "#808080")

        st.markdown(f"""
        <div style="background:{color}22; border-left:5px solid {color};
                    padding:20px; border-radius:8px; margin:10px 0;">
            <h2 style="color:{color}; margin:0">{emoji} {emotion.upper()}</h2>
            <p style="margin:5px 0; font-size:16px">
                Confidence: <b>{result['confidence']*100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if result["sarcasm"]:
            st.warning("⚠️ Sarcasm detected — emotion adjusted.")
        if result["negation"]:
            st.info("ℹ️ Negation detected — emotion adjusted.")
        if result["adjusted"]:
            st.caption(f"Raw prediction: **{result['raw_emotion']}** → "
                       f"adjusted to **{result['final_emotion']}**")

        st.markdown("### All Emotion Scores")
        scores_df = pd.DataFrame(list(result["all_scores"].items()),
                                  columns=["Emotion", "Score"]
                                  ).sort_values("Score", ascending=True)
        fig = px.bar(scores_df, x="Score", y="Emotion", orientation="h",
                     color="Emotion", color_discrete_map=EMOTION_COLOR,
                     range_x=[0, 1])
        fig.update_layout(showlegend=False, height=300,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    elif predict_btn:
        st.warning("Please enter some text first.")

# ── TAB 2: Batch Prediction ───────────────────────────────────────────────────
with tab2:
    st.title("📂 Batch Emotion Detection")
    st.markdown("Upload a CSV with a `text` column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a column named `text`.")
        else:
            st.success(f"Loaded {len(df)} rows.")
            st.dataframe(df.head())

            if st.button("Run Batch Prediction", type="primary"):
                progress = st.progress(0)
                results  = []
                for i, row in df.iterrows():
                    r = predict(str(row["text"]), model, tokenizer,
                                sarcasm_detector, negation_handler,
                                id_to_label, device)
                    results.append({"text": row["text"],
                                    "emotion": r["final_emotion"],
                                    "confidence": r["confidence"],
                                    "adjusted": r["adjusted"]})
                    progress.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)
                st.success("Done!")
                st.dataframe(results_df)

                dist = results_df["emotion"].value_counts().reset_index()
                dist.columns = ["Emotion", "Count"]
                fig = px.pie(dist, names="Emotion", values="Count",
                             title="Emotion Distribution",
                             color="Emotion", color_discrete_map=EMOTION_COLOR)
                st.plotly_chart(fig, use_container_width=True)

                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv,
                                   "batch_results.csv", "text/csv")

# ── TAB 3: Prediction Log ─────────────────────────────────────────────────────
with tab3:
    st.title("📊 Live Prediction Log")
    if "prediction_log" not in st.session_state or \
       not st.session_state.prediction_log:
        st.info("No predictions yet. Use Single Prediction tab first.")
    else:
        log_df = pd.DataFrame(st.session_state.prediction_log)
        st.dataframe(log_df, use_container_width=True)

        dist = log_df["emotion"].value_counts().reset_index()
        dist.columns = ["Emotion", "Count"]
        fig = px.bar(dist, x="Emotion", y="Count", color="Emotion",
                     color_discrete_map=EMOTION_COLOR,
                     title="Emotion Distribution (This Session)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Predictions", len(log_df))
        col2.metric("Avg Confidence", f"{log_df['confidence'].mean()*100:.1f}%")

        if st.button("🗑️ Clear Log"):
            st.session_state.prediction_log = []
            st.rerun()

# ── TAB 4: A/B Comparison ────────────────────────────────────────────────────
with tab4:
    st.title("⚖️ A/B Model Comparison")
    st.markdown("Compare two model versions on the same input.")

    col1, col2   = st.columns(2)
    model_a_path = col1.text_input("Model A", value="models/best_model.pt")
    model_b_path = col2.text_input("Model B", value="models/best_model_v2.pt")
    ab_text      = st.text_area("Text to compare:", height=100)

    if st.button("Compare") and ab_text.strip():
        results = {}
        for name, path in [("Model A", model_a_path), ("Model B", model_b_path)]:
            if os.path.exists(path):
                m, tok, sd, nh, lm, itl, dev = load_model(path, label_map_path)
                results[name] = predict(ab_text, m, tok, sd, nh, itl, dev)
            else:
                st.warning(f"{name} not found at `{path}`")

        if len(results) == 2:
            col1, col2 = st.columns(2)
            for col, (name, result) in zip([col1, col2], results.items()):
                emotion = result["final_emotion"]
                color   = EMOTION_COLOR.get(emotion, "#808080")
                emoji   = EMOTION_EMOJI.get(emotion, "❓")
                with col:
                    st.markdown(f"### {name}")
                    st.markdown(f"""
                    <div style="background:{color}22; border-left:4px solid {color};
                                padding:15px; border-radius:6px;">
                        <h3 style="color:{color}; margin:0">{emoji} {emotion.upper()}</h3>
                        <p>Confidence: <b>{result['confidence']*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    scores_df = pd.DataFrame(
                        list(result["all_scores"].items()),
                        columns=["Emotion", "Score"]
                    ).sort_values("Score", ascending=False)
                    st.dataframe(scores_df, use_container_width=True,
                                 hide_index=True)

            if results["Model A"]["final_emotion"] == results["Model B"]["final_emotion"]:
                st.success("✅ Both models agree.")
            else:
                st.warning("⚠️ Models disagree — check confidence scores.")

    # Load saved A/B report
    ab_summary_path = "logs/ab_testing/ab_summary.json"
    if os.path.exists(ab_summary_path):
        st.markdown("---")
        st.markdown("### Last Full A/B Test Results")
        with open(ab_summary_path) as f:
            summary = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model A Accuracy", f"{summary['accuracy_a']:.3f}")
        c2.metric("Model B Accuracy", f"{summary['accuracy_b']:.3f}",
                  delta=f"{summary['accuracy_b']-summary['accuracy_a']:+.3f}")
        c3.metric("Model A F1", f"{summary['f1_a']:.3f}")
        c4.metric("Model B F1", f"{summary['f1_b']:.3f}",
                  delta=f"{summary['f1_b']-summary['f1_a']:+.3f}")
        st.info(f"💡 {summary['recommendation']}")
        if os.path.exists("logs/ab_testing/ab_comparison.png"):
            st.image("logs/ab_testing/ab_comparison.png")
