# usage: document + sentence-level sentiment (HF emotion -> VADER fallback)
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is available for fallback sentiment analysis
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Try loading Hugging Face emotion classifier (preferred)
_TRANS_OK = False
try:
    from transformers import pipeline as hf_pipeline
    _emo = hf_pipeline("text-classification",
                       model="j-hartmann/emotion-english-distilroberta-base",
                       top_k=None)
    _TRANS_OK = True
except Exception:
    _emo = None
    _TRANS_OK = False

# Initialize VADER sentiment analyzer (rule-based fallback)
_VADER = SentimentIntensityAnalyzer()

# Unified sentiment analysis interface
def analyze_sentiment(text: str, sent_threshold=0.5, max_sentences=0):
    # Split text into sentences
    sentences = sent_tokenize(text)
    if max_sentences and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    sentence_rows = []   # Holds per-sentence sentiment
    doc_accum = defaultdict(list)  # Aggregates document-level scores

    if _TRANS_OK:
        # Transformer-based emotion classification
        for s in sentences:
            try:
                # HuggingFace pipeline expects max 512 tokens
                out = _emo(s[:512])[0]
                # Select best (highest score) label
                best = max(out, key=lambda x: x["score"])
                if best["score"] >= sent_threshold:
                    sentence_rows.append({"sentence": s, "emotion": best["label"], "score": float(best["score"])})
                # Accumulate scores for averaging
                for d in out:
                    doc_accum[d["label"]].append(float(d["score"]))
            except Exception:
                continue
        # Average scores for doc-level sentiment
        doc_scores = {k: float(np.mean(v)) for k, v in doc_accum.items()} if doc_accum else {}
        method = "transformers_emotion"
    else:
        # Fallback: VADER polarity analysis
        vs = _VADER.polarity_scores(text)
        doc_scores = {"vader_neg": vs["neg"], "vader_neu": vs["neu"], "vader_pos": vs["pos"], "vader_compound": vs["compound"]}
        for s in sentences:
            ss = _VADER.polarity_scores(s)
            strength = max(ss["neg"], ss["pos"])
            if strength >= sent_threshold:
                label = "positive" if ss["pos"] >= ss["neg"] else "negative"
                sentence_rows.append({"sentence": s, "emotion": label, "score": float(strength)})
        method = "vader"

    # Return:
    # - doc_scores: aggregate sentiment profile
    # - sentence_rows: per-sentence emotion classification
    # - method: which backend was used
    return doc_scores, pd.DataFrame(sentence_rows), method
