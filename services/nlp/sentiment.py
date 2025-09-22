# usage: document + sentence-level sentiment (HF emotion -> VADER fallback)
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

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

_VADER = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str, sent_threshold=0.5, max_sentences=0):
    sentences = sent_tokenize(text)
    if max_sentences and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    sentence_rows = []
    doc_accum = defaultdict(list)

    if _TRANS_OK:
        for s in sentences:
            try:
                out = _emo(s[:512])[0]
                best = max(out, key=lambda x: x["score"])
                if best["score"] >= sent_threshold:
                    sentence_rows.append({"sentence": s, "emotion": best["label"], "score": float(best["score"])})
                for d in out:
                    doc_accum[d["label"]].append(float(d["score"]))
            except Exception:
                continue
        doc_scores = {k: float(np.mean(v)) for k, v in doc_accum.items()} if doc_accum else {}
        method = "transformers_emotion"
    else:
        vs = _VADER.polarity_scores(text)
        doc_scores = {"vader_neg": vs["neg"], "vader_neu": vs["neu"], "vader_pos": vs["pos"], "vader_compound": vs["compound"]}
        for s in sentences:
            ss = _VADER.polarity_scores(s)
            strength = max(ss["neg"], ss["pos"])
            if strength >= sent_threshold:
                label = "positive" if ss["pos"] >= ss["neg"] else "negative"
                sentence_rows.append({"sentence": s, "emotion": label, "score": float(strength)})
        method = "vader"

    return doc_scores, pd.DataFrame(sentence_rows), method
