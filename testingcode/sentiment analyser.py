import os
import pandas as pd
import docx2txt
from transformers import pipeline
import numpy as np
import re
import nltk
from pathlib import Path

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


def clean_text(text):
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


sentiment_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


def read_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif path.endswith(".docx"):
        return docx2txt.process(path)
    return None


def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def load_documents(root_dir):
    docs = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".txt", ".docx")):
                path = os.path.join(subdir, file)
                raw_text = read_file(path)
                if raw_text and raw_text.strip():
                    text = clean_text(raw_text)
                    region = os.path.basename(subdir)
                    title = os.path.splitext(file)[0]
                    docs.append({"title": title, "region": region, "text": text})
    print("Loaded docs:", len(docs))
    for d in docs[:3]:
        print("TITLE:", d["title"])
        print("REGION:", d["region"])
        print("TEXT SAMPLE:", d["text"][:200], "\n---\n")

    return pd.DataFrame(docs)





root_directory = Path(__file__).resolve().parent.parent / 'TextualDocuments'

print("Using root directory:", root_directory)

df = load_documents(root_directory)

results = []
sentence_hits = []

for _, row in df.iterrows():
    chunks = chunk_text(row["text"], chunk_size=300)
    doc_scores = []
    for chunk in chunks:
        try:
            out = sentiment_pipeline(chunk[:512])
            if isinstance(out, list) and isinstance(out[0], list):
                chunk_scores = out[0]
            else:
                chunk_scores = out
            doc_scores.append(chunk_scores)
        except Exception as e:
            print(f"Skipping chunk due to error: {e}")

    if doc_scores:
        emotions = doc_scores[0]
        avg_scores = {emo["label"]: [] for emo in emotions}
        for chunk_scores in doc_scores:
            for emo in chunk_scores:
                avg_scores[emo["label"]].append(emo["score"])
        avg_scores = {k: float(np.mean(v)) for k, v in avg_scores.items()}

        results.append({
            "title": row["title"],
            "region": row["region"],
            **avg_scores
        })

    sentences = sent_tokenize(row["text"])
    for sent in sentences:
        out = sentiment_pipeline(sent[:512])[0]
        best = max(out, key=lambda x: x["score"])
        if best["score"] > 0.5:
            sentence_hits.append({
                "title": row["title"],
                "region": row["region"],
                "sentence": sent,
                "emotion": best["label"],
                "score": best["score"]
            })


results_df = pd.DataFrame(results)
results_df.to_csv("emotion_results.csv", index=False)
print("\nDocument-level results:\n", results_df.head())

sentences_df = pd.DataFrame(sentence_hits)
sentences_df.to_csv("emotional_sentences.csv", index=False)
print("\nSentence-level highlights saved to emotional_sentences.csv")
