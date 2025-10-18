# NLP Pipeline – Text Processing, Features, and Sentiment

This repository provides a compact NLP toolkit and a CLI script to analyze text: tokenize/lemmatize, compute n‑grams and POS distributions, and extract document‑ and sentence‑level emotions.

## Key Components

- **`preprocessing.py`** – spaCy-first, NLTK‑fallback pipeline for sentence splitting, tokenization, stopword removal, lemmatization, POS tagging, vocabulary stats.
- **`features.py`** – N‑gram construction and frequency counting (unigram/bigram/trigram/…).
- **`sentiment.py`** – Document + sentence‑level sentiment using a Hugging Face emotion classifier with VADER fallback.
- **`analyze_texts.py`** – CLI that can ingest .txt files and raw files (PDF/DOCX/RTF) via optional ETL helpers when `--from-raw` is used.
- **`__init__.py`** – Convenient exports for library-style usage.

> The CLI writes tables for word frequency, n‑grams, POS counts, sentence‑level emotion highlights, and a JSON summary per input file.

---

## Quickstart

### 1) Python & Dependencies

```bash
# Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Base runtime
pip install -U pip wheel

# Core libs used by this repo
pip install spacy nltk pandas numpy torch transformers

# (Optional) If you plan to parse PDFs/DOCX in --from-raw mode, install your ETL deps here as well.
pip install pypdf python-docx

# Download the small English model for spaCy
python -m spacy download en_core_web_sm
```

### 2) NLTK Resources

`preprocessing.py` and `sentiment.py` auto‑download necessary NLTK data at runtime (punkt, stopwords, wordnet, taggers, VADER). No manual steps required in most cases.

### 3) CLI Guide

This guide summarizes the command-line flags and outputs for the bundled scripts.

#### `analyze_texts.py` (clean `.txt` or `--from-raw`)

**Synopsis**

```bash
python analyze_texts.py --input <file_or_dir> [--outdir PATH] \
  [--ngrams "1,2,3"] [--topn N] [--sent-threshold FLOAT] [--max-sentences INT] [--from-raw]
```

**Modes**

- Default: expects clean `.txt` (single file or folder). 
- `--from-raw`: tries to read and normalize raw inputs (PDF/DOC/DOCX/RTF/TXT) using ETL helpers discovered on your PYTHONPATH.

**Outputs (per file)**

- `<TAG>_wordfreq_top{N}.csv`
- `<TAG>_{unigram|bigram|trigram|...}_top{N}.csv`
- `<TAG>_pos_counts.csv`
- `<TAG>_emotional_sentences.csv`
- `<TAG>_summary.json`
- `<TAG>_doc_emotion_vertical.csv`

**Tuning**

- Increase `--sent-threshold` to keep only stronger sentence hits.
- Use `--max-sentences` to cap per‑file runtime on very long documents.
- Reduce `--ngrams` to speed up feature extraction.


## Outputs Explained

- **Word frequency**: lemma counts and vocabulary statistics (type-token ratio).
- **N‑grams**: common multi‑word terms via underscore-joined tokens.
- **POS counts**: coarse POS distribution for stylistic/genre signals.
- **Sentence emotions**: the most confident sentences by emotion.
- **Doc sentiment**: averaged emotion scores (transformer) or VADER polarity scores if transformer is unavailable.

---

## Tips & Troubleshooting

- Extremely long files are processed in **chunks** with spaCy; the max length is raised to accommodate large corpora.
- If spaCy fails on pathological inputs, the code **falls back to NLTK** tokenization, lemmatization, and tagging.
- If the Hugging Face pipeline/model is unavailable, sentiment **falls back to VADER** polarity.
- For better reproducibility, pin library versions (see example below) and record your CLI flags in a run log.
- If you need faster runs:
  - Increase `--sent-threshold` and set `--max-sentences` to a lower value.
  - Reduce `--ngrams` to `1,2` or just `1`.
  - Use smaller `--topn`.

