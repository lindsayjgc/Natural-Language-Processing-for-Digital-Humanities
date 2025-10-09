# NLP Pipeline – Text Processing, Features, and Sentiment

This repository provides a compact NLP toolkit and two CLI scripts to analyze text: tokenize/lemmatize, compute n‑grams and POS distributions, and extract document‑ and sentence‑level emotions.

## Key Components

- **`preprocessing.py`** – spaCy-first, NLTK‑fallback pipeline for sentence splitting, tokenization, stopword removal, lemmatization, POS tagging, vocabulary stats.
- **`features.py`** – N‑gram construction and frequency counting (unigram/bigram/trigram/…).
- **`sentiment.py`** – Document + sentence‑level sentiment using a Hugging Face emotion classifier with VADER fallback.
- **`analyze_texts.py`** – CLI for already ETL‑cleaned `.txt` inputs; writes CSV/JSON outputs.
- **`naturalLanguageProcessingOfDocuments.py`** – CLI that can also ingest raw files (PDF/DOCX/RTF) via optional ETL helpers when `--from-raw` is used.
- **`__init__.py`** – Convenient exports for library-style usage.

> The CLIs write tables for word frequency, n‑grams, POS counts, sentence‑level emotion highlights, and a JSON summary per input file.

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
# pip install pypdf python-docx

# Download the small English model for spaCy
python -m spacy download en_core_web_sm
```

> GPU is optional. If PyTorch detects CUDA, `analyze_texts.py` will print that it's using the GPU; otherwise CPU is used.

### 2) NLTK Resources

`preprocessing.py` and `sentiment.py` auto‑download necessary NLTK data at runtime (punkt, stopwords, wordnet, taggers, VADER). No manual steps required in most cases.

### 3) Run on Clean `.txt` Files

```bash
# Analyze a single cleaned text file
python analyze_texts.py --input path/to/clean.txt --outdir data/outputs

# Analyze all .txt files in a folder, use top-100 terms, and only pick confident emotional sentences
python analyze_texts.py --input path/to/folder --outdir data/outputs   --topn 100 --sent-threshold 0.6 --ngrams 1,2,3
```

This produces (per input file):

- `<TAG>_wordfreq_top{N}.csv`
- `<TAG>_{unigram|bigram|trigram|...}_top{N}.csv`
- `<TAG>_pos_counts.csv`
- `<TAG>_emotional_sentences.csv`
- `<TAG>_summary.json` (doc metadata & emotion backend)
- `<TAG>_doc_emotion_vertical.csv` (sorted doc‑level emotion scores)

### 4) Run on Raw Documents (PDF/DOCX/RTF/TXT)

```bash
# Use the ETL-style path to read/normalize raw files before NLP
python naturalLanguageProcessingOfDocuments.py   --input path/to/file_or_folder   --outdir nlp_outputs   --from-raw   --ngrams 1,2,3 --topn 50 --sent-threshold 0.5
```

> In `--from-raw` mode, the script expects ETL helpers to be available on your PYTHONPATH (e.g., `services.etl.readers/normalizers`). If you don’t have them, stick to clean `.txt` inputs or wire in your ETL package.

---

## Library Usage (import and call)

```python
from preprocessing import process_text
from features import count_ngrams, count_pos
from sentiment import analyze_sentiment

text = "Your document text..."

prep = process_text(text, lowercase=True, remove_punct=True, remove_nums=True, remove_stop=True)
grams = count_ngrams(prep["lemmas"], ngram_ns=(1,2,3))
pos_counts = count_pos(prep["pos_seq"])
doc_sent, sent_df, method = analyze_sentiment(text, sent_threshold=0.5, max_sentences=0)

print(prep["vocab_size"], prep["token_count"], prep["type_token_ratio"])
print(doc_sent)          # document-level emotion profile
print(sent_df.head())    # high-confidence emotional sentences
```

### Exported Symbols

If you package these modules, `__init__.py` exposes:

```python
process_text, count_ngrams, count_pos, analyze_sentiment, __version__
```

---

## Outputs Explained

- **Word frequency**: lemma counts and vocabulary statistics (type-token ratio).
- **N‑grams**: common multi‑word terms via `_`-joined tokens (e.g., `like_cats`).
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

### Example `requirements.txt`

```
spacy>=3.7
nltk>=3.9
pandas>=2.2
numpy>=1.26
torch>=2.2        # optional for GPU; CPU works too
transformers>=4.44
```

---

## Repository Structure

```
.
├── analyze_texts.py                         # CLI for clean .txt inputs
├── naturalLanguageProcessingOfDocuments.py  # CLI; can use --from-raw
├── preprocessing.py                         # spaCy/NLTK processing
├── features.py                              # n-grams & POS counters
├── sentiment.py                             # emotions & fallback
└── __init__.py                              # exports for library usage
```

---

## License

Add your preferred license here (MIT/Apache-2.0/BSD-3/etc.).
