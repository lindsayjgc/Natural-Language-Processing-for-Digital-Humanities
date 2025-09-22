# usage: python .\naturalLanguageProcessingOfDocuments.py --input "filename" --outdir "outputdir"
import os, re, json, argparse, hashlib, zipfile, html
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    import docx2txt
except Exception:
    docx2txt = None

_DOCX_OK = False
try:
    import docx as _py_docx
    _DOCX_OK = True
except Exception:
    _py_docx = None
    _DOCX_OK = False

_TEXTRACT_OK = False
try:
    import textract
    _TEXTRACT_OK = True
except Exception:
    textract = None
    _TEXTRACT_OK = False

_SPACY_OK = False
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    _nlp = None
    _SPACY_OK = False

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

import nltk
for pkg, locator in [
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("universal_tagset", "help/tagsets/upenn_tagset.pickle"),
    ("vader_lexicon", "sentiment/vader_lexicon.zip")
]:
    try:
        nltk.data.find(locator)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

_STOPWORDS = set(stopwords.words("english"))
_WORDNET_LEM = WordNetLemmatizer()
_VADER = SentimentIntensityAnalyzer()

def _read_txt(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    with open(p, "rb") as f:
        return f.read().decode("utf-8", "ignore")

def _read_docx_with_docx2txt(p: Path) -> str:
    return docx2txt.process(str(p)) or ""

def _read_docx_with_python_docx(p: Path) -> str:
    d = _py_docx.Document(str(p))
    return "\n".join(par.text for par in d.paragraphs)

def _read_docx_by_zip(p: Path) -> str:
    with zipfile.ZipFile(str(p)) as z:
        try:
            data = z.read("word/document.xml").decode("utf-8", "ignore")
        except KeyError:
            # Some docs store body in alt parts; fallback to concatenated text of all XML
            data = "".join(z.read(n).decode("utf-8", "ignore")
                           for n in z.namelist() if n.endswith(".xml"))
    # crude XML text extraction
    data = re.sub(r"</w:p>", "\n", data)
    data = re.sub(r"<[^>]+>", "", data)
    return html.unescape(data)

def _read_rtf_simple(p: Path) -> str:
    raw = p.read_bytes()
    try:
        txt = raw.decode("utf-8", "ignore")
    except Exception:
        txt = raw.decode("latin-1", "ignore")
    txt = re.sub(r"\\'[0-9a-fA-F]{2}", lambda m: bytes.fromhex(m.group(0)[2:]).decode('latin-1','ignore'), txt)
    txt = re.sub(r"\\[A-Za-z]+-?\d* ?", "", txt)
    txt = re.sub(r"[{}]", "", txt)
    return re.sub(r"\s+\n", "\n", txt)

def _sniff_kind(p: Path) -> str:
    with open(p, "rb") as f:
        head = f.read(8)
    if head.startswith(b"PK"):
        return "docx_zip"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole_doc"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head.startswith(b"%PDF"):
        return "pdf"
    return "text_or_unknown"

def read_text(path: Path) -> str:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".txt", ".md", ".rst"):
        return _read_txt(p)
    if suf == ".docx":
        kind = _sniff_kind(p)
        if kind != "docx_zip":
            if kind == "rtf":
                return _read_rtf_simple(p)
            if kind == "ole_doc" and _TEXTRACT_OK:
                return textract.process(str(p)).decode("utf-8", "ignore")
            # best-effort plain text fallback
            return _read_txt(p)
        try:
            if docx2txt is not None:
                return _read_docx_with_docx2txt(p)
        except Exception:
            pass
        if _DOCX_OK:
            try:
                return _read_docx_with_python_docx(p)
            except Exception:
                pass
        return _read_docx_by_zip(p)
    if suf == ".doc":
        if _TEXTRACT_OK:
            return textract.process(str(p)).decode("utf-8", "ignore")
        raise RuntimeError("Legacy .doc detected. Install `textract` (and its OS deps) or convert to .docx/.txt.")
    if suf == ".rtf":
        return _read_rtf_simple(p)
    if suf == ".pdf":
        if _TEXTRACT_OK:
            return textract.process(str(p)).decode("utf-8", "ignore")
        raise RuntimeError("PDF detected. Install `textract` or convert to .txt.")
    return _read_txt(p)

def strip_gutenberg_headers(text: str) -> str:
    lines = text.splitlines()
    start_re = re.compile(r"""^(\*\*\*\s*START\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|\*\*\*\s*START\s+OF\s+.*EBOOK|START\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK)""", re.IGNORECASE)
    end_re = re.compile(r"""^(\*\*\*\s*END\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|\*\*\*\s*END\s+OF\s+.*EBOOK|END\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|End\s+of\s+the\s+Project\s+Gutenberg\s+EBook|End\s+of\s+Project\s+Gutenberg'?s)""", re.IGNORECASE)
    start_idx, end_idx = 0, len(lines)
    for i, ln in enumerate(lines):
        if start_re.search(ln.strip()):
            start_idx = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if end_re.search(lines[i].strip()):
            end_idx = i
            break
    if start_idx == 0 and end_idx == len(lines):
        return text.strip()
    body = "\n".join(lines[start_idx:end_idx]).strip()
    return body if body else text.strip()

def basic_clean(text: str, unwrap_lines=True) -> str:
    text = re.sub(r'\r\n?', '\n', text)
    if unwrap_lines:
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def words_from_tokens(tokens, *, lowercase=True, remove_punct=True, remove_nums=True):
    out = []
    for t in tokens:
        w = t.lower() if lowercase else t
        if remove_punct and re.fullmatch(r"\W+", w):
            continue
        if remove_nums and re.fullmatch(r"\d+([.,]\d+)?", w):
            continue
        out.append(w)
    return out

def wn_pos_for(word, treebank_tag: str):
    tag = treebank_tag[:1].upper()
    return {'J':'a','N':'n','V':'v','R':'r'}.get(tag, 'n')

def lemmatize_tokens(tokens, pos_tags=None):
    if _SPACY_OK:
        doc = _nlp(" ".join(tokens))
        return [t.lemma_ if t.lemma_ not in ("-PRON-",) else t.lower_ for t in doc]
    if pos_tags is None:
        pos_tags = pos_tag(tokens)
    lemmas = []
    for w, tag in pos_tags:
        lemmas.append(_WORDNET_LEM.lemmatize(w, wn_pos_for(w, tag)))
    return lemmas

def ngrams(tokens, n=2):
    return ["_".join(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

def hash_stem(path: Path):
    stem = path.stem
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{h}"

def sentiment_doc_and_sentences(text: str, sent_threshold=0.5, max_sentences=0):
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

def process_text(text: str,
                 *,
                 lowercase=True,
                 remove_punct=True,
                 remove_nums=True,
                 remove_stop=True,
                 ngram_ns=(1,2,3)):
    if _SPACY_OK:
        doc = _nlp(text)
        sent_strs = [s.text.strip() for s in doc.sents if s.text.strip()]
        tokens_raw = [t.text for t in doc]
        pos_seq = [t.pos_ for t in doc if not t.is_space]
    else:
        sent_strs = [s.strip() for s in sent_tokenize(text) if s.strip()]
        tokens_raw = word_tokenize(text)
        pos_seq = [p for _, p in pos_tag([w for w in tokens_raw if re.search(r"\S", w)], tagset="universal")]
    tokens = words_from_tokens(tokens_raw, lowercase=lowercase, remove_punct=remove_punct, remove_nums=remove_nums)
    tokens_nostop = [t for t in tokens if (t not in _STOPWORDS)] if remove_stop else list(tokens)
    lemmas = lemmatize_tokens(tokens_nostop)
    freq_lemmas = Counter(lemmas)
    vocab_size = len(freq_lemmas)
    token_count = len(lemmas)
    ttr = (vocab_size / token_count) if token_count else 0.0
    ngram_counts = {}
    for n in ngram_ns:
        if n <= 1:
            ngram_counts["unigram"] = freq_lemmas
        else:
            grams = Counter(ngrams(lemmas, n=n))
            key = {2:"bigram",3:"trigram"}.get(n, f"ngram_{n}")
            ngram_counts[key] = grams
    pos_counts = Counter(pos_seq)
    return {
        "sentences": sent_strs,
        "tokens": tokens,
        "tokens_nostop": tokens_nostop,
        "lemmas": lemmas,
        "freq_lemmas": freq_lemmas,
        "ngrams": ngram_counts,
        "pos_counts": pos_counts,
        "vocab_size": vocab_size,
        "token_count": token_count,
        "type_token_ratio": ttr
    }

def process_path(input_path: Path,
                 outdir: Path,
                 *,
                 strip_gut: bool = True,
                 lowercase=True,
                 remove_punct=True,
                 remove_nums=True,
                 remove_stop=True,
                 ngram_ns=(1,2,3),
                 sent_threshold=0.5,
                 max_sentences=0,
                 topn=50):
    raw = read_text(input_path)
    if strip_gut:
        raw = strip_gutenberg_headers(raw)
    text = basic_clean(raw, unwrap_lines=True)
    out = process_text(text,
                       lowercase=lowercase,
                       remove_punct=remove_punct,
                       remove_nums=remove_nums,
                       remove_stop=remove_stop,
                       ngram_ns=ngram_ns)
    doc_sent, sent_df, sent_method = sentiment_doc_and_sentences(text, sent_threshold=sent_threshold, max_sentences=max_sentences)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = hash_stem(input_path)
    wf = pd.DataFrame(out["freq_lemmas"].most_common(topn), columns=["lemma","count"])
    wf.to_csv(outdir / f"{tag}_wordfreq_top{topn}.csv", index=False)
    for name, counter in out["ngrams"].items():
        top = pd.DataFrame(counter.most_common(topn), columns=[name,"count"])
        top.to_csv(outdir / f"{tag}_{name}_top{topn}.csv", index=False)
    pos_df = pd.DataFrame(sorted(out["pos_counts"].items(), key=lambda x: (-x[1], x[0])), columns=["POS","count"])
    pos_df.to_csv(outdir / f"{tag}_pos_counts.csv", index=False)
    meta = {
        "file": str(input_path),
        "sentiment_method": sent_method,
        "doc_sentiment": doc_sent,
        "vocab_size": out["vocab_size"],
        "token_count": out["token_count"],
        "type_token_ratio": out["type_token_ratio"],
    }
    (outdir / f"{tag}_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    sent_df.to_csv(outdir / f"{tag}_emotional_sentences.csv", index=False)
    return meta

def run_tests(tmp_out: Path):
    samples = [
        "I absolutely loved the museum exhibit! The curation was brilliant, though the queue was long.",
        "War is devastating; families suffer. Still, communities show resilience in the face of tragedy."
    ]
    ok = True
    for i, s in enumerate(samples, 1):
        p = tmp_out / f"sample_{i}.txt"
        p.write_text(s, encoding="utf-8")
        _ = process_path(p, tmp_out, strip_gut=True, topn=10, max_sentences=10)
        tag = hash_stem(p)
        expect = [
            f"{tag}_wordfreq_top10.csv",
            f"{tag}_unigram_top10.csv",
            f"{tag}_bigram_top10.csv",
            f"{tag}_trigram_top10.csv",
            f"{tag}_pos_counts.csv",
            f"{tag}_summary.json",
            f"{tag}_emotional_sentences.csv",
        ]
        for fn in expect:
            if not (tmp_out / fn).exists():
                print("❌ Missing output:", fn); ok = False
    print("✅ Tests passed." if ok else "❌ Tests failed.")
    return ok

def main():
    ap = argparse.ArgumentParser(description="NLP pipeline: .txt/.docx/.doc/.rtf/.pdf preprocessing, freq, n-grams, POS, sentiment.")
    ap.add_argument("--input", help="Path to file OR directory (recursive).")
    ap.add_argument("--outdir", default="nlp_outputs", help="Output directory.")
    ap.add_argument("--outputdir", dest="outdir", help="Alias for --outdir.")
    ap.add_argument("--no-strip-gutenberg", action="store_true", help="Do NOT strip Project Gutenberg headers/footers.")
    ap.add_argument("--lower", action="store_true", help="Lowercase tokens (default True).")
    ap.add_argument("--keep-punct", action="store_true", help="Keep punctuation tokens.")
    ap.add_argument("--keep-nums", action="store_true", help="Keep numeric tokens.")
    ap.add_argument("--keep-stop", action="store_true", help="Keep stopwords.")
    ap.add_argument("--ngrams", default="1,2,3", help="Comma list, e.g. 1,2,3 or 1,2")
    ap.add_argument("--topn", type=int, default=50, help="Top-N rows for frequency outputs.")
    ap.add_argument("--sent-threshold", type=float, default=0.5, help="Min score to include sentence in highlights.")
    ap.add_argument("--max-sentences", type=int, default=0, help="Cap sentences analyzed for sentiment (0=no cap).")
    ap.add_argument("--run-tests", action="store_true", help="Run built-in tests and exit.")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if args.run_tests:
        outdir.mkdir(parents=True, exist_ok=True)
        ok = run_tests(outdir)
        raise SystemExit(0 if ok else 1)
    if not args.input:
        raise SystemExit("Provide --input (file or directory) or use --run-tests")
    ipath = Path(args.input)
    if ipath.is_dir():
        exts = ("*.txt","*.docx","*.doc","*.rtf","*.pdf")
        paths = []
        for pat in exts:
            paths.extend(ipath.rglob(pat))
        if not paths:
            raise SystemExit("No supported files found (.txt/.docx/.doc/.rtf/.pdf).")
    else:
        paths = [ipath]
    ngram_ns = tuple(sorted({int(n.strip()) for n in args.ngrams.split(",") if n.strip()}))
    for p in paths:
        meta = process_path(
            p, outdir,
            strip_gut=not args.no_strip_gutenberg,
            lowercase=True,
            remove_punct=not args.keep_punct,
            remove_nums=not args.keep_nums,
            remove_stop=not args.keep_stop,
            ngram_ns=ngram_ns,
            sent_threshold=args.sent_threshold,
            max_sentences=args.max_sentences,
            topn=args.topn
        )
        print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
