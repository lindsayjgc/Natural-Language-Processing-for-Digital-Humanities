# Alternative CLI script for NLP analysis.
# Unlike analyze_texts.py, this one can handle both ETL-cleaned texts and raw files
# (PDF, DOCX, RTF, etc.) by using ETL readers/normalizers if --from-raw is specified.

import argparse, json, hashlib
from pathlib import Path
import pandas as pd

# Try local imports, fallback to package-relative if needed
try:
    from preprocessing import process_text
    from features import count_ngrams, count_pos
    from sentiment import analyze_sentiment
except ImportError:
    from .preprocessing import process_text
    from .features import count_ngrams, count_pos
    from .sentiment import analyze_sentiment


# ---- optional ETL helpers (for --from-raw mode) ----
def _maybe_import_etl():
    """
    Try importing ETL utilities (for PDFs, DOCX, raw text cleaning).
    Fallback: manually extend sys.path.
    """
    try:
        from services.etl.readers import read_text_smart
        from services.etl.normalizers import strip_gutenberg_headers, basic_clean, remove_footnotes
        return read_text_smart, strip_gutenberg_headers, basic_clean, remove_footnotes
    except Exception:
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from services.etl.readers import read_text_smart
        from services.etl.normalizers import strip_gutenberg_headers, basic_clean, remove_footnotes
        return read_text_smart, strip_gutenberg_headers, basic_clean, remove_footnotes


# ---- local utilities ----
def _read_clean_txt(p: Path) -> str:
    """
    Read a cleaned text file robustly, trying multiple encodings.
    """
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_bytes().decode("utf-8","ignore")

def _hash_stem(path: Path) -> str:
    """
    Generate a stable short hash from file path to use in filenames.
    """
    stem = path.stem
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{h}"


# ---- core analysis pipeline ----
def _analyze_text_blob(text: str, tag: str, outdir: Path, *, ngram_ns, topn, sent_threshold, max_sentences):
    """
    Run NLP analysis on raw text and write CSV/JSON outputs.
    """
    prep = process_text(text)
    ngram_counts = count_ngrams(prep["lemmas"], ngram_ns)
    pos_counts   = count_pos(prep["pos_seq"])
    doc_sent, sent_df, sent_method = analyze_sentiment(text, sent_threshold=sent_threshold, max_sentences=max_sentences)

    outdir.mkdir(parents=True, exist_ok=True)

    # Word frequencies
    wf = pd.DataFrame(prep["freq_lemmas"].most_common(topn), columns=["lemma","count"])
    wf.to_csv(outdir / f"{tag}_wordfreq_top{topn}.csv", index=False)

    # N-grams
    for name, counter in ngram_counts.items():
        top = pd.DataFrame(counter.most_common(topn), columns=[name,"count"])
        top.to_csv(outdir / f"{tag}_{name}_top{topn}.csv", index=False)

    # POS
    pos_df = pd.DataFrame(sorted(pos_counts.items(), key=lambda x: (-x[1], x[0])), columns=["POS","count"])
    pos_df.to_csv(outdir / f"{tag}_pos_counts.csv", index=False)

    # Metadata summary
    meta = {
        "sentiment_method": sent_method,
        "doc_sentiment": doc_sent,
        "vocab_size": prep["vocab_size"],
        "token_count": prep["token_count"],
        "type_token_ratio": prep["type_token_ratio"],
    }
    (outdir / f"{tag}_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sentence-level sentiment
    sent_df.to_csv(outdir / f"{tag}_emotional_sentences.csv", index=False)

    # Vertical doc-level emotion scores
    lines = ["emotion,score"]
    for emo, score in sorted(doc_sent.items(), key=lambda kv: -kv[1]):
        lines.append(f"{emo},{score:.12f}")
    (outdir / f"{tag}_doc_emotion_vertical.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return meta


def process_path(ipath: Path,
                 outdir: Path,
                 *,
                 from_raw: bool,
                 ngram_ns,
                 topn,
                 sent_threshold,
                 max_sentences):
    """
    Process a single file path (either cleaned text or raw document).
    """
    tag = _hash_stem(ipath)
    if from_raw:
        # Use ETL pipeline for raw docs
        read_text_smart, strip_gut, basic_clean, remove_footnotes = _maybe_import_etl()
        raw = read_text_smart(ipath)
        text = strip_gut(raw)
        text = remove_footnotes(text)
        text = basic_clean(text, unwrap_lines=True)
    else:
        # Already ETL-cleaned .txt
        text = _read_clean_txt(ipath)

    meta = _analyze_text_blob(
        text, tag, outdir,
        ngram_ns=ngram_ns, topn=topn,
        sent_threshold=sent_threshold, max_sentences=max_sentences
    )
    meta["file"] = str(ipath)
    return meta


def main():
    """
    CLI entrypoint: handle args, support --from-raw to use ETL pipeline.
    """
    ap = argparse.ArgumentParser(description="NLP analysis on ETL-clean texts (default) or raw inputs.")
    ap.add_argument("--input", required=True, help="File or directory.")
    ap.add_argument("--outdir", default="nlp_outputs", help="Output directory.")
    ap.add_argument("--ngrams", default="1,2,3", help="Comma list, e.g. 1,2,3")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--sent-threshold", type=float, default=0.5)
    ap.add_argument("--max-sentences", type=int, default=0)
    ap.add_argument("--from-raw", action="store_true", help="If set, preprocess like ETL (readers/normalizers) before NLP.")
    args = ap.parse_args()

    ipath = Path(args.input)
    outdir = Path(args.outdir)
    ngram_ns = tuple(sorted({int(n.strip()) for n in args.ngrams.split(",") if n.strip()}))

    # Collect input files depending on mode
    if ipath.is_dir():
        pats = ("*.txt",) if not args.from_raw else ("*.txt","*.docx","*.doc","*.rtf","*.pdf")
        paths = []
        for pat in pats:
            paths.extend(ipath.rglob(pat))
        if not paths:
            raise SystemExit("No files found for given mode (try --from-raw for non-.txt).")
    else:
        paths = [ipath]

    # Run processing on each file
    for p in paths:
        meta = process_path(
            p, outdir,
            from_raw=args.from_raw,
            ngram_ns=ngram_ns,
            topn=args.topn,
            sent_threshold=args.sent_threshold,
            max_sentences=args.max_sentences
        )
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
