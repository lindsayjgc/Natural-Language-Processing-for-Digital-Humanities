import argparse, json, hashlib
from pathlib import Path
import pandas as pd

# Local imports (with fallback)
try:
    from preprocessing import process_text
    from features import count_ngrams, count_pos
    from sentiment import analyze_sentiment
except ImportError:
    from .preprocessing import process_text
    from .features import count_ngrams, count_pos
    from .sentiment import analyze_sentiment


def _maybe_import_etl():
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


def _read_clean_txt(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_bytes().decode("utf-8", "ignore")


def _hash_stem(path: Path) -> str:
    stem = path.stem
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{h}"


def _analyze_text_blob(text: str, tag: str, outdir: Path, *, ngram_ns, topn, sent_threshold, max_sentences):
    """Run NLP analysis on text and write multiple JSON outputs."""
    prep = process_text(text)
    ngram_counts = count_ngrams(prep["lemmas"], ngram_ns)
    pos_counts = count_pos(prep["pos_seq"])
    doc_sent, sent_df, sent_method = analyze_sentiment(
        text, sent_threshold=sent_threshold, max_sentences=max_sentences
    )

    outdir.mkdir(parents=True, exist_ok=True)

    wf_data = [{"lemma": lemma, "count": count} for lemma, count in prep["freq_lemmas"].most_common(topn)]
    (outdir / f"{tag}_wordfreq.json").write_text(json.dumps(wf_data, ensure_ascii=False, indent=2), encoding="utf-8")

    ngram_data = {
        name: [{"ngram": ng, "count": count} for ng, count in counter.most_common(topn)]
        for name, counter in ngram_counts.items()
    }
    (outdir / f"{tag}_ngrams.json").write_text(json.dumps(ngram_data, ensure_ascii=False, indent=2), encoding="utf-8")

    pos_data = [{"POS": pos, "count": count} for pos, count in sorted(pos_counts.items(), key=lambda x: (-x[1], x[0]))]
    (outdir / f"{tag}_pos.json").write_text(json.dumps(pos_data, ensure_ascii=False, indent=2), encoding="utf-8")

    sent_data = sent_df.to_dict(orient="records")
    (outdir / f"{tag}_sentiment_sentences.json").write_text(
        json.dumps(sent_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "file_tag": tag,
        "sentiment_method": sent_method,
        "doc_sentiment": doc_sent,
        "vocab_size": prep["vocab_size"],
        "token_count": prep["token_count"],
        "type_token_ratio": prep["type_token_ratio"],
    }
    (outdir / f"{tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def process_path(ipath: Path,
                 outdir: Path,
                 *,
                 from_raw: bool,
                 ngram_ns,
                 topn,
                 sent_threshold,
                 max_sentences):
    tag = _hash_stem(ipath)
    if from_raw:
        read_text_smart, strip_gut, basic_clean, remove_footnotes = _maybe_import_etl()
        raw = read_text_smart(ipath)
        text = strip_gut(raw)
        text = remove_footnotes(text)
        text = basic_clean(text, unwrap_lines=True)
    else:
        text = _read_clean_txt(ipath)

    meta = _analyze_text_blob(
        text, tag, outdir,
        ngram_ns=ngram_ns, topn=topn,
        sent_threshold=sent_threshold, max_sentences=max_sentences
    )
    meta["file_path"] = str(ipath)
    return meta


def main():
    ap = argparse.ArgumentParser(description="NLP analysis — outputs separate JSONs per file.")
    ap.add_argument("--input", required=True, help="File or directory.")
    ap.add_argument("--outdir", default="nlp_outputs_jsons", help="Output directory.")
    ap.add_argument("--ngrams", default="1,2,3", help="Comma list, e.g. 1,2,3")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--sent-threshold", type=float, default=0.5)
    ap.add_argument("--max-sentences", type=int, default=0)
    ap.add_argument("--from-raw", action="store_true", help="If set, preprocess raw docs before NLP.")
    args = ap.parse_args()

    ipath = Path(args.input)
    outdir = Path(args.outdir)
    ngram_ns = tuple(sorted({int(n.strip()) for n in args.ngrams.split(",") if n.strip()}))

    if ipath.is_dir():
        pats = ("*.txt",) if not args.from_raw else ("*.txt", "*.docx", "*.doc", "*.rtf", "*.pdf")
        paths = []
        for pat in pats:
            paths.extend(ipath.rglob(pat))
        if not paths:
            raise SystemExit("No files found for given mode (try --from-raw).")
    else:
        paths = [ipath]

    all_summaries = []
    for p in paths:
        summary = process_path(
            p, outdir,
            from_raw=args.from_raw,
            ngram_ns=ngram_ns,
            topn=args.topn,
            sent_threshold=args.sent_threshold,
            max_sentences=args.max_sentences
        )
        all_summaries.append(summary)

    # Combined summary of all processed files
    summary_path = outdir / "all_summaries.json"
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
