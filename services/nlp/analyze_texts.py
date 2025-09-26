# Command-line script for analyzing cleaned text files using the NLP pipeline.
# Generates word frequencies, n-grams, POS distributions, and sentiment outputs.

import argparse, json, torch
from pathlib import Path
import pandas as pd

# Display whether CUDA (GPU) or CPU is being used for transformer models
print("Device set to use", "cuda" if torch.cuda.is_available() else "cpu")

# Try relative import first, fallback to sys.path manipulation if run directly
try:
    from services.nlp.preprocessing import process_text
    from services.nlp.features import count_ngrams, count_pos
    from services.nlp.sentiment import analyze_sentiment
    from services.shared.io_utils import hash_stem
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from services.nlp.preprocessing import process_text
    from services.nlp.features import count_ngrams, count_pos
    from services.nlp.sentiment import analyze_sentiment
    from services.shared.io_utils import hash_stem


def analyze_file(txt_path: Path, outdir: Path, ngram_ns, topn, sent_threshold, max_sentences, chunk_chars):
    """
    Run full NLP analysis on a single text file and write outputs to CSV/JSON.
    """
    # Ensure input is .txt (ETL should have cleaned beforehand)
    if txt_path.suffix.lower() != ".txt":
        raise SystemExit(f"TXT-only input for NLP stage. Use ETL first. Offending file: {txt_path.name}")
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    # Preprocess text (tokenize, lemmatize, remove stopwords, chunk with spaCy)
    prep = process_text(
        text,
        lowercase=True,
        remove_punct=True,
        remove_nums=True,
        remove_stop=True,
        chunk_chars=chunk_chars,
        batch_size=8
    )

    # Extract n-grams and POS tag counts
    grams = count_ngrams(prep["lemmas"], ngram_ns=ngram_ns)
    pos_counts = count_pos(prep["pos_seq"])

    # Run document + sentence-level sentiment analysis
    doc_sent, sent_df, method = analyze_sentiment(text, sent_threshold=sent_threshold, max_sentences=max_sentences)

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)
    tag = hash_stem(txt_path)  # Short hash for filenames to avoid collisions

    # --- Save outputs ---
    # Word frequency table
    pd.DataFrame(prep["freq_lemmas"].most_common(topn), columns=["lemma", "count"]).to_csv(
        outdir / f"{tag}_wordfreq_top{topn}.csv", index=False)

    # N-grams tables
    for name, counter in grams.items():
        pd.DataFrame(counter.most_common(topn), columns=[name, "count"]).to_csv(outdir / f"{tag}_{name}_top{topn}.csv",
                                                                                index=False)

    # POS counts table
    pd.DataFrame(sorted(pos_counts.items(), key=lambda x: (-x[1], x[0])), columns=["POS", "count"]).to_csv(
        outdir / f"{tag}_pos_counts.csv", index=False)

    # Sentence-level emotional highlights
    sent_df.to_csv(outdir / f"{tag}_emotional_sentences.csv", index=False)

    # Document summary (JSON with metadata + sentiment method)
    meta = {
        "file": str(txt_path),
        "sentiment_method": method,
        "doc_sentiment": doc_sent,
        "vocab_size": prep["vocab_size"],
        "token_count": prep["token_count"],
        "type_token_ratio": prep["type_token_ratio"],
    }
    (outdir / f"{tag}_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save vertical emotion scores (CSV for readability)
    lines = ["emotion,score"]
    for emo, score in sorted(doc_sent.items(), key=lambda kv: -kv[1]):
        lines.append(f"{emo},{score:.12f}")
    vertical_text = "\n".join(lines) + "\n"
    (outdir / f"{tag}_doc_emotion_vertical.csv").write_text(vertical_text, encoding="utf-8")


def main():
    """
    CLI entrypoint: parses arguments and runs analysis on file(s) or directory.
    """
    ap = argparse.ArgumentParser(description="Analyze clean .txt files: wordfreq, n-grams, POS, sentiment.")
    ap.add_argument("--input", required=True, help="Clean .txt file or directory (from ETL).")
    ap.add_argument("--outdir", default="data/outputs", help="Output directory for CSV/JSON.")
    ap.add_argument("--ngrams", default="1,2,3", help="Comma list: e.g., 1,2,3 or 1,2")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--sent-threshold", type=float, default=0.5, help="Confidence cutoff for sentence emotions")
    ap.add_argument("--max-sentences", type=int, default=0, help="Limit sentences analyzed for speed (0 = all)")
    ap.add_argument("--chunk-chars", type=int, default=120000,
                    help="spaCy chunk size in characters. Set 0 for single-pass.")
    args = ap.parse_args()

    ip = Path(args.input)
    outdir = Path(args.outdir)
    ngram_ns = tuple(sorted({int(n.strip()) for n in args.ngrams.split(",") if n.strip()}))

    # Collect paths to process
    if ip.is_dir():
        paths = list(ip.rglob("*.txt"))
        if not paths:
            raise SystemExit("No .txt files found. Run ETL first.")
    else:
        if ip.suffix.lower() != ".txt":
            raise SystemExit("TXT-only for NLP stage. Run ETL to produce clean .txt first.")
        paths = [ip]

    # Run analysis on each file
    for p in paths:
        analyze_file(p, outdir, ngram_ns, args.topn, args.sent_threshold, args.max_sentences, args.chunk_chars)


if __name__ == "__main__":
    main()
