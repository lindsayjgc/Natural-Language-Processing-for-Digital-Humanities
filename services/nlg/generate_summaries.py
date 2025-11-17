import argparse
from pathlib import Path
from io_utils import process_file
from models import DEFAULT_MODEL

def run_pipeline():
    ap = argparse.ArgumentParser(
        description="Transformer-based abstractive summarizer (JSON output, summary only)."
    )
    ap.add_argument("--input", required=True, help="Input file or directory of .txt files.")
    ap.add_argument("--outdir", default="summaries_abstractive_json",
                    help="Directory for JSON summaries.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Model name (e.g. pszemraj/led-large-book-summary, facebook/bart-large-cnn, t5-small, google/pegasus-xsum).")
    ap.add_argument("--max-length", type=int, default=240,
                    help="Maximum length of each generated summary chunk.")
    ap.add_argument("--min-length", type=int, default=80,
                    help="Minimum length of each generated summary chunk.")
    args = ap.parse_args()

    ipath = Path(args.input)
    outdir = Path(args.outdir)

    # Find all text files
    paths = [ipath] if ipath.is_file() else list(ipath.rglob("*.txt"))
    if not paths:
        raise SystemExit("No .txt files found.")

    print(f"[INFO] Processing {len(paths)} file(s)...")
    for p in paths:
        process_file(p, outdir, args.model, args.max_length, args.min_length)

    print("[INFO] All JSON summaries completed.")


if __name__ == "__main__":
    run_pipeline()
