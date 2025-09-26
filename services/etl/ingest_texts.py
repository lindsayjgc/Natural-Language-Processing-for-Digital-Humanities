# usage: python services/etl/ingest_texts.py --input "path|dir" --outdir "data/clean" [--keep-footnotes] [--no-strip-gutenberg]
import argparse, json
from pathlib import Path
from hashlib import sha1

# Prefer package imports; fall back to path-hack if run as a script
try:
    from services.etl.readers import read_text_smart                 # format-agnostic reader
    from services.etl.normalizers import strip_gutenberg_headers, basic_clean, remove_footnotes
    from services.shared.io_utils import hash_stem                   # stable hashed filename stem
except ImportError:
    # Allow running this file directly without installing the package
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from services.etl.readers import read_text_smart
    from services.etl.normalizers import strip_gutenberg_headers, basic_clean, remove_footnotes
    from services.shared.io_utils import hash_stem

def process_one(ipath: Path, outdir: Path, *, strip_gut=True, keep_footnotes=False):
    """
    ETL one source file -> canonical UTF-8 .txt + JSON sidecar with minimal provenance.

    Steps:
      1) Read raw text (supporting .txt/.docx/.doc/.rtf/.pdf).
      2) Optionally strip Project Gutenberg boilerplate (headers/footers).
      3) Optionally remove footnotes.
      4) Normalize whitespace / unwrap lines.
      5) Save as `<hash_stem>.txt` + `<hash_stem>_meta.json`.

    Args:
        ipath: input file path.
        outdir: directory where cleaned text + meta are written.
        strip_gut: if True, remove Gutenberg headers/footers.
        keep_footnotes: if True, retain footnotes (skip removal).
    """
    raw = read_text_smart(ipath)
    text = strip_gutenberg_headers(raw) if strip_gut else raw
    if not keep_footnotes:
        text = remove_footnotes(text)
    text = basic_clean(text, unwrap_lines=True)

    tag = hash_stem(ipath)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write cleaned text
    cpath = outdir / f"{tag}.txt"
    cpath.write_text(text, encoding="utf-8")

    # Record minimal provenance/metrics for traceability
    meta = {
        "source_path": str(ipath),
        "source_sha1": sha1(ipath.read_bytes()).hexdigest() if ipath.exists() else "",
        "clean_path": str(cpath),
        "steps": [
            "strip_gutenberg" if strip_gut else "no_strip",
            "remove_footnotes" if not keep_footnotes else "keep_footnotes",
            "basic_clean",
        ],
        "char_count": len(text),
    }
    (outdir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"ETL ✓ {ipath.name} -> {cpath.name}")

def main():
    """CLI: ingest a single file or walk a directory of supported files."""
    ap = argparse.ArgumentParser(description="Ingest & clean texts into canonical .txt + meta.")
    ap.add_argument("--input", required=True, help="File or directory (.txt/.docx/.doc/.rtf/.pdf).")
    ap.add_argument("--outdir", default="data/clean", help="Output directory for clean texts.")
    ap.add_argument("--keep-footnotes", action="store_true", help="Do not remove footnotes.")
    ap.add_argument("--no-strip-gutenberg", action="store_true", help="Do not strip Project Gutenberg headers/footers.")
    args = ap.parse_args()

    ip = Path(args.input)
    outdir = Path(args.outdir)

    # Expand to a file list (recursively for directories) with simple pattern set
    if ip.is_file():
        paths = [ip]
    else:
        patterns = ("*.txt","*.docx","*.doc","*.rtf","*.pdf")
        paths = []
        for pat in patterns:
            paths.extend(ip.rglob(pat))
        if not paths:
            raise SystemExit("No supported files found.")

    for p in paths:
        process_one(p, outdir, strip_gut=not args.no_strip_gutenberg, keep_footnotes=args.keep_footnotes)

if __name__ == "__main__":
    main()
