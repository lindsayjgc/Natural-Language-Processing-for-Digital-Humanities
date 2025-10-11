# Text ETL Mini‑Package

Lightweight helpers to **read**, **normalize**, and **ingest** textual sources (Project Gutenberg books, RTF/DOCX dumps, PDFs, etc.) into a canonical UTF‑8 `.txt` plus a small JSON sidecar with provenance/metrics.

---

## What’s inside

- `readers.py` — format‑agnostic loader that reads `.txt/.md/.rst`, `.docx`, legacy `.doc`, `.rtf`, and `.pdf` (via optional backends) into plain text.
- `normalizers.py` — normalization utilities: strip Project Gutenberg boilerplate, remove footnotes, unwrap soft line breaks, and tidy whitespace.
- `ingest_texts.py` — single‑file/dir ETL CLI: read → normalize → write `<hash>.txt` and `<hash>_meta.json` in an output folder.
- `__init__.py` — package surface that exposes common entry points and semantic version.

---

## Features

- **Robust reading** of multiple formats with graceful fallbacks (DOCX via `docx2txt` → `python-docx` → raw XML unzip; RTF with a lightweight decoder; `.doc`/PDF via `textract` when available).
- **Gutenberg boilerplate stripping** using tolerant START/END markers.
- **Footnote removal** (inline markers like `[12]`, `[Footnote: …]`, trailing FOOTNOTES sections, list‑style notes).
- **Whitespace normalization** with optional soft line unwrapping to restore paragraphs.
- **Deterministic output naming** using a stable hashed stem so duplicates collapse to the same filenames.
- **JSON sidecar** with source path, SHA‑1 of the original file, applied steps, and cleaned character count.

---

## Installation

These files are designed to live under a package path like `services/etl/`. You can use them directly in a repo or install as an editable module:

```bash
# from the repo root (that contains services/)
pip install -e .
```

### Optional backends

Some formats use optional libraries:

- **DOCX**: `docx2txt` (fast path) and `python-docx` (fallback).
- **Legacy .doc / PDF**: `textract` (if installed).

Install suggestions:

```bash
pip install docx2txt python-docx
# for broader legacy/PDF support:
pip install textract
```

> If `textract` is not installed, attempting to read `.doc` or `.pdf` will raise a helpful error. DOCX still works via the layered strategy.

---

## Quick start

### 1) CLI: ingest a file or a directory

```bash
python services/etl/ingest_texts.py --input path/to/file_or_dir --outdir data/clean
# keep original footnotes
python services/etl/ingest_texts.py --input corpus/ --outdir data/clean --keep-footnotes
# treat a Gutenberg book as-is (skip header/footer strip)
python services/etl/ingest_texts.py --input book.txt --no-strip-gutenberg
```

**Outputs** (per source file):

```
data/clean/
  <hash>.txt            # cleaned UTF-8 text
  <hash>_meta.json      # provenance + metrics
```

The meta JSON contains:

```json
{
  "source_path": "...",
  "source_sha1": "...",
  "clean_path": "...",
  "steps": ["strip_gutenberg", "remove_footnotes", "basic_clean"],
  "char_count": 123456
}
```

### 2) Python API

```python
from pathlib import Path
from services.etl import (
    read_text_smart,          # robust reader
    strip_gutenberg_headers,  # normalizers
    remove_footnotes,
    basic_clean,
    ingest_one                # ETL helper for a single file
)

# Read any supported file into text
raw = read_text_smart(Path("data/raw/book.docx"))

# Clean ad hoc
body = strip_gutenberg_headers(raw)
body = remove_footnotes(body)
clean = basic_clean(body, unwrap_lines=True)

# Save via the helper
outdir = Path("data/clean")
ingest_one(Path("data/raw/book.docx"), outdir, strip_gut=True, keep_footnotes=False)
```

---

## Behavior details

### Supported inputs

- Plain text: `.txt`, `.md`, `.rst`
- Wordprocessing: `.docx` (multiple strategies), legacy `.doc` (requires `textract`)
- Rich text: `.rtf` (built‑in simple decoder)
- PDF: `.pdf` (requires `textract`)

### DOCX strategy chain

1. `docx2txt` (if installed) →
2. `python-docx` (if available) →
3. unzip `word/document.xml` and strip tags →
4. if sniffing shows the file isn’t a real DOCX zip (e.g., RTF mis‑named), route to the appropriate reader.

### Gutenberg/footnotes rules of thumb

- START/END markers are matched with tolerant regular expressions; if not found, the function returns a trimmed original.
- Footnote remover handles inline “Footnote: …”, bracketed markers like `[iv]`, back‑half FOOTNOTES sections, and line‑leading list‑style notes.

---

## Project layout (suggested)

```
services/
  etl/
    __init__.py
    readers.py
    normalizers.py
    ingest_texts.py
data/
  raw/    # your source files
  clean/  # ETL outputs (.txt + _meta.json)
```

---

## Troubleshooting

- **`RuntimeError: Legacy .doc detected. Install textract...`** — install `textract` or convert to `.docx`/`.txt`.
- **PDFs produce empty text** — some scans are image‑only; use OCR (e.g., `tesseract`) before ETL.
- **Weird encoding artifacts** — the TXT reader cycles multiple encodings and finally decodes with `"ignore"` to salvage bytes; you may still need to convert truly exotic encodings.
- **Preserving hard line breaks** — call `basic_clean(..., unwrap_lines=False)`.

---

## Development

- Exported API (import from `services.etl`):

```python
from services.etl import read_text_smart, strip_gutenberg_headers, remove_footnotes, basic_clean, ingest_one
```

- Version is tracked in `services/etl/__init__.py`.
- CLI entry point is `services/etl/ingest_texts.py:main`.

### Tests (sketch)

- Unit tests for each normalizer (Gutenberg strip, footnote removal, whitespace).
- Golden‑file tests for `read_text_smart` across formats (with/without optional deps).
- CLI smoke test over small fixture corpus.

---

## License

Add your preferred OSS license here (e.g., MIT).

---

## Acknowledgements

- Project Gutenberg for public‑domain texts used during development.
- The authors of `docx2txt`, `python-docx`, and `textract` for robust extraction libraries.
