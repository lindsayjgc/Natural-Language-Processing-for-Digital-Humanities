# CLI Guide

This guide summarizes the command-line flags and outputs for the bundled scripts.

## `analyze_texts.py` (clean `.txt` only)

**Synopsis**

```bash
python analyze_texts.py --input <file_or_dir> [--outdir PATH] [--ngrams "1,2,3"] [--topn N]   [--sent-threshold FLOAT] [--max-sentences INT] [--chunk-chars INT]
```

**Flags**

- `--input` (required): a single `.txt` file or a directory containing `.txt` files.
- `--outdir` (default: `data/outputs`): where CSV/JSON outputs are written.
- `--ngrams` (default: `1,2,3`): comma list of n‑gram sizes.
- `--topn` (default: `50`): number of top items for wordfreq and each n‑gram table.
- `--sent-threshold` (default: `0.5`): minimum confidence for a sentence to be included in the emotional highlights.
- `--max-sentences` (default: `0`): limit the number of sentences analyzed (0 means all).
- `--chunk-chars` (default: `120000`): per‑chunk character size for spaCy processing (set 0 to process in a single pass).

**Notes**

- On start, it prints whether the device is `cuda` or `cpu` (based on PyTorch availability).
- Non‑`.txt` files will be rejected—run your ETL first to produce clean `.txt`.

---

## `naturalLanguageProcessingOfDocuments.py` (clean `.txt` or `--from-raw`)

**Synopsis**

```bash
python naturalLanguageProcessingOfDocuments.py --input <file_or_dir> [--outdir PATH] \
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
