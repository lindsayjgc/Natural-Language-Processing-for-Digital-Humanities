#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Directory sentiment/emotion analyzer for Gutenberg-style .txt files.

Filename format expected (best effort parsing, robust to missing parts):
    Title__Author__pg12345.txt

Key features:
- Recursively walks an input directory for .txt files
- Parses title/author/pgid from filename when available
- Strips Project Gutenberg headers/footers (heuristics + markers)
- Cleans and chunks text (word-based) for transformer inference
- Uses emotion classifier (default: j-hartmann/emotion-english-distilroberta-base)
- Aggregates average emotion scores per document
- Extracts high-confidence emotional sentences
- Saves: emotion_results.csv and emotional_sentences.csv in output directory
- Optional: dry-run (no model needed) just to verify parsing/cleaning

Usage:
    python sentiment_analyser_dir.py --input /path/to/texts --output ./out
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---- Optional imports with graceful fallback ----
HAVE_TRANSFORMERS = True
try:
    from transformers import pipeline
except Exception:
    HAVE_TRANSFORMERS = False

HAVE_NLTK = True
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except Exception:
    HAVE_NLTK = False
    sent_tokenize = None

GUTENBERG_START_PATTERNS = [
    r'\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*',
    r'\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*',
]
GUTENBERG_END_PATTERNS = [
    r'\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*',
    r'\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*',
]

def parse_filename(meta_name: str) -> Tuple[str, str, Optional[str]]:
    """Parse Title__Author__pgID from filename stem."""
    stem = Path(meta_name).stem
    parts = stem.split('__')
    title = parts[0].replace('_', ' ').strip() if parts else stem.replace('_', ' ').strip()
    author = parts[1].replace('_', ' ').strip() if len(parts) > 1 else ''
    pgid = None
    if len(parts) > 2:
        m = re.search(r'pg(\d+)', parts[2], re.IGNORECASE)
        if m:
            pgid = m.group(1)
    return title, author, pgid

def strip_gutenberg(text: str) -> str:
    """Strip Gutenberg header/footer using known markers."""
    lines = text.splitlines()
    start_idx, end_idx = 0, len(lines)

    for i, line in enumerate(lines):
        if any(re.search(pat, line) for pat in GUTENBERG_START_PATTERNS):
            start_idx = i + 1
            break
    for j in range(len(lines)-1, -1, -1):
        if any(re.search(pat, lines[j]) for pat in GUTENBERG_END_PATTERNS):
            end_idx = j
            break

    core = lines[start_idx:end_idx]
    cleaned = '\n'.join(core)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def chunk_words(text: str, chunk_words: int = 350) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def analyze_document(text: str, emotion_model, min_sentence_score: float, chunk_words_n: int) -> Tuple[Dict[str, float], List[Dict]]:
    chunks = chunk_words(text, chunk_words=chunk_words_n)
    doc_scores_list: List[List[Dict]] = []
    for chunk in chunks:
        out = emotion_model(chunk[:512])
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
            doc_scores_list.append(out[0])
        else:
            doc_scores_list.append(out)

    avg_scores: Dict[str, float] = {}
    if doc_scores_list:
        labels = [d['label'] for d in doc_scores_list[0]]
        for lab in labels:
            vals = []
            for per_chunk in doc_scores_list:
                for item in per_chunk:
                    if item['label'] == lab:
                        vals.append(float(item['score']))
                        break
            avg_scores[lab] = sum(vals)/len(vals) if vals else 0.0

    highlights: List[Dict] = []
    if HAVE_NLTK and sent_tokenize is not None:
        for sent in sent_tokenize(text):
            out = emotion_model(sent[:512])
            cand = out[0] if isinstance(out, list) and isinstance(out[0], list) else out
            best = max(cand, key=lambda x: x['score'])
            if best['score'] >= min_sentence_score:
                highlights.append({'sentence': sent.strip(), 'emotion': best['label'], 'score': float(best['score'])})
    return avg_scores, highlights

def main(args):
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    ensure_output_dir(output_dir)

    txt_files = [p for p in input_dir.rglob('*.txt') if p.is_file()]
    if not txt_files:
        print(f'No .txt files found under: {input_dir}')
        return

    emotion_model = None
    if not args.dry_run:
        if not HAVE_TRANSFORMERS:
            print('ERROR: transformers not available. Install with: pip install transformers torch --upgrade')
            return
        emotion_model = pipeline('text-classification', model=args.model, top_k=None)
        if not HAVE_NLTK:
            print('WARNING: nltk not available; sentence-level highlights skipped.')

    results_rows, sentence_rows = [], []

    for p in sorted(txt_files):
        title, author, pgid = parse_filename(p.name)

        print(f"[INFO] Starting: {p.name}")  # <-- add this

        raw = p.read_text(encoding='utf-8', errors='ignore')
        text = strip_gutenberg(raw) if args.strip_gutenberg else raw

        meta = {'file': str(p), 'title': title, 'author': author, 'pgid': pgid or ''}
        if args.dry_run:
            print(f"[DRY-RUN] {title} — {author} (pg{pgid}) :: {p.name}")
            continue

        avg_scores, highlights = analyze_document(
            text, emotion_model, args.min_sentence_score, args.chunk_words
        )
        row = {**meta, **avg_scores}
        results_rows.append(row)

        for h in highlights:
            sentence_rows.append({**meta, 'sentence': h['sentence'], 'emotion': h['emotion'], 'score': h['score']})

        print(f"[DONE] {p.name} — chunks={len(chunk_words(text, args.chunk_words))}, highlights={len(highlights)}")

    if results_rows:
        results_csv = output_dir / 'emotion_results.csv'
        all_keys = set().union(*(row.keys() for row in results_rows))
        cols = ['file', 'title', 'author', 'pgid'] + sorted(k for k in all_keys if k not in ['file','title','author','pgid'])
        with results_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(results_rows)

    if sentence_rows:
        sentences_csv = output_dir / 'emotional_sentences.csv'
        cols = ['file', 'title', 'author', 'pgid', 'emotion', 'score', 'sentence']
        with sentences_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(sentence_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default=r"C:\Users\vedan\PycharmProjects\Natural-Language-Processing-for-Digital-Humanities\data\literature\Literature_Romance",
        help='Input directory (default: Romance corpus)'
    )
    parser.add_argument(
        '--output',
        default=r"C:\Users\vedan\PycharmProjects\Natural-Language-Processing-for-Digital-Humanities\data\literature\Literature_Romance\out",
        help='Output directory (default: Romance out folder)'
    )
    parser.add_argument('--model', default='j-hartmann/emotion-english-distilroberta-base')
    parser.add_argument('--chunk-words', type=int, default=350)
    parser.add_argument('--min-sentence-score', type=float, default=0.5)
    parser.add_argument('--strip-gutenberg', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(f"[INFO] Input directory: {args.input}")
    print(f"[INFO] Output directory: {args.output}")
    main(args)