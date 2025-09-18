#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gutenberg category importer with:
- Category -> bookshelf/subject mapping
- Skip-list for whole buckets
- Skip buckets already completed across runs (per-bucket manifest + _DONE file)
- Book-level exclusions (poetry, plays, humour, lit-other, hist-other, archaeology)
- Robust networking: retries, backoff, longer timeouts, host fallback
- Saves each bucket into its own subfolder
- Strips Project Gutenberg headers/footers from saved text
- Windows-safe filenames with truncation + hash to avoid path-length errors
- Prints a line when each book is completed
"""

import os
import re
import time
import json
import signal
import hashlib
import requests
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------- USER CONFIG ---------
DEST = r"C:\Users\vedan\PycharmProjects\Natural-Language-Processing-for-Digital-Humanities\data\literature"
os.makedirs(DEST, exist_ok=True)

# How many books to fetch per bucket
PER_CATEGORY_LIMIT = 50

# Skip entire buckets whose names match these substrings (case-insensitive)
SKIP_BUCKET_KEYWORDS = [
    "poetry", "plays", "dramas", "films",
    "humour", "humor",
    "literature - other", "lit-other",
    "history - other", "hist-other",
    "archaeology", "archaeology & anthropology",
    "freewizry",
    "mythologies/folklore",
    "science-fic/fantasy",
    "history - anc.",
    "history - med.",
    "history religious",
    "history - war",
    "history - schools & universities",
]

EXCLUDED_PATTERNS = [
    # existing
    r"\bpoetry\b", r"\bpoems?\b",
    r"\bplays?\b|\bdrama(s)?\b|\bfilms?\b",
    r"\bhumou?r\b|\bcomedy\b",
    r"\bliterature\s*-\s*other\b|\blit[-\s]?other\b",
    r"\bhistory\s*-\s*other\b|\bhist[-\s]?other\b",
    r"\barchaeolog(y|ical|y & anthropology)\b",

    # NEW: Fairy tales & close variants
    r"\bfairy[-\s]?(tales?|stories)\b",
    r"\bfolk[-\s]?tales?\b|\bfolktales?\b",
    r"\bmytholog(?:y|ies)\b|\blegends?\b|\bfolklore\b",

    # NEW: Children's / Juvenile / YA (common in PG subjects)
    r"\bjuvenile\b|\bjuvenile\s+fiction\b",
    r"\bchildren'?s\s+(stories|literature|fiction)\b",
    r"\byoung\s+adult(s)?\b|\byoung[-\s]?people'?s\b",

    # NEW: Historical fiction (“history novels”)
    r"\bhistorical\s+fiction\b",
    r"\bhistorical\s+novels?\b",
    r"\bhistory\s*--\s*fiction\b",
    # a few catalog variants occasionally seen
    r"\bhistorical\s+romance\b",
]
EXCLUDED_REGEX = re.compile("|".join(EXCLUDED_PATTERNS), re.IGNORECASE)

def is_excluded(book):
    # Check subjects, bookshelves, and title (some fairy tales/historical fiction show in titles too)
    title = book.get("title") or ""
    shelves = " | ".join(book.get("bookshelves") or [])
    subjects = " | ".join(book.get("subjects") or [])
    haystack = f"{title} | {shelves} | {subjects}"
    return bool(EXCLUDED_REGEX.search(haystack))

# --------- CATEGORY CONFIG ---------
CATEGORY_CONFIG = {
    "Literature/Adventure": {"topics": ["Adventure"], "subject_contains": ["Adventure stories"]},
    "Literature/American Literature": {"topics": ["American literature"], "subject_contains": ["United States -- Fiction", "American literature"]},
    "Literature/British Literature": {"topics": ["English literature"], "subject_contains": ["England -- Fiction", "British literature", "English literature"]},
    "Literature/French Literature": {"topics": ["French literature"], "subject_contains": ["French literature", "France -- Fiction"]},
    "Literature/German Literature": {"topics": ["German literature"], "subject_contains": ["German literature", "Germany -- Fiction"]},
    "Literature/Russian Literature": {"topics": ["Russian literature"], "subject_contains": ["Russian literature", "Russia -- Fiction"]},
    "Literature/Classics of Literature": {"topics": ["Best Books Ever Listings", "Harvard Classics"], "subject_contains": ["Classics"]},
    "Literature/Biographies": {"topics": ["Biographies"], "subject_contains": ["Biography"]},
    "Literature/Novels": {"topics": ["Fiction"], "subject_contains": ["Fiction"]},
    "Literature/Short Stories": {"topics": ["Short Stories"], "subject_contains": ["Short stories"]},
    "Literature/Romance": {"topics": ["Romance"], "subject_contains": ["Love stories", "Romance fiction"]},
    "Literature/Science-Fiction & Fantasy": {"topics": ["Science Fiction", "Fantasy"], "subject_contains": ["Science fiction", "Fantasy"]},
    "Literature/Crime, Thrillers & Mystery": {"topics": ["Crime Fiction", "Detective Fiction", "Mystery"], "subject_contains": ["Detective", "Mystery", "Crime"]},
    "Literature/Mythology, Legends & Folklore": {"topics": ["Mythology", "Folklore", "Fairy Tales"], "subject_contains": ["Mythology", "Legends", "Folklore", "Fairy tales"]},
    "Literature/Children & Young Adult Reading": {"topics": ["Children's Literature", "Children's Instructional Books"], "subject_contains": ["Children's literature", "Young adults"]},
}

# --------- NETWORK HARDENING ---------
GUTENDEX_BASES = ["https://gutendex.com", "https://api.gutendex.com"]
DEFAULT_TIMEOUT = (6, 75)  # (connect, read) seconds

def make_session():
    s = requests.Session()
    retry = Retry(
        total=7, connect=5, read=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "gutendex-downloader/1.8 (timeout-hardened)"})
    return s

SESSION = make_session()

# --------- HELPERS ---------
def is_excluded(book):
    shelves = " | ".join(book.get("bookshelves") or [])
    subjects = " | ".join(book.get("subjects") or [])
    return bool(EXCLUDED_REGEX.search(f"{shelves} | {subjects}"))

def pick_text_format(formats_dict):
    candidates = []
    for k, v in formats_dict.items():
        if not k.startswith("text/plain"):
            continue
        if v.endswith(".zip") or v.endswith(".gz"):
            continue
        candidates.append((k.lower(), v))
    if not candidates:
        return None
    for k, v in candidates:
        if "charset=utf-8" in k:
            return v
    return candidates[0][1]

def safe_name(s):
    s = re.sub(r"[^\w\-. ]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip(" ._")
    return s[:180] if s else "untitled"

def strip_gutenberg_headers(text: str) -> str:
    lines = text.splitlines()
    start_idx, end_idx = 0, len(lines)
    start_pat = re.compile(r"^\*\*\*\s*START\s+OF\s+.*EBOOK", re.IGNORECASE)
    end_pat = re.compile(r"^\*\*\*\s*END\s+OF\s+.*EBOOK", re.IGNORECASE)
    for i, line in enumerate(lines):
        if start_pat.search(line.strip()):
            start_idx = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if end_pat.search(lines[i].strip()):
            end_idx = i
            break
    body = "\n".join(lines[start_idx:end_idx]).strip()
    return body if body else text.strip()

def build_path(bucket_dir, title, authors, pgid, ext=".txt"):
    base = f"{safe_name(title)}__{safe_name(authors)}__pg{pgid}{ext}"
    full = os.path.join(bucket_dir, base)
    max_total = 240 if os.name == "nt" else 4096
    if len(full) <= max_total:
        return full
    stem, dotext = os.path.splitext(base)
    h = hashlib.sha1(f"{title}|{authors}|{pgid}".encode("utf-8")).hexdigest()[:8]
    overhead = len(os.path.join(bucket_dir, "")) + len("__") + len(h) + len(dotext)
    allow = max_total - overhead
    keep = max(16, allow)
    truncated = stem[:keep]
    base2 = f"{truncated}__{h}{dotext}"
    return os.path.join(bucket_dir, base2)

# ---------- Subject matching helpers ----------
def _subjects_text(book):
    return " | ".join(book.get("subjects") or [])

def _compile_subject_matcher(needles):
    # Use word boundaries so 'Queens' != 'Queensland'; still match long LCSH phrases.
    pats = []
    for n in needles or []:
        n = n.strip()
        if not n:
            continue
        pats.append(re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)", re.IGNORECASE))
    return pats

def _subjects_match_any(book, pats):
    s = _subjects_text(book)
    return any(p.search(s) for p in pats) if pats else True

# ---------- HTTP helpers ----------
def _fetch_json(url):
    bases = [""] if url.startswith("http") else GUTENDEX_BASES
    errs = []
    for base in bases:
        full = url if base == "" else f"{base}{url if url.startswith('/') else '/' + url}"
        try:
            r = SESSION.get(full, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                errs.append(f"JSON decode failed for {full}: {e}")
                continue
            if isinstance(data, dict) and data.get("detail"):
                errs.append(f"API detail for {full}: {data.get('detail')}")
                continue
            return data
        except Exception as e:
            errs.append(f"{type(e).__name__} on {full}: {e}")
            continue
    raise RuntimeError("All Gutendex bases failed:\n  - " + "\n  - ".join(errs))

def paged(url):
    next_url = url
    while next_url:
        data = _fetch_json(next_url)
        for item in data.get("results", []):
            yield item
        next_url = data.get("next")
        time.sleep(0.25)

# ---------- Queries (use BOTH topics and subject_contains) ----------
def query_by_topics(topics, lang="en"):
    seen = set()
    for t in topics or []:
        url_path = f"/books?{urlencode({'languages': lang, 'topic': t})}"
        for book in paged(url_path):
            bid = book.get("id")
            if bid in seen:
                continue
            seen.add(bid)
            yield book

def query_by_subject_contains(needles, lang="en"):
    # 1) topic= for each needle; 2) fallback search= for each needle; verify subjects with regex
    pats = _compile_subject_matcher(needles)
    seen = set()

    for n in needles or []:
        url_path = f"/books?{urlencode({'languages': lang, 'topic': n})}"
        for book in paged(url_path):
            bid = book.get("id")
            if bid in seen:
                continue
            if _subjects_match_any(book, pats):
                seen.add(bid)
                yield book

    for n in needles or []:
        url_path = f"/books?{urlencode({'languages': lang, 'search': n})}"
        for book in paged(url_path):
            bid = book.get("id")
            if bid in seen:
                continue
            if _subjects_match_any(book, pats):
                seen.add(bid)
                yield book

def iter_candidates(topics, subject_needles, lang="en"):
    # Combine both sources of candidates
    yielded = set()
    for book in query_by_topics(topics, lang=lang):
        bid = book.get("id")
        if bid in yielded:
            continue
        yielded.add(bid)
        yield book
    for book in query_by_subject_contains(subject_needles, lang=lang):
        bid = book.get("id")
        if bid in yielded:
            continue
        yielded.add(bid)
        yield book

# ---------- Per-bucket manifest ----------
def manifest_paths(bucket_dir):
    return os.path.join(bucket_dir, "_manifest.json"), os.path.join(bucket_dir, "_DONE")

def load_manifest(bucket_dir):
    mpath, dpath = manifest_paths(bucket_dir)
    done = os.path.exists(dpath)
    data = {"saved_ids": [], "completed": done}
    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    if not data.get("saved_ids"):
        ids = set()
        for fn in os.listdir(bucket_dir):
            m = re.search(r"__pg(\d+)\.txt$", fn)
            if m:
                ids.add(int(m.group(1)))
        data["saved_ids"] = sorted(ids)
    if done or len(data["saved_ids"]) >= PER_CATEGORY_LIMIT:
        data["completed"] = True
    return data

def save_manifest(bucket_dir, data):
    mpath, dpath = manifest_paths(bucket_dir)
    try:
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    if data.get("completed"):
        try:
            with open(dpath, "w") as f:
                f.write("done\n")
        except Exception:
            pass

# ---------- Saving ----------
def save_book(book, dest_dir):
    title = book["title"]
    authors = ", ".join(a.get("name","") for a in book.get("authors", [])) or "Unknown"
    pgid = book.get("id")
    fmt = pick_text_format(book.get("formats", {}))
    if not fmt:
        return False, "no-plain-text", ""
    path = build_path(dest_dir, title, authors, pgid, ext=".txt")
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(path):
        return True, "exists", path
    r = SESSION.get(fmt, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    cleaned = strip_gutenberg_headers(r.text)
    try:
        with open(path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(cleaned)
        return True, "saved", path
    except FileNotFoundError:
        h = hashlib.sha1(f"{title}|{authors}|{pgid}".encode("utf-8")).hexdigest()[:8]
        fallback = os.path.join(dest_dir, f"pg{pgid}_{h}.txt")
        with open(fallback, "w", encoding="utf-8", errors="ignore") as f:
            f.write(cleaned)
        return True, "saved", fallback

# ---------- Orchestration ----------
def download_bucket(bucket_name, topics, subject_needles, limit=PER_CATEGORY_LIMIT):
    print(f"\n=== Bucket: {bucket_name} (limit {limit}) ===")
    bucket_dir = os.path.join(DEST, safe_name(bucket_name))
    os.makedirs(bucket_dir, exist_ok=True)

    manifest = load_manifest(bucket_dir)
    if manifest.get("completed"):
        print(f"Skipping (already completed): {bucket_name} — {len(manifest.get('saved_ids', []))} saved")
        return

    saved_ids = set(manifest.get("saved_ids", []))
    saved_count = len(saved_ids)

    any_candidates = False
    for book in iter_candidates(topics, subject_needles, lang="en"):
        any_candidates = True
        if saved_count >= limit:
            break
        if is_excluded(book):
            continue
        pgid = int(book.get("id"))
        if pgid in saved_ids:
            continue

        ok, status, path_used = save_book(book, bucket_dir)
        if ok and status in ("saved", "exists"):
            saved_ids.add(pgid)
            saved_count += 1
            title = book["title"]
            authors = ", ".join(a.get("name","") for a in book.get("authors", [])) or "Unknown"
            print(f"[{saved_count}/{limit}] Saved: {title} — {authors}  ->  {os.path.basename(path_used)}")

            manifest["saved_ids"] = sorted(saved_ids)
            save_manifest(bucket_dir, manifest)

    if not any_candidates:
        print(f"[warn] No candidates found via topics/subjects for: {bucket_name}. "
              f"Consider relaxing needles or adding generic topics (e.g., 'Fiction').")

    if saved_count >= limit:
        manifest["completed"] = True
        save_manifest(bucket_dir, manifest)
        print(f"Reached limit for {bucket_name} — marked as DONE.")
    else:
        print(f"✅ Downloaded {saved_count} so far for bucket: {bucket_name} (not yet at limit {limit})")

def main():
    interrupted = False
    def _sigint(_sig, _frm):
        nonlocal interrupted
        interrupted = True
        print("\n[!] Ctrl+C detected — finishing current item then stopping...")
    signal.signal(signal.SIGINT, _sigint)

    for bucket, conf in CATEGORY_CONFIG.items():
        if interrupted:
            break
        if any(key in bucket.lower() for key in SKIP_BUCKET_KEYWORDS):
            print(f"Skipping bucket due to skip list: {bucket}")
            continue
        download_bucket(
            bucket_name=bucket,
            topics=conf.get("topics", []),
            subject_needles=conf.get("subject_contains", []),
            limit=PER_CATEGORY_LIMIT,
        )

    print("\nStopped by user." if interrupted else "\nAll buckets processed.")

if __name__ == "__main__":
    main()
