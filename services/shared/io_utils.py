# usage: helper functions for hashing and filenames
from pathlib import Path
import hashlib
import re

def safe_filename(name: str, maxlen: int = 180) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", name)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip(" ._")
    return s[:maxlen] if s else "untitled"

def hash_stem(p: Path) -> str:
    stem = p.stem
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{h}"
