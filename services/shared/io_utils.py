# usage: helper functions for hashing and filenames
from pathlib import Path
import hashlib
import re


def safe_filename(name: str, maxlen: int = 180) -> str:
    """
    Sanitize a string for use as a filename.

    Steps:
    - Replace invalid characters (anything not alphanumeric, underscore, dash, dot, or space) with "_".
    - Replace whitespace with "_".
    - Collapse multiple underscores.
    - Strip trailing/leading dots, spaces, or underscores.
    - Truncate result to `maxlen` (default 180 chars).
    - If nothing remains, return "untitled".
    """
    s = re.sub(r"[^\w\-. ]+", "_", name)  # keep only safe chars
    s = re.sub(r"\s+", "_", s)  # collapse whitespace
    s = re.sub(r"_+", "_", s).strip(" ._")  # collapse underscores, trim
    return s[:maxlen] if s else "untitled"


def hash_stem(p: Path) -> str:
    """
    Generate a hashed filename stem.

    Example:
        Path("example.txt") → "example_ab12cd"

    Steps:
    - Extract the stem (filename without extension).
    - Compute SHA-1 hash of the full path (ensures uniqueness).
    - Take the first 6 hex chars of the hash for brevity.
    - Append to the stem with an underscore.
    """
    stem = p.stem
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{h}"
