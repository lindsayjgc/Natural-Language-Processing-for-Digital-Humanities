from pathlib import Path

def read_text_file(path: Path) -> str:
    """
    Read a text file robustly, trying multiple encodings.
    """
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8","ignore")