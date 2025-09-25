# usage: robust text reader for .txt/.docx/.doc/.rtf/.pdf
import zipfile, html, re
from pathlib import Path

# Optional accelerators/backends (gracefully degrade if missing)
try:
    import docx2txt                     # fast & simple for many .docx files
except Exception:
    docx2txt = None

_DOCX_OK = False
try:
    import docx as _py_docx             # python-docx fallback
    _DOCX_OK = True
except Exception:
    _py_docx = None
    _DOCX_OK = False

_TEXTRACT_OK = False
try:
    import textract                     # catch-all for legacy .doc/.pdf (if installed)
    _TEXTRACT_OK = True
except Exception:
    textract = None
    _TEXTRACT_OK = False

def _read_txt(p: Path) -> str:
    """Try common encodings; fall back to 'ignore' decoding to salvage bytes."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_bytes().decode("utf-8", "ignore")

def _read_docx_with_docx2txt(p: Path) -> str:
    """Primary .docx path via docx2txt (handles most cases quickly)."""
    return docx2txt.process(str(p)) or ""

def _read_docx_with_python_docx(p: Path) -> str:
    """Secondary .docx path using python-docx (preserves paragraph structure)."""
    d = _py_docx.Document(str(p))
    return "\n".join(par.text for par in d.paragraphs)

def _read_docx_by_zip(p: Path) -> str:
    """
    Final .docx fallback: unzip document.xml and strip XML tags to recover text.
    Useful when both docx2txt and python-docx fail, or file is slightly malformed.
    """
    with zipfile.ZipFile(str(p)) as z:
        try:
            data = z.read("word/document.xml").decode("utf-8", "ignore")
        except KeyError:
            # Some files store text in other XMLs; concatenate all XML payloads
            data = "".join(z.read(n).decode("utf-8", "ignore") for n in z.namelist() if n.endswith(".xml"))
    # Replace paragraph closes with newlines; then drop all tags
    data = re.sub(r"</w:p>", "\n", data)
    data = re.sub(r"<[^>]+>", "", data)
    return html.unescape(data)

def _read_rtf_simple(p: Path) -> str:
    """
    Lightweight RTF debracer/decoder:
    - Decode hex escapes like \\'hh
    - Strip control words (\\wordN)
    - Remove braces and tidy whitespace
    """
    raw = p.read_bytes()
    try:
        txt = raw.decode("utf-8", "ignore")
    except Exception:
        txt = raw.decode("latin-1", "ignore")
    txt = re.sub(r"\\'[0-9a-fA-F]{2}", lambda m: bytes.fromhex(m.group(0)[2:]).decode('latin-1','ignore'), txt)
    txt = re.sub(r"\\[A-Za-z]+-?\d* ?", "", txt)
    txt = re.sub(r"[{}]", "", txt)
    return re.sub(r"\s+\n", "\n", txt)

def _sniff_kind(p: Path) -> str:
    """
    Quick magic-bytes sniffing to distinguish docx zip / OLE .doc / RTF / PDF.
    Used to route edge-case .docx files to the right reader.
    """
    with open(p, "rb") as f:
        head = f.read(8)
    if head.startswith(b"PK"):
        return "docx_zip"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole_doc"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head.startswith(b"%PDF"):
        return "pdf"
    return "text_or_unknown"

def read_text_smart(path: Path) -> str:
    """
    Read text from a variety of formats with graceful degradation:
      - .txt/.md/.rst → open with encoding fallbacks
      - .docx → docx2txt → python-docx → raw-zip parse
      - .doc  → textract if available, else raise
      - .rtf  → simple RTF stripper
      - .pdf  → textract if available, else raise
      - else  → treat as plain text

    Raises:
        RuntimeError: when a legacy/binary format is detected without a viable backend.
    """
    p = Path(path)
    suf = p.suffix.lower()

    # Plain text family
    if suf in (".txt", ".md", ".rst"):
        return _read_txt(p)

    # DOCX with multiple strategies + kind sniffing for weird cases
    if suf == ".docx":
        kind = _sniff_kind(p)
        if kind != "docx_zip":
            # Actually not a standard docx zip (e.g., RTF renamed .docx)
            if kind == "rtf":
                return _read_rtf_simple(p)
            if kind == "ole_doc" and _TEXTRACT_OK:
                return textract.process(str(p)).decode("utf-8", "ignore")
            return _read_txt(p)
        # Standard .docx zip → attempt readers in order of robustness
        try:
            if docx2txt is not None:
                return _read_docx_with_docx2txt(p)
        except Exception:
            pass
        if _DOCX_OK:
            try:
                return _read_docx_with_python_docx(p)
            except Exception:
                pass
        return _read_docx_by_zip(p)

    # Legacy .doc via textract only
    if suf == ".doc":
        if _TEXTRACT_OK:
            return textract.process(str(p)).decode("utf-8", "ignore")
        raise RuntimeError("Legacy .doc detected. Install `textract` or convert to .txt/.docx.")

    # RTF simple decoder
    if suf == ".rtf":
        return _read_rtf_simple(p)

    # PDF via textract (no built-in parser here)
    if suf == ".pdf":
        if _TEXTRACT_OK:
            return textract.process(str(p)).decode("utf-8", "ignore")
        raise RuntimeError("PDF detected. Install `textract` or convert to .txt.")

    # Default: attempt to read as text
    return _read_txt(p)
