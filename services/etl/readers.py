# usage: robust text reader for .txt/.docx/.doc/.rtf/.pdf
import zipfile, html, re
from pathlib import Path

try:
    import docx2txt
except Exception:
    docx2txt = None

_DOCX_OK = False
try:
    import docx as _py_docx
    _DOCX_OK = True
except Exception:
    _py_docx = None
    _DOCX_OK = False

_TEXTRACT_OK = False
try:
    import textract
    _TEXTRACT_OK = True
except Exception:
    textract = None
    _TEXTRACT_OK = False

def _read_txt(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_bytes().decode("utf-8", "ignore")

def _read_docx_with_docx2txt(p: Path) -> str:
    return docx2txt.process(str(p)) or ""

def _read_docx_with_python_docx(p: Path) -> str:
    d = _py_docx.Document(str(p))
    return "\n".join(par.text for par in d.paragraphs)

def _read_docx_by_zip(p: Path) -> str:
    with zipfile.ZipFile(str(p)) as z:
        try:
            data = z.read("word/document.xml").decode("utf-8", "ignore")
        except KeyError:
            data = "".join(z.read(n).decode("utf-8", "ignore") for n in z.namelist() if n.endswith(".xml"))
    data = re.sub(r"</w:p>", "\n", data)
    data = re.sub(r"<[^>]+>", "", data)
    return html.unescape(data)

def _read_rtf_simple(p: Path) -> str:
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
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".txt", ".md", ".rst"):
        return _read_txt(p)
    if suf == ".docx":
        kind = _sniff_kind(p)
        if kind != "docx_zip":
            if kind == "rtf":
                return _read_rtf_simple(p)
            if kind == "ole_doc" and _TEXTRACT_OK:
                return textract.process(str(p)).decode("utf-8", "ignore")
            return _read_txt(p)
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
    if suf == ".doc":
        if _TEXTRACT_OK:
            return textract.process(str(p)).decode("utf-8", "ignore")
        raise RuntimeError("Legacy .doc detected. Install `textract` or convert to .txt/.docx.")
    if suf == ".rtf":
        return _read_rtf_simple(p)
    if suf == ".pdf":
        if _TEXTRACT_OK:
            return textract.process(str(p)).decode("utf-8", "ignore")
        raise RuntimeError("PDF detected. Install `textract` or convert to .txt.")
    return _read_txt(p)
