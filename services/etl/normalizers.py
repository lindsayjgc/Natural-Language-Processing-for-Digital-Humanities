# usage: text normalization (gutenberg strip, footnotes removal, whitespace)
import re

def strip_gutenberg_headers(text: str) -> str:
    lines = text.splitlines()
    start_re = re.compile(r"""^(\*\*\*\s*START\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|\*\*\*\s*START\s+OF\s+.*EBOOK|START\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK)""", re.IGNORECASE)
    end_re = re.compile(r"""^(\*\*\*\s*END\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|\*\*\*\s*END\s+OF\s+.*EBOOK|END\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|End\s+of\s+the\s+Project\s+Gutenberg\s+EBook|End\s+of\s+Project\s+Gutenberg'?s)""", re.IGNORECASE)
    start_idx, end_idx = 0, len(lines)
    for i, ln in enumerate(lines):
        if start_re.search(ln.strip()):
            start_idx = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if end_re.search(lines[i].strip()):
            end_idx = i
            break
    if start_idx == 0 and end_idx == len(lines):
        return text.strip()
    body = "\n".join(lines[start_idx:end_idx]).strip()
    return body if body else text.strip()

def remove_footnotes(text: str) -> str:
    text = re.sub(r'\[Footnote:.*?\]', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'(?<!\w)\[(?:\d{1,3}|[ivxlcdm]{1,5}|[a-z])\](?!\w)', '', text, flags=re.IGNORECASE)
    m = re.search(r'^\s*(FOOTNOTES?|NOTES?)\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
    if m and m.start() > len(text) * 0.5:
        text = text[:m.start()].rstrip()
    lines, out, i = text.splitlines(), [], 0
    leader = re.compile(r'^\s*\[\s*(\d{1,3}|[ivxlcdm]{1,5}|[a-z])\s*\]\s*', re.IGNORECASE)
    while i < len(lines):
        if leader.match(lines[i]):
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
        else:
            out.append(lines[i]); i += 1
    return "\n".join(out)

def basic_clean(text: str, unwrap_lines: bool = True) -> str:
    text = re.sub(r'\r\n?', '\n', text)
    if unwrap_lines:
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
