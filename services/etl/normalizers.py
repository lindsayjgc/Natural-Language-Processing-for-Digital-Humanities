# usage: text normalization (gutenberg strip, footnotes removal, whitespace)
import re

def strip_gutenberg_headers(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate by slicing between
    detected START/END markers (robust to variant phrasings).
    Falls back to returning stripped input if markers not found.
    """
    lines = text.splitlines()

    # Regexes tolerate common START/END variants from Gutenberg dumps
    start_re = re.compile(
        r"""^(\*\*\*\s*START\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|\*\*\*\s*START\s+OF\s+.*EBOOK|START\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK)""",
        re.IGNORECASE,
    )
    end_re = re.compile(
        r"""^(\*\*\*\s*END\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|\*\*\*\s*END\s+OF\s+.*EBOOK|END\s+OF\s+(?:THE|THIS)?\s*PROJECT\s+GUTENBERG\s+EBOOK|End\s+of\s+the\s+Project\s+Gutenberg\s+EBook|End\s+of\s+Project\s+Gutenberg'?s)""",
        re.IGNORECASE,
    )

    start_idx, end_idx = 0, len(lines)
    # Find START boundary
    for i, ln in enumerate(lines):
        if start_re.search(ln.strip()):
            start_idx = i + 1
            break
    # Find END boundary (scan from bottom)
    for i in range(len(lines) - 1, -1, -1):
        if end_re.search(lines[i].strip()):
            end_idx = i
            break

    # If no markers detected, just trim outer whitespace
    if start_idx == 0 and end_idx == len(lines):
        return text.strip()

    # Return body slice
    body = "\n".join(lines[start_idx:end_idx]).strip()
    return body if body else text.strip()

def remove_footnotes(text: str) -> str:
    """
    Heuristically remove inline and trailing-section footnotes.

    Handles:
      - Inline “[Footnote: …]” spans (DOTALL).
      - Inline bracketed markers like “[12]”, “[iv]”, “[a]”.
      - Trailing 'FOOTNOTES'/'NOTES' sections located in back half of doc.
      - Leading bracketed list-style footnotes at start of lines.
    """
    # Remove inline “Footnote: …”
    text = re.sub(r'\[Footnote:.*?\]', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove inline bracketed markers (digits/roman/letters)
    text = re.sub(r'(?<!\w)\[(?:\d{1,3}|[ivxlcdm]{1,5}|[a-z])\](?!\w)', '', text, flags=re.IGNORECASE)

    # Drop trailing FOOTNOTES/NOTES section if it appears after mid-document
    m = re.search(r'^\s*(FOOTNOTES?|NOTES?)\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
    if m and m.start() > len(text) * 0.5:
        text = text[:m.start()].rstrip()

    # Remove line-leading “[n] …” blocks
    lines, out, i = text.splitlines(), [], 0
    leader = re.compile(r'^\s*\[\s*(\d{1,3}|[ivxlcdm]{1,5}|[a-z])\s*\]\s*', re.IGNORECASE)
    while i < len(lines):
        if leader.match(lines[i]):
            # Skip this footnote paragraph (consume until blank line)
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            # Skip following blank lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1
        else:
            out.append(lines[i]); i += 1
    return "\n".join(out)

def basic_clean(text: str, unwrap_lines: bool = True) -> str:
    """
    Normalize newlines/whitespace and optionally unwrap soft line breaks.

    - Convert CR/LF variants to '\n'
    - If unwrap_lines=True, join single newlines into spaces (keep paragraph breaks)
    - Collapse multiple blank lines to a single blank line
    - Collapse runs of spaces/tabs
    - Trim leading/trailing whitespace
    """
    text = re.sub(r'\r\n?', '\n', text)
    if unwrap_lines:
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # turn lone newlines into spaces
    text = re.sub(r'\n\s*\n+', '\n\n', text)          # squeeze multiple blank lines
    text = re.sub(r'[ \t]+', ' ', text)               # collapse spaces/tabs
    return text.strip()
