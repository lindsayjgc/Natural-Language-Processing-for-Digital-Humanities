from typing import List

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks, trying not to break sentences.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        k = j
        window = min(400, n - j)
        if window > 0:
            tail = text[j:j+window]
            dot = tail.find(". ")
            if dot != -1:
                k = j + dot + 2
        chunks.append(text[i:k])
        if k >= n:
            break
        i = max(0, k - overlap)
    return chunks
