"""
Package export surface for ETL helpers.
Exposes common reader/normalizer/ingest entry points at the package level.
"""
from .readers import read_text_smart          # robust loader for txt/docx/doc/rtf/pdf
from .normalizers import (                     # text normalization utilities
    strip_gutenberg_headers,
    remove_footnotes,
    basic_clean,
)
from .ingest_texts import process_one as ingest_one  # single-file ETL helper

__all__ = [
    "read_text_smart",
    "strip_gutenberg_headers",
    "remove_footnotes",
    "basic_clean",
    "ingest_one",
]

# Semantic version for this mini-package
__version__ = "0.1.0"
