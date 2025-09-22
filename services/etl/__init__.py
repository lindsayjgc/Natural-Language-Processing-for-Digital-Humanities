from .readers import read_text_smart
from .normalizers import strip_gutenberg_headers, remove_footnotes, basic_clean
from .ingest_texts import process_one as ingest_one

__all__ = [
    "read_text_smart",
    "strip_gutenberg_headers",
    "remove_footnotes",
    "basic_clean",
    "ingest_one",
]

__version__ = "0.1.0"
