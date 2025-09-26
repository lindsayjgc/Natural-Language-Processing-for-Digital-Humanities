# Expose core NLP utilities for easier imports when using this package.

from .preprocessing import process_text        # Text cleaning, tokenization, lemmatization, POS tagging
from .features import count_ngrams, count_pos  # Feature extraction: n-grams and POS counts
from .sentiment import analyze_sentiment       # Sentiment/emotion analysis

# Define what symbols are exported when `from package import *` is used
__all__ = [
    "process_text",
    "count_ngrams",
    "count_pos",
    "analyze_sentiment",
]

# Package version identifier
__version__ = "0.1.0"
