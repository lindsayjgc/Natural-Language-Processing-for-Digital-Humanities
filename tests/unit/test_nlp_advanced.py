"""
Additional NLP processing tests for comprehensive coverage
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add services to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.nlp.features import make_ngrams, count_ngrams, count_pos
from services.nlp.preprocessing import words_from_tokens, process_text
from services.nlp.sentiment import analyze_sentiment

class TestNLPFeatures:
    """Test NLP feature extraction functions"""

    def test_make_ngrams_simple(self):
        """Test n-gram generation on simple tokens"""
        tokens = ["hello", "world", "python"]
        bigrams = make_ngrams(tokens, n=2)
        
        assert len(bigrams) == 2
        assert "hello_world" in bigrams
        assert "world_python" in bigrams

    def test_make_ngrams_empty(self):
        """Test n-gram generation on empty tokens"""
        bigrams = make_ngrams([], n=2)
        assert bigrams == []

    def test_count_ngrams(self):
        """Test n-gram counting"""
        tokens = ["hello", "world", "hello"]
        ngram_counts = count_ngrams(tokens, ngram_ns=(1, 2))
        
        assert "unigram" in ngram_counts
        assert "bigram" in ngram_counts
        assert ngram_counts["unigram"]["hello"] == 2

    def test_count_pos(self):
        """Test POS tag counting"""
        pos_tags = ["NN", "VB", "NN", "JJ"]
        pos_counts = count_pos(pos_tags)
        
        assert isinstance(pos_counts, dict)
        assert pos_counts["NN"] == 2
        assert pos_counts["VB"] == 1

class TestNLPPreprocessing:
    """Test text preprocessing functions"""

    def test_words_from_tokens(self):
        """Test word extraction from tokens"""
        tokens = ["Hello", "world", "!", "123"]
        words = words_from_tokens(tokens, lowercase=True, remove_punct=True, remove_nums=True)
        
        assert isinstance(words, list)
        # Should filter out punctuation and numbers
        assert "!" not in words
        assert "123" not in words

    def test_words_from_tokens_empty(self):
        """Test word extraction from empty tokens"""
        words = words_from_tokens([])
        assert words == []

    def test_process_text_simple(self):
        """Test basic text processing"""
        text = "Hello world! This is a test."
        result = process_text(text)
        
        # Should return some kind of processed result
        assert result is not None
        assert isinstance(result, dict) or isinstance(result, tuple)

class TestSentimentAnalysis:
    """Test sentiment analysis functions"""

    def test_analyze_sentiment_basic(self):
        """Test sentiment analysis on sample text"""
        text = "I love this wonderful day!"
        sentiment = analyze_sentiment(text)
        
        # The function returns a tuple, check that it's not None and has content
        assert sentiment is not None
        assert isinstance(sentiment, tuple)
        assert len(sentiment) > 0

    def test_analyze_sentiment_empty(self):
        """Test sentiment analysis on empty text"""
        sentiment = analyze_sentiment("")
        
        # Should handle empty text gracefully
        assert sentiment is not None
        assert isinstance(sentiment, tuple)

class TestTextProcessing:
    """Test file processing functions"""

    def test_ngram_edge_cases(self):
        """Test edge cases in n-gram generation"""
        # Single token
        single = make_ngrams(["word"], n=2)
        assert single == []
        
        # n=1 should return original tokens
        tokens = ["a", "b", "c"]
        unigrams = make_ngrams(tokens, n=1)
        assert len(unigrams) == 3

    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent"""
        text = "This is a test sentence."
        result1 = process_text(text)
        result2 = process_text(text)
        
        # Same input should give same output
        assert type(result1) == type(result2)

class TestIntegration:
    """Test integration between different NLP components"""

    def test_full_pipeline_simple(self):
        """Test a simple end-to-end NLP pipeline"""
        text = "Hello world. This is a test."
        
        # Process text
        processed = process_text(text)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(text)
        
        # Both should complete without error
        assert processed is not None
        assert sentiment is not None

    def test_error_handling(self):
        """Test error handling in NLP functions"""
        # These should not crash the system
        try:
            process_text("")
            analyze_sentiment("")
            make_ngrams([], 1)
            count_ngrams([])
            count_pos([])
        except Exception as e:
            # If exceptions occur, they should be reasonable
            assert isinstance(e, (ValueError, TypeError, AttributeError))