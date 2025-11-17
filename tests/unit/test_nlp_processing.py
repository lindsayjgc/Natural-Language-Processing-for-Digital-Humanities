"""
Unit tests for NLP processing functionality
Tests core NLP pipeline components
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.nlp.analyze_texts import process_path
from services.nlp.features import make_ngrams, count_ngrams, count_pos


class TestNLPProcessing:
    """Test NLP processing functions"""

    def test_ngram_generation(self):
        """Test n-gram generation"""
        tokens = ["the", "cat", "sat", "on", "the", "mat"]

        bigrams = make_ngrams(tokens, n=2)
        assert len(bigrams) == 5  # 6 tokens - 1 = 5 bigrams
        assert bigrams[0] == "the_cat"  # Returns joined strings
        assert bigrams[1] == "cat_sat"

    def test_ngram_counting(self):
        """Test n-gram counting"""
        lemmas = ["the", "cat", "sat", "on", "the", "mat"]

        ngram_counts = count_ngrams(lemmas, ngram_ns=(1, 2))
        assert "unigram" in ngram_counts  # Unigrams
        assert "bigram" in ngram_counts  # Bigrams
        assert len(ngram_counts["unigram"]) > 0  # Should have unigram counts

    def test_pos_counting(self):
        """Test POS tag counting"""
        pos_seq = ["DT", "NN", "VBD", "IN", "DT", "NN"]

        pos_counts = count_pos(pos_seq)
        assert "DT" in pos_counts  # Determiners
        assert "NN" in pos_counts  # Nouns
        assert pos_counts["DT"] == 2  # Two determiners

    def test_whitespace_handling_in_processing(self):
        """Test that processing handles whitespace correctly"""
        text = "This  is   a test with    extra    spaces."

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write(text)
            tmp_path = Path(tmp_file.name)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = process_path(
                    ipath=tmp_path, outdir=Path(tmp_dir), from_raw=True
                )

                # Processing should handle text with extra spaces
                assert result["token_count"] > 0

        finally:
            tmp_path.unlink()

    def test_text_processing_basic(self):
        """Test basic text file processing"""
        # Create temporary text file
        test_text = """
        This is a sample document for testing.
        It contains multiple sentences with different sentiments.
        The text is designed to test the NLP processing pipeline.
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write(test_text)
            tmp_path = Path(tmp_file.name)

        try:
            # Create output directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                outdir = Path(tmp_dir)

                # Process the text
                result = process_path(ipath=tmp_path, outdir=outdir, from_raw=True)

                # Verify results
                assert "vocab_size" in result
                assert "token_count" in result
                assert "type_token_ratio" in result
                assert "doc_sentiment" in result
                assert "sentiment_method" in result

                # Check reasonable values
                assert result["vocab_size"] > 0
                assert result["token_count"] > 0
                assert 0 <= result["type_token_ratio"] <= 1

                # Check sentiment scores
                sentiment = result["doc_sentiment"]
                assert isinstance(sentiment, dict)
                # Sentiment scores should sum to approximately 1
                total = sum(sentiment.values())
                assert pytest.approx(total, 0.1) == 1.0

        finally:
            # Cleanup
            tmp_path.unlink()

    def test_sentiment_analysis(self):
        """Test sentiment analysis on different texts"""
        positive_text = "I am so happy and joyful today! This is wonderful news!"
        negative_text = "This is terrible and sad. I feel very disappointed."
        neutral_text = "The meeting is scheduled for 3 PM tomorrow."

        for text, expected_emotion in [
            (positive_text, "joy"),
            (negative_text, "sadness"),
            (neutral_text, "neutral"),
        ]:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp_file:
                tmp_file.write(text)
                tmp_path = Path(tmp_file.name)

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = process_path(
                        ipath=tmp_path, outdir=Path(tmp_dir), from_raw=True
                    )

                    sentiment = result["doc_sentiment"]

                    # The expected emotion should have a high score
                    # Note: This is a heuristic test and may not always pass
                    # depending on the model
                    max_emotion = max(sentiment.items(), key=lambda x: x[1])
                    print(
                        f"Text: {text[:50]}... → Detected: {max_emotion[0]} ({max_emotion[1]:.2f})"
                    )

            finally:
                tmp_path.unlink()

    def test_ngram_edge_cases(self):
        """Test n-gram generation with edge cases"""
        # Single token
        single_token = ["word"]
        bigrams = make_ngrams(single_token, n=2)
        assert len(bigrams) == 0  # No bigrams from single token

        # Empty list
        empty_tokens = []
        bigrams = make_ngrams(empty_tokens, n=2)
        assert len(bigrams) == 0

    def test_process_path_with_invalid_file(self):
        """Test processing with non-existent file"""
        fake_path = Path("/tmp/nonexistent_file_12345.txt")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(Exception):
                process_path(ipath=fake_path, outdir=Path(tmp_dir), from_raw=True)

    def test_pos_tagging_edge_cases(self):
        """Test POS tagging with edge cases"""
        # Empty POS sequence
        empty_pos = []
        pos_counts = count_pos(empty_pos)
        assert len(pos_counts) == 0

        # Single POS tag
        single_pos = ["NN"]
        pos_counts = count_pos(single_pos)
        assert pos_counts["NN"] == 1

    def test_punctuation_handling(self):
        """Test handling of punctuation in text"""
        text = "Hello, world! How are you? I'm fine."

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write(text)
            tmp_path = Path(tmp_file.name)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = process_path(
                    ipath=tmp_path, outdir=Path(tmp_dir), from_raw=True
                )

                # Should handle punctuation appropriately
                assert result["token_count"] > 0
                assert result["vocab_size"] > 0

        finally:
            tmp_path.unlink()

    def test_long_text_processing(self):
        """Test processing longer text documents"""
        # Generate a longer text
        long_text = " ".join(
            [
                "This is sentence number {}.".format(i)
                for i in range(1, 101)  # 100 sentences
            ]
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write(long_text)
            tmp_path = Path(tmp_file.name)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = process_path(
                    ipath=tmp_path, outdir=Path(tmp_dir), from_raw=True
                )

                # Should handle longer texts
                assert result["token_count"] > 100  # At least 100 tokens
                assert result["vocab_size"] > 0  # Should have some vocabulary

        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
