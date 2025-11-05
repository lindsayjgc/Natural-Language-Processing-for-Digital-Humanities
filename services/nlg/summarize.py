from textwrap import wrap
from transformers import pipeline
import torch
from preprocess import preprocess_text

def abstractive_summarize(
    text: str,
    model_name: str = "pszemraj/led-large-book-summary",
    max_length: int = 240,
    min_length: int = 80,
    device: int | None = None,
    summarizer=None
) -> str:
    """
    Generate an abstractive summary of the given text using a transformer summarization model.

    If a summarizer pipeline is provided, it will be reused instead of reloaded.

    Args:
        text: Input text to summarize.
        model_name: Hugging Face model name for summarization.
        max_length: Max tokens per summary chunk.
        min_length: Min tokens per summary chunk.
        device: Device ID (0 = GPU, -1 = CPU). Auto-detected if None.
        summarizer: Optional preloaded Hugging Face pipeline.

    Returns:
        The combined abstractive summary.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Load summarizer only if not provided
    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=device
        )

    clean_text = preprocess_text(text)
    # Chunk size of 3500 characters is chosen to stay within the model's input limit (e.g., LED model's max input length).
    chunks = wrap(clean_text, 3500)
    partials = []

    for i, chunk in enumerate(chunks, 1):
        print(f"   [Chunk {i}/{len(chunks)}] summarizing...")
        result = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            truncation=True,
            do_sample=False
        )[0]["summary_text"]
        partials.append(result)

    return " ".join(partials)
