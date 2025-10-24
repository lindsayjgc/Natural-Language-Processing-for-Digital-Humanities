from textwrap import wrap
from transformers import pipeline
import torch
from preprocess import preprocess_text

def abstractive_summarize(text: str, model_name="pszemraj/led-large-book-summary",
                          max_length=240, min_length=80, device=None):
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=device
    )

    clean_text = preprocess_text(text)
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
