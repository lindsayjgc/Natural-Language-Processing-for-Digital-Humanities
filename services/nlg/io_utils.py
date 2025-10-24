import json
from pathlib import Path
from summarize import abstractive_summarize
from preprocess import count_sentences

def process_file(ipath: Path, outdir: Path, model_name: str,
                 max_length: int, min_length: int):
    """Summarize a text file and write JSON output."""
    outdir.mkdir(parents=True, exist_ok=True)
    text = ipath.read_text(encoding="utf-8")

    print(f"[INFO] Summarizing {ipath.name} using model: {model_name}")
    summary = abstractive_summarize(
        text,
        model_name=model_name,
        max_length=max_length,
        min_length=min_length
    )

    meta = {
        "file_name": ipath.name,
        "file_path": str(ipath.resolve()),
        "model": model_name,
        "input_sentence_count": count_sentences(text),
        "summary_sentence_count": count_sentences(summary),
        "max_length": max_length,
        "min_length": min_length,
        "summary_text": summary
    }

    out_file = outdir / f"{ipath.stem}_summary.json"
    out_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Saved JSON summary: {out_file.name}")
    return meta
