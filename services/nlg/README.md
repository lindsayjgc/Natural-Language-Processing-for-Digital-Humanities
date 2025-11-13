# Transformer-Based Abstractive Summarizer

A modular Python pipeline for generating **abstractive summaries** from large text files using **Hugging Face transformer models**.
The system supports batch summarization, sentence counting, and structured JSON output containing both metadata and generated summaries.

---

## 🚀 Features

* **Automatic summarization** using state-of-the-art transformer models
* **Batch processing** of `.txt` files from a directory
* **JSON output** with metadata and sentence statistics
* **Customizable** summarization parameters (`max_length`, `min_length`)
* **Supports multiple summarization models** (e.g., LED, BART, T5, Pegasus)
* **spaCy-based preprocessing and sentence segmentation**

---

## 🧩 Project Structure

```
.
├── generate_summaries.py    # CLI entry point; orchestrates the pipeline
├── io_utils.py              # File I/O: summarize and save results as JSON
├── models.py                # Supported model names and defaults
├── preprocess.py            # spaCy preprocessing & sentence counting
├── summarize.py             # Abstractive summarization logic using Hugging Face
```

---

## ⚙️ Installation

1. **Clone this repository**

   ```bash
   git clone <your_repo_url>
   cd <your_repo_name>
   ```

2. **Install dependencies**

   ```bash
   pip install torch transformers spacy
   python -m spacy download en_core_web_sm
   ```

---

## 💡 Usage

You can summarize a single text file **or all `.txt` files in a directory**.

### Command-Line Interface

```bash
python generate_summaries.py --input ./data --outdir ./summaries --model facebook/bart-large-cnn
```

### Arguments

| Argument       | Description                             | Default                           |
| -------------- | --------------------------------------- | --------------------------------- |
| `--input`      | Input file or directory of `.txt` files | **Required**                      |
| `--outdir`     | Directory for JSON summaries            | `summaries_abstractive_json`      |
| `--model`      | Hugging Face model name                 | `pszemraj/led-large-book-summary` |
| `--max-length` | Max tokens per summary chunk            | `240`                             |
| `--min-length` | Min tokens per summary chunk            | `80`                              |

---

## 🧠 Example Output

For each input file, the system produces a JSON file:

```json
{
  "file_name": "example.txt",
  "file_path": "/absolute/path/example.txt",
  "model": "facebook/bart-large-cnn",
  "input_sentence_count": 102,
  "summary_sentence_count": 8,
  "max_length": 240,
  "min_length": 80,
  "summary_text": "This is the generated summary..."
}
```

Saved as:

```
summaries_abstractive_json/example_summary.json
```

---

## 🧩 Supported Models

Defined in `models.py`:

* `pszemraj/led-large-book-summary`
* `facebook/bart-large-cnn`
* `t5-small`
* `google/pegasus-xsum`

You can easily extend this list with other transformer models from [Hugging Face](https://huggingface.co/models).

---

## 🧪 Example

```bash
python generate_summaries.py \
  --input ./texts/input.txt \
  --outdir ./outputs \
  --model google/pegasus-xsum \
  --max-length 200 \
  --min-length 50
```

**Output:**

```
[INFO] Summarizing input.txt using model: google/pegasus-xsum
[Chunk 1/2] summarizing...
[DONE] Saved JSON summary: input_summary.json
[INFO] All JSON summaries completed.
```

---

## 🛠️ Notes

* Long texts are automatically **chunked** into 3500-character segments to fit model input limits.
* Sentence counts are computed using **spaCy's sentencizer** for consistency.
* Summaries are fully **abstractive**, not extractive.

---

## 📜 License

MIT License — feel free to modify and extend.

