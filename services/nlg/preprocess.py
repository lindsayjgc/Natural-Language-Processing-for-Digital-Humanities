import spacy

"""
def preprocess_text(text: str, max_len: int = 4096) -> str:
    text = " ".join(text.split())
    return text[:max_len * 4]

def count_sentences(text: str) -> int:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    return len([s for s in doc.sents])"""

# Load the spaCy model and add the sentencizer once at module level
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

def preprocess_text(text: str, max_len: int = 4096) -> str:
    text = " ".join(text.split())
    return text[:max_len * 4]
def count_sentences(text: str) -> int:
    if not text.strip():
        return 0
    doc = nlp(text)
    return sum(1 for _ in doc.sents)
