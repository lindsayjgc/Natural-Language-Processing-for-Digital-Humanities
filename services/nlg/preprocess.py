import spacy

def preprocess_text(text: str, max_len: int = 4096) -> str:
    text = " ".join(text.split())
    return text[:max_len * 4]

def count_sentences(text: str) -> int:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    return len([s for s in doc.sents])
