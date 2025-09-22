# usage: tokenization, lemmatization, stopwords, POS (spaCy -> NLTK fallback)
import re
from collections import Counter

_SPACY_OK = False
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    _nlp = None
    _SPACY_OK = False

import nltk
for pkg, locator in [
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("universal_tagset", "help/tagsets/upenn_tagset.pickle"),
]:
    try:
        nltk.data.find(locator)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

_STOPWORDS = set(stopwords.words("english"))
_WORDNET_LEM = WordNetLemmatizer()

def words_from_tokens(tokens, *, lowercase=True, remove_punct=True, remove_nums=True):
    out = []
    for t in tokens:
        w = t.lower() if lowercase else t
        if remove_punct and re.fullmatch(r"\W+", w):
            continue
        if remove_nums and re.fullmatch(r"\d+([.,]\d+)?", w):
            continue
        out.append(w)
    return out

def _wn_pos(treebank_tag: str):
    tag = treebank_tag[:1].upper()
    return {'J':'a','N':'n','V':'v','R':'r'}.get(tag, 'n')

def lemmatize_tokens(tokens):
    if _SPACY_OK:
        doc = _nlp(" ".join(tokens))
        return [t.lemma_ if t.lemma_ not in ("-PRON-",) else t.lower_ for t in doc]
    pos_tags = pos_tag(tokens)
    return [WordNetLemmatizer().lemmatize(w, _wn_pos(tag)) for w, tag in pos_tags]

def process_text(text: str,
                 *,
                 lowercase=True,
                 remove_punct=True,
                 remove_nums=True,
                 remove_stop=True):
    if _SPACY_OK:
        doc = _nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        tokens_raw = [t.text for t in doc]
        pos_seq = [t.pos_ for t in doc if not t.is_space]
    else:
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        tokens_raw = word_tokenize(text)
        pos_seq = [p for _, p in pos_tag([w for w in tokens_raw if re.search(r"\S", w)], tagset="universal")]

    tokens = words_from_tokens(tokens_raw, lowercase=lowercase, remove_punct=remove_punct, remove_nums=remove_nums)
    tokens_nostop = [t for t in tokens if (t not in _STOPWORDS)] if remove_stop else list(tokens)
    lemmas = lemmatize_tokens(tokens_nostop)

    freq_lemmas = Counter(lemmas)
    vocab_size = len(freq_lemmas)
    token_count = len(lemmas)
    ttr = (vocab_size / token_count) if token_count else 0.0

    return {
        "sentences": sentences,
        "tokens": tokens,
        "tokens_nostop": tokens_nostop,
        "lemmas": lemmas,
        "pos_seq": pos_seq,
        "freq_lemmas": freq_lemmas,
        "vocab_size": vocab_size,
        "token_count": token_count,
        "type_token_ratio": ttr
    }
