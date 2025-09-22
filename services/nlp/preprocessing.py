# usage: tokenization, lemmatization, stopwords, POS (spaCy stream -> NLTK fallback)
import re
from collections import Counter

_SPACY_OK = False
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])
    if "sentencizer" not in _nlp.pipe_names:
        _nlp.add_pipe("sentencizer")
    _nlp.max_length = 1_000_000_000
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

def _wn_pos(treebank_tag: str):
    tag = treebank_tag[:1].upper()
    return {'J':'a','N':'n','V':'v','R':'r'}.get(tag, 'n')

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

def _spacy_stream(text: str,
                  *,
                  lowercase=True,
                  remove_punct=True,
                  remove_nums=True,
                  remove_stop=True,
                  chunk_chars=120_000,
                  batch_size=8):
    sentences = []
    tokens_basic = []
    tokens_nostop = []
    lemmas = []
    pos_seq = []

    if not chunk_chars or chunk_chars <= 0:
        chunks = [text]  # single pass (you asked for "no limit")
    else:
        chunks = (text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars))

    for doc in _nlp.pipe(chunks, batch_size=batch_size):
        for s in doc.sents:
            s_text = s.text.strip()
            if s_text:
                sentences.append(s_text)

        for t in doc:
            if t.is_space:
                continue
            w_norm = t.text.lower() if lowercase else t.text

            basic_keep = True
            if remove_punct and (t.is_punct or re.fullmatch(r"\W+", w_norm)):
                basic_keep = False
            if remove_nums and (t.like_num or re.fullmatch(r"\d+([.,]\d+)?", w_norm)):
                basic_keep = False
            if basic_keep:
                tokens_basic.append(w_norm)

            nostop_keep = basic_keep and (not remove_stop or (w_norm not in _STOPWORDS and not t.is_stop))
            if nostop_keep:
                lemma = t.lemma_ if t.lemma_ != "-PRON-" else w_norm
                tokens_nostop.append(w_norm)
                lemmas.append(lemma)

            pos_seq.append(t.pos_)

    return sentences, tokens_basic, tokens_nostop, lemmas, pos_seq

def process_text(text: str,
                 *,
                 lowercase=True,
                 remove_punct=True,
                 remove_nums=True,
                 remove_stop=True,
                 chunk_chars=120_000,
                 batch_size=8):
    if _SPACY_OK:
        try:
            sentences, tokens_basic, tokens_nostop, lemmas, pos_seq = _spacy_stream(
                text,
                lowercase=lowercase,
                remove_punct=remove_punct,
                remove_nums=remove_nums,
                remove_stop=remove_stop,
                chunk_chars=chunk_chars,
                batch_size=batch_size,
            )
        except ValueError:
            # pathological token edge case → NLTK fallback
            sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
            tokens_raw = word_tokenize(text)
            pos_seq = [p for _, p in pos_tag([w for w in tokens_raw if re.search(r"\S", w)], tagset="universal")]
            tokens_basic = words_from_tokens(tokens_raw, lowercase=lowercase, remove_punct=remove_punct, remove_nums=remove_nums)
            tokens_nostop = [t for t in tokens_basic if (t not in _STOPWORDS)] if remove_stop else list(tokens_basic)
            lemmas = [WordNetLemmatizer().lemmatize(w) for w in tokens_nostop]
    else:
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        tokens_raw = word_tokenize(text)
        pos_seq = [p for _, p in pos_tag([w for w in tokens_raw if re.search(r"\S", w)], tagset="universal")]
        tokens_basic = words_from_tokens(tokens_raw, lowercase=lowercase, remove_punct=remove_punct, remove_nums=remove_nums)
        tokens_nostop = [t for t in tokens_basic if (t not in _STOPWORDS)] if remove_stop else list(tokens_basic)
        lemmas = [WordNetLemmatizer().lemmatize(w) for w in tokens_nostop]

    freq_lemmas = Counter(lemmas)
    vocab_size = len(freq_lemmas)
    token_count = len(lemmas)
    ttr = (vocab_size / token_count) if token_count else 0.0

    return {
        "sentences": sentences,
        "tokens": tokens_basic,
        "tokens_nostop": tokens_nostop,
        "lemmas": lemmas,
        "pos_seq": pos_seq,
        "freq_lemmas": freq_lemmas,
        "vocab_size": vocab_size,
        "token_count": token_count,
        "type_token_ratio": ttr
    }
