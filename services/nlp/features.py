# Feature extraction utilities: n-grams and POS counts
from collections import Counter

def make_ngrams(tokens, n=2):
    """
    Construct n-grams of length n from a list of tokens.
    Example: tokens=["I","like","cats"], n=2 → ["I_like", "like_cats"]
    """
    return ["_".join(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

def count_ngrams(lemmas, ngram_ns=(1,2,3)):
    """
    Count frequency of unigrams, bigrams, trigrams, etc.
    Returns a dict of {ngram_name: Counter}.
    """
    out = {}
    for n in ngram_ns:
        if n <= 1:
            out["unigram"] = Counter(lemmas)
        else:
            out[{2:"bigram",3:"trigram"}.get(n, f"ngram_{n}")] = Counter(make_ngrams(lemmas, n))
    return out

def count_pos(pos_seq):
    """
    Count occurrences of POS tags in a sequence.
    """
    return Counter(pos_seq)
