# usage: n-grams, POS counts, aggregation
from collections import Counter

def make_ngrams(tokens, n=2):
    return ["_".join(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

def count_ngrams(lemmas, ngram_ns=(1,2,3)):
    out = {}
    for n in ngram_ns:
        if n <= 1:
            out["unigram"] = Counter(lemmas)
        else:
            out[{2:"bigram",3:"trigram"}.get(n, f"ngram_{n}")] = Counter(make_ngrams(lemmas, n))
    return out

def count_pos(pos_seq):
    return Counter(pos_seq)
