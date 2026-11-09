"""BM25 keyword search over chunk texts."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"\w+", re.UNICODE)

# A short list is enough: BM25's IDF already down-weights very common words.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "for",
        "from",
        "has",
        "have",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def _fold(text: str) -> str:
    """Lower-case and strip accents, so 'Belém' matches 'belem'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(token: str) -> str:
    """Very light stemming: fold simple English plurals ('palaces' -> 'palace')."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [
        _stem(token)
        for token in _WORD.findall(_fold(text))
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


class BM25Index:
    """Okapi BM25 ranking (Robertson & Zaragoza, 2009) over a fixed set of documents."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs = [Counter(tokenize(doc)) for doc in documents]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._avg_length = (sum(self._lengths) / len(self._docs)) if self._docs else 0.0

        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(doc.keys())
        n = len(self._docs)
        # The "+1" variant keeps IDF positive even for terms found in most documents.
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._docs)

    def scores(self, query: str) -> list[float]:
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        results = [0.0] * len(self._docs)
        if not terms or not self._avg_length:
            return results
        for i, doc in enumerate(self._docs):
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / self._avg_length)
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if tf:
                    score += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
            results[i] = score
        return results

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Indices and scores of the top ``k`` documents that match at least one term."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: item[1], reverse=True)
        return [(i, score) for i, score in ranked[:k] if score > 0]
