"""
Pure-Python text-similarity primitives used by the memory module.

The original MEXTRA code relies on two retrieval / scoring functions:

  * EHRAgent  -> ``Levenshtein.distance`` (edit distance), see
    ``EHRAgent/ehragent/medagent.py::retrieve_examples``.
  * RAP       -> cosine similarity over SBERT/MiniLM embeddings, see
    ``RAP/attacking/run_attack.py::generate_examples``.

To keep this reproduction runnable with **zero third-party dependencies**
(the sandbox has no numpy/torch/sentence-transformers and no network), we
re-implement both here in the standard library:

  * ``levenshtein`` is an exact dynamic-programming edit distance, identical
    in behaviour to the ``python-Levenshtein`` package the authors use.
  * ``NgramEmbedder`` is a deterministic lexical embedder (character n-grams
    + word unigrams, L2-normalised). It is a *stand-in* for SBERT: it
    preserves the property the attack actually exploits -- that prepending a
    topic word shifts the query's vector toward topically-similar memory
    records -- without downloading a 90MB transformer. A real SBERT encoder
    can be dropped in via ``MemoryModule(embedder=...)`` for a faithful run.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence


try:
    from Levenshtein import distance as _lev_c   # C-backed, ~100x faster
except Exception:
    _lev_c = None


def levenshtein(a: str, b: str) -> int:
    """Exact Levenshtein (edit) distance between two strings.

    Matches ``Levenshtein.distance`` from the original ``medagent.py``. Uses the
    C-backed ``python-Levenshtein`` when available, else the pure-Python fallback.
    """
    if _lev_c is not None:
        return _lev_c(a, b)
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la:
        return lb
    if not lb:
        return la
    # Keep the inner row short for speed.
    if la < lb:
        a, b, la, lb = b, a, lb, la
    # Single-row in-place DP, O(la*lb) time, O(lb) space.
    row = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        prev_diag = row[0]   # value of cell (i-1, j-1)
        row[0] = i
        for j in range(1, lb + 1):
            cur = row[j]     # value of cell (i-1, j), saved before overwrite
            if ca == b[j - 1]:
                row[j] = prev_diag
            else:
                row[j] = 1 + min(prev_diag, row[j - 1], cur)
            prev_diag = cur
    return row[lb]


_WORD_RE = re.compile(r"[a-z0-9]+")


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    text = f" {text.lower().strip()} "
    return [text[i:i + n] for i in range(len(text) - n + 1)]


class NgramEmbedder:
    """Deterministic bag-of-(char-ngram + word) sparse embedder.

    This is intentionally simple and dependency-free. It produces a sparse
    L2-normalised vector (a ``dict`` of feature -> weight) so that
    ``cosine`` reduces to a sparse dot product.
    """

    def __init__(self, char_n: int = 3, word_weight: float = 2.0):
        self.char_n = char_n
        self.word_weight = word_weight

    def encode(self, text: str) -> Dict[str, float]:
        feats: Counter = Counter()
        for g in _char_ngrams(text, self.char_n):
            feats[f"c:{g}"] += 1.0
        for w in _WORD_RE.findall(text.lower()):
            feats[f"w:{w}"] += self.word_weight
        norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
        return {k: v / norm for k, v in feats.items()}


def cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity of two sparse (already-L2-normalised) vectors."""
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(w * vec_b.get(k, 0.0) for k, w in vec_a.items())


def topk_indices(scores: Sequence[float], k: int, largest: bool = True) -> List[int]:
    """Indices of the top-``k`` scores. Ties broken by original order, which
    mirrors the stable ``sorted`` used in the original code."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=largest)
    return order[:k]
