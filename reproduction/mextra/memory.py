"""
The agent memory module M = {(q_i, s_i)}.

This is a faithful, simplified re-implementation of the retrieval logic that
appears (duplicated, in messy form) in the original repo:

  * edit-distance retrieval  -> ``EHRAgent/ehragent/medagent.py::retrieve_examples``
    (``Levenshtein.distance(query, question)``, ascending, take first ``num_shots``).
  * cosine retrieval         -> ``RAP/attacking/run_attack.py::generate_examples``
    (``cos_sim`` over SBERT embeddings, ``torch.topk``).

A "record" stores the private user query plus its (irrelevant to the attack)
solution. The attack only ever cares about ``record.query`` -- exactly as the
paper states: "Once the user queries are obtained, the corresponding agent
responses can be easily reproduced."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .textsim import NgramEmbedder, cosine, levenshtein, topk_indices


@dataclass
class Record:
    rid: str
    query: str                       # the *private* user query q_i (the target)
    solution: str = ""               # s_i (not needed by the attack)
    category: str = ""               # optional topical tag (used by synthetic data)


class MemoryModule:
    """Long-term memory with pluggable top-k retrieval.

    Parameters
    ----------
    records   : the stored (q_i, s_i) records.
    method    : 'edit_distance' or 'cosine' -- the scoring function f(q, q_i).
    embedder  : object with ``.encode(text) -> sparse vec`` (only for cosine).
                Defaults to the dependency-free :class:`NgramEmbedder`; pass a
                wrapper around SBERT here for a fully faithful run.
    """

    def __init__(
        self,
        records: List[Record],
        method: str = "edit_distance",
        embedder: Optional[object] = None,
    ):
        self.records = records
        self.method = method
        self.embedder = embedder or NgramEmbedder()
        self._emb_cache: Optional[List[Dict[str, float]]] = None
        if method == "cosine":
            self._emb_cache = [self.embedder.encode(r.query) for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def retrieve(self, query: str, k: int) -> List[Record]:
        """Return the top-``k`` records for ``query`` under ``self.method``.

        Mirrors the originals: edit distance ranks *ascending* (smaller =
        closer), cosine ranks *descending* (larger = closer).
        """
        idx = topk_indices(self.score_all(query), k,
                           largest=(self.method == "cosine"))
        return [self.records[i] for i in idx]

    def score_all(self, query: str) -> List[float]:
        """Raw score of ``query`` against every record (smaller=closer for edit
        distance, larger=closer for cosine). Exposed so callers can rank a
        prefix cheaply (used by the size/depth sweeps in experiments.py)."""
        if self.method == "edit_distance":
            return [levenshtein(query, r.query) for r in self.records]
        if self.method == "cosine":
            qv = self.embedder.encode(query)
            return [cosine(qv, ev) for ev in self._emb_cache]  # type: ignore[arg-type]
        raise ValueError(f"unknown retrieve method: {self.method}")
