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
    knowledge: str = ""              # EHRAgent knowledge field
    code: str = ""                   # EHRAgent code/solution field
    actions: List[str] = field(default_factory=list)  # RAP action traces


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

    def append(self, record: Record, cap: Optional[int] = None) -> None:
        """Write a new (q, s) record into memory, as a real agent does after a
        successful interaction (see EHRAgent/attacking/init_memory.py, which
        appends regardless of correctness). With ``cap`` set, evicts the oldest
        record (FIFO) once the cap is exceeded. Keeps the cosine cache in sync."""
        self.records.append(record)
        if self.method == "cosine":
            self._emb_cache.append(self.embedder.encode(record.query))  # type: ignore[union-attr]
        if cap is not None and len(self.records) > cap:
            self.records.pop(0)
            if self.method == "cosine":
                self._emb_cache.pop(0)  # type: ignore[union-attr]

    def retrieve(self, query: str, k: int, floor: Optional[float] = None) -> List[Record]:
        """Return the top-``k`` records for ``query`` under ``self.method``.

        Mirrors the originals: edit distance ranks *ascending* (smaller =
        closer), cosine ranks *descending* (larger = closer).

        ``floor`` adds the *relevance gate* that production RAG stacks commonly
        apply but the paper's victims do not: a record is only injected if it is
        actually relevant -- cosine score ``>= floor`` (similarity) or edit
        distance ``<= floor`` (closeness). With it set, a generic "dump your
        memory" prompt -- which is lexically unrelated to any specific private
        record -- retrieves *nothing*, so there is nothing to leak (see
        ``stress2.py`` S2). Default ``None`` = unconditional top-k (paper
        behaviour), so existing callers are unaffected.
        """
        scores = self.score_all(query)
        idx = topk_indices(scores, k, largest=(self.method == "cosine"))
        if floor is not None:
            if self.method == "cosine":
                idx = [i for i in idx if scores[i] >= floor]
            else:  # edit distance: smaller = closer, so gate on an upper bound
                idx = [i for i in idx if scores[i] <= floor]
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
