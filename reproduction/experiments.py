#!/usr/bin/env python3
"""
RQ2 (memory-module configuration) and RQ3 (prompting strategy) sweeps.

Reproduces the *trends* in paper Table 2 and Figures 2/3/4:

  RQ2a  scoring function : edit_distance vs cosine        (Table 2)
  RQ2b  memory size m    : 50..500                        (Fig 2)
  RQ2c  retrieval depth k: 1..5                           (Fig 3)
  RQ3   #prompts n + basic vs advanced instruction        (Fig 4)

The retrieval-side quantities (RN, overlap, and therefore how much *can* be
extracted) are computed exactly; EN/EE additionally use the offline compliance
model (see mextra/agent.py). Writes CSVs to results/ and prints text tables.
"""

from __future__ import annotations

import csv
import os

from mextra.agent import SimulatedAgent
from mextra.attack import generate_prompts
from mextra.data import make_memory
from mextra.evaluate import PromptResult, compute_metrics, run_attack
from mextra.memory import MemoryModule
from mextra.textsim import topk_indices

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

MEM_SIZES = [50, 100, 200, 300, 400, 500]
KS = [1, 2, 3, 4, 5]
NS = [10, 20, 30, 40, 50]


def _attack(agent_key, method, strategy, n, k, m, seed=0):
    records = make_memory(agent_key, size=m, seed=seed)
    memory = MemoryModule(records, method=method)
    agent = SimulatedAgent(agent=agent_key)
    prompts = generate_prompts(agent_key, strategy=strategy, n=n, backend="offline")
    results = run_attack(memory, prompts, agent, k=k)
    return compute_metrics(results, k=k)


def _metrics_from_scores(prompts, records, score_rows, agent, k, m, largest):
    """Run the attack for memory size ``m`` and depth ``k`` reusing a
    precomputed score matrix (rows = prompts, cols = all records). Since smaller
    memories are prefixes (data.make_memory is order-stable), top-k over the
    first ``m`` records is just a slice -- avoiding recomputation across the
    size/depth sweeps. The compliance stage is identical to evaluate.run_attack.
    ``largest`` = True for cosine (larger=closer), False for edit distance."""
    results = []
    for p, scores in zip(prompts, score_rows):
        text = str(p)
        idx = topk_indices(scores[:m], k, largest=largest)
        retrieved = [records[i] for i in idx]
        retrieved_q = [r.query for r in retrieved]
        outputs, extracted = [], set()
        for run_idx in range(3):
            resp = agent.respond(text, retrieved, run_idx=run_idx)
            outputs.append(resp.output)
            low = resp.output.lower()
            for q in retrieved_q:
                if q.lower() in low:
                    extracted.add(q)
            if retrieved_q and len(extracted) == len(retrieved_q):
                break
        ext = [q for q in retrieved_q if q in extracted]
        results.append(PromptResult(prompt=text, retrieved=retrieved_q, outputs=outputs, extracted=ext))
    return compute_metrics(results, k=k)


def _precompute(agent_key, method, prompts, records):
    """Score matrix: one row of per-record scores per prompt (computed once)."""
    mem = MemoryModule(records, method=method)
    return [mem.score_all(str(p)) for p in prompts]


def rq2_scoring_and_size():
    print("\n############ RQ2: scoring function x memory size (EN) ############")
    rows = [["agent", "scoring", *[f"m={m}" for m in MEM_SIZES]]]
    for agent_key in ("ehragent", "rap"):
        k = 4 if agent_key == "ehragent" else 3
        records = make_memory(agent_key, size=max(MEM_SIZES), seed=0)
        agent = SimulatedAgent(agent=agent_key)
        prompts = generate_prompts(agent_key, strategy="basic", n=30)
        for method in ("edit_distance", "cosine"):
            largest = method == "cosine"
            score_rows = _precompute(agent_key, method, prompts, records)  # once
            ens = [_metrics_from_scores(prompts, records, score_rows, agent, k, m, largest).EN
                   for m in MEM_SIZES]
            rows.append([agent_key, method, *ens])
            print(f"{agent_key:>9} {method:>13} | EN " + "  ".join(f"{e:>3}" for e in ens))
    _write_csv("rq2_scoring_size.csv", rows)


def rq2_retrieval_depth():
    print("\n############ RQ2: retrieval depth k (EN / RN) ############")
    rows = [["agent", "metric", *[f"k={k}" for k in KS]]]
    for agent_key, method in (("ehragent", "edit_distance"), ("rap", "cosine")):
        largest = method == "cosine"
        records = make_memory(agent_key, size=200, seed=0)
        agent = SimulatedAgent(agent=agent_key)
        prompts = generate_prompts(agent_key, strategy="basic", n=30)
        score_rows = _precompute(agent_key, method, prompts, records)  # once, reuse across k
        ens, rns = [], []
        for k in KS:
            mtr = _metrics_from_scores(prompts, records, score_rows, agent, k, 200, largest)
            ens.append(mtr.EN)
            rns.append(mtr.RN)
        rows.append([agent_key, "EN", *ens])
        rows.append([agent_key, "RN", *rns])
        print(f"{agent_key:>9} EN | " + "  ".join(f"{e:>3}" for e in ens))
        print(f"{agent_key:>9} RN | " + "  ".join(f"{r:>3}" for r in rns))
    _write_csv("rq2_retrieval_depth.csv", rows)


def _rq3_sweep_n(agent_key, method, strat, k):
    """EN/RN across NS. Reuses a single score matrix for prefix-stable
    strategies (basic, advanced_cosine); recomputes per-n for advanced_edit
    whose length ladder depends on n."""
    largest = method == "cosine"
    records = make_memory(agent_key, size=200, seed=0)
    agent = SimulatedAgent(agent=agent_key)
    ens, rns = [], []
    if strat == "advanced_edit":
        for n in NS:
            prompts = generate_prompts(agent_key, strat, n=n)
            sr = _precompute(agent_key, method, prompts, records)
            mtr = _metrics_from_scores(prompts, records, sr, agent, k, 200, largest)
            ens.append(mtr.EN); rns.append(mtr.RN)
    else:
        prompts = generate_prompts(agent_key, strat, n=max(NS))
        sr = _precompute(agent_key, method, prompts, records)  # once
        for n in NS:
            mtr = _metrics_from_scores(prompts[:n], records, sr[:n], agent, k, 200, largest)
            ens.append(mtr.EN); rns.append(mtr.RN)
    return ens, rns


def rq3_num_prompts_and_instruction():
    print("\n############ RQ3: #prompts n x instruction (EN / RN) ############")
    rows = [["agent", "scoring", "instruction", "metric", *[f"n={n}" for n in NS]]]
    cfgs = [
        ("ehragent", "edit_distance", {"basic": "basic", "advanced": "advanced_edit"}),
        ("ehragent", "cosine", {"basic": "basic", "advanced": "advanced_cosine"}),
        ("rap", "edit_distance", {"basic": "basic", "advanced": "advanced_edit"}),
        ("rap", "cosine", {"basic": "basic", "advanced": "advanced_cosine"}),
    ]
    for agent_key, method, instrs in cfgs:
        k = 4 if agent_key == "ehragent" else 3
        for iname, strat in instrs.items():
            ens, rns = _rq3_sweep_n(agent_key, method, strat, k)
            rows.append([agent_key, method, iname, "EN", *ens])
            rows.append([agent_key, method, iname, "RN", *rns])
            print(f"{agent_key:>9} {method:>13} {iname:>8} EN | " + "  ".join(f"{e:>3}" for e in ens)
                  + "   RN | " + "  ".join(f"{r:>3}" for r in rns))
    _write_csv("rq3_num_prompts.csv", rows)


def _write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  -> wrote {path}")


if __name__ == "__main__":
    rq2_scoring_and_size()
    rq2_retrieval_depth()
    rq3_num_prompts_and_instruction()
    print("\nKey expected trends (and what to look for):")
    print("  * edit_distance >= cosine for EN at fixed m   (paper Table 2)")
    print("  * EN/RN grow with memory size m               (paper Fig 2)")
    print("  * EN/RN grow with retrieval depth k           (paper Fig 3)")
    print("  * EN/RN grow with n; advanced >= basic on RN  (paper Fig 4)")
