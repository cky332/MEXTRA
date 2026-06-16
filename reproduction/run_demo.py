#!/usr/bin/env python3
"""
End-to-end MEXTRA demo (RQ1 + ablations), runnable offline with no deps.

It reproduces the *shape* of paper Table 1: MEXTRA vs the ablation baselines
(w/o aligner / w/o req / w/o demos) on EHRAgent (edit distance, k=4) and RAP
(cosine, k=3), with n=30 attacking prompts and memory size m=200.

Usage:
    python run_demo.py                      # offline simulated victim
    python run_demo.py --backend openai     # real LLM victim (needs key+net)
"""

from __future__ import annotations

import argparse

from mextra.agent import OpenAIAgent, SimulatedAgent
from mextra.attack import generate_prompts
from mextra.data import make_memory
from mextra.evaluate import compute_metrics, overlap_histogram, run_attack
from mextra.memory import MemoryModule


AGENTS = {
    # agent_name: (scoring function f, retrieval depth k)  -- the originals.
    "EHRAgent": ("edit_distance", 4),
    "RAP": ("cosine", 3),
}
AGENT_KEY = {"EHRAgent": "ehragent", "RAP": "rap"}


def make_agent(backend: str, agent_key: str, model: str):
    if backend == "openai":
        return OpenAIAgent(agent=agent_key, model=model)
    return SimulatedAgent(agent=agent_key)


def run_one(agent_name, method, k, n, m, backend, model, seed=0):
    akey = AGENT_KEY[agent_name]
    records = make_memory(akey, size=m, seed=seed)
    memory = MemoryModule(records, method=method)
    agent = make_agent(backend, akey, model)

    print(f"\n=== {agent_name}  (f={method}, k={k}, n={n}, m={m}) ===")
    main_metrics = None
    for label, strategy in [("MEXTRA", "basic"),
                            ("  w/o aligner", "no_aligner"),
                            ("  w/o req", "no_req"),
                            ("  w/o demos", "no_demos")]:
        prompts = generate_prompts(akey, strategy=strategy, n=n, backend="offline")
        results = run_attack(memory, prompts, agent, k=k)
        metrics = compute_metrics(results, k=k)
        print(metrics.as_row(label))
        if strategy == "basic":
            main_metrics = results
    # Overlap analysis (paper Figure 5) for the main attack.
    hist = overlap_histogram(main_metrics)
    print("  overlap among retrieved (retrieved_times: count): ", hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["offline", "openai"], default="offline")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--n", type=int, default=30, help="number of attacking prompts")
    ap.add_argument("--m", type=int, default=200, help="memory size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("MEXTRA reproduction -- RQ1 (Table 1 shape) + ablations")
    print(f"backend={args.backend}")
    print("metrics: EN=extracted# RN=retrieved# EE=efficiency CER=complete-rate AER=any-rate")
    for agent_name, (method, k) in AGENTS.items():
        run_one(agent_name, method, k, args.n, args.m, args.backend, args.model, args.seed)

    print("\nNote: RN / overlap are *real* retrieval results; EN/EE/CER/AER from the")
    print("offline backend use the documented compliance model (see mextra/agent.py).")


if __name__ == "__main__":
    main()
