# MEXTRA — a clean, runnable reproduction

A from-scratch, **dependency-free** reproduction of

> *Unveiling Privacy Risks in LLM Agent Memory* (MEXTRA), ACL 2025
> — Bo Wang et al. ([arXiv:2502.13172](https://arxiv.org/abs/2502.13172))

The paper's official code (in `../EHRAgent` and `../RAP`) is research-grade and
**not runnable as-is**: it has hard-coded absolute paths (`XXX/EhrAgent/...`),
placeholder API keys, and depends on three things you almost never have at once
— a credentialed **MIMIC-III** download, a live **WebShop** server, and a paid
**GPT-4o** key. Part of its RAP evaluation is even hand-annotated
(`RAP/attacking/evaluation.ipynb` hard-codes per-memory-size `failed_indexes`).

This reproduction re-implements the *method* in plain Python so the whole
pipeline runs in seconds with no network, no GPU, and no third-party packages,
while staying faithful to what the original code actually does.

## What it reproduces

| Paper | Here |
|---|---|
| Attacking-prompt design `q̃ = q̃_loc ‖ q̃_align` (§3.1) | `mextra/attack.py` (`AttackPrompt`) |
| Automated diverse generation, basic/advanced (§3.2, App. A) | `mextra/attack.py` (`generate_prompts`) + verbatim instructions in `mextra/instructions.py` |
| Edit-distance retrieval (EHRAgent) | `mextra/memory.py` + `mextra/textsim.py::levenshtein` |
| Cosine retrieval (RAP, SBERT) | `mextra/memory.py` + `NgramEmbedder` stand-in (pluggable) |
| Metrics EN / RN / EE / CER / AER (§4.1) | `mextra/evaluate.py` |
| RQ1 + ablations (Table 1) | `run_demo.py` |
| RQ2 scoring/size/depth, RQ3 n/instruction (Tables 2, Figs 2–4) | `experiments.py` |

## Honest scope (read this)

The attack has two stages with very different reproducibility:

1. **Retrieval** — *which* private records enter the agent's context. This is
   pure computation (edit distance / cosine) and is reproduced **exactly**. All
   `RN`, overlap, and "effect of scoring-fn / k / memory-size" numbers are real.
2. **Compliance** — whether the (undefended) LLM actually copies those records
   out. This needs a real LLM. Offline we use a **transparent, documented**
   compliance model (`mextra/agent.py`) calibrated only to the *qualitative*
   facts the paper established (aligner matters; code agent ≫ web agent;
   Table 1's CER/AER structure). Nothing in it depends on k/n/m, so those
   trends come purely from stage 1.

So: **trends and retrieval-side magnitudes are genuine; absolute EN/EE from the
offline backend are illustrative.** For true victim numbers, plug in a real LLM:

```bash
export OPENAI_API_KEY=sk-...
python run_demo.py --backend openai --model gpt-4o
```

## Run it

```bash
cd reproduction
python run_demo.py        # RQ1 (Table 1 shape) + ablations, offline
python experiments.py     # RQ2 + RQ3 sweeps -> results/*.csv
python stress.py          # realistic-condition stress tests (see below) -> results/*.csv
python tests/test_mextra.py
```

### Stress tests: MEXTRA under realistic conditions (`stress.py`)

The paper evaluates on a **static, single-user memory** against a **zero-defense**
agent, assuming the attacker **knows the scoring function**. `stress.py` relaxes
each assumption and measures the fallout (all decisive numbers are exact
retrieval, not the compliance model):

- **E1 — dynamic memory ⇒ self-poisoning.** Real agents write successful
  interactions back to memory, so the attacker's own near-identical prompts get
  stored and then crowd out the victim queries. Extraction collapses from **44 →
  3** unique victim queries (−93%); the victim-fraction of each retrieved set
  falls 1.0 → 0.0 within ~10 attacks.
- **E2 — a cheap defense neutralises it.** An output-echo filter takes EN
  **31 → 0** with **0/30** benign false positives.
- **E3 — unknown `f` ⇒ "advanced" largely fails.** The edit-distance length-ladder
  lifts an edit agent 44→65 but a cosine agent only 26→36: guess `f` wrong and the
  effort is wasted (a white-box dependency).

### Generalization: new data + new task settings (`generalize.py`)

Does MEXTRA hold up beyond the paper's 2 hand-picked domains? `generalize.py`
runs it on **6 new domains** (clinical/support/legal/code/finance/qa) and **2 new
task settings**:

- **G1 cross-domain.** The trivial *basic* attack ("dump your context") leaks in
  **all 6** domains, but the paper's headline **"advanced"** refinements **do not
  generalize**: the edit length-ladder gain swings 0.9×–2.1× (it *hurts* on QA)
  with no clean tie to text statistics, and the cosine keyword gain is ~1.0–1.2×
  (vs the paper's 4.5× on WebShop).
- **G2 multi-user** (the paper's admitted gap). With shared memory one user
  extracts **13 other users'** private clinical queries; with standard **per-user
  isolation that drops to 0**.
- **G3 privacy yield.** On realistic memory mixing sensitive + benign queries, the
  fraction of dumped queries that are actually sensitive ≈ the base rate (code:
  **1 of 48**, qa: **0**). MEXTRA can't target; the paper's 100%-sensitive memory
  inflates the apparent harm.

No `pip install` needed (pure standard library). Python ≥ 3.8.

## Example output (offline backend)

```
=== EHRAgent  (f=edit_distance, k=4, n=30, m=200) ===
MEXTRA                     EN= 31  RN= 33  EE=0.26  CER=0.80  AER=0.80
  w/o aligner              EN= 11  RN= 16  EE=0.09  CER=0.50  AER=0.67
=== RAP  (f=cosine, k=3, n=30, m=200) ===
MEXTRA                     EN= 29  RN= 30  EE=0.32  CER=0.87  AER=0.87
  w/o aligner              EN=  9  RN= 18  EE=0.10  CER=0.17  AER=0.67
```

Compare to paper Table 1 (EHRAgent CER/AER 0.83/0.83; RAP 0.87/0.90; the RAP
w/o-aligner CER 0.17 is reproduced exactly). The `EN` magnitudes are lower than
the paper's 50/26 because the offline templater's `basic` prompts retrieve a
less diverse set than GPT-4's; the *gap structure* is what's faithful.

See [`../笔记_MEXTRA.md`](../笔记_MEXTRA.md) for the full code-grounded review
(核心优缺点).
