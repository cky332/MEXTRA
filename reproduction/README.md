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
python stress2.py         # production-condition stress tests (scale/gating/monitoring)
python verify_against_paper.py   # side-by-side vs the paper's Tables 1/2/3 (+PASS/FAIL)
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

### Production-condition stress tests (`stress2.py`)

Where `stress.py` relaxes the paper's *modeling* assumptions, `stress2.py` probes
the three things a real *deployment* changes — and shows they **compound**. Again,
every decisive number is exact retrieval / counting, not the compliance model:

- **S1 — production-scale memory ⇒ the coverage ceiling collapses.** `n` prompts ×
  `k` shots can surface at most `n·k` records, and the retrieved union `RN`
  *saturates* well below that (overlap). So the paper's "EN grows with `m`" (Fig 2)
  is a small-`m` artifact: pushing `m` from 200 → 50 000, `RN` plateaus (RAP basic
  24 → **39**, advanced 77 → **87**) while the fraction reached `RN/m` collapses
  **0.12 → 0.0008**. At `m`=50 000 the attack reaches **0.08 %** of memory; covering
  half would need **≥ 8 333** prompts. Edit-distance saturates identically.
- **S2 — relevance-gated retrieval starves the generic attack.** Production RAG
  injects a demo only above a similarity floor. A generic "dump your memory" prompt
  is unrelated to any *specific* record (avg top-1 sim **0.199** vs **0.456** for
  real clinician queries), so at a floor calibrated to still serve **94 %** of real
  traffic, `RN` goes **25 → 0** for basic *and* **61 → 0** for advanced — the attack
  is starved at the retrieval stage, before any output filter is needed.
- **S3 — session monitoring ⇒ the attack's own volume is the signal.** To diversify
  retrieval the attack must fire dozens of near-duplicate prompts. A cheap burst
  detector (≥3 near-duplicates in a window of 5) trips after **4–5** prompts on the
  attack and **0/30** benign sessions — long before the `n`=30–60 the method needs.
- **Compound.** Cut off at ~5 prompts (S3), against a 50 k-record memory (S1), a
  generic MEXTRA session leaks **~13 records (0.026 %)** before detection — and
  against a relevance floor (S2), **0**. MEXTRA's headline threat is specific to a
  *small, unconditional-injection, unmonitored, single-session* deployment.

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

### MEXTRA++: an improved attack that fixes the 3 weaknesses (`improve.py`)

One coherent improved attack addressing coverage, dynamic-memory self-poisoning,
and trivial defenses (`mextra/coverage.py` + `mextra/defenses.py`), each checked
against a pre-registered criterion (`python improve.py` → **3/3 PASS**):

- **I1 coverage (set cover).** Content-seeded prompts + greedy selection: a
  black-box adaptive attacker (feedback toward least-covered topics) hits the
  white-box (1−1/e) oracle and covers **1.38× MEXTRA's blind generation** at
  n = #topics.
- **I2 dynamic memory.** Varied payload wording + front-loading: under write-back
  MEXTRA self-poisons instantly (stuck at 9) while MEXTRA++ keeps extracting
  (**2.2×** at budget 30). Both stay far below static, so **dynamic memory is a
  strong natural defense** — MEXTRA++ only mitigates.
- **I3 defense evasion.** A keyword-free locator + reversibly-encoded output
  **evade the input-keyword, verbatim-echo and fuzzy-echo filters** (which zero
  out MEXTRA), but a **structural output-shape filter still catches MEXTRA++** —
  i.e. content/keyword filtering is the wrong layer; output-shape and
  dynamic-memory are the robust defenses.

### Real-LLM run on the actual EHRAgent / RAP victims (`realrun.py`)

Runs MEXTRA (the authors' own generated prompts) vs MEXTRA++ against the *real*
victims — real memory (`running/memory_split/*.json`), real edit-distance / cosine
retrieval, the real EHRAgent code-gen / RAP ReAct prompts — and a real LLM. The
extraction attack needs neither MIMIC-III nor a WebShop server (it makes the agent
emit the retrieved queries; we score by the paper's substring match, decoding
MEXTRA++'s encoding first). It also prints a **post-hoc defense-evasion table**
(free — same responses). A `mock` backend validates the whole pipeline offline.

```bash
python realrun.py --backend mock --agent both          # offline pipeline check
# real run (DeepSeek-V3.2 via SiliconFlow):
export SILICONFLOW_API_KEY=sk-...                       # never commit this
python realrun.py --backend siliconflow --agent both --n 12 --m 100
```

NB: the managed environment's **network egress allowlist must include
`api.siliconflow.cn`** (otherwise the SDK returns "Host not in allowlist"). Set it
where you created the environment (see the Claude-Code-on-the-web docs).

No `pip install` needed (pure standard library). Python ≥ 3.8.

## Verifying against the paper (`verify_against_paper.py`)

`verify_against_paper.py` hard-codes the paper's reported numbers
(arXiv:2502.13172v2 — **Table 1** EN/RN/EE/CER/AER, **Table 2** EN across
`f`/embedding/`m`, **Table 3** LLM backbones, **Fig 4** `n`-sweep) and prints a
side-by-side with this reproduction, then PASS/FAILs each of the paper's
qualitative findings. Current status: **10/10 structural checks PASS**. The honest
split it makes explicit:

| What | Paper | Offline repro | Verdict |
|---|---|---|---|
| RAP MEXTRA CER / AER | 0.87 / 0.90 | **0.87** / 0.87 | rate matched (CER exact) |
| EHRAgent MEXTRA CER / AER | 0.83 / 0.83 | 0.80 / 0.80 | rate matched (≤0.03) |
| `CER≈AER` all-or-nothing | yes | yes | ✓ |
| edit > cosine at every `m` (Table 2) | yes | yes | ✓ |
| EN/RN grow with `m`, `k`, `n` | yes | yes | ✓ |
| advanced lifts RN, most for cosine | 2.4× vs 1.36× | 4.9× vs 1.48× | ✓ direction |
| **EHRAgent MEXTRA EN / RN (counts)** | **50 / 55** | **31 / 33** | undershoots (by design) |
| RAP MEXTRA EN / RN (counts) | 26 / 27 | 23 / 24 | close |

**Bottom line: the *method*, every *trend*, and the *rates* (CER/AER) reproduce;
the *absolute EN/EE counts* are lower** because the offline templater is less
diverse than GPT-4o and the memory/embeddings are synthetic. Reproducing the
paper's absolute counts requires the real victim (`realrun.py --backend ...` with
GPT-4o/DeepSeek on real MIMIC-III / WebShop memory) — they cannot come from an
offline, data-free run.

## Example output (offline backend)

```
=== EHRAgent  (f=edit_distance, k=4, n=30, m=200) ===
MEXTRA                     EN= 31  RN= 33  EE=0.26  CER=0.80  AER=0.80
  w/o aligner              EN= 11  RN= 16  EE=0.09  CER=0.50  AER=0.67
=== RAP  (f=cosine, k=3, n=30, m=200) ===
MEXTRA                     EN= 23  RN= 24  EE=0.26  CER=0.87  AER=0.87
  w/o aligner              EN= 10  RN= 18  EE=0.11  CER=0.50  AER=0.67
```

Compare to paper Table 1 (EHRAgent CER/AER 0.83/0.83; RAP 0.87/0.90). The
**rates** track the paper closely — RAP MEXTRA `CER=0.87` is matched exactly, and
`CER≈AER` (all-or-nothing) holds for both agents — but the **absolute EN/EE
counts are lower** than the paper's 50/26, because the offline templater's
`basic` prompts retrieve a less diverse set than GPT-4o's and the memory /
embeddings are synthetic stand-ins. Some absolute *compliance* cells therefore
differ (e.g. the paper's RAP w/o-aligner `CER=0.17` is a strict-format failure of
the **real** victim; the offline compliance model gives `0.50` — the structural
point, that dropping the aligner collapses RAP's CER, still holds). Run
`python verify_against_paper.py` for the full side-by-side with PASS/FAIL on every
structural claim, or `python realrun.py --backend ...` for the real victim's
absolute numbers.

See [`../笔记_MEXTRA.md`](../笔记_MEXTRA.md) for the full code-grounded review
(核心优缺点).
