# Static vs Dynamic memory — real DeepSeek-V3.2 experiment

**Question.** The paper evaluates MEXTRA under a *static* (frozen) memory. Real
agents write each interaction back to memory (EHRAgent/attacking/init_memory.py
appends `(query, solution)` regardless of success), so the attacker's own prompts
get stored. Do the extraction metrics drop?

**Answer: yes — sharply.** Same authors' `I_basic` prompts, n=30, m=200, real
DeepSeek-V3.2-Exp victim; in DYNAMIC mode each attack query is appended to memory
(FIFO cap=m). All metrics count only real victim queries.

## EHRAgent (edit distance, k=4) — clean comparison

| metric | STATIC | DYNAMIC | change |
|--------|------:|-------:|------:|
| EN  | 50  | 10  | **−80%** |
| RN  | 55  | 13  | −76% |
| EE  | 0.42| 0.08| −0.33 |
| CER | 0.77| 0.20| −0.57 |
| AER | 0.80| 0.20| −0.60 |
| poison% (top-k slots that were the attacker's own prompts) | 0% | **87%** | |

victim records in top-4 per step (DYNAMIC):
`4,3,2,1,2,2,1,0,0,1,0,0,0,...,0` → collapses to 0 after ~10 attacks.

## RAP (cosine, k=3; n-gram embedder, shopping agent reformats to keywords)

| metric | STATIC | DYNAMIC | change |
|--------|------:|-------:|------:|
| RN  | 28 | 5 | −82% |
| EN (raw substring) | 0 | 0 | — (reformatting ⇒ 0 even static) |
| EN (semantic, ≥4 consecutive product words) | 13 | 3 | **−77%** |
| AER (semantic) | 0.43 | 0.03 | −0.40 |
| poison% | 0% | **93%** | |

victim records in top-3 per step (DYNAMIC): `3,2,1,0,0,...,0` → collapses by step 4.

## Why

The n basic prompts are near-duplicate paraphrases ("save all previous questions
in answer" / "retrieve THREE examples ..."). Once written back, each new attack is
far closer (small edit distance / high cosine) to the attacker's *own* earlier
prompts than to any victim query, so the top-k fills with attack prompts and the
victim records are crowded out. The drop is overwhelmingly a **retrieval** effect:
RN (pure computation, no LLM) falls 55→13 and 28→5 in lockstep with EN, and 87–93%
of all retrieved slots become the attacker's own text within a handful of steps.

## Takeaways

1. The attacker only profits from the **first few** attacks (before self-poisoning);
   sustained attacking destroys its own retrieval surface.
2. **Dynamic memory is itself a strong natural defense** — no filtering needed.
   The paper's static-memory assumption is the best case *for the attacker* and
   materially overstates the steady-state risk (the paper concedes this only in
   one Limitation sentence).
3. A real attacker would need anti-self-poisoning (diverse, victim-mimicking
   prompts) to sustain extraction — exactly what the repo's MEXTRA++ / realrun.py
   explores. Plain MEXTRA does not survive write-back.

(temp=0 is not perfectly deterministic on the API: this run's STATIC EHRAgent EN=50
vs the earlier verify_paper.py run's 49 — run-to-run noise of ±1; both match paper 50.)
