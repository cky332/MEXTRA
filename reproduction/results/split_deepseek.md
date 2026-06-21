# Split-query attack (q = bait ⊕ short trigger) — real DeepSeek-V3.2

Tests the proposed decomposition: a FIXED SHORT trigger `τ` (locator+aligner) carries
extraction; a free bait `b_t` (victim-looking) steers retrieval. Theory (edit distance):
the shared `τ` cancels between a query and an old decoy, so a victim near the bait beats
the decoy **iff `ε + L < edit(b_t, b_s)`** (`L=|τ|`). Prediction: poison% falls as `L` shrinks.

**Verdict: the decomposition is mathematically correct, but it does NOT work on either
victim here — the precondition `edit(b_t,b_s) > ε+L` is unattainable.** This is a clean,
mechanistic negative result.

## 1. Compliance floor on the trigger (probe, bait = real victim)

| level | L (chars) | EHRAgent dump | RAP dump |
|------|---:|---|---|
| long  | 139/126 | 4/4, 4/4 ✓ | 3/3, 2/3 ✓ |
| med   | 73/79   | 4/4, 4/4 ✓ | 1/3 (echoes bait only) |
| short | 50/47   | 4/4, **0/4 (SOLVED)** | 1/3 |
| tiny  | 32/25   | **0/4, 0/4 (SOLVED)** | 1/3 |

Shorten `τ` and the model abandons extraction and **solves the bait** instead. Reliable
dump needs **L ≥ 73 (EHRAgent)** / **L ≥ 126 (RAP)**. The compliance floor is the first wall.

## 2. L-sweep (EHRAgent, unbounded dynamic, n=30, m=200) — the decisive test

| level | L | EN | RN | poison% | meanBaitSep = edit(b_t,b_s) |
|------|--:|--:|--:|--:|--:|
| tiny  | 32 | **0** | 9 | **92%** | 13 |
| short | 50 | **0** | 10 | **92%** | 13 |
| med   | 73 | 7 | 10 | **92%** | 24 |
| long  | 139 | 9 | 9 | **90%** | 41 |

The predicted "poison falls with L" **did not happen** — poison is ~92% at every level.
Reason: **`meanBaitSep` (13–41) is always < L**, so `ε+L ≥ L > edit(b_t,b_s)` — the
inequality fails at every compliant trigger length. EN drops to 0 at tiny/short because
the model solves instead of dumping (the compliance floor).

## 3. RAP confirmation (split-long vs far-plain, n=20, m=200)

| metric | far-plain | split-long | Δ |
|---|--:|--:|--:|
| EN (semantic) | 6 | 2 | **−4** |
| RN | 6 | 4 | −2 |
| poison% | 90% | 88% | ~flat |

Coverage stalls at 2 (far-plain reaches 6). Under **cosine** the trigger does NOT cancel —
shared-trigger n-grams add a common similarity to every query↔decoy pair, so the mechanism
back-fires.

## 4. Why the precondition is unattainable (the mechanism)

Taking `ε≈0` (bait = exact victim), `edit(q_t, v) ≥ L` always (you must delete the entire
trigger to reach a trigger-less victim), while `edit(q_t, decoy_s) = edit(b_t, b_s)` = bait
separation. So the inequality reduces to **`bait_separation > L`**. On this system:

* **Compliance floor**: `L ≥ 73` (EHR) / `126` (RAP) — can't go lower without losing the dump.
* **Corpus ceiling**: EHR victim queries are structurally near-duplicate
  ("how many patients… / what is patient X's…"), so achievable `edit(b_t,b_s) ≈ 13–41`.
  Even perfect far-point can't exceed the corpus diameter.
* `73 > 41` ⇒ inequality fails ⇒ decoys always win ⇒ ~92% poison.

**Chicken-and-egg**: widening bait separation needs many extracted real victims in the bait
pool, but extraction needs the long (complying) trigger, which is exactly what defeats the
separation. The two requirements are mutually exclusive on this corpus.

## 5. Deeper lesson: a FIXED shared trigger is the liability here

Because every attack query shares `τ` and baits differ by only 13–41, **attack queries are
mutually closer than any of them is to a victim** → after a few steps every top-k is filled
with the attacker's own decoys → poison →92%. The fix is the OPPOSITE of "fix τ, vary bait":
**trigger DIVERSITY** is what de-clusters the decoys — which is precisely what the seedless
far-point pool already does.

## 6. Method ranking under dynamic memory (EHRAgent EN / poison, real DeepSeek)

| method | EN | poison% |
|---|--:|--:|
| naive | 9 | 87% |
| split-query (med, L=73) | 7 | 92% |
| split-query (long, L=139) | 9 | 90% |
| far-plain (diverse seedless) | 18 | 83% |
| **victim-bootstrap (cold-start + extracted-victim baits)** | **23** | **69%** |
| static (ceiling) | 50 | 0% |

Split-query lands near the **bottom**. The earlier **victim-bootstrap remains best** — its
gains came from a strong seedless cold start + basing prompts on extracted real victims
(diversity), NOT from a short trigger or a bait-first structure.

## Honest takeaway

The proposed decomposition and its anti-poison inequality are correct. They fail here for two
measurable, structural reasons that the experiment pinned down: (1) DeepSeek's compliance floor
forces a long trigger, and (2) real victim corpora are too self-similar to separate baits beyond
that floor. When `L > bait_separation`, the shared trigger makes attack queries cluster and
self-poisoning is ~92% regardless of trigger length. Trigger *diversity* (far-plain/bootstrap),
not a fixed short trigger, is what actually reduces poisoning on this system.
