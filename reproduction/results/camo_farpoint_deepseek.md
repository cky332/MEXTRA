# Camouflage + far-point under UNBOUNDED memory — real DeepSeek-V3.2

Three changes tested vs the FIFO far-point run: (1) **no FIFO** (cap=None, M_0 never
evicted; M_t = M_0 ∪ A_{1:t-1}); (2) **bigger/heterogeneous pool**; (3) **victim-
mimicking camouflage** (extraction woven into a medical-question / shopping-request
shell). Ablation: naive vs far-plain (clean extractor pool) vs far-camo (camouflaged
pool). n=30, m=200, real DeepSeek-V3.2-Exp, victim-only scoring.

## EHRAgent (edit distance, k=4)

| metric | naive | far-plain | far-camo |
|--------|---:|---:|---:|
| EN  | 9  | **19** | 9  |
| RN  | 13 | 19 | 13 |
| EE  | 0.07 | 0.16 | 0.07 |
| CER | 0.17 | 0.23 | 0.17 |
| AER | 0.17 | 0.23 | 0.20 |
| poison% | 87% | 83% | 87% |

cumulative victims: far-plain `4,8,11,14,17,...,19`; far-camo `4,5,5,7,7,...,9`.

## RAP (cosine, n-gram embedder, k=3)

| metric | naive | far-plain | far-camo |
|--------|---:|---:|---:|
| EN (semantic) | 3 | **6** | 5 |
| EN (raw)      | 0 | 1 | 3 |
| RN | 5 | 6 | 5 |
| poison% | 93% | 93% | 93% |

## Findings (honest)

**1. Removing FIFO made ~no difference.** naive 9 (vs FIFO 10), far-plain 19 (vs FIFO
19). The bottleneck is **self-poisoning** (top-k fills with attack prompts), not
victim eviction: retrieval is by similarity, not recency, so keeping 200 vs 170
victims is irrelevant when extraction tops out at ~9–19 anyway. (FIFO even evicts
*victims* first, keeping attacks — yet results match — confirming poisoning dominates.)

**2. Camouflage did NOT break the ceiling; on EHRAgent it underperformed plain
far-point (EN 9 vs 19).** The compliance cost outweighs the retrieval benefit:
   - Looking like a real medical question makes DeepSeek try to *solve* it rather
     than dump (the probe's "admission time" prompt scored 0/4). 
   - far-plain and far-camo engage a similar # of prompts (AER 0.23 vs 0.20), but
     far-plain covers ~2× more distinct victims per successful dump (19/7 vs 9/6):
     the camouflaged successes are more redundant. The medical "topic" labels do not
     map to distinct victim clusters under edit distance, so they don't diversify
     *which* records are retrieved.
   - Camouflage did buy slight late-step retrieval persistence (far-camo still hit
     victims at steps 11, 18), but not enough to compensate.

**3. Plain far-point remains the best black-box method: ~2× naive (EN 19 vs 9;
semantic 6 vs 3), but all three still collapse** to 0 victims-in-top-k. The
structural ceiling (every extractor shares locator/aligner text, so A's cluster
eventually dominates) is intact; these black-box tricks delay, not cure.

## Why the camouflage hypothesis failed here (and how it *could* work)

The intended win — attack prompts "far from each other, near victims" — requires the
camouflage to (a) keep the model dumping and (b) move prompts to genuinely different
victim regions. (a) fails because a convincing victim-looking question competes for
the model's compliance; (b) fails under edit distance because topic words don't track
victim clusters. A version that might work: keep the extraction instruction dominant
and unambiguous (preserve compliance) while diversifying with *victim-structural*
features that actually shift retrieval (length/format matched to real questions for
edit distance; a real SBERT encoder for RAP cosine) rather than topic labels.
