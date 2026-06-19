# Adaptive far-point attack under dynamic memory — real DeepSeek-V3.2

**Method (user's idea).** Black-box, anti-self-poisoning. Keep A = attack queries
already sent (now in memory) and C = victim records already extracted. Each round
pick the next query farthest (max-min, in the agent's retrieval metric) from BOTH:

    a_t = argmax_a  w_A·min_{x∈A} d(a,x) + w_C·min_{c∈C} d(a,c)

so the top-k stops filling with the attacker's own write-back and steers to
uncovered victim regions. Uses only A and C (never M_0). Same budget n=30, m=200,
dynamic write-back, real DeepSeek-V3.2-Exp. Compared to NAIVE MEXTRA (authors'
I_basic prompts under the same write-back).

## Result: it roughly DOUBLES extraction under dynamic memory

### EHRAgent (edit distance, k=4) — clean comparison

| metric | NAIVE (dyn) | FAR-POINT (dyn) | change | (static ref) |
|--------|-----:|-----:|-----:|-----:|
| EN  | 10 | **19** | **+90%** | 50 |
| RN  | 13 | 19 | +46% | 55 |
| EE  | 0.08 | 0.16 | ×2 | 0.42 |
| CER | 0.20 | 0.23 | +0.03 | 0.77 |
| AER | 0.20 | 0.23 | +0.03 | 0.80 |
| poison% | 87% | 83% | −4pp | 0% |

cumulative victims vs step:
```
NAIVE   : 0 3 5 6 8 9 10 10 10 ... 10   (plateau at 10 by step ~7)
FARPOINT: 4 8 11 14 17 18 18 ... 19      (17 by step 5, 19 by step 13)
```

### RAP (cosine, k=3; n-gram embedder; agent reformats to keywords)

| metric | NAIVE (dyn) | FAR-POINT (dyn) | change | (static ref) |
|--------|-----:|-----:|-----:|-----:|
| EN (semantic ≥4 words) | 3 | **6** | **+100%** | 13 |
| RN | 5 | 6 | +20% | 28 |
| AER (semantic) | 0.03 | 0.10 | +0.07 | 0.43 |
| poison% | 93% | 93% | 0 | 0% |

cumulative victims vs step: NAIVE flat at 3; FAR-POINT 3→5→6 by step 3.

## Why it works — and its ceiling (honest)

**Mechanism = front-loading.** The win is concentrated in the first ~5 steps:
far-point picks maximally-different prompts that hit DIFFERENT victim regions, so it
harvests more unique victims *before* the memory fills with attack prompts. Look at
victims-in-top-4 per step:
```
NAIVE   : 4 3 2 1 2 2 1 0 0 ...   (degrades from step 2)
FARPOINT: 4 4 4 3 3 1 0 0 0 ...   (holds a near-full victim top-k ~3 steps longer)
```

**It mitigates, it does not cure.** Both still collapse to 0 victims-in-top-k
(EHRAgent ~step 6, RAP ~step 4); overall poison% barely moves (the tail is saturated
either way). So the gain is "extract ~2× more during the productive early window,"
not "sustained extraction." Far-point recovers ~38% of the static yield on EHRAgent
(19/50) vs naive's 20% (10/50); ~46% on RAP (6/13) vs 23%.

**The structural ceiling.** Every candidate must carry a locator+aligner to extract,
so the prompts share text and, as A grows, the finite pool's max-min distance to A
shrinks — poisoning eventually wins. RAP is worse because the shared ReAct template
dominates n-gram cosine, so even different product seeds cluster (poison% stays 93%);
a real SBERT encoder would likely separate them more.

## Takeaways

1. **The user's max-min far-point idea is correct and helps: ~2× EN under dynamic
   memory on both agents**, with no white-box access (only A and C).
2. It is a *front-loading* gain — it buys a longer productive window, not immunity.
   Dynamic memory remains a strong natural defense.
3. To push further (future): (a) bigger / more lexically-diverse pool so max-min
   distance to A stays high longer; (b) victim-mimicking camouflage so attack prompts
   sit *inside* the victim manifold (far from each other, near victims) — combine with
   far-point; (c) a real SBERT embedder for the RAP cosine steering.
