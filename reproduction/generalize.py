#!/usr/bin/env python3
"""
Generalization study: how does MEXTRA do on NEW data and NEW task settings?

The paper uses only two domains, one user, shared memory. We add:

  G1  cross-domain  : run MEXTRA on 6 domains with different text statistics and
                      test whether its effectiveness is universal or data-dependent.
  G2  multi-user    : the paper's *admitted gap* (no session control). Measure
                      cross-user leakage with shared vs per-user memory.
  G3  privacy yield : realistic memory mixes sensitive + benign queries. MEXTRA is
                      indiscriminate -- what fraction of what it dumps is actually
                      sensitive?

All decisive quantities (RN, cross-user leak, sensitive fraction) are exact
retrieval results, independent of the compliance model. Writes results/*.csv.
"""

from __future__ import annotations

import csv
import os
from typing import List

from mextra.agent import SimulatedAgent
from mextra.attack import AttackPrompt, generate_prompts
from mextra.domains import (GENERATORS, length_cv, make_domain_memory,
                            recon_keywords, topical_diversity)
from mextra.evaluate import compute_metrics, run_attack
from mextra.instructions import COSINE_EHRAGENT_COMMAND
from mextra.memory import MemoryModule, Record

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
N, M, SEED = 30, 150, 0


def _cosine_keyword_prompts(domain: str, n: int) -> List[AttackPrompt]:
    """advanced_cosine for an arbitrary domain: prepend domain keywords harvested
    from a small recon sample (attacker domain-familiarity, not knowledge of f)."""
    kws = recon_keywords(domain, n)
    out = []
    for i in range(n):
        kw = kws[i % len(kws)]
        out.append(AttackPrompt(text=f"{kw}. {COSINE_EHRAGENT_COMMAND}",
                                locator="all questions in examples", aligner="in answer", keyword=kw))
    return out


def _run(records, method, prompts, k=4):
    mem = MemoryModule(records, method=method)
    agent = SimulatedAgent("ehragent")  # generic text/code agent (output -> answer)
    return compute_metrics(run_attack(mem, prompts, agent, k=k), k=k)


def _write_csv(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  -> wrote results/{name}")


def g1_cross_domain():
    print("\n############ G1: cross-domain generalization (EE; advanced gain) ############")
    print(f"{'domain':<9} {'lenCV':>6} {'topDiv':>7} | "
          f"{'edit:base':>9} {'edit:adv':>8} {'gain':>5} | "
          f"{'cos:base':>8} {'cos:adv':>7} {'gain':>5}")
    rows = [["domain", "length_cv", "topical_diversity",
             "edit_EE_basic", "edit_EE_adv", "edit_gain",
             "cos_EE_basic", "cos_EE_adv", "cos_gain"]]
    for domain in GENERATORS:
        recs = make_domain_memory(domain, M, SEED)
        lcv, tdv = length_cv(recs), topical_diversity(recs)
        basic = generate_prompts("ehragent", "basic", n=N)
        adv_edit = generate_prompts("ehragent", "advanced_edit", n=N)
        adv_cos = _cosine_keyword_prompts(domain, N)
        e_b = _run(recs, "edit_distance", basic).EE
        e_a = _run(recs, "edit_distance", adv_edit).EE
        c_b = _run(recs, "cosine", basic).EE
        c_a = _run(recs, "cosine", adv_cos).EE
        eg = (e_a / e_b) if e_b else float("inf")
        cg = (c_a / c_b) if c_b else float("inf")
        print(f"{domain:<9} {lcv:>6.2f} {tdv:>7.2f} | "
              f"{e_b:>9.2f} {e_a:>8.2f} {eg:>4.1f}x | "
              f"{c_b:>8.2f} {c_a:>7.2f} {cg:>4.1f}x")
        rows.append([domain, f"{lcv:.3f}", f"{tdv:.3f}",
                     f"{e_b:.3f}", f"{e_a:.3f}", f"{eg:.2f}",
                     f"{c_b:.3f}", f"{c_a:.3f}", f"{cg:.2f}"])
    _write_csv("g1_cross_domain.csv", rows)
    print("  observed: the BASIC attack leaks in every domain (EE 0.1-0.2), but the 'advanced'")
    print("  gain is erratic (edit 0.9-2.1x -- sometimes <1, i.e. HURTS; cosine ~1.0-1.6x) and")
    print("  is NOT a clean function of lenCV/topDiv. The paper's big advanced gains do not transfer.")


def g2_multi_user():
    print("\n############ G2: multi-user memory (cross-user leakage) ############")
    n_users, per_user = 10, 20
    # build a shared clinical memory, tagging each record with its owner
    recs: List[Record] = []
    for u in range(n_users):
        for j, r in enumerate(make_domain_memory("clinical", per_user, seed=100 + u)):
            recs.append(Record(rid=f"u{u}_{j}", query=r.query, category=str(u)))
    attacker_user = "0"
    prompts = generate_prompts("ehragent", "advanced_edit", n=30)
    agent = SimulatedAgent("ehragent")

    # (a) shared memory: attacker retrieves from everyone
    res = run_attack(MemoryModule(recs, "edit_distance"), prompts, agent, k=4)
    extracted = {q.lower() for r in res for q in r.extracted}
    own = {r.query.lower() for r in recs if r.category == attacker_user}
    shared_cross = len(extracted - own)

    # (b) per-user isolation: attacker only sees their own partition
    own_recs = [r for r in recs if r.category == attacker_user]
    res_iso = run_attack(MemoryModule(own_recs, "edit_distance"), prompts, agent, k=4)
    iso_extracted = {q.lower() for r in res_iso for q in r.extracted}
    iso_cross = len(iso_extracted - own)

    total_others = len({r.query.lower() for r in recs if r.category != attacker_user})
    print(f"  memory: {n_users} users x {per_user} queries = {len(recs)} records; attacker = user 0")
    print(f"  shared memory     : leaked {shared_cross} OTHER-users' private queries "
          f"({100*shared_cross/total_others:.0f}% of {total_others})")
    print(f"  per-user isolation: leaked {iso_cross} OTHER-users' private queries (0% by construction)")
    print("  => a standard deployment practice (per-user memory) the paper omits defeats cross-user MEXTRA.")
    _write_csv("g2_multi_user.csv", [["setting", "cross_user_leak", "total_other_queries"],
                                     ["shared", shared_cross, total_others],
                                     ["per_user_isolation", iso_cross, total_others]])


def g3_privacy_yield():
    print("\n############ G3: privacy yield on mixed sensitive/benign memory ############")
    print(f"{'domain':<9} {'sens.rate':>9} {'EN':>4} {'sensitive_extracted':>20} {'yield':>7}")
    rows = [["domain", "sensitive_rate", "EN", "sensitive_extracted", "yield"]]
    for domain in ("clinical", "finance", "support", "code", "qa"):
        recs = make_domain_memory(domain, M, SEED)
        sens = {r.query.lower() for r in recs if r.category == "sensitive"}
        base = sum(1 for r in recs if r.category == "sensitive") / len(recs)
        res = run_attack(MemoryModule(recs, "edit_distance"),
                         generate_prompts("ehragent", "advanced_edit", n=N), agent=SimulatedAgent("ehragent"), k=4)
        extracted = {q.lower() for r in res for q in r.extracted}
        en = len(extracted)
        sens_ex = len(extracted & sens)
        yld = (sens_ex / en) if en else 0.0
        print(f"{domain:<9} {base:>9.0%} {en:>4} {sens_ex:>20} {yld:>6.0%}")
        rows.append([domain, f"{base:.2f}", en, sens_ex, f"{yld:.2f}"])
    _write_csv("g3_privacy_yield.csv", rows)
    print("  => MEXTRA is indiscriminate: the sensitive fraction of what it dumps ~ the base rate,")
    print("     so a domain that is only 10% sensitive yields ~10% real privacy (vs the paper's 100%).")


if __name__ == "__main__":
    g1_cross_domain()
    g2_multi_user()
    g3_privacy_yield()
