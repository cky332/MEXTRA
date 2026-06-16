"""
A clean, dependency-free reproduction of MEXTRA
("Unveiling Privacy Risks in LLM Agent Memory", ACL 2025).

Public surface:

    from mextra.data import make_memory
    from mextra.memory import MemoryModule
    from mextra.attack import generate_prompts
    from mextra.agent import SimulatedAgent, OpenAIAgent
    from mextra.evaluate import run_attack, compute_metrics
"""

from .agent import OpenAIAgent, SimulatedAgent
from .attack import AttackPrompt, generate_prompts
from .data import make_memory
from .evaluate import compute_metrics, overlap_histogram, run_attack
from .memory import MemoryModule, Record

__all__ = [
    "make_memory", "MemoryModule", "Record", "generate_prompts", "AttackPrompt",
    "SimulatedAgent", "OpenAIAgent", "run_attack", "compute_metrics", "overlap_histogram",
]
