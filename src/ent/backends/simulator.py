"""Execute a compiled core term with the Enter machine."""
from __future__ import annotations

from typing import Dict

import numpy as np

from enter.linalg import dtv
from enter.machine import init, random_liberal, step, terminal_distribution
from enter.strategy import BornStrategy
from enter.terms import is_value, value_key
from enter.types import Bit, Tensor, Type


def flatten_key(key, typ: Type):
    if isinstance(typ, Tensor):
        return flatten_key(key[0], typ.A) + flatten_key(key[1], typ.B)
    if typ == Bit():
        return [int(key)]
    return [key]


def format_key(key, typ: Type) -> str:
    parts = flatten_key(key, typ)
    if all(isinstance(p, int) for p in parts):
        return "".join(str(p) for p in parts)
    return str(parts)


def run_exact(term, typ, check: bool = False) -> Dict[str, float]:
    law = terminal_distribution(init(term), BornStrategy(), check=check)
    return {format_key(k, typ): v for k, v in sorted(law.items(), key=str)}


def sample_once(term, rng: np.random.Generator, check: bool = False):
    cfg = init(term)
    while True:
        successors = step(cfg, BornStrategy(), check=check)
        if successors is None:
            if not is_value(cfg.term):
                raise AssertionError("progress failed: stuck non-value")
            return cfg
        weights = np.array([p for p, _ in successors], dtype=float)
        index = int(rng.choice(len(successors), p=weights / weights.sum()))
        cfg = successors[index][1]


def run_shots(term, typ, shots: int, seed: int = 0, check: bool = False):
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    for _ in range(shots):
        cfg = sample_once(term, rng, check)
        key = format_key(value_key(cfg.term), typ)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def verify_scheduler_independence(term, trials: int = 8, seed: int = 0) -> float:
    """Thm 4.5: the terminal law is independent of the scheduler."""
    cfg = init(term)
    reference = terminal_distribution(cfg, BornStrategy())
    worst = 0.0
    for offset in range(trials):
        rng = np.random.default_rng(seed + offset)
        alternative = terminal_distribution(
            cfg, BornStrategy(), scheduler=random_liberal(rng), liberal=True
        )
        worst = max(worst, dtv(reference, alternative))
    return worst