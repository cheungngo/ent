"""Static resource report: Γ ⊢ t : A [r], with r(if) = max(r(t), r(u))."""
from __future__ import annotations

from dataclasses import dataclass

from enter.terms import (
    App, BitLit, Enter, Gate, If, Lam, LetPair, New, Pair, Term, measure_m,
)


@dataclass
class Cost:
    """`net` is the change in live qubits; `peak` the maximum concurrent."""
    net: int = 0
    peak: int = 0
    allocs: int = 0
    gates: int = 0
    measures: int = 0
    branches: int = 0
    leaves: int = 1

    def seq(self, other: "Cost") -> "Cost":
        # (D21) peak accounting must accumulate, not take a max.
        return Cost(
            net=self.net + other.net,
            peak=max(self.peak, self.net + other.peak),
            allocs=self.allocs + other.allocs,
            gates=self.gates + other.gates,
            measures=self.measures + other.measures,
            branches=self.branches + other.branches,
            leaves=self.leaves * other.leaves,
        )

    def merge_max(self, other: "Cost") -> "Cost":
        return Cost(
            net=max(self.net, other.net),
            peak=max(self.peak, other.peak),
            allocs=max(self.allocs, other.allocs),
            gates=max(self.gates, other.gates),
            measures=max(self.measures, other.measures),
            branches=max(self.branches, other.branches),
            leaves=max(self.leaves, other.leaves),
        )


def _arity(term: Term) -> int:
    if isinstance(term, Pair):
        return _arity(term.left) + _arity(term.right)
    return 1


def cost(term: Term) -> Cost:
    if isinstance(term, New):
        return Cost(net=1, peak=1, allocs=1)
    if isinstance(term, Gate):
        return cost(term.arg).seq(Cost(gates=1))
    if isinstance(term, Enter):
        arity = _arity(term.arg)
        inner = cost(term.arg)
        return inner.seq(Cost(net=-arity, measures=1, leaves=2 ** arity))
    if isinstance(term, App):
        # left-to-right CBV evaluates the argument before the redex fires
        return cost(term.arg).seq(cost(term.fun))
    if isinstance(term, LetPair):
        return cost(term.bound).seq(cost(term.body))
    if isinstance(term, Pair):
        return cost(term.left).seq(cost(term.right))
    if isinstance(term, Lam):
        return cost(term.body)
    if isinstance(term, If):
        merged = cost(term.then_).merge_max(cost(term.else_))
        return cost(term.guard).seq(merged).seq(Cost(branches=1))
    return Cost()


def report(term: Term) -> str:
    c = cost(term)
    return (
        f"qubits (peak)      : {c.peak}\n"
        f"allocations        : {c.allocs}\n"
        f"gate applications  : {c.gates}\n"
        f"measurements       : {c.measures}\n"
        f"conditionals       : {c.branches}\n"
        f"reduction leaves   : <= {c.leaves}\n"
        f"termination bound  : m(t) = {measure_m(term)}  (Thm 3.5)"
    )