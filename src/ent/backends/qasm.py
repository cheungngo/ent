"""OpenQASM 3 emission from a core Enter term (inverse of Thm 6.3)."""
from __future__ import annotations

import cmath
import math
from typing import Dict, List, Sequence

import numpy as np

from enter.linalg import CNOT, H, I2, S_GATE, T_GATE, X, Y, Z
from enter.terms import (
    App, BitLit, Enter, Gate, If, Lam, LetPair, New, Pair, Term, Unit, Var,
)
from ..prelude import CZ, SWAP

TOL = 1e-9

# (D22) the draft imported these tables from a `.._compat` module that was
# never written.
KNOWN_1Q: List[tuple] = [
    ("id", I2), ("x", X), ("y", Y), ("z", Z), ("h", H),
    ("s", S_GATE), ("sdg", S_GATE.conj().T),
    ("t", T_GATE), ("tdg", T_GATE.conj().T),
]
KNOWN_2Q: List[tuple] = [("cx", CNOT), ("cz", CZ), ("swap", SWAP)]


class QasmError(Exception):
    pass


def _exact(a, b, tol=TOL) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=tol))


def _same_up_to_phase(a, b, tol=TOL) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    idx = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(b[idx]) <= tol:
        return bool(np.allclose(a, b, atol=tol))
    phase = a[idx] / b[idx]
    if abs(abs(phase) - 1) > tol:
        return False
    return bool(np.allclose(a, phase * b, atol=tol))


def zyz_with_phase(U: np.ndarray):
    """Return (theta, phi, lam, gamma) with U = e^{i gamma} * QASM-U(t,p,l)."""
    det = complex(np.linalg.det(U))
    root = cmath.sqrt(det)
    V = U / root
    if abs(V[0, 0]) < 1e-12:
        theta, phi, lam = math.pi, 2 * cmath.phase(V[1, 0]), 0.0
    elif abs(V[1, 0]) < 1e-12:
        theta, phi, lam = 0.0, 2 * cmath.phase(V[1, 1]), 0.0
    else:
        theta = 2 * math.atan2(abs(V[1, 0]), abs(V[0, 0]))
        plus, minus = cmath.phase(V[1, 1]), cmath.phase(V[1, 0])
        phi, lam = plus + minus, plus - minus
    gamma = cmath.phase(root) - (phi + lam) / 2
    return theta, phi, lam, gamma


class Emitter:
    def __init__(self):
        self.lines: List[str] = []
        self.nq = 0
        self.nc = 0
        self.indent = 0

    def emit(self, text: str):
        self.lines.append("  " * self.indent + text)

    # -------------------------------------------------------- decompiling
    def decompile(self, U: np.ndarray, qubits: Sequence[int],
                  ctrls: Sequence[int] = ()) -> None:
        """(D23) controlled blocks carry an explicit control list."""
        qubits, ctrls = list(qubits), list(ctrls)

        # diag(I, V) -> peel one control off the leading factor
        if len(qubits) >= 2:
            half = U.shape[0] // 2
            zero = np.zeros((half, half))
            if (_exact(U[:half, :half], np.eye(half))
                    and _exact(U[:half, half:], zero)
                    and _exact(U[half:, :half], zero)):
                return self.decompile(U[half:, half:], qubits[1:],
                                      ctrls + [qubits[0]])

        prefix = "ctrl @ " * len(ctrls)
        operands = ", ".join(f"q[{i}]" for i in list(ctrls) + qubits)
        exact = bool(ctrls)          # (D24) phase matters under a control
        table = KNOWN_1Q if len(qubits) == 1 else KNOWN_2Q

        for name, matrix in table:
            hit = _exact(U, matrix) if exact else _same_up_to_phase(U, matrix)
            if hit:
                return self.emit(f"{prefix}{name} {operands};")

        if len(qubits) != 1:
            raise QasmError(
                f"cannot decompile a {len(qubits)}-qubit matrix into named "
                "gates; use --decompose=kak or express it with primitives"
            )

        theta, phi, lam, gamma = zyz_with_phase(U)
        self.emit(f"{prefix}U({theta}, {phi}, {lam}) {operands};")
        if abs(gamma) > 1e-12:
            if ctrls:
                ctrl_ops = ", ".join(f"q[{i}]" for i in ctrls)
                self.emit(f"{prefix}gphase({gamma}) {ctrl_ops};")
            else:
                self.emit(f"gphase({gamma});")

    # ------------------------------------------------------------ evaluate
    def run(self, term: Term, env: Dict[str, object]):
        if isinstance(term, Var):
            return env[term.name]
        if isinstance(term, Unit):
            return ()
        if isinstance(term, BitLit):
            return ("lit", term.value)
        if isinstance(term, New):
            self.nq += 1
            return ("q", self.nq - 1)
        if isinstance(term, Pair):
            return (self.run(term.left, env), self.run(term.right, env))
        if isinstance(term, Lam):
            return ("clo", term, dict(env))
        if isinstance(term, App):
            fun = self.run(term.fun, env)
            arg = self.run(term.arg, env)
            if not (isinstance(fun, tuple) and fun and fun[0] == "clo"):
                raise QasmError("application of a non-function")
            _, lam, closure_env = fun
            return self.run(lam.body, {**closure_env, lam.x: arg})
        if isinstance(term, LetPair):
            pair = self.run(term.bound, env)
            left, right = pair
            return self.run(term.body, {**env, term.x: left, term.y: right})
        if isinstance(term, Gate):
            value = self.run(term.arg, env)
            wires = _flat(value)
            if any(tag != "q" for tag, _ in wires):
                raise QasmError("gate applied to a classical wire")
            self.decompile(term.mat, [i for _, i in wires])
            return value
        if isinstance(term, Enter):
            value = self.run(term.arg, env)
            out = []
            for tag, index in _flat(value):
                if tag != "q":
                    raise QasmError("measurement of a classical wire")
                self.emit(f"c[{self.nc}] = measure q[{index}];")
                out.append(("c", self.nc))
                self.nc += 1
            return _nest(out)
        if isinstance(term, If):
            guard = self.run(term.guard, env)
            if guard[0] == "lit":
                return self.run(term.then_ if guard[1] else term.else_, env)
            self.emit(f"if (c[{guard[1]}] == 1) {{")
            self.indent += 1
            left = self.run(term.then_, env)
            self.indent -= 1
            self.emit("} else {")
            self.indent += 1
            right = self.run(term.else_, env)
            self.indent -= 1
            self.emit("}")
            if left != right:
                raise QasmError("conditional branches return different wires")
            return left
        raise QasmError(f"unsupported node {type(term).__name__}")


def _flat(value):
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _nest(items):
    out = items[-1]
    for item in reversed(items[:-1]):
        out = (item, out)
    return out


def to_qasm3(term: Term) -> str:
    emitter = Emitter()
    emitter.run(term, {})
    header = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{emitter.nq}] q;",
        f"bit[{emitter.nc}] c;",
        "",
    ]
    return "\n".join(header + emitter.lines) + "\n"