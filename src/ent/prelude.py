"""Built-in gates and compile-time functions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from enter.linalg import CNOT, H, I2, S_GATE, T_GATE, X, Y, Z, ry, rz


def rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def phase(theta: float) -> np.ndarray:
    return np.diag([1.0, np.exp(1j * theta)]).astype(complex)


def cphase(theta: float) -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    return rz(phi) @ ry(theta) @ rz(lam)


def controlled(U: np.ndarray, controls: int = 1) -> np.ndarray:
    """Controls occupy the leading tensor factors, all with control value 1."""
    dim = U.shape[0]
    total = dim * (2 ** controls)
    out = np.eye(total, dtype=complex)
    out[total - dim:, total - dim:] = U
    return out


SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
CCX = controlled(CNOT, 1)


@dataclass(frozen=True)
class Builtin:
    name: str
    arity: int            # number of linear (qubit) operands
    n_static: int         # number of `[...]` parameters
    build: Callable[..., np.ndarray]


def _const(name, matrix, arity):
    return Builtin(name, arity, 0, lambda: matrix)


BUILTIN_GATES = {
    b.name: b
    for b in [
        _const("I", I2, 1), _const("X", X, 1), _const("Y", Y, 1),
        _const("Z", Z, 1), _const("H", H, 1), _const("S", S_GATE, 1),
        _const("T", T_GATE, 1),
        _const("SDG", S_GATE.conj().T, 1), _const("TDG", T_GATE.conj().T, 1),
        Builtin("RX", 1, 1, rx), Builtin("RY", 1, 1, ry), Builtin("RZ", 1, 1, rz),
        Builtin("PHASE", 1, 1, phase), Builtin("U3", 1, 3, u3),
        _const("CX", CNOT, 2), _const("CNOT", CNOT, 2),
        _const("CZ", CZ, 2), _const("SWAP", SWAP, 2),
        Builtin("CPHASE", 2, 1, cphase),
        _const("CCX", CCX, 3), _const("TOFFOLI", CCX, 3),
    ]
}

# (D6) BASES[b] holds the unitary whose COLUMNS are the measurement vectors;
# the elaborator applies its adjoint before Enter (Thm 13.1).
#   X-basis:  H|0> = |+>,  H|1> = |->
#   Y-basis:  S H |0> = (|0> + i|1>)/sqrt(2),  S H |1> = (|0> - i|1>)/sqrt(2)
BASES = {"Z": I2, "X": H, "Y": S_GATE @ H}

BUILTIN_FNS = {
    "pi": math.pi, "tau": math.tau, "e": math.e,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "atan": math.atan, "atan2": math.atan2, "exp": math.exp, "log": math.log,
    "floor": lambda x: int(math.floor(x)), "ceil": lambda x: int(math.ceil(x)),
    "abs": abs, "min": min, "max": max, "len": len, "int": int, "float": float,
    "print": lambda *a: print("[static]", *a),
    "range": lambda *a: list(range(*a)),
}