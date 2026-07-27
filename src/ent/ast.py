"""Surface syntax tree."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .errors import Span

# ------------------------------------------------------------------ types
@dataclass(frozen=True)
class TName:
    name: str
    span: Span


@dataclass(frozen=True)
class TArrow:
    A: "Ty"
    B: "Ty"
    span: Span


@dataclass(frozen=True)
class TTuple:
    items: tuple
    span: Span


@dataclass(frozen=True)
class TArray:
    elem: "Ty"
    size: "Expr"
    span: Span


Ty = object


# ------------------------------------------------------------ expressions
@dataclass
class ELit:
    value: object
    span: Span


@dataclass
class EVar:
    name: str
    span: Span


@dataclass
class ETuple:
    items: List["Expr"]
    span: Span


@dataclass
class EArray:
    items: List["Expr"]
    span: Span


@dataclass
class EIndex:
    base: "Expr"
    index: "Expr"
    span: Span


@dataclass
class ECall:
    fn: "Expr"
    sargs: List["Expr"]
    args: List["Expr"]
    span: Span


@dataclass
class EMeasure:
    arg: "Expr"
    basis: Optional[str]
    span: Span


@dataclass
class EAlloc:
    count: Optional["Expr"]
    span: Span


@dataclass
class ECopy:
    arg: "Expr"
    span: Span


@dataclass
class EIf:
    cond: "Expr"
    then_: "Block"
    else_: Optional["Block"]
    span: Span


@dataclass
class ELam:
    params: List[tuple]
    body: "Expr"
    span: Span


@dataclass
class EBin:
    op: str
    left: "Expr"
    right: "Expr"
    span: Span


@dataclass
class EUn:
    op: str
    arg: "Expr"
    span: Span


@dataclass
class EBlockExpr:
    block: "Block"
    span: Span


@dataclass
class ESuspend:
    block: "Block"
    span: Span


Expr = object


# ------------------------------------------------------------- statements
@dataclass
class PVar:
    name: str
    span: Span


@dataclass
class PWild:
    span: Span


@dataclass
class PTuple:
    items: List[object]
    span: Span


@dataclass
class SLet:
    pat: object
    ann: Optional[Ty]
    expr: Expr
    is_const: bool
    span: Span


@dataclass
class SGate:
    name: str
    sargs: List[Expr]
    args: List[Expr]
    adjoint: bool
    controls: List[Expr]
    span: Span


@dataclass
class SIf:
    cond: Expr
    then_: "Block"
    else_: Optional["Block"]
    span: Span


@dataclass
class SFor:
    var: str
    start: Expr
    end: Expr
    body: "Block"
    span: Span


@dataclass
class SDiscard:
    args: List[Expr]
    is_forget: bool
    span: Span


@dataclass
class SAssert:
    cond: Expr
    message: Optional[str]
    span: Span


@dataclass
class SExpr:
    expr: Expr
    span: Span


@dataclass
class SReturn:
    expr: Optional[Expr]
    span: Span


@dataclass
class Block:
    stmts: List[object]
    tail: Optional[Expr]
    span: Span


# ------------------------------------------------------------------ items
@dataclass
class FnDecl:
    name: str
    sparams: List[tuple]
    params: List[tuple]
    ret: Optional[Ty]
    body: Block
    is_gate: bool
    span: Span


@dataclass
class DefGate:
    name: str
    sparams: List[tuple]
    arity: int
    matrix: Expr
    span: Span


@dataclass
class ConstDecl:
    name: str
    expr: Expr
    span: Span


@dataclass
class Import:
    path: str
    span: Span


@dataclass
class Program:
    items: List[object] = field(default_factory=list)