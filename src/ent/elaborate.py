"""Staged elaboration: Ent AST -> closed Enter-calculus core terms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from enter.terms import (
    App, BitLit, Enter, If, Lam, LetPair, New, Pair, Term, Unit, Var,
    gate as mk_gate, tup,
)
from enter.types import Bit, Lolli, One, Q, Tensor, Type
from enter.typing_rules import typecheck

from .ast import (
    Block, ConstDecl, DefGate, EAlloc, EArray, EBin, EBlockExpr, ECall, ECopy,
    EIf, EIndex, ELam, ELit, EMeasure, ESuspend, ETuple, EUn, EVar, FnDecl,
    Import, PTuple, PVar, PWild, Program, SAssert, SDiscard, SExpr, SFor,
    SGate, SIf, SLet, SReturn, TArray, TArrow, TName, TTuple,
)
from .errors import EntError, Label, Span
from .prelude import BASES, BUILTIN_FNS, BUILTIN_GATES, controlled

Cont = Callable[[], Term]
_NOVALUE = object()          # sentinel: "this branch yields nothing"


# ----------------------------------------------------------------- values
@dataclass
class Cell:
    """A live linear resource: a mutable handle onto a core variable."""
    name: str
    cname: str
    typ: Type
    decl: Optional[Span]
    live: bool = True
    killed: Optional[Span] = None
    touched: bool = False
    retired: bool = False      # (D14) rolled back by `restore`


@dataclass
class Closure:
    decl: FnDecl
    sargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateVal:
    """Either a builtin, a user `gate` body, or an explicit matrix."""
    name: str
    builtin: Optional[object] = None
    decl: Optional[FnDecl] = None
    matrix: Optional[np.ndarray] = None
    arity: int = 1


class Scope:
    __slots__ = ("table", "parent", "owned")

    def __init__(self, parent: Optional["Scope"] = None):
        self.table: Dict[str, Any] = {}
        self.parent = parent
        self.owned: List[Cell] = []

    def get(self, name: str):
        scope = self
        while scope is not None:
            if name in scope.table:
                return scope.table[name]
            scope = scope.parent
        return None


def right_nest(types: Sequence[Type]) -> Type:
    out = types[-1]
    for typ in reversed(types[:-1]):
        out = Tensor(typ, out)
    return out


# ------------------------------------------------------------ elaborator
class Elaborator:
    MAX_INLINE_DEPTH = 256

    def __init__(self, program: Program, filename: str = "<input>"):
        self.program = program
        self.filename = filename
        self.counter = 0
        self.cells: List[Cell] = []
        self.globals = Scope()
        self.gate_mode = 0        # >0 while lowering a `gate` body
        self.ops: List[tuple] = []
        self.depth = 0
        self.stats = dict(gates=0, measures=0, allocs=0, ifs=0)

    # ---------------------------------------------------------- utilities
    def fresh(self, base: str = "x") -> str:
        self.counter += 1
        return f"{base}${self.counter}"

    def err(self, code, message, span=None, label="", notes=()):
        raise EntError(code, message, span,
                       [Label(span, label)] if span else [], list(notes))

    def new_cell(self, name: str, typ: Type, span: Optional[Span]) -> Cell:
        cell = Cell(name, self.fresh(name), typ, span)
        self.cells.append(cell)
        return cell

    def use(self, cell: Cell, span) -> str:
        if not cell.live:
            self.err(
                "E0104", f"use of consumed resource `{cell.name}`", span,
                "used here",
                [f"consumed at {cell.killed}",
                 "note: measurement and gate application consume arguments"],
            )
        cell.live = False
        cell.killed = span
        cell.touched = True
        return cell.cname

    # ------------------------------------------------------ core builders
    def bind(self, term: Term, typ: Type, name: str, span,
             k: Callable[[Cell], Term]) -> Term:
        # (D11) `1` has no eliminator, so it can never be let-bound.
        if typ == One():
            self.err("E9002", "internal: cannot bind a value of type 1", span)
        cell = self.new_cell(name, typ, span)
        return App(Lam(cell.cname, typ, k(cell)), term)

    def chain(self, term: Term, binders: List[Tuple[str, Type]], k: Cont) -> Term:
        if not binders:
            self.err("E9002", "internal: empty binder list")
        if len(binders) == 1:
            name, typ = binders[0]
            return App(Lam(name, typ, k()), term)
        name, _typ = binders[0]
        rest = self.fresh("_t")
        return LetPair(name, rest, term,
                       self.chain(Var(rest), binders[1:], k))

    def rebind(self, term: Term, cells: List[Cell], k: Cont) -> Term:
        """Bind the components of a right-nested tuple back into `cells`."""
        fresh_names = [self.fresh(c.name) for c in cells]

        def inner() -> Term:
            for cell, name in zip(cells, fresh_names):
                cell.cname, cell.live, cell.killed = name, True, None
                cell.touched = True
            return k()

        return self.chain(term, list(zip(fresh_names, [c.typ for c in cells])),
                          inner)

    def unpack_new(self, term: Term, names: List[str], typ: Type,
                   span, k: Callable[[List[Cell]], Term]) -> Term:
        cells = [self.new_cell(name, typ, span) for name in names]
        return self.chain(term, [(c.cname, typ) for c in cells],
                          lambda: k(cells))

    # ----------------------------------------------------- value plumbing
    def flatten(self, value, span) -> List[Cell]:
        if isinstance(value, Cell):
            return [value]
        if isinstance(value, (list, tuple)):
            out: List[Cell] = []
            for item in value:
                out.extend(self.flatten(item, span))
            return out
        self.err("E0202", f"expected a linear value, found {value!r}", span)

    @staticmethod
    def flatten_soft(value) -> List[Cell]:
        if isinstance(value, Cell):
            return [value]
        if isinstance(value, (list, tuple)):
            out: List[Cell] = []
            for item in value:
                out.extend(Elaborator.flatten_soft(item))
            return out
        return []

    def leaf_types(self, value, span) -> List[Type]:
        if isinstance(value, (list, tuple)):
            out: List[Type] = []
            for item in value:
                out.extend(self.leaf_types(item, span))
            return out
        if isinstance(value, Cell):
            return [value.typ]
        if isinstance(value, bool) or value in (0, 1):
            return [Bit()]
        self.err("E0202", f"value {value!r} has no runtime representation", span)

    def value_term(self, value, span) -> Tuple[Term, Type]:
        if isinstance(value, Cell):
            return Var(self.use(value, span)), value.typ
        if isinstance(value, bool):
            return BitLit(int(value)), Bit()
        if isinstance(value, int) and value in (0, 1):
            return BitLit(value), Bit()
        if value is None:
            return Unit(), One()
        if isinstance(value, (list, tuple)):
            if not value:
                return Unit(), One()
            parts = [self.value_term(item, span) for item in value]
            term, typ = parts[-1]
            for sub_term, sub_typ in reversed(parts[:-1]):
                term, typ = Pair(sub_term, term), Tensor(sub_typ, typ)
            return term, typ
        self.err("E0202", f"value {value!r} has no runtime representation", span)

    def value_term_type(self, value, span) -> Type:
        snap = self.snapshot()
        _, typ = self.value_term(value, span)
        self.restore(snap)
        return typ

    @staticmethod
    def shape_of(value):
        if value is None:
            return _NOVALUE
        if isinstance(value, (list, tuple)):
            return tuple(Elaborator.shape_of(v) for v in value)
        return None

    @staticmethod
    def rebuild(shape, leaves: List[Any]):
        if shape is _NOVALUE:
            return None
        if shape is None:
            return leaves.pop(0)
        return tuple(Elaborator.rebuild(s, leaves) for s in shape)

    # --------------------------------------------------------------- snap
    def snapshot(self):
        return (
            len(self.cells),
            [(c, c.cname, c.live, c.killed, c.touched) for c in self.cells],
            self.counter,
        )

    def restore(self, snap):
        length, states, counter = snap
        for cell in self.cells[length:]:
            # (D14) retire rolled-back cells: `Scope.owned` may still hold them
            cell.live = False
            cell.retired = True
        del self.cells[length:]
        for cell, cname, live, killed, touched in states:
            cell.cname, cell.live = cname, live
            cell.killed, cell.touched = killed, touched
        self.counter = counter

    # ================================================== static evaluation
    def sval(self, expr, sc: Scope):
        """Evaluate a purely static expression; linear effects are rejected."""
        holder: Dict[str, Any] = {}
        snap = self.snapshot()
        self.elab_expr(expr, sc, lambda value: holder.setdefault("v", value)
                       and Unit() or Unit())
        # (D8) a static position must not consume or allocate anything.
        dirty = [c for c in self.cells[:snap[0]] if c.touched
                 and not snap[1][self.cells.index(c)][4]] if False else []
        created = len(self.cells) - snap[0]
        self.restore(snap)
        value = holder.get("v")
        if isinstance(value, Cell) or created:
            self.err(
                "E0501",
                "a runtime value was used where a compile-time value is required",
                getattr(expr, "span", None), "not known at compile time",
                ["note: static positions are array sizes, loop bounds "
                 "and `[...]` parameters"],
            )
        return value

    # ================================================= expression -> term
    def elab_expr(self, expr, sc: Scope, k: Callable[[Any], Term]) -> Term:
        span = getattr(expr, "span", None)

        if isinstance(expr, ELit):
            return k(expr.value)

        if isinstance(expr, EVar):
            if expr.name in BUILTIN_FNS:
                return k(BUILTIN_FNS[expr.name])
            value = sc.get(expr.name)
            if value is None:
                if expr.name in BUILTIN_GATES:
                    return k(GateVal(expr.name, builtin=BUILTIN_GATES[expr.name]))
                self.err("E0100", f"unbound name `{expr.name}`", span, "not found")
            if isinstance(value, Cell) and value.retired:
                self.err("E0104", f"`{expr.name}` is no longer available", span)
            return k(value)

        if isinstance(expr, ETuple):
            def go(i, acc):
                if i == len(expr.items):
                    return k(tuple(acc))
                return self.elab_expr(expr.items[i], sc,
                                      lambda v: go(i + 1, acc + [v]))
            return go(0, [])

        if isinstance(expr, EArray):
            return self.elab_expr(ETuple(expr.items, span), sc,
                                  lambda v: k(list(v)))

        if isinstance(expr, EIndex):
            def with_base(base):
                index = self.sval(expr.index, sc)
                if not isinstance(base, (list, tuple)):
                    self.err("E0202", "indexing a non-array value", span)
                if not isinstance(index, int) or not 0 <= index < len(base):
                    self.err("E0501",
                             f"index {index} out of range 0..{len(base)}", span)
                return k(base[index])
            return self.elab_expr(expr.base, sc, with_base)

        if isinstance(expr, EAlloc):
            if self.gate_mode:
                self.err("E0401", "`gate` bodies must be allocation-free", span)
            count = 1 if expr.count is None else self.sval(expr.count, sc)
            if not isinstance(count, int) or count < 0:
                self.err("E0501", "qubit count must be a static non-negative int",
                         span)
            self.stats["allocs"] += count

            def go(i, acc):
                if i == count:
                    return k(acc[0] if expr.count is None else acc)
                return self.bind(New(), Q(), "q", span,
                                 lambda cell: go(i + 1, acc + [cell]))
            return go(0, [])

        if isinstance(expr, EMeasure):
            if self.gate_mode:
                self.err("E0401", "`gate` bodies must be measurement-free", span)

            def with_target(value):
                cells = self.flatten(value, span)
                for cell in cells:
                    if cell.typ != Q():
                        self.err("E0202", f"`{cell.name}` is not a qubit", span)
                name = expr.basis or "Z"
                if name not in BASES:
                    self.err("E0101", f"unknown measurement basis `{name}`", span)
                arg = tup(*[Var(self.use(c, span)) for c in cells])
                if name != "Z":
                    # (Thm 13.1) rotated readout = apply U† then Enter
                    single = BASES[name].conj().T
                    full = single
                    for _ in range(len(cells) - 1):
                        full = np.kron(full, single)
                    arg = mk_gate(full, arg, f"{name}-basis")
                    self.stats["gates"] += 1
                self.stats["measures"] += 1
                names = [f"m{i}" for i in range(len(cells))]
                return self.unpack_new(
                    Enter(arg), names, Bit(), span,
                    lambda bits: k(bits[0] if len(bits) == 1 else bits),
                )
            return self.elab_expr(expr.arg, sc, with_target)

        if isinstance(expr, ECopy):
            def with_bit(value):
                if not isinstance(value, Cell) or value.typ != Bit():
                    self.err("E0202", "`copy` applies to a Bit", span,
                             notes=["note: quantum resources cannot be copied "
                                    "(Thm 5.1)"])
                guard = Var(self.use(value, span))
                self.stats["ifs"] += 1
                term = If(guard,
                          Pair(BitLit(1), BitLit(1)),
                          Pair(BitLit(0), BitLit(0)))
                return self.unpack_new(term, ["c0", "c1"], Bit(), span,
                                       lambda cs: k(tuple(cs)))
            return self.elab_expr(expr.arg, sc, with_bit)

        if isinstance(expr, EUn):
            def with_arg(value):
                if not isinstance(value, Cell):
                    return k(-value if expr.op == "-" else (not value))
                # (D19) `-` has no runtime meaning on Bit/Qubit.
                if expr.op != "!" or value.typ != Bit():
                    self.err("E0202",
                             f"unary `{expr.op}` is not defined on {value.typ}",
                             span)
                self.stats["ifs"] += 1
                return self.bind(If(Var(self.use(value, span)),
                                    BitLit(0), BitLit(1)),
                                 Bit(), "n", span, k)
            return self.elab_expr(expr.arg, sc, with_arg)

        if isinstance(expr, EBin):
            return self.elab_binop(expr, sc, k)

        if isinstance(expr, EIf):
            return self.elab_conditional(expr.cond, expr.then_, expr.else_,
                                         sc, k, span)

        if isinstance(expr, EBlockExpr):
            return self.elab_block(expr.block, Scope(sc), k)

        if isinstance(expr, ELam):
            return self.elab_lambda(expr, sc, k)

        if isinstance(expr, ECall):
            return self.elab_call(expr, sc, k)

        if isinstance(expr, ESuspend):
            self.err("E9001", "`suspend` requires the reflection backend (§14)",
                     span)

        self.err("E0001", f"cannot elaborate {type(expr).__name__}", span)

    # ------------------------------------------------------------ binops
    def elab_binop(self, expr: EBin, sc: Scope, k):
        span = expr.span

        def with_left(left):
            def with_right(right):
                dynamic = isinstance(left, Cell) or isinstance(right, Cell)
                if not dynamic:
                    return k(self.static_binop(expr.op, left, right, span))
                if expr.op not in ("&&", "||", "^", "==", "!="):
                    self.err("E0202",
                             f"operator `{expr.op}` is not defined on Bit", span)
                return self.dynamic_bool(expr.op, left, right, span, k)
            return self.elab_expr(expr.right, sc, with_right)

        return self.elab_expr(expr.left, sc, with_left)

    @staticmethod
    def static_binop(op, a, b, span):
        table = {
            "+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
            "/": lambda: a / b if isinstance(a, float) or isinstance(b, float)
                 else a // b,
            "%": lambda: a % b,
            "^": lambda: (a ^ b) if isinstance(a, int) and isinstance(b, int)
                 else (bool(a) != bool(b)),
            "<<": lambda: a << b, ">>": lambda: a >> b,
            "==": lambda: a == b, "!=": lambda: a != b, "<": lambda: a < b,
            ">": lambda: a > b, "<=": lambda: a <= b, ">=": lambda: a >= b,
            "&&": lambda: a and b, "||": lambda: a or b,
        }
        if op not in table:
            raise EntError("E0202", f"unknown operator `{op}`", span)
        return table[op]()

    def dynamic_bool(self, op, left, right, span, k):
        """`b1 op b2`, encoded with additive conditionals only.

        Both branches of every emitted If consume the same context, so the
        additive rule of §6.1 applies verbatim.
        """
        def lit(value):
            return BitLit(int(bool(value)))

        def as_bit(value):
            return Var(self.use(value, span)) if isinstance(value, Cell) \
                else lit(value)

        left_t = as_bit(left)
        right_dyn = isinstance(right, Cell)
        right_t = as_bit(right)

        def keep(when_one: int, when_zero: int) -> Term:
            """A constant/negated branch that still consumes the right operand."""
            if right_dyn:
                self.stats["ifs"] += 1
                return If(right_t, lit(when_one), lit(when_zero))
            return lit(when_one if bool(right) else when_zero)

        def pass_through() -> Term:
            return right_t

        if op == "&&":
            term = If(left_t, pass_through(), keep(0, 0))
        elif op == "||":
            term = If(left_t, keep(1, 1), pass_through())
        elif op in ("^", "!="):
            term = If(left_t, keep(0, 1), pass_through())
        else:                                       # "=="
            term = If(left_t, pass_through(), keep(0, 1))

        self.stats["ifs"] += 1
        return self.bind(term, Bit(), "b", span, k)

    # ------------------------------------------------------------ lambda
    def elab_lambda(self, expr: ELam, sc: Scope, k):
        span = expr.span
        child = Scope(sc)
        binders = []
        for name, ty in expr.params:
            typ = self.resolve_type(ty, sc)
            cell = self.new_cell(name, typ, span)
            child.table[name] = cell
            child.owned.append(cell)
            binders.append((cell, typ))

        holder: Dict[str, Type] = {}

        def body_k(value):
            term, typ = self.value_term(value, span)
            holder["typ"] = typ
            return term

        body = self.elab_expr(expr.body, child, body_k)
        self.check_scope_exit(child)

        if holder["typ"] == One():
            # (D11) the caller could never bind the result.
            self.err("E0603", "a linear function must return Bit/Qubit data",
                     span)

        term, result_typ = body, holder["typ"]
        for cell, typ in reversed(binders):
            term = Lam(cell.cname, typ, term)
            result_typ = Lolli(typ, result_typ)

        return self.bind(term, result_typ, "f", span, k)

    # -------------------------------------------------------------- call
    def elab_call(self, expr: ECall, sc: Scope, k):
        span = expr.span

        def with_callee(callee):
            if isinstance(callee, GateVal):
                return self.apply_gate_expr(callee, expr, sc, k)
            if isinstance(callee, Closure):
                return self.inline(callee, expr, sc, k)
            if callable(callee):
                return k(callee(*[self.sval(a, sc) for a in expr.args]))
            if isinstance(callee, (int, float, str, bool)):
                # (D30) `pi(1)` used to silently ignore its arguments.
                if expr.args or expr.sargs:
                    self.err("E0202", "this value is not callable", span)
                return k(callee)
            if isinstance(callee, Cell) and isinstance(callee.typ, Lolli):
                def with_arg(arg):
                    term, typ = self.value_term(
                        arg if len(expr.args) == 1 else tuple(arg), span)
                    if typ != callee.typ.A:
                        self.err("E0202",
                                 f"argument type {typ} does not match "
                                 f"{callee.typ.A}", span)
                    return self.bind(App(Var(self.use(callee, span)), term),
                                     callee.typ.B, "r", span, k)
                target = expr.args[0] if len(expr.args) == 1 \
                    else ETuple(expr.args, span)
                return self.elab_expr(target, sc, with_arg)
            self.err("E0202", "this value is not callable", span)

        return self.elab_expr(expr.fn, sc, with_callee)

    def inline(self, closure: Closure, expr: ECall, sc: Scope, k):
        span = expr.span
        self.depth += 1
        if self.depth > self.MAX_INLINE_DEPTH:
            self.err("E0602", "static inlining depth exceeded", span,
                     notes=["note: recursion at the static level must terminate"])

        decl = closure.decl
        # (D18) static and linear arities are both checked.
        if len(expr.sargs) != len(decl.sparams):
            self.err("E0201",
                     f"`{decl.name}` expects {len(decl.sparams)} static "
                     f"arguments, found {len(expr.sargs)}", span)
        if len(expr.args) != len(decl.params):
            self.err("E0201",
                     f"`{decl.name}` expects {len(decl.params)} arguments, "
                     f"found {len(expr.args)}", span)

        child = Scope(self.globals)
        for (name, _), value in zip(decl.sparams, expr.sargs):
            child.table[name] = self.sval(value, sc)
        child.table.update(closure.sargs)

        def bind_args(i, acc):
            if i == len(decl.params):
                for (name, _), value in zip(decl.params, acc):
                    child.table[name] = value

                def after(value):
                    self.depth -= 1
                    return k(value)

                return self.elab_block(decl.body, child, after, owns=False)
            return self.elab_expr(expr.args[i], sc,
                                  lambda v: bind_args(i + 1, acc + [v]))

        return bind_args(0, [])

    # ------------------------------------------------------- gate lowering
    def lower_gate(self, gv: GateVal, sargs: List[Any], cells: List[Cell],
                   span) -> List[tuple]:
        """Return a flat list of (matrix, cells, label) primitive operations."""
        if gv.builtin is not None:
            if len(sargs) != gv.builtin.n_static:
                self.err("E0201",
                         f"gate `{gv.name}` takes {gv.builtin.n_static} static "
                         f"arguments, found {len(sargs)}", span)
            if len(cells) != gv.builtin.arity:
                self.err("E0201",
                         f"gate `{gv.name}` takes {gv.builtin.arity} qubits, "
                         f"found {len(cells)}", span)
            return [(gv.builtin.build(*sargs), list(cells), gv.name)]

        if gv.matrix is not None:
            if len(cells) != gv.arity:
                self.err("E0201", f"gate `{gv.name}` arity mismatch", span)
            return [(gv.matrix, list(cells), gv.name)]

        decl = gv.decl
        if len(sargs) != len(decl.sparams):
            self.err("E0201", f"gate `{gv.name}` static arity mismatch", span)

        child = Scope(self.globals)
        for (name, _), value in zip(decl.sparams, sargs):
            child.table[name] = value

        cursor = 0
        for name, ty in decl.params:
            typ = self.resolve_type(ty, child)
            width = self.type_width(typ)
            group = cells[cursor:cursor + width]
            child.table[name] = group if (width > 1 or isinstance(ty, TArray)) \
                else group[0]
            cursor += width
        if cursor != len(cells):
            self.err("E0201", f"gate `{gv.name}` arity mismatch", span)

        saved_ops, self.ops = self.ops, []
        self.gate_mode += 1
        self.elab_block(decl.body, child, lambda value: Unit(), owns=False)
        self.gate_mode -= 1
        ops, self.ops = self.ops, saved_ops
        return ops

    def apply_gate_expr(self, gv: GateVal, expr: ECall, sc: Scope, k):
        stmt = SGate(gv.name, expr.sargs, expr.args, False, [], expr.span)
        return self.elab_gate_stmt(stmt, sc, lambda: k(None))

    def emit_ops(self, ops: List[tuple], k: Cont) -> Term:
        if not ops:
            return k()
        matrix, cells, label = ops[0]
        rest = ops[1:]

        seen: Set[int] = set()
        for cell in cells:
            if id(cell) in seen:
                self.err("E0102",
                         f"qubit `{cell.name}` is used twice in one operation",
                         cell.decl, "contraction is not permitted on Q",
                         ["note: Enter Calculus §1.2 — premises have disjoint "
                          "contexts"])
            seen.add(id(cell))

        if self.gate_mode:
            self.ops.append((matrix, list(cells), label))
            return self.emit_ops(rest, k)

        span = cells[0].decl
        arg = tup(*[Var(self.use(cell, span)) for cell in cells])
        self.stats["gates"] += 1
        return self.rebind(mk_gate(matrix, arg, label), cells,
                           lambda: self.emit_ops(rest, k))

    # =================================================== statement -> term
    def elab_block(self, block: Block, sc: Scope, k, owns: bool = True) -> Term:
        def tail() -> Term:
            if block.tail is not None:
                return self.elab_expr(block.tail, sc,
                                      lambda v: self.finish(sc, v, k, owns))
            return self.finish(sc, None, k, owns)

        return self.elab_stmts(block.stmts, sc, tail)

    def finish(self, sc: Scope, value, k, owns: bool) -> Term:
        if owns:
            # (D13) values returned from the block escape the ownership check
            # and are handed to the parent scope.
            escaping = self.flatten_soft(value)
            self.check_scope_exit(sc, {id(c) for c in escaping})
            if sc.parent is not None:
                sc.parent.owned.extend(escaping)
        return k(value)

    def check_scope_exit(self, sc: Scope, escaping: Set[int] = frozenset()):
        for cell in sc.owned:
            if cell.live and not cell.retired and id(cell) not in escaping:
                self.err(
                    "E0103", f"linear resource `{cell.name}` is never consumed",
                    cell.decl, "declared here",
                    ["note: weakening is not permitted for quantum resources",
                     "help: `measure`, `discard`, or return it"],
                )

    def elab_stmts(self, stmts: List[Any], sc: Scope, k: Cont) -> Term:
        if not stmts:
            return k()
        head, rest = stmts[0], stmts[1:]

        def cont() -> Term:
            return self.elab_stmts(rest, sc, k)

        if isinstance(head, SLet):
            if head.is_const:
                sc.table[head.pat.name] = self.sval(head.expr, sc)
                return cont()

            def bound(value):
                if head.ann is not None and isinstance(value, Cell):
                    want = self.resolve_type(head.ann, sc)
                    if want != value.typ:
                        self.err("E0202",
                                 f"annotation {want} does not match {value.typ}",
                                 head.span)
                self.bind_pattern(head.pat, value, sc)
                return cont()

            return self.elab_expr(head.expr, sc, bound)

        if isinstance(head, SExpr):
            return self.elab_expr(head.expr, sc, lambda v: cont())

        if isinstance(head, SReturn):
            if rest:
                self.err("E0001",
                         "`return` must be the final statement of a block",
                         head.span)
            if head.expr is None:
                return k()
            return self.elab_expr(head.expr, sc, lambda v: k(v))

        if isinstance(head, SGate):
            return self.elab_gate_stmt(head, sc, cont)

        if isinstance(head, SFor):
            start = self.sval(head.start, sc)
            end = self.sval(head.end, sc)
            if not isinstance(start, int) or not isinstance(end, int):
                self.err("E0501", "loop bounds must be static integers",
                         head.span)

            def loop(i: int) -> Term:
                if i >= end:
                    return cont()
                child = Scope(sc)
                child.table[head.var] = i
                return self.elab_block(head.body, child, lambda v: loop(i + 1))

            return loop(start)

        if isinstance(head, SIf):
            return self.elab_conditional(head.cond, head.then_, head.else_,
                                         sc, lambda v: cont(), head.span)

        if isinstance(head, SDiscard):
            def go(i: int) -> Term:
                if i >= len(head.args):
                    return cont()
                return self.elab_expr(
                    head.args[i], sc,
                    lambda value: self.discard_value(value, head.span,
                                                     lambda: go(i + 1)),
                )
            return go(0)

        if isinstance(head, SAssert):
            if not self.sval(head.cond, sc):
                self.err("E0701", head.message or "static assertion failed",
                         head.span)
            return cont()

        self.err("E0001", f"cannot elaborate {type(head).__name__}",
                 getattr(head, "span", None))

    # ------------------------------------------------------------ helpers
    def bind_pattern(self, pat, value, sc: Scope):
        if isinstance(pat, PWild):
            sc.owned.extend(self.flatten_soft(value))
            return
        if isinstance(pat, PVar):
            existing = sc.table.get(pat.name)
            if isinstance(existing, Cell) and existing.live:
                self.err("E0105",
                         f"`{pat.name}` shadows a live linear resource",
                         pat.span)
            sc.table[pat.name] = value
            # (D15) only a single cell inherits the surface name.
            if isinstance(value, Cell):
                value.name = pat.name
            sc.owned.extend(self.flatten_soft(value))
            return
        if isinstance(pat, PTuple):
            if not isinstance(value, (list, tuple)) or len(value) != len(pat.items):
                self.err("E0202", "tuple pattern does not match the value",
                         pat.span)
            for sub_pat, sub_value in zip(pat.items, value):
                self.bind_pattern(sub_pat, sub_value, sc)
            return
        self.err("E0001", "bad pattern", getattr(pat, "span", None))

    def forget(self, cell: Cell, span, k: Cont) -> Term:
        guard = Var(self.use(cell, span))
        self.stats["ifs"] += 1
        body = k()                       # shared subterm: no size blow-up
        return If(guard, body, body)

    def discard_value(self, value, span, k: Cont) -> Term:
        cells = self.flatten(value, span)
        qubits = [c for c in cells if c.typ == Q()]
        bits = [c for c in cells if c.typ == Bit()]

        def drop_bits(remaining: List[Cell]) -> Term:
            if not remaining:
                return k()
            return self.forget(remaining[0], span,
                               lambda: drop_bits(remaining[1:]))

        if not qubits:
            return drop_bits(bits)

        self.stats["measures"] += 1
        source = tup(*[Var(self.use(cell, span)) for cell in qubits])
        names = [f"_d{i}" for i in range(len(qubits))]
        return self.unpack_new(Enter(source), names, Bit(), span,
                               lambda fresh: drop_bits(fresh + bits))

    def elab_gate_stmt(self, stmt: SGate, sc: Scope, k: Cont) -> Term:
        span = stmt.span
        target = sc.get(stmt.name)

        # (D16) `f[n](x);` in statement position where f is a `fn`.
        if isinstance(target, Closure):
            if stmt.adjoint or stmt.controls:
                self.err("E0402", "`adj`/`ctrl` require a `gate`, not a `fn`",
                         span)
            call = ECall(EVar(stmt.name, span), stmt.sargs, stmt.args, span)
            return self.elab_expr(call, sc, lambda v: k())

        if isinstance(target, GateVal):
            gv = target
        elif stmt.name in BUILTIN_GATES:
            gv = GateVal(stmt.name, builtin=BUILTIN_GATES[stmt.name])
        else:
            self.err("E0101", f"unknown gate `{stmt.name}`", span, "not a gate")

        sargs = [self.sval(a, sc) for a in stmt.sargs]

        def with_args(values):
            cells = self.flatten(values, span)
            for cell in cells:
                if cell.typ != Q():
                    self.err("E0202",
                             f"`{cell.name}` is a {cell.typ}, gates require "
                             "Qubit", span)

            n_controls = sum(c if isinstance(c, int) else self.sval(c, sc)
                             for c in stmt.controls)
            if n_controls >= len(cells):
                self.err("E0201", "not enough target qubits for the controls",
                         span)
            controls, targets = cells[:n_controls], cells[n_controls:]
            ops = self.lower_gate(gv, sargs, targets, span)

            if stmt.adjoint:
                ops = [(m.conj().T, cs, f"{lbl}†")
                       for m, cs, lbl in reversed(ops)]
            if controls:
                ops = [(controlled(m, len(controls)), controls + cs,
                        "ctrl@" + lbl) for m, cs, lbl in ops]
            return self.emit_ops(ops, k)

        return self.elab_expr(ETuple(stmt.args, span), sc, with_args)

    # ------------------------------------------------------- conditionals
    def elab_conditional(self, cond, then_blk, else_blk, sc: Scope, k, span):
        def with_cond(guard_value):
            if not isinstance(guard_value, Cell):
                chosen = then_blk if guard_value else else_blk
                if chosen is None:
                    return k(None)
                return self.elab_block(chosen, Scope(sc), k)

            if guard_value.typ != Bit():
                self.err("E0202", "conditional guard must be a Bit", span)

            # (D10) Gamma |- b : Bit  is discharged *before* the branches.
            guard = Var(self.use(guard_value, span))

            def run_branch(block, signature):
                snap = self.snapshot()
                entry = {id(c): c.live for c in self.cells}
                saved_touch = [(c, c.touched) for c in self.cells]
                for cell, _ in saved_touch:
                    cell.touched = False       # (D9) branch-local tracking
                captured: Dict[str, Any] = {}

                def branch_tail(value):
                    escaped = {id(c) for c in self.flatten_soft(value)}
                    touched = [c for c in self.cells
                               if id(c) in entry and c.touched and c.live
                               and id(c) not in escaped]
                    outs = list(signature) if signature is not None else touched
                    captured["outs"] = outs
                    captured["dead"] = {id(c) for c in self.cells
                                        if id(c) in entry and entry[id(c)]
                                        and not c.live}
                    captured["shape"] = self.shape_of(value)
                    captured["ltypes"] = ([] if value is None
                                          else self.leaf_types(value, span))
                    payload = list(outs) + ([] if value is None else [value])
                    if payload:
                        term, typ = self.value_term(payload, span)
                    else:
                        # (D11/D12) pad to Bit: `1` cannot be bound, and the
                        # branch term must still be emitted for its effects.
                        term, typ = BitLit(0), Bit()
                        captured["pad"] = True
                    captured["typ"] = typ
                    return term

                child = Scope(sc)
                term = (branch_tail(None) if block is None
                        else self.elab_block(block, child, branch_tail))
                self.restore(snap)
                return term, captured

            # pass 1 --- discover the touched signature of each branch
            _, cap_then = run_branch(then_blk, None)
            _, cap_else = run_branch(else_blk, None)

            if cap_then["dead"] != cap_else["dead"]:
                self.err(
                    "E0301", "conditional branches consume different resources",
                    span, "branches disagree",
                    ["note: §6.1 additive rule — Γ ⊢ b : Bit, ∆ ⊢ u : A, "
                     "∆ ⊢ u' : A",
                     "help: measure or discard the same resources in both "
                     "branches"],
                )
            if cap_then["shape"] != cap_else["shape"]:
                self.err("E0302",
                         "conditional branches yield different value shapes",
                         span)

            merged: Dict[int, Cell] = {}
            for cell in cap_then["outs"] + cap_else["outs"]:
                merged.setdefault(id(cell), cell)
            signature = list(merged.values())

            # pass 2 --- emit with the agreed output signature
            term_then, cap_then = run_branch(then_blk, signature)
            term_else, cap_else = run_branch(else_blk, signature)

            if cap_then["typ"] != cap_else["typ"]:
                self.err("E0302",
                         f"branch types differ: {cap_then['typ']} vs "
                         f"{cap_else['typ']}", span)

            self.stats["ifs"] += 1
            joint = If(guard, term_then, term_else)

            shape = cap_then["shape"]
            leaf_types = cap_then["ltypes"]

            if not signature and shape is _NOVALUE:
                # padded Bit: bind it and immediately forget it
                return self.bind(joint, Bit(), "_u", span,
                                 lambda cell: self.forget(cell, span,
                                                          lambda: k(None)))

            fresh = [self.fresh(c.name) for c in signature]
            binders = [(n, c.typ) for n, c in zip(fresh, signature)]
            leaf_names = [self.fresh("v") for _ in leaf_types]
            binders.extend(zip(leaf_names, leaf_types))

            def after() -> Term:
                for cell, name in zip(signature, fresh):
                    cell.cname, cell.live, cell.killed = name, True, None
                    cell.touched = True
                leaves: List[Any] = []
                for name, typ in zip(leaf_names, leaf_types):
                    cell = self.new_cell("v", typ, span)
                    cell.cname = name
                    leaves.append(cell)
                return k(self.rebuild(shape, leaves))

            return self.chain(joint, binders, after)

        return self.elab_expr(cond, sc, with_cond)

    # -------------------------------------------------------------- types
    def resolve_type(self, ty, sc: Scope) -> Type:
        if isinstance(ty, TName):
            table = {"Qubit": Q(), "Bit": Bit(), "Unit": One()}
            if ty.name not in table:
                self.err("E0202", f"unknown type `{ty.name}`", ty.span)
            return table[ty.name]
        if isinstance(ty, TTuple):
            return right_nest([self.resolve_type(t, sc) for t in ty.items])
        if isinstance(ty, TArrow):
            return Lolli(self.resolve_type(ty.A, sc),
                         self.resolve_type(ty.B, sc))
        if isinstance(ty, TArray):
            size = self.sval(ty.size, sc)
            if not isinstance(size, int) or size < 1:
                self.err("E0501", "array size must be a positive static int",
                         ty.span)
            return right_nest([self.resolve_type(ty.elem, sc)] * size)
        self.err("E0202", "unsupported type", getattr(ty, "span", None))

    def type_width(self, typ: Type) -> int:
        if isinstance(typ, Tensor):
            return self.type_width(typ.A) + self.type_width(typ.B)
        return 1

    # ================================================================ main
    def compile(self, entry: str = "main") -> Tuple[Term, Type, dict]:
        for item in self.program.items:
            if isinstance(item, FnDecl):
                self.globals.table[item.name] = (
                    GateVal(item.name, decl=item) if item.is_gate
                    else Closure(item)
                )
            elif isinstance(item, DefGate):
                raw = self.sval(item.matrix, self.globals)
                matrix = np.asarray(raw, dtype=complex)
                expected = (2 ** item.arity, 2 ** item.arity)
                if matrix.shape != expected:
                    raise EntError("E0203",
                                   f"`{item.name}` must be {expected}, "
                                   f"found {matrix.shape}", item.span)
                self.globals.table[item.name] = GateVal(
                    item.name, matrix=matrix, arity=item.arity)
            elif isinstance(item, ConstDecl):
                self.globals.table[item.name] = self.sval(item.expr, self.globals)
            elif isinstance(item, Import):
                continue          # already inlined by ent.modules.link

        main = self.globals.table.get(entry)
        if not isinstance(main, Closure):
            raise EntError("E0100", f"no entry point `fn {entry}`")
        if main.decl.params:
            raise EntError("E0601",
                           f"`{entry}` must take no linear parameters",
                           main.decl.span)

        holder: Dict[str, Type] = {}

        def done(value):
            term, typ = self.value_term(value, main.decl.span)
            holder["typ"] = typ
            return term

        scope = Scope(self.globals)
        term = self.elab_block(main.decl.body, scope, done)

        for cell in self.cells:
            if cell.live and not cell.retired:
                raise EntError(
                    "E0103", f"linear resource `{cell.name}` escapes `{entry}`",
                    cell.decl, notes=["help: measure, discard, or return it"])

        # self-verification against the calculus itself
        inferred = typecheck(term, {})
        if inferred != holder["typ"]:
            raise EntError("E9000",
                           f"internal: core type {inferred} != surface type "
                           f"{holder['typ']}")

        return term, holder["typ"], dict(self.stats)


def compile_source(source: str, filename: str = "<input>",
                   entry: str = "main", roots=None):
    """Parse, link, elaborate and self-verify.  Returns (term, typ, stats, src)."""
    from .modules import link
    program, sources = link(filename, source, roots)
    term, typ, stats = Elaborator(program, filename).compile(entry)
    return term, typ, stats, sources