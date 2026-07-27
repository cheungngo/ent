"""Recursive-descent parser for Ent."""
from __future__ import annotations

from typing import List, Optional

from .ast import (
    Block, ConstDecl, DefGate, EAlloc, EArray, EBin, EBlockExpr, ECall, ECopy,
    EIf, EIndex, ELam, ELit, EMeasure, ESuspend, ETuple, EUn, EVar, Expr,
    FnDecl, Import, PTuple, PVar, PWild, Program, SAssert, SDiscard, SExpr,
    SFor, SGate, SIf, SLet, SReturn, TArray, TArrow, TName, TTuple,
)
from .errors import EntError, Label
from .lexer import Tok, lex

_PRECEDENCE = [
    ("||",), ("&&",), ("^",), ("==", "!="),
    ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"), ("*", "/", "%"),
]


class Parser:
    def __init__(self, tokens: List[Tok]):
        self.toks = tokens
        self.i = 0

    # ------------------------------------------------------------ helpers
    @property
    def cur(self) -> Tok:
        return self.toks[self.i]

    def at(self, kind, value=None) -> bool:
        return self.cur.is_(kind, value)

    def eat(self, kind, value=None) -> Optional[Tok]:
        if self.at(kind, value):
            self.i += 1
            return self.toks[self.i - 1]
        return None

    def expect(self, kind, value=None) -> Tok:
        token = self.eat(kind, value)
        if token is None:
            want = value or kind
            raise EntError(
                "E0001",
                f"expected {want!r}, found {self.cur.value!r}",
                self.cur.span,
                [Label(self.cur.span, "unexpected token")],
            )
        return token

    # -------------------------------------------------------------- items
    def parse_program(self) -> Program:
        program = Program()
        while not self.at("EOF"):
            program.items.append(self.parse_item())
        return program

    def parse_item(self):
        span = self.cur.span
        if self.eat("KW", "import"):
            parts = [self.expect("ID").value]
            while self.eat("OP", "."):
                parts.append(self.expect("ID").value)
            self.expect("OP", ";")
            return Import(".".join(parts), span)

        if self.eat("KW", "const"):
            name = self.expect("ID").value
            if self.eat("OP", ":"):
                self.parse_type()
            self.expect("OP", "=")
            expr = self.parse_expr()
            self.expect("OP", ";")
            return ConstDecl(name, expr, span)

        if self.eat("KW", "defgate"):
            name = self.expect("ID").value
            sparams = self.parse_sparams()
            self.expect("OP", "(")
            arity = self.expect("INT").value
            self.expect("OP", ")")
            self.expect("OP", "=")
            matrix = self.parse_expr()
            self.expect("OP", ";")
            return DefGate(name, sparams, arity, matrix, span)

        is_gate = self.eat("KW", "gate") is not None
        if not is_gate:
            self.expect("KW", "fn")
        name = self.expect("ID").value
        sparams = self.parse_sparams()
        params = self.parse_params()
        ret = self.parse_type() if self.eat("OP", "->") else None
        body = self.parse_block()
        return FnDecl(name, sparams, params, ret, body, is_gate, span)

    def parse_sparams(self):
        out = []
        if self.eat("OP", "["):
            while not self.at("OP", "]"):
                name = self.expect("ID").value
                kind = self.parse_type() if self.eat("OP", ":") else None
                out.append((name, kind))
                if not self.eat("OP", ","):
                    break
            self.expect("OP", "]")
        return out

    def parse_params(self):
        out = []
        self.expect("OP", "(")
        while not self.at("OP", ")"):
            name = self.expect("ID").value
            self.expect("OP", ":")
            out.append((name, self.parse_type()))
            if not self.eat("OP", ","):
                break
        self.expect("OP", ")")
        return out

    # -------------------------------------------------------------- types
    def parse_type(self):
        left = self.parse_type_atom()
        if self.eat("OP", "-o"):
            return TArrow(left, self.parse_type(), left.span)
        return left

    def parse_type_atom(self):
        span = self.cur.span
        if self.eat("OP", "("):
            items = []
            while not self.at("OP", ")"):
                items.append(self.parse_type())
                if not self.eat("OP", ","):
                    break
            self.expect("OP", ")")
            base = items[0] if len(items) == 1 else TTuple(tuple(items), span)
        else:
            token = self.eat("KW") or self.expect("ID")
            base = TName(token.value, span)

        while self.eat("OP", "["):
            size = self.parse_expr()
            self.expect("OP", "]")
            base = TArray(base, size, span)
        return base

    # --------------------------------------------------------- statements
    def parse_block(self) -> Block:
        span = self.expect("OP", "{").span
        stmts, tail = [], None
        while not self.at("OP", "}"):
            if self.at("KW", "let") or self.at("KW", "const"):
                stmts.append(self.parse_let())
                continue
            if self.at("KW", "if"):
                stmts.append(self.parse_if_stmt())
                continue
            if self.at("KW", "for"):
                stmts.append(self.parse_for())
                continue
            if self.at("KW", "discard") or self.at("KW", "forget"):
                stmts.append(self.parse_discard())
                continue
            if self.at("KW", "assert"):
                start = self.expect("KW", "assert").span
                cond = self.parse_expr()
                # (D3) the message is optional and comma-separated only.
                message = None
                if self.eat("OP", ","):
                    message = self.expect("STR").value
                self.expect("OP", ";")
                stmts.append(SAssert(cond, message, start))
                continue
            if self.at("KW", "return"):
                start = self.expect("KW", "return").span
                expr = None if self.at("OP", ";") else self.parse_expr()
                self.expect("OP", ";")
                stmts.append(SReturn(expr, start))
                continue
            if self.looks_like_gate_stmt():
                stmts.append(self.parse_gate_stmt())
                continue

            expr = self.parse_expr()
            if self.eat("OP", ";"):
                # (D5) statement spans follow the expression, not the block.
                stmts.append(SExpr(expr, getattr(expr, "span", span)))
            else:
                tail = expr
                break

        self.expect("OP", "}")
        return Block(stmts, tail, span)

    def looks_like_gate_stmt(self) -> bool:
        """`adj`/`ctrl` prefix, or IDENT followed by `[` or by a bare name."""
        if self.at("KW", "adj") or self.at("KW", "ctrl"):
            return True
        if not self.at("ID"):
            return False
        nxt = self.toks[self.i + 1]
        return nxt.is_("OP", "[") or nxt.kind == "ID"

    def parse_gate_stmt(self):
        span = self.cur.span
        adjoint = False
        controls: List[Expr] = []
        while True:
            if self.eat("KW", "adj"):
                adjoint = not adjoint
                continue
            if self.eat("KW", "ctrl"):
                count = 1
                if self.eat("OP", "("):
                    count = self.parse_expr()
                    self.expect("OP", ")")
                controls.append(count)
                continue
            break

        name = self.expect("ID").value
        sargs = []
        if self.eat("OP", "["):
            while not self.at("OP", "]"):
                sargs.append(self.parse_expr())
                if not self.eat("OP", ","):
                    break
            self.expect("OP", "]")

        args = []
        if self.eat("OP", "("):
            while not self.at("OP", ")"):
                args.append(self.parse_expr())
                if not self.eat("OP", ","):
                    break
            self.expect("OP", ")")
        else:
            while not self.at("OP", ";"):
                args.append(self.parse_expr())
                if not self.eat("OP", ","):
                    break
        self.expect("OP", ";")
        return SGate(name, sargs, args, adjoint, controls, span)

    def parse_let(self):
        span = self.cur.span
        is_const = self.eat("KW", "const") is not None
        if not is_const:
            self.expect("KW", "let")
        pat = self.parse_pattern()
        ann = self.parse_type() if self.eat("OP", ":") else None
        self.expect("OP", "=")
        expr = self.parse_expr()
        self.expect("OP", ";")
        return SLet(pat, ann, expr, is_const, span)

    def parse_pattern(self):
        span = self.cur.span
        if self.eat("OP", "("):
            items = []
            while not self.at("OP", ")"):
                items.append(self.parse_pattern())
                if not self.eat("OP", ","):
                    break
            self.expect("OP", ")")
            return items[0] if len(items) == 1 else PTuple(items, span)
        if self.at("ID") and self.cur.value == "_":
            self.i += 1
            return PWild(span)
        return PVar(self.expect("ID").value, span)

    def parse_if_stmt(self):
        span = self.expect("KW", "if").span
        cond = self.parse_expr()
        then_ = self.parse_block()
        else_ = None
        if self.eat("KW", "else"):
            if self.at("KW", "if"):
                inner = self.parse_if_stmt()
                else_ = Block([inner], None, span)
            else:
                else_ = self.parse_block()
        return SIf(cond, then_, else_, span)

    def parse_for(self):
        span = self.expect("KW", "for").span
        var = self.expect("ID").value
        self.expect("KW", "in")
        start = self.parse_expr()
        self.expect("OP", "..")
        end = self.parse_expr()
        return SFor(var, start, end, self.parse_block(), span)

    def parse_discard(self):
        span = self.cur.span
        is_forget = self.eat("KW", "forget") is not None
        if not is_forget:
            self.expect("KW", "discard")
        args = [self.parse_expr()]
        while self.eat("OP", ","):
            args.append(self.parse_expr())
        self.expect("OP", ";")
        return SDiscard(args, is_forget, span)

    # -------------------------------------------------------- expressions
    def parse_expr(self, level: int = 0):
        if level >= len(_PRECEDENCE):
            return self.parse_unary()
        left = self.parse_expr(level + 1)
        while self.cur.kind == "OP" and self.cur.value in _PRECEDENCE[level]:
            span = self.cur.span
            op = self.eat("OP").value
            right = self.parse_expr(level + 1)
            left = EBin(op, left, right, getattr(left, "span", span))
        return left

    def parse_unary(self):
        span = self.cur.span
        if self.eat("OP", "!"):
            return EUn("!", self.parse_unary(), span)
        if self.eat("OP", "-"):
            return EUn("-", self.parse_unary(), span)
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_atom()
        while True:
            if self.at("OP", "("):
                span = self.eat("OP", "(").span
                args = []
                while not self.at("OP", ")"):
                    args.append(self.parse_expr())
                    if not self.eat("OP", ","):
                        break
                self.expect("OP", ")")
                expr = ECall(expr, [], args, span)
            elif self.at("OP", "["):
                span = self.eat("OP", "[").span
                first = self.parse_expr()
                # (D5) the draft carried a dead `and False` disjunct here.
                if self.at("OP", ","):
                    sargs = [first]
                    while self.eat("OP", ","):
                        sargs.append(self.parse_expr())
                    self.expect("OP", "]")
                    expr = ECall(expr, sargs, self.parse_arglist(), span)
                else:
                    self.expect("OP", "]")
                    if self.at("OP", "("):
                        expr = ECall(expr, [first], self.parse_arglist(), span)
                    else:
                        expr = EIndex(expr, first, span)
            else:
                return expr

    def parse_arglist(self):
        self.expect("OP", "(")
        args = []
        while not self.at("OP", ")"):
            args.append(self.parse_expr())
            if not self.eat("OP", ","):
                break
        self.expect("OP", ")")
        return args

    def parse_atom(self):
        span = self.cur.span
        if self.at("INT") or self.at("FLOAT") or self.at("STR"):
            return ELit(self.eat(self.cur.kind).value, span)
        if self.eat("KW", "true"):
            return ELit(True, span)
        if self.eat("KW", "false"):
            return ELit(False, span)
        if self.eat("KW", "measure"):
            arg = self.parse_postfix()
            basis = self.expect("ID").value if self.eat("KW", "in") else None
            return EMeasure(arg, basis, span)
        if self.eat("KW", "qubit"):
            self.expect("OP", "(")
            self.expect("OP", ")")
            return EAlloc(None, span)
        if self.eat("KW", "qubits"):
            self.expect("OP", "(")
            count = self.parse_expr()
            self.expect("OP", ")")
            return EAlloc(count, span)
        if self.eat("KW", "copy"):
            return ECopy(self.parse_postfix(), span)
        if self.eat("KW", "suspend"):
            return ESuspend(self.parse_block(), span)
        if self.at("KW", "if"):
            stmt = self.parse_if_stmt()
            return EIf(stmt.cond, stmt.then_, stmt.else_, span)
        if self.eat("OP", "\\"):
            params = []
            while not self.at("OP", "->"):
                name = self.expect("ID").value
                self.expect("OP", ":")
                params.append((name, self.parse_type()))
                if not self.eat("OP", ","):
                    break
            self.expect("OP", "->")
            return ELam(params, self.parse_expr(), span)
        if self.at("OP", "["):                       # (D4) array literal
            self.eat("OP", "[")
            items = []
            while not self.at("OP", "]"):
                items.append(self.parse_expr())
                if not self.eat("OP", ","):
                    break
            self.expect("OP", "]")
            return EArray(items, span)
        if self.at("OP", "{"):
            return EBlockExpr(self.parse_block(), span)
        if self.eat("OP", "("):
            items = []
            while not self.at("OP", ")"):
                items.append(self.parse_expr())
                if not self.eat("OP", ","):
                    break
            self.expect("OP", ")")
            return items[0] if len(items) == 1 else ETuple(items, span)
        token = self.eat("ID")
        if token is None:
            raise EntError("E0001", f"unexpected {self.cur.value!r}", span)
        return EVar(token.value, span)


def parse(source: str, filename: str = "<input>") -> Program:
    return Parser(lex(source, filename)).parse_program()