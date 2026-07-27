"""Tokenizer for Ent."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .errors import EntError, Span

KEYWORDS = {
    "fn", "gate", "defgate", "let", "const", "if", "else", "for", "in",
    "return", "measure", "qubit", "qubits", "discard", "forget", "copy",
    "import", "adj", "ctrl", "assert", "true", "false", "suspend", "as",
    "Unit", "Bit", "Qubit", "int", "float", "bool", "str",
}

# (D2) the lambda backslash must be part of the operator class, otherwise
# `\x:Qubit -> e` cannot be lexed at all.
_MASTER = re.compile(
    r"""
      (?P<ws>[ \t\r]+)
    | (?P<nl>\n)
    | (?P<line_comment>//[^\n]*)
    | (?P<block_comment>/\*(?:.|\n)*?\*/)
    | (?P<float>\d+\.\d+(?:[eE][-+]?\d+)?)
    | (?P<int>0x[0-9a-fA-F]+|\d+)
    | (?P<ident>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<op>-o(?![A-Za-z_0-9])|->|=>|\.\.|<=|>=|==|!=|&&|\|\||<<|>>|::
        |[-+*/%^<>=!,;:(){}\[\]@#?|&~.\\])
    """,
    re.X,
)


@dataclass(frozen=True)
class Tok:
    kind: str          # ID KW INT FLOAT STR OP EOF
    value: object
    span: Span

    def is_(self, kind: str, value=None) -> bool:
        return self.kind == kind and (value is None or self.value == value)


def lex(source: str, filename: str = "<input>") -> List[Tok]:
    tokens: List[Tok] = []
    line, col, pos = 1, 1, 0

    while pos < len(source):
        match = _MASTER.match(source, pos)
        if match is None:
            raise EntError(
                "E0001",
                f"unexpected character {source[pos]!r}",
                Span(filename, line, col),
            )

        kind = match.lastgroup
        text = match.group()
        span = Span(filename, line, col, len(text))

        if kind == "nl":
            line, col, pos = line + 1, 1, match.end()
            continue
        if kind in ("ws", "line_comment"):
            col, pos = col + len(text), match.end()
            continue
        if kind == "block_comment":
            line += text.count("\n")
            col, pos = 1, match.end()
            continue

        if kind == "ident":
            tokens.append(Tok("KW" if text in KEYWORDS else "ID", text, span))
        elif kind == "int":
            tokens.append(Tok("INT", int(text, 0), span))
        elif kind == "float":
            tokens.append(Tok("FLOAT", float(text), span))
        elif kind == "string":
            body = text[1:-1].encode().decode("unicode_escape")
            tokens.append(Tok("STR", body, span))
        else:
            tokens.append(Tok("OP", text, span))

        col, pos = col + len(text), match.end()

    tokens.append(Tok("EOF", None, Span(filename, line, col, 0)))
    return tokens