"""Source spans and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Span:
    file: str
    line: int
    col: int
    length: int = 1

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class Label:
    span: Optional[Span]
    text: str = ""


class EntError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        span: Optional[Span] = None,
        labels: Optional[List[Label]] = None,
        notes: Optional[List[str]] = None,
    ):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.span = span
        # (D1) labels with no span cannot be rendered; drop them here.
        self.labels = [l for l in (labels or []) if l.span is not None]
        self.notes = list(notes or [])

    def render(self, sources: Dict[str, str]) -> str:
        out = [f"error[{self.code}]: {self.message}"]
        if self.span is not None:
            out.append(f"  --> {self.span}")
        for label in self.labels:
            lines = sources.get(label.span.file, "").splitlines()
            if 0 < label.span.line <= len(lines):
                out.append(f"{label.span.line:>4} | {lines[label.span.line - 1]}")
                caret = " " * (label.span.col - 1) + "^" * max(1, label.span.length)
                out.append(f"     | {caret} {label.text}")
        for note in self.notes:
            out.append(f"   = {note}")
        return "\n".join(out)