"""Textual module linking for `import` (D27)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .ast import Import, Program
from .errors import EntError
from .parser import parse

STD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "std")


def _resolve(path: str, roots: Sequence[str]) -> Optional[str]:
    parts = path.split(".")
    if parts and parts[0] == "std":
        candidates = [os.path.join(STD_DIR, *parts[1:]) + ".ent"]
    else:
        candidates = [os.path.join(root, *parts) + ".ent" for root in roots]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def link(filename: str, source: str,
         roots: Optional[Sequence[str]] = None) -> Tuple[Program, Dict[str, str]]:
    """Depth-first inclusion of imported modules, de-duplicated by real path."""
    roots = list(roots or [os.path.dirname(os.path.abspath(filename)) or "."])
    sources: Dict[str, str] = {filename: source}
    merged = Program([])
    seen: Set[str] = set()

    def visit(name: str, text: str) -> None:
        key = os.path.abspath(name)
        if key in seen:
            return
        seen.add(key)
        for item in parse(text, name).items:
            if isinstance(item, Import):
                target = _resolve(item.path, roots)
                if target is None:
                    raise EntError(
                        "E0800", f"cannot resolve module `{item.path}`", item.span
                    )
                with open(target, encoding="utf-8") as handle:
                    body = handle.read()
                sources[target] = body
                visit(target, body)
            else:
                merged.items.append(item)

    visit(filename, source)
    return merged, sources