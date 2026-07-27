# Ent: A Usable Quantum Programming Language on Top of the Enter Calculus

## Language Specification, Reference Implementation, and Debug Log

> Reconstructed automatically from **085XvA.tex** by
> *TeX → Repo* on 2026-07-27T02:24:06.523Z.
> The LaTeX document is the single source of truth; this repository is a
> mechanical extraction of the code listings it embeds.

## Abstract

Ent (source files .ent, compiler entc) is a surface language whose only semantics is ``desugar to a closed Enter-calculus term and reduce it''. Every theorem of the Enter calculus therefore applies to every Ent program for free, and the compiler literally re-checks its own output by handing the emitted core term back to enter.typing_rules.typecheck. This document specifies the surface syntax, the staged (compile-time / run-time) type discipline, the complete desugaring into the core calculus, the compiler architecture, and a reference implementation in Python that extends the existing enter-calculus artefact package. Section sec:guarantees states precisely which guarantees are inherited from which theorem, and which are not available. Appendix app:debug is a consolidated log of thirty corrections (D1--D30) applied to the first draft of the implementation; the listings in Section sec:impl are the corrected sources.

## Contents

```
    docs/notes/note-01.txt
    docs/notes/note-02.txt
    docs/notes/note-03.txt
    docs/sessions/session-01.txt
    docs/sessions/session-02.txt
    docs/sessions/session-03.txt
    examples/bell.ent
    examples/dj.ent
    examples/ghz.ent
    examples/teleport.ent
    extracted/listing-01.ent
    extracted/listing-02.ent
    extracted/listing-03.ent
    extracted/listing-04.ent
    extracted/listing-05.ebnf
    extracted/listing-06.qasm
    pyproject.toml
    src/ent/__init__.py
    src/ent/ast.py
    src/ent/backends/__init__.py
    src/ent/backends/qasm.py
    src/ent/backends/simulator.py
    src/ent/cli.py
    src/ent/cost.py
    src/ent/elaborate.py
    src/ent/errors.py
    src/ent/lexer.py
    src/ent/modules.py
    src/ent/parser.py
    src/ent/prelude.py
    src/ent/std/gates.ent
    src/ent/std/grover.ent
    src/ent/std/qft.ent
    src/enter/README.md
    tests/test_ent.py
```

## Provenance

* Listings extracted: **32**
* Files written: **35**
* Corrections/debug identifiers referenced by the document: D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30
* Every file is mapped back to its originating listing in `manifest.json`.

## Caveats

* Files marked **TODO / stub** are declared in the document's layout tree but no
  source was embedded for them.
* External packages referenced by the code are **not** included — see
  `DEPENDENCIES.md`.
* Shell transcripts and prose listings were placed under `docs/`; they are
  illustrative, not executable.
