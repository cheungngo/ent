# External dependencies

Detected by scanning `import` statements in the extracted Python sources.
Modules that are **not** part of this extraction must be installed or copied in
separately — the specification document only embeds the sources it lists.

## Third-party / companion packages

| Module | Imported by |
|---|---|
| `enter` | `src/ent/backends/qasm.py`, `src/ent/backends/simulator.py`, `src/ent/cli.py`, `src/ent/cost.py`, `src/ent/elaborate.py`, `src/ent/prelude.py` |
| `numpy` | `src/ent/backends/qasm.py`, `src/ent/backends/simulator.py`, `src/ent/elaborate.py`, `src/ent/prelude.py` |

## Standard library (informational)

`argparse`, `cmath`, `dataclasses`, `math`, `os`, `re`, `sys`, `threading`, `typing`

---

**Note.** Any module above that is described in the document as an *external* or
*unchanged artefact* is intentionally absent from this ZIP. Obtain it from the
upstream project and place it so that it is importable (e.g. under `src/`),
then `pip install -e .`.
