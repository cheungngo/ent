"""entc — the Ent compiler driver."""
from __future__ import annotations

import argparse
import sys
import threading

from enter.typing_rules import typecheck

from .cost import report
from .elaborate import compile_source
from .errors import EntError

sys.setrecursionlimit(1_000_000)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _with_big_stack(fn, *args, size: int = 256 * 1024 * 1024):
    """(D26) CPS elaboration is deeply recursive; give it a real stack."""
    box = {}

    def run():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:        # re-raised on the caller's thread
            box["error"] = exc

    try:
        threading.stack_size(size)
    except (ValueError, RuntimeError):
        pass
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="entc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("run", "check", "verify", "cost", "emit"):
        p = sub.add_parser(name)
        p.add_argument("file")
        p.add_argument("--entry", default="main")
        if name == "run":
            p.add_argument("--shots", type=int, default=None)
            p.add_argument("--exact", action="store_true")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--check", action="store_true",
                           help="assert configuration invariants on every step")
        if name == "emit":
            p.add_argument("--target", default="qasm3",
                           choices=["qasm3", "core", "pretty"])
            p.add_argument("-o", "--out", default=None)

    args = parser.parse_args(argv)
    check_flag = getattr(args, "check", False)      # (D25)

    # (D25) read the source once, outside the diagnostic handler.
    try:
        source = _read(args.file)
    except OSError as error:
        print(f"error: cannot read {args.file}: {error}", file=sys.stderr)
        return 2

    try:
        term, typ, stats, sources = _with_big_stack(
            compile_source, source, args.file, args.entry
        )
    except EntError as error:
        print(error.render({args.file: source}), file=sys.stderr)
        return 1

    if args.cmd == "check":
        print(f"ok: {args.file} : {typ}")
        print(f"core type check: {typecheck(term, {})}")
        print(f"static counts: {stats}")
        return 0

    if args.cmd == "cost":
        print(report(term))
        return 0

    if args.cmd == "emit":
        if args.target == "core":
            text = repr(term)
        elif args.target == "pretty":
            from enter.gen import pretty
            text = pretty(term)
        else:
            from .backends.qasm import to_qasm3
            text = to_qasm3(term)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text)
        else:
            print(text, end="" if text.endswith("\n") else "\n")
        return 0

    if args.cmd == "verify":
        from .backends.simulator import verify_scheduler_independence
        worst = verify_scheduler_independence(term)
        print(f"scheduler independence (Thm 4.5): max dTV = {worst:.2e}")
        print("preservation/progress/normalization (Thms 3.3-3.5): "
              "asserted per step")
        return 0 if worst < 1e-9 else 1

    from .backends.simulator import run_exact, run_shots
    if args.exact or args.shots is None:
        for key, value in run_exact(term, typ, check=check_flag).items():
            print(f"{key}  {value:.6f}")
    else:
        counts = run_shots(term, typ, args.shots, args.seed, check=check_flag)
        for key, value in counts.items():
            print(f"{key}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())