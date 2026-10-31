"""PTQKit CLI entry (`qs`). Scaffold only."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qs",
        description="PTQKit — PTQ methods library (scaffold)",
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run a recipe (not yet implemented)")
    run_p.add_argument("recipe", help="Path to recipe YAML")
    run_p.add_argument("--set", action="append", default=[], help="Override key=value")

    st_p = sub.add_parser("status", help="Show cached/stale/missing stages")
    st_p.add_argument("recipe", help="Path to recipe YAML")

    sub.add_parser("report", help="Render results summaries to Markdown")
    sub.add_parser("version", help="Print package version")

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0
    if args.cmd == "version":
        from ptqkit import __version__

        print(__version__)
        return 0

    print(
        f"[ptqkit] `{args.cmd}` is scaffold-only.\n"
        "Use staging/ebss and staging/fastvit scripts for now; "
        "see README.md and recipes/.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
