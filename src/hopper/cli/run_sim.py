"""
Module: hopper.cli.run_sim

Developer: ehtkarim
Date: April 29, 2026

Provides the hopper-sim command-line entry point for running configured simulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import logging

from ..nodes.pipeline import run_from_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CRES synthetic signal simulator")
    p.add_argument("config", type=str, help="Path to YAML config file")
    p.add_argument("--quiet", action="store_true", help="Suppress structured solver summary output")
    p.add_argument("--log-level", default="WARNING", help="Python logging level (default: WARNING)")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.WARNING))
    ctx = run_from_config(Path(args.config))

    solver_info = ctx.get("solver_info")
    if solver_info and not args.quiet:
        print("Solver summary:")
        print(json.dumps(solver_info, indent=2, sort_keys=True))

    npz_path = ctx.get("npz_path")
    root_path = ctx.get("root_path")
    log_path = ctx.get("log_path")
    if npz_path:
        print(f"Wrote NPZ:  {npz_path}")
    if root_path:
        print(f"Wrote ROOT: {root_path}")
    if log_path:
        print(f"Wrote LOG:  {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
