from __future__ import annotations

import argparse
from pathlib import Path

from ..nodes.pipeline import run_from_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CRES synthetic signal simulator")
    p.add_argument("config", type=str, help="Path to YAML config file")
    args = p.parse_args(argv)

    ctx = run_from_config(Path(args.config))

    npz_path = ctx.get("npz_path")
    root_path = ctx.get("root_path")
    if npz_path:
        print(f"Wrote NPZ:  {npz_path}")
    if root_path:
        print(f"Wrote ROOT: {root_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
