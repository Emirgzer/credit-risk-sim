"""Command-line entry point: `python -m creditsim generate|validate|train`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .generator import GeneratorConfig, generate
from .model import train_and_evaluate
from .validate import validate


def _cmd_generate(args: argparse.Namespace) -> int:
    df = generate(GeneratorConfig(n=args.n, seed=args.seed, target_default_rate=args.default_rate,
                                  edge_case_share=args.edge_cases))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} borrowers to {out} (default rate {df['default'].mean():.2%}, "
          f"{(df['edge_case'] != '').sum():,} edge cases)")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    problems = validate(pd.read_csv(args.data))
    if problems:
        print("validation failed:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("validation passed")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data, keep_default_na=True)
    problems = validate(df)
    if problems:
        print("refusing to train, data failed validation:", *problems, sep="\n  - ")
        return 1

    result = train_and_evaluate(df, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    for name, table in result.calibration.items():
        table.to_csv(out_dir / f"calibration_{name}.csv")
    result.test_frame.to_csv(out_dir / "test_scores.csv", index=False)

    print(f"{'model':<22}{'AUC':>8}{'Gini':>8}{'KS':>8}{'Brier':>8}")
    for name, m in result.metrics.items():
        print(f"{name:<22}{m['roc_auc']:>8.3f}{m['gini']:>8.3f}{m['ks']:>8.3f}{m['brier']:>8.4f}")
    print(f"\nreports written to {out_dir}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditsim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic borrower dataset")
    g.add_argument("--n", type=int, default=50_000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--default-rate", type=float, default=0.08)
    g.add_argument("--edge-cases", type=float, default=0.02, help="share of rows turned into edge cases")
    g.add_argument("--out", default="data/borrowers.csv")
    g.set_defaults(func=_cmd_generate)

    v = sub.add_parser("validate", help="run data quality checks on a CSV")
    v.add_argument("--data", default="data/borrowers.csv")
    v.set_defaults(func=_cmd_validate)

    t = sub.add_parser("train", help="train and evaluate PD models")
    t.add_argument("--data", default="data/borrowers.csv")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out-dir", default="reports")
    t.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
