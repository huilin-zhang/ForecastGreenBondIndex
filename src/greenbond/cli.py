"""Command-line entry points for demo, local experiments, ETL, and extraction."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .data import TARGET, align_predictors, normalize, read_csv, read_legacy, synthetic_data
from .pipeline import run


def default_output():
    return Path("reports") / datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f")


def main():
    parser = argparse.ArgumentParser(description="Forecasting the Green Bond Index")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "run", "legacy"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--lookback", type=int, default=10)
        command.add_argument("--rf-trials", type=int, default=8)
        command.add_argument("--neural", action="store_true")
        command.add_argument("--epochs", type=int, default=20)
        command.add_argument("--lstm-trials", type=int, default=4)
        if name == "demo":
            command.add_argument("--rows", type=int, default=900)
        elif name == "run":
            command.add_argument("--input", type=Path, required=True)
        else:
            command.add_argument("--data-dir", type=Path, default=Path("Data"))
            command.add_argument("--macro-lag-days", type=int, default=45)
    etl = sub.add_parser("prepare", help="Join local CSV inputs on the index calendar")
    etl.add_argument("--target", type=Path, required=True)
    etl.add_argument("--daily", type=Path, required=True)
    etl.add_argument("--monthly", type=Path)
    etl.add_argument("--macro-lag-days", type=int, default=45)
    etl.add_argument("--output", type=Path, required=True)
    api = sub.add_parser("extract", help="Explicitly request licensed LSEG data")
    api.add_argument("--instruments", type=Path, default=Path("config/instruments.example.json"))
    api.add_argument("--config", type=Path, default=Path("Data/lseg-data.config.json"))
    api.add_argument("--start", required=True)
    api.add_argument("--end", required=True)
    api.add_argument("--cache", type=Path, default=Path("data/raw/lseg"))
    args = parser.parse_args()
    try:
        if args.command == "extract":
            from .lseg import download
            frame = download(args.instruments, args.start, args.end, args.cache, args.config)
            frame.to_csv(args.cache / "daily.csv", index_label="date")
            print(f"Predictors saved to {args.cache / 'daily.csv'}")
            return
        if args.command == "prepare":
            import pandas as pd
            if args.output.exists():
                raise ValueError("Prepared output already exists; choose a fresh path.")
            target = read_csv(args.target)[[TARGET]]
            daily = normalize(pd.read_csv(args.daily))
            monthly = normalize(pd.read_csv(args.monthly)) if args.monthly else None
            frame = align_predictors(target, daily, monthly, args.macro_lag_days)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index_label="date")
            print(f"Prepared {len(frame)} observations: {args.output}")
            return
        if args.command == "demo":
            frame, source = synthetic_data(args.rows, args.seed), "synthetic"
        elif args.command == "run":
            frame, source = read_csv(args.input), "local_csv"
        else:
            frame, source = read_legacy(args.data_dir, args.macro_lag_days), "local_legacy_exports"
        output = args.output or default_output()
        metrics = run(frame, output, source=source, seed=args.seed,
                      lookback=args.lookback, rf_trials=args.rf_trials, neural=args.neural,
                      epochs=args.epochs, lstm_trials=args.lstm_trials,
                      macro_lag_days=getattr(args, "macro_lag_days", None))
        print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"Report: {output / 'report.md'}")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
