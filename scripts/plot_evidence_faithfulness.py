from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import read_csv_or_empty, write_csv  # noqa: E402
from scripts.run_evidence_faithfulness import plot_faithfulness  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evidence-chain faithfulness from faithfulness_bar_data.csv.")
    parser.add_argument("--output_dir", default="outputs/submission_unified")
    parser.add_argument("--data_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_evidence_faithfulness(args.output_dir, data_csv=args.data_csv)


def plot_evidence_faithfulness(output_dir: str | Path, data_csv: str | Path | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    target = output_dir / "figure_data" / "faithfulness_bar_data.csv"
    source = Path(data_csv) if data_csv else target
    frame = read_csv_or_empty(source)
    if frame.empty:
        frame = pd.DataFrame([{"status": "unavailable", "reason": "faithfulness_bar_data.csv missing or empty"}])
    write_csv(target, frame)
    plot_faithfulness(output_dir)
    _write_report(output_dir, "plot_evidence_faithfulness", frame, target)
    return frame


def _write_report(output_dir: Path, name: str, frame: pd.DataFrame, data_path: Path) -> None:
    report = output_dir / "reports" / f"{name}_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    status = "unavailable" if frame.empty or frame.get("status", pd.Series(dtype=str)).astype(str).eq("unavailable").all() else "ok"
    report.write_text(f"# {name}\n\n- status: {status}\n- figure_data: `{data_path}`\n- rows: {len(frame)}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
