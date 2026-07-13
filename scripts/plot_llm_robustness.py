from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import read_csv_or_empty, write_csv  # noqa: E402
from scripts.run_llm_annotation_robustness import plot_robustness_figures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LLM annotation robustness figures from robustness_curve_data.csv.")
    parser.add_argument("--output_dir", default="outputs/submission_unified")
    parser.add_argument("--curve_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_llm_robustness(args.output_dir, curve_csv=args.curve_csv)


def plot_llm_robustness(output_dir: str | Path, curve_csv: str | Path | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    target = output_dir / "figure_data" / "robustness_curve_data.csv"
    source = Path(curve_csv) if curve_csv else target
    frame = read_csv_or_empty(source)
    if frame.empty:
        frame = pd.DataFrame([{"status": "unavailable", "reason": "robustness_curve_data.csv missing or empty"}])
    write_csv(target, frame)
    plot_robustness_figures(output_dir)
    _write_report(output_dir, "plot_llm_robustness", frame, target)
    return frame


def _write_report(output_dir: Path, name: str, frame: pd.DataFrame, data_path: Path) -> None:
    report = output_dir / "reports" / f"{name}_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    status = "unavailable" if frame.empty or frame.get("status", pd.Series(dtype=str)).astype(str).eq("unavailable").all() else "ok"
    report.write_text(f"# {name}\n\n- status: {status}\n- figure_data: `{data_path}`\n- rows: {len(frame)}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
