from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRIC_NAMES = ["Macro-F1", "AUROC", "AUPRC"]


def ensure_artifact_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "tables_csv": root / "tables_csv",
        "tables_latex": root / "tables_latex",
        "figures_pdf": root / "figures_pdf",
        "figures_png": root / "figures_png",
        "figure_data": root / "figure_data",
        "reports": root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def write_latex(path: str | Path, frame: pd.DataFrame, highlight_metrics: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = frame.copy()
    if highlight_metrics:
        table = highlight_best_and_second(table)
    out.write_text(dataframe_to_latex(table), encoding="utf-8")
    return out


def write_markdown(path: str | Path, frame: pd.DataFrame) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dataframe_to_markdown(frame), encoding="utf-8")
    return out


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| status |\n|---|\n| missing |\n"
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_markdown_cell(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines) + "\n"


def dataframe_to_latex(frame: pd.DataFrame) -> str:
    if frame.empty:
        frame = pd.DataFrame([{"status": "--"}])
    columns = [str(col) for col in frame.columns]
    align = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(_latex_escape(col) for col in columns) + r" \\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        cells = [_latex_cell(row[col]) for col in frame.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def highlight_best_and_second(frame: pd.DataFrame, group_col: str = "dataset") -> pd.DataFrame:
    result = frame.copy()
    if group_col not in result:
        return result
    for metric in METRIC_NAMES:
        mean_col = f"{metric}_mean"
        display_col = f"{metric}_mean_std"
        if mean_col not in result or display_col not in result:
            continue
        for _, index in result.groupby(group_col, dropna=False).groups.items():
            values = pd.to_numeric(result.loc[index, mean_col], errors="coerce")
            finite = values.dropna().sort_values(ascending=False)
            if finite.empty:
                continue
            best_index = finite.index[0]
            second_index = finite.index[1] if len(finite) > 1 else None
            best_value = _missing_to_dash(result.at[best_index, display_col])
            result.at[best_index, display_col] = rf"\textbf{{{best_value}}}" if best_value != "--" else "--"
            if second_index is not None:
                second_value = _missing_to_dash(result.at[second_index, display_col])
                result.at[second_index, display_col] = rf"\underline{{{second_value}}}" if second_value != "--" else "--"
    return result


def write_unavailable_csv(path: str | Path, reason: str, **metadata: object) -> pd.DataFrame:
    row = {"status": "unavailable", "reason": reason, **metadata}
    frame = pd.DataFrame([row])
    write_csv(path, frame)
    return frame


def copy_if_exists(src: str | Path, dst: str | Path) -> bool:
    source = Path(src)
    target = Path(dst)
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def save_figure(fig, pdf_path: str | Path, png_path: str | Path) -> None:
    pdf = Path(pdf_path)
    png = Path(png_path)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=220)


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def mean_std_text(values: Iterable[float], scale: float = 100.0) -> str:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return "--"
    mean = float(array.mean()) * scale
    if array.size < 2:
        return f"{mean:.2f} +/- NA".replace("+/-", "±")
    std = float(array.std(ddof=1)) * scale
    return f"{mean:.2f} ± {std:.2f}"


def _latex_cell(value: object) -> str:
    text = _missing_to_dash(value)
    if text in {"--", "NA"}:
        return text
    if text.startswith("\\textbf{") or text.startswith("\\underline{"):
        return text
    return _latex_escape(text)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _markdown_cell(value: object) -> str:
    text = _missing_to_dash(value)
    return text.replace("|", "\\|")


def _missing_to_dash(value: object) -> str:
    if value is None or value is pd.NA:
        return "--"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "--"
    text = str(value)
    if text.lower() in {"nan", "none", "missing", "not_applicable", "model_not_applicable_to_dataset", "<na>"}:
        return "--"
    return text
