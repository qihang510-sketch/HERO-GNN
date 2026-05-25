from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.risk_card import format_risk_card
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a simple HERO-GNN explanation card.")
    parser.add_argument("--output", default="outputs/explanations/example.json", help="Output JSON path.")
    parser.add_argument("--node-id", type=int, default=0, help="Node id to explain.")
    parser.add_argument("--score", type=float, default=0.0, help="Risk score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card = format_risk_card(args.node_id, args.score, evidence_chain=[])
    write_json(args.output, card)
    print(f"Wrote explanation card to {args.output}")


if __name__ == "__main__":
    main()
