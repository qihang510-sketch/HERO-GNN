from __future__ import annotations

import logging


def get_logger(name: str = "hero_gnn") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)

