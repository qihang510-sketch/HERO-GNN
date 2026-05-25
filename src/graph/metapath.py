from __future__ import annotations


def parse_metapath(spec: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in spec.split("-") if part.strip())

