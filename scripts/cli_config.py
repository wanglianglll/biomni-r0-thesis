from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    suffix = p.suffix.lower()
    if suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config requires Python 3.11+.")
        return tomllib.loads(p.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config format: {suffix}. Use .json or .toml")


def apply_config_defaults(parser, argv=None):
    argv = list(argv or [])
    config_path = None
    remaining = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config requires a file path")
            config_path = argv[i + 1]
            i += 2
            continue
        remaining.append(arg)
        i += 1
    config = load_config(config_path)
    if config:
        parser.set_defaults(**config)
    return parser, remaining, config_path, config
