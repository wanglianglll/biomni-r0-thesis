#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path("/root/autodl-tmp/Biomni-main/models/Qwen3.5-27B")
REQUIRED = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]
REQUIRED += [f"model.safetensors-{i:05d}-of-00011.safetensors" for i in range(1, 12)]


def main() -> int:
    missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
    incomplete = sorted(str(p.relative_to(MODEL_DIR)) for p in MODEL_DIR.glob("**/*.part"))
    incomplete += sorted(str(p.relative_to(MODEL_DIR)) for p in MODEL_DIR.glob("**/*.incomplete"))
    size_gb = sum(p.stat().st_size for p in MODEL_DIR.glob("**/*") if p.is_file()) / 1024**3
    report = {
        "model_dir": str(MODEL_DIR),
        "size_gb": round(size_gb, 2),
        "required_count": len(REQUIRED),
        "missing_count": len(missing),
        "missing": missing,
        "incomplete_count": len(incomplete),
        "incomplete_examples": incomplete[:20],
        "ready": not missing and not incomplete,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
