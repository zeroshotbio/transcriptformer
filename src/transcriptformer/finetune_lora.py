"""Entry point for ``python -m transcriptformer.finetune_lora``."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Execute the repository-level ``finetune_lora.py`` script."""
    script = Path(__file__).resolve().parents[2] / "finetune_lora.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":  # pragma: no cover - CLI gateway
    main()
