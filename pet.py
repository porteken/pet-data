"""Run the full PET data pipeline in sequence."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PIPELINE_STEPS: list[tuple[str, ...]] = [
    ("cites.py", "cities.py"),
    ("pull_weather.py",),
    ("pull_mrt.py",),
    ("combine.py",),
    ("calculate_pet.py",),
]


def resolve_script(root: Path, candidates: tuple[str, ...]) -> Path:
    """Return the first script that exists from `candidates`."""
    for script_name in candidates:
        script_path = root / script_name
        if script_path.exists():
            return script_path

    options = " or ".join(candidates)
    msg = f"Missing pipeline script: expected {options} in {root}"
    raise FileNotFoundError(msg)


def run_script(script_path: Path) -> None:
    """Execute one script with the same Python interpreter."""
    subprocess.run([sys.executable, str(script_path)], check=True)  # noqa: S603


def main() -> None:
    """Run pipeline scripts in order and fail fast on errors."""
    root = Path(__file__).resolve().parent
    total = len(PIPELINE_STEPS)

    for index, candidates in enumerate(PIPELINE_STEPS, start=1):
        script_path = resolve_script(root, candidates)
        logger.info("[%d/%d] Running %s...", index, total, script_path.name)
        run_script(script_path)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
