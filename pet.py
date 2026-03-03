"""Run the full PET data pipeline in sequence."""

from __future__ import annotations

import logging
import runpy
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PIPELINE_STEPS: list[str] = [
    "cities.py",
    "pull_weather.py",
    "pull_mrt.py",
    "combine.py",
    "calculate_pet.py",
]


def run_script(script_name: str, step_num: int, total_steps: int) -> None:
    """Execute a Python script using the current interpreter."""
    root = Path(__file__).resolve().parent
    script_path = root / script_name

    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        raise FileNotFoundError(script_name)

    logger.info("=" * 50)
    logger.info("STEP %d/%d: %s", step_num, total_steps, script_name)
    logger.info("=" * 50)

    start_time = time.time()

    original_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path)]
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        if exc.code in (None, 0):
            return
        logger.exception("Pipeline failed during step: %s", script_name)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Pipeline failed during step: %s", script_name)
        raise SystemExit(1) from exc
    finally:
        sys.argv = original_argv

    elapsed = time.time() - start_time
    logger.info("Step %s finished in %.2f seconds.\n", script_name, elapsed)


def main() -> None:
    """Run pipeline scripts in order and fail fast on errors."""
    total_start = time.time()
    total_steps = len(PIPELINE_STEPS)

    for index, script_name in enumerate(PIPELINE_STEPS, start=1):
        run_script(script_name, index, total_steps)

    total_elapsed = time.time() - total_start
    logger.info("=" * 50)
    logger.info("Pipeline complete. Total time: %.2f seconds.", total_elapsed)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
