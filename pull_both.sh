#!/usr/bin/env bash
# Convenience wrapper: run the full pipeline (fetch -> process -> load -> views)
# for both PET and wetbulb. This is the default behaviour of pull_all.sh; the
# wrapper exists for naming symmetry with pull_pet.sh / pull_wetbulb.sh.
# Equivalent to: ./pull_all.sh --products both
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/pull_all.sh" --products both "$@"
