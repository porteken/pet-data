#!/usr/bin/env bash
# Convenience wrapper: run the full pipeline (fetch -> process -> load -> views)
# for the PET product only. The wetbulb table is left untouched.
# Equivalent to: PRODUCTS=pet ./pull_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRODUCTS=pet exec "${HERE}/pull_all.sh" "$@"
