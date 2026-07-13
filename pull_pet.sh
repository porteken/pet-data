#!/usr/bin/env bash
# Convenience wrapper: run the full pipeline (fetch -> process -> load -> views).
# Equivalent to: ./pull_all.sh --products pet
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/pull_all.sh" --products pet "$@"
