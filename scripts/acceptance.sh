#!/usr/bin/env bash
# Buyer day-one acceptance: env + demo + report schema.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -x "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$ROOT/.venv-buyer/bin/python" ]]; then
  PY="$ROOT/.venv-buyer/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
echo "using $PY"
"$PY" -c "import memaudit; print('memaudit', memaudit.__version__)"
"$PY" -m memaudit doctor --output-dir examples
echo "acceptance: doctor finished"
