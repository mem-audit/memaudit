#!/usr/bin/env bash
# memaudit publish helper — build, check, and upload to PyPI with guardrails.
#
# Usage:
#   ./scripts/publish.sh          # real PyPI
#   ./scripts/publish.sh --test   # TestPyPI rehearsal
#
# Guardrails:
#   * refuses to run outside the repo root
#   * refuses if this exact version already exists on the target index
#   * always rebuilds dist/ from scratch and runs `twine check`
#   * requires you to type the version string to confirm before uploading
#
# Credentials: twine prompts (username: __token__, password: pypi-...), or export
# TWINE_USERNAME=__token__ and TWINE_PASSWORD before running. Never commit tokens.

set -euo pipefail

REPO="pypi"
INDEX_JSON="https://pypi.org/pypi"
INDEX_NAME="PyPI"
if [[ "${1:-}" == "--test" ]]; then
  REPO="testpypi"
  INDEX_JSON="https://test.pypi.org/pypi"
  INDEX_NAME="TestPyPI"
elif [[ -n "${1:-}" ]]; then
  echo "error: unknown argument '$1' (only --test is supported)" >&2
  exit 2
fi

# --- locate repo root and sanity-check the project ---------------------------
cd "$(dirname "$0")/.."
if [[ ! -f pyproject.toml ]]; then
  echo "error: pyproject.toml not found — run from the memaudit repo" >&2
  exit 1
fi

NAME="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["name"])')"
VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
if [[ "$NAME" != "memaudit" ]]; then
  echo "error: project name is '$NAME', expected 'memaudit' — wrong directory?" >&2
  exit 1
fi
echo "==> $NAME $VERSION  →  $INDEX_NAME"

# --- tooling ------------------------------------------------------------------
for mod in build twine; do
  python3 -c "import $mod" 2>/dev/null || {
    echo "error: python module '$mod' missing. Fix: python3 -m pip install --upgrade build twine" >&2
    exit 1
  }
done

# --- refuse if the version is already published --------------------------------
HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' "$INDEX_JSON/$NAME/$VERSION/json" || echo 000)"
case "$HTTP_CODE" in
  200)
    echo "error: $NAME $VERSION already exists on $INDEX_NAME — bump the version in pyproject.toml first." >&2
    exit 1
    ;;
  404)
    echo "==> $NAME $VERSION not on $INDEX_NAME yet — ok to publish."
    ;;
  000)
    echo "error: could not reach $INDEX_NAME to verify the version (network?). Refusing to guess." >&2
    exit 1
    ;;
  *)
    echo "warning: unexpected HTTP $HTTP_CODE checking $INDEX_NAME; continuing (first-ever upload returns 404 only once the project page exists)."
    ;;
esac

# --- clean build + check --------------------------------------------------------
if [[ -d dist ]]; then
  echo "==> removing stale dist/"
  rm -rf dist
fi
echo "==> building sdist + wheel"
python3 -m build
echo "==> twine check"
python3 -m twine check dist/*

echo
echo "==> about to upload to $INDEX_NAME:"
ls -l dist/
echo
read -r -p "Type the version ($VERSION) to confirm upload, anything else aborts: " CONFIRM
if [[ "$CONFIRM" != "$VERSION" ]]; then
  echo "aborted — nothing uploaded."
  exit 1
fi

python3 -m twine upload --repository "$REPO" dist/*
echo "==> done. Verify in a clean venv:  pip install $NAME==$VERSION"
