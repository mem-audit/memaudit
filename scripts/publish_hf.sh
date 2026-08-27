#!/usr/bin/env bash
# Publish memaudit Hugging Face org card + Gradio demo Space.
#
# Requires:
#   export HF_TOKEN=hf_...   # write access to organization `memaudit`
#   pip install -U "huggingface_hub>=0.24"
#
# Creates / updates:
#   - Space memaudit/README          (organization card, sdk=static)
#   - Space memaudit/memaudit-demo   (Gradio report viewer, CPU basic)
#
# Usage (from repo root):
#   ./scripts/publish_hf.sh

set -euo pipefail

cd "$(dirname "$0")/.."
if [[ ! -f pyproject.toml ]]; then
  echo "error: pyproject.toml not found — run from the memaudit repo" >&2
  exit 1
fi
if [[ ! -d hf/space || ! -f hf/space/app.py ]]; then
  echo "error: hf/space/ missing — expected Gradio Space assets" >&2
  exit 1
fi
if [[ ! -f hf/org/README.md ]]; then
  echo "error: hf/org/README.md missing — expected org card" >&2
  exit 1
fi

TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "${TOKEN}" ]]; then
  echo "error: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set." >&2
  echo "Ask the founder for a Hugging Face token with write access to org memaudit," >&2
  echo "then:  export HF_TOKEN=hf_... && ./scripts/publish_hf.sh" >&2
  exit 1
fi
export HF_TOKEN="$TOKEN"
export HUGGING_FACE_HUB_TOKEN="$TOKEN"

python3 - <<'PY'
import sys
try:
    import huggingface_hub
except ImportError:
    print("error: huggingface_hub missing. Fix: python3 -m pip install -U 'huggingface_hub>=0.24'", file=sys.stderr)
    sys.exit(1)
print(f"==> huggingface_hub {huggingface_hub.__version__}")
PY

ORG="memaudit"
DEMO_SPACE="${ORG}/memaudit-demo"
ORG_CARD="${ORG}/README"

echo "==> whoami"
python3 - <<'PY'
from huggingface_hub import whoami
info = whoami()
name = info.get("name") or info.get("fullname") or "?"
orgs = [o.get("name") for o in (info.get("orgs") or []) if isinstance(o, dict)]
print(f"    user: {name}")
print(f"    orgs: {', '.join(orgs) if orgs else '(none listed)'}")
if "memaudit" not in orgs and name != "memaudit":
    print("warning: org 'memaudit' not listed on this token — create_repo may fail with 403.")
PY

echo "==> create/ensure org card Space ${ORG_CARD}"
python3 - <<'PY'
from huggingface_hub import create_repo, HfApi
api = HfApi()
repo_id = "memaudit/README"
try:
    create_repo(repo_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    print(f"    ok: {repo_id}")
except Exception as e:
    print(f"error creating {repo_id}: {e}")
    raise
PY

echo "==> upload org card README"
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path=str(Path("hf/org").resolve()),
    repo_id="memaudit/README",
    repo_type="space",
    commit_message="Update memaudit organization card",
)
print("    uploaded hf/org → memaudit/README")
PY

echo "==> create/ensure demo Space ${DEMO_SPACE}"
python3 - <<'PY'
from huggingface_hub import create_repo
repo_id = "memaudit/memaudit-demo"
try:
    create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    print(f"    ok: {repo_id}")
except Exception as e:
    print(f"error creating {repo_id}: {e}")
    raise
PY

echo "==> upload Gradio Space (hf/space → ${DEMO_SPACE})"
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path=str(Path("hf/space").resolve()),
    repo_id="memaudit/memaudit-demo",
    repo_type="space",
    commit_message="Publish memaudit Gradio report demo Space",
)
print("    uploaded hf/space → memaudit/memaudit-demo")
PY

echo "==> request CPU basic hardware (best-effort)"
python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo_id = "memaudit/memaudit-demo"
# Hardware APIs vary by hub version; try common entry points.
tried = []
for fn_name, kwargs in [
    ("request_space_hardware", {"repo_id": repo_id, "hardware": "cpu-basic"}),
    ("set_space_sleep_time", None),  # placeholder skip
]:
    fn = getattr(api, fn_name, None)
    if fn is None or kwargs is None:
        continue
    tried.append(fn_name)
    try:
        fn(**kwargs)
        print(f"    {fn_name}({kwargs}) → ok")
        break
    except Exception as e:
        print(f"    {fn_name} skipped/failed: {e}")
else:
    if not tried:
        print("    no request_space_hardware on this hub version — Space defaults to CPU basic for Gradio.")
    else:
        print("    hardware request failed; confirm CPU basic in Space Settings if needed.")
PY

echo
echo "==> live URLs"
echo "    Org:   https://huggingface.co/memaudit"
echo "    Card:  https://huggingface.co/spaces/memaudit/README"
echo "    Demo:  https://huggingface.co/spaces/memaudit/memaudit-demo"
echo
echo "==> polling Space runtime status (up to ~90s)"
python3 - <<'PY'
import time
from huggingface_hub import HfApi

api = HfApi()
repo_id = "memaudit/memaudit-demo"
deadline = time.time() + 90
last = None
while time.time() < deadline:
    try:
        info = api.space_info(repo_id)
        runtime = getattr(info, "runtime", None)
        stage = None
        if runtime is not None:
            stage = getattr(runtime, "stage", None) or getattr(runtime, "status", None)
            if stage is None and isinstance(runtime, dict):
                stage = runtime.get("stage") or runtime.get("status")
        last = stage or "unknown"
        print(f"    runtime: {last}")
        if str(last).upper() in {"RUNNING", "LIVE", "RUNTIME_STAGE_RUNNING"}:
            break
        if str(last).upper() in {"CONFIG_ERROR", "BUILD_ERROR", "RUNTIME_ERROR", "FAILED"}:
            print("    Space reported an error stage — check the Space build logs.")
            break
    except Exception as e:
        print(f"    poll error: {e}")
        last = str(e)
    time.sleep(8)
print(f"    final polled status: {last}")
print("    open: https://huggingface.co/spaces/memaudit/memaudit-demo")
PY

echo "==> done"
