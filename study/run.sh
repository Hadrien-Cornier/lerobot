#!/usr/bin/env bash
# Run the pinned teacher baseline. The parent session owns job authorization.
set -euo pipefail

LINGBOT_STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_REPO_DIR="$(dirname "$LINGBOT_STUDY_DIR")"
LINGBOT_WORK_DIR="${LINGBOT_PROFILE_WORK_DIR:-$LINGBOT_STUDY_DIR/runtime}"

if [[ "${LINGBOT_DRY_RUN:-0}" == "1" ]]; then
    python3 "$LINGBOT_STUDY_DIR/prepare.py" --work-root "$LINGBOT_WORK_DIR" --dry-run
    exit 0
fi

cd "$LINGBOT_REPO_DIR"
mkdir -p "$LINGBOT_WORK_DIR"
if [[ -e "$LINGBOT_WORK_DIR/results" ]]; then
    printf '%s\n' 'Use a new work directory. The results directory already exists.' >&2
    exit 64
fi
LINGBOT_PYTHON="$LINGBOT_WORK_DIR/venv/bin/python"

# Always print any complete or partial result before an optional upload.
lingbot_finish() {
    LINGBOT_EXIT_CODE=$?
    trap - EXIT
    if [[ -f "$LINGBOT_WORK_DIR/results/results.json" ]]; then
        printf '%s\n' 'LINGBOT_PROFILE_RESULTS_JSON_BEGIN'
        cat "$LINGBOT_WORK_DIR/results/results.json"
        printf '%s\n' 'LINGBOT_PROFILE_RESULTS_JSON_END'
    else
        printf '%s\n' 'LINGBOT_PROFILE_RESULTS_JSON_MISSING'
    fi
    if [[ -x "$LINGBOT_PYTHON" ]]; then
        "$LINGBOT_PYTHON" "$LINGBOT_STUDY_DIR/prepare.py" --work-root "$LINGBOT_WORK_DIR" --upload || true
    else
        printf '%s\n' 'ARTIFACT_UPLOAD_SKIPPED=runtime_not_ready'
    fi
    exit "$LINGBOT_EXIT_CODE"
}
trap lingbot_finish EXIT
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Do not use shell tracing. HF_TOKEN remains in the environment.
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --disable-pip-version-check "uv==0.11.23"
fi
uv python install 3.12
uv venv --python 3.12 "$LINGBOT_WORK_DIR/venv"
LINGBOT_PYTHON="$LINGBOT_WORK_DIR/venv/bin/python"
uv pip install --python "$LINGBOT_PYTHON" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
    "transformers==5.5.4" "huggingface-hub==1.30.0" \
    "numpy==2.2.6" "scipy==1.18.1" "av==15.1.0" "PyYAML==6.0.3" \
    ".[training,lingbot_vla_v2]"
uv pip freeze --python "$LINGBOT_PYTHON" > "$LINGBOT_WORK_DIR/environment-freeze.txt"

"$LINGBOT_PYTHON" - <<'PY'
import importlib.metadata
import json
import sys
import torch
assert sys.version_info >= (3, 12)
expected = {"torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.5.4", "huggingface-hub": "1.30.0"}
versions = {name: importlib.metadata.version(name) for name in expected}
for name, value in expected.items():
    assert versions[name].split("+")[0] == value, (name, versions[name], value)
assert torch.cuda.is_available(), "The baseline requires a CUDA device"
print(json.dumps({"runtime_versions": versions, "python": sys.version, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}))
PY

"$LINGBOT_PYTHON" "$LINGBOT_STUDY_DIR/prepare.py" --work-root "$LINGBOT_WORK_DIR"
"$LINGBOT_PYTHON" "$LINGBOT_STUDY_DIR/prepare_full.py" --work-root "$LINGBOT_WORK_DIR"
"$LINGBOT_PYTHON" "$LINGBOT_STUDY_DIR/run_full_matrix.py" \
    --config "$LINGBOT_WORK_DIR/full-policy-runtime-template.json"
