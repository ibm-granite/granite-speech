#!/usr/bin/env bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
python_bin="${PYTHON:-python3}"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

cd "${repo_root}"
uv build --out-dir "${tmp_dir}/dist"

uv venv "${tmp_dir}/venv" --python "${python_bin}"
venv_python="${tmp_dir}/venv/bin/python"

uv pip install --python "${venv_python}" "${tmp_dir}"/dist/granite_speech-*.whl pytest

cp -R "${repo_root}/tests" "${tmp_dir}/tests"
cp "${repo_root}/pyproject.toml" "${tmp_dir}/pyproject.toml"
cd "${tmp_dir}"

"${venv_python}" -m pytest tests -m "not real_weights"

run_real_weights="${GRANITE_SPEECH_RUN_REAL_WEIGHTS:-}"
case "${run_real_weights}" in
  1|true|TRUE|yes|YES|on|ON)
    "${venv_python}" -m pytest tests/test_real_weights_smoke.py -m real_weights
    ;;
  *)
    echo "Skipping real-weights smoke; set GRANITE_SPEECH_RUN_REAL_WEIGHTS=1 to enable it."
    ;;
esac
