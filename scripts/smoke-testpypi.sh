#!/usr/bin/env bash
set -euo pipefail

# Run the real-weights smoke tests (base + plus) against the granite-speech
# build published to Test PyPI, rather than the local source tree. Mirrors the
# smoke-base / smoke-plus GitHub workflows but installs the *published* artifact
# so a release can be verified end-to-end before it goes to real PyPI.
#
# The package comes from Test PyPI; its runtime deps (torch, soundfile, ...) are
# not mirrored there, so they resolve from real PyPI via --extra-index-url.
#
# Requirements:
#   - uv on PATH
#   - llama-cli on PATH (llama.cpp build >= 9850 for the plus smoke), or set
#     GRANITE_SPEECH_SMOKE_LLAMA_CPP_BINARY / GRANITE_SPEECH_PLUS_SMOKE_LLAMA_CPP_BINARY
#
# Usage:
#   scripts/smoke-testpypi.sh                 # latest version on Test PyPI
#   scripts/smoke-testpypi.sh 0.0.2           # pin an exact version
#   GRANITE_SPEECH_SMOKE_SUITE=base scripts/smoke-testpypi.sh   # base only

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
python_bin="${PYTHON:-python3}"
version="${1:-}"
suite="${GRANITE_SPEECH_SMOKE_SUITE:-both}"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

spec="granite-speech"
if [ -n "${version}" ]; then
  spec="granite-speech==${version}"
fi

uv venv "${tmp_dir}/venv" --python "${python_bin}"
venv_python="${tmp_dir}/venv/bin/python"

# Package from Test PyPI, dependencies from real PyPI. --index-strategy
# unsafe-best-match lets uv consider both indexes when resolving deps rather
# than pinning everything to the first index that contains the top-level name.
uv pip install \
  --python "${venv_python}" \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  "${spec}" pytest huggingface_hub

# Copy tests + pyproject into the temp dir so the installed package (not the
# repo's ./granite_speech) is what gets imported, and so pytest picks up the
# real_weights marker config.
cp -R "${repo_root}/tests" "${tmp_dir}/tests"
cp "${repo_root}/pyproject.toml" "${tmp_dir}/pyproject.toml"
cd "${tmp_dir}"

# Fail loudly if we somehow imported the repo source instead of the Test PyPI wheel.
resolved="$("${venv_python}" -c 'import granite_speech; print(granite_speech.__version__, granite_speech.__file__)')"
echo "Resolved granite_speech: ${resolved}"
case "${resolved}" in
  *"${repo_root}"*)
    echo "error: imported granite_speech from the repo source, not the Test PyPI install" >&2
    exit 1
    ;;
esac

run_base() {
  echo "== base real-weights smoke =="
  "${venv_python}" -m pytest tests/test_real_weights_smoke.py -m real_weights -v
}

run_plus() {
  echo "== plus real-weights smoke =="
  "${venv_python}" -m pytest tests/test_real_weights_smoke_plus.py -m real_weights -v
}

case "${suite}" in
  base) run_base ;;
  plus) run_plus ;;
  both) run_base; run_plus ;;
  *)
    echo "error: GRANITE_SPEECH_SMOKE_SUITE must be base, plus, or both (got '${suite}')" >&2
    exit 1
    ;;
esac
