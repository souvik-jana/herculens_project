#!/usr/bin/env bash
# Rebuild the /tmp/venv used by all comparison-analysis runs in the Linux
# sandbox (aarch64, python 3.10, no root, github release assets blocked).
#
# The repo's own uv.lock targets python >=3.13 + jax cuda and cannot be used
# here; this env pins the closest CPU stack that gwemfish runs on:
#   herculens==0.2.3 (pypi) + jax<0.7  (+ Sersic compat shim in
#   shared/system_config.py::apply_herculens_compat)
#
# Run each line separately if the 45 s per-call limit bites; uv caches make
# reruns cheap. Usage:  bash shared/setup_sandbox_env.sh

set -e
export UV_CACHE_DIR=/tmp/uv-cache

uv venv /tmp/venv --python /usr/bin/python3.10
uv pip install --python /tmp/venv/bin/python "jax<0.7" numpyro optax
uv pip install --python /tmp/venv/bin/python matplotlib astropy corner \
    nautilus-sampler scienceplots joblib
uv pip install --python /tmp/venv/bin/python "herculens==0.2.3" helens utax \
    jaxtronomy lenstronomy jaxopt blackjax
uv pip install --python /tmp/venv/bin/python autolens

# Smoke test (from repo root):
#   PYTHONPATH=src:comparison-analysis /tmp/venv/bin/python -c \
#     "from shared.system_config import setup_jax, build_em_ctx; setup_jax(); build_em_ctx(); print('ok')"
