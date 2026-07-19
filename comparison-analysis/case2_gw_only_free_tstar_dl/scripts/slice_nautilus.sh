#!/usr/bin/env bash
# One checkpoint-safe 45-s slice of a Case-2f nautilus stage.
#
#   bash slice_nautilus.sh naut-helens|lenstronomy [seconds]
#
# The sandbox kills each call at 45 s, and SIGTERM can land mid-HDF5-write,
# corrupting the nautilus checkpoint ("bad object header version number" —
# observed 2026-07-18, cost a 35k-call run). Guard: before each slice,
# validate the checkpoint with h5py; good -> refresh .bak, corrupt -> restore
# .bak. Worst case loses one slice of progress.
#
# Regime- and machine-portable: REPO is derived from this script's own
# location (the sandbox mount name changes between sessions), and the
# checkpoint name tracks CA2_REGIME/CA2_BUDGET exactly like
# common_case2f.case_paths(), so `CA2_REGIME=scan_opt bash slice_nautilus.sh ...`
# resumes the scan_opt checkpoint and never touches the precise one.
# Defaults are unchanged (precise/full), so the Case-2f precise reproduce
# commands still do what they did.
set -u
STAGE="$1"
SECS="${2:-40}"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$CASE_DIR/../.." && pwd)"
PY=/tmp/venv/bin/python
CASE=comparison-analysis/case2_gw_only_free_tstar_dl
REGIME="${CA2_REGIME:-precise}"
BUDGET="${CA2_BUDGET:-full}"
TAG=$([ "$STAGE" = "naut-helens" ] && echo helens || echo lenstronomy)
CK="${TMPDIR:-/tmp}/ca2f_${REGIME}_${BUDGET}_${TAG}.hdf5"
LOG="/tmp/ca2f_${REGIME}_${TAG}.log"

# The JAX persistent compile cache can also be corrupted by the 45-s SIGTERM
# landing mid-write; a poisoned kernel made the jitted likelihood return
# garbage and trip the parity gate (observed: "max relative diff = 7.8e+288"
# then "1.000e+00" on identical, deterministic inputs). Recompiling costs a
# few seconds; a corrupt cache costs the whole slice. Always start clean.
rm -rf "${TMPDIR:-/tmp}/jax_cache"

if [ -f "$CK" ]; then
    if "$PY" -c "import h5py; h5py.File('$CK','r').close()" 2>/dev/null; then
        cp "$CK" "$CK.bak"
    elif [ -f "$CK.bak" ]; then
        echo "checkpoint corrupt -> restoring $CK.bak"
        cp "$CK.bak" "$CK"
    else
        echo "checkpoint corrupt, no backup -> starting fresh"
        rm -f "$CK"
    fi
fi

cd "$REPO" && timeout "$SECS" env PYTHONUNBUFFERED=1 CA2_REGIME="$REGIME" \
    PYTHONPATH=src:comparison-analysis:$CASE/scripts \
    "$PY" "$CASE/scripts/run_case2f.py" "$STAGE" > "$LOG" 2>&1
rc=$?
tr '\r' '\n' < "$LOG" | grep -E "^(Sampling|Computing|Bounding)" | tail -1
if [ $rc -eq 0 ]; then
    echo "SLICE FINISHED (stage completed)"
    tr '\r' '\n' < "$LOG" | tail -4
else
    echo "slice cut at ${SECS}s (rc=$rc)"
fi
