#!/usr/bin/env bash
# run from R_compare/
# usage:
#   COV=HC1 SCALE=response bash run.sh > output.txt 2>&1
#   COV=none bash run.sh > output.txt 2>&1

set -euo pipefail

# Convenience wrapper to run R then Julia

RSCPT="r/r_pipeline.R"
JLSCPT="jl/jl_pipeline.jl"

# Defaults (can be overridden via env)
COV_DEFAULT="none"         # HC3 | HC1 | HC0 | none
SCALE_DEFAULT="response"   # response | link

COV="${COV:-$COV_DEFAULT}"
SCALE="${SCALE:-$SCALE_DEFAULT}"

echo "[config] COV=${COV} SCALE=${SCALE}"

echo "[1/2] Running R pipeline"
if [[ "${COV}" == "none" || -z "${COV}" ]]; then
  Rscript "$RSCPT" --type "${SCALE}"
else
  Rscript "$RSCPT" --vcov "${COV}" --type "${SCALE}"
fi

echo "[2/2] Running Julia pipeline"
if [[ "${COV}" == "none" || -z "${COV}" ]]; then
  julia --project -t auto "$JLSCPT" --scale "${SCALE}"
else
  julia --project -t auto "$JLSCPT" --cov "${COV}" --scale "${SCALE}"
fi

echo "Done. See results_r and results_julia."
