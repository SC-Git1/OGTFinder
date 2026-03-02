#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

shopt -s nullglob
SLURM_FILES=("${SCRIPT_DIR}"/*.slurm)
shopt -u nullglob

if [[ ${#SLURM_FILES[@]} -eq 0 ]]; then
  echo "No .slurm files found in ${SCRIPT_DIR}"
  exit 1
fi

echo "Submitting ${#SLURM_FILES[@]} SLURM files from ${SCRIPT_DIR}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry run enabled; no jobs will be submitted."
fi

cd "${REPO_ROOT}"

for slurm_file in "${SLURM_FILES[@]}"; do
  rel_path="${slurm_file#${REPO_ROOT}/}"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[DRY-RUN] sbatch ${rel_path}"
  else
    echo "sbatch ${rel_path}"
    sbatch "${slurm_file}"
  fi
done

