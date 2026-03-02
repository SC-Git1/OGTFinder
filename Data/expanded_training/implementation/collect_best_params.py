"""One-time utility: consolidate all DL best_params.json files into a single JSON.

Scans deep_learning/out/{model}/{model}_best_params.json and writes a single
all_best_params.json keyed by model name. This makes ensemble_models_fixed
self-contained on the cluster without needing deep_learning/out/ present.

Resolves paths relative to this script's location — safe to run from anywhere.

Usage:
    python collect_best_params.py
"""
from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # dl_ogt/
DL_OUT_ROOT = REPO_ROOT / "deep_learning" / "out"
OUTPUT_FILE = SCRIPT_DIR / "all_best_params.json"

# All models referenced by any reg_dlstack_*.py ensemble config
ALL_REFERENCED = {
    "attention_mlp", "baseline_mlp", "danets", "ft_transformer", "gandalf",
    "gated_mlp", "grownet", "hopular", "lassonet", "node", "realmlp",
    "resnet_mlp", "rtdl_resnet", "saint", "snn", "sparse_mlp", "tabm",
    "tabnet", "tabnet_inspired", "tabr", "vime", "wide_deep",
}


def main() -> None:
    if not DL_OUT_ROOT.exists():
        print(f"ERROR: deep_learning output directory not found: {DL_OUT_ROOT}")
        raise SystemExit(1)

    consolidated: dict[str, dict] = {}
    found: list[str] = []

    for model_dir in sorted(DL_OUT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name
        params_file = model_dir / f"{model_key}_best_params.json"
        if params_file.exists():
            with open(params_file) as f:
                consolidated[model_key] = json.load(f)
            found.append(model_key)

    missing = sorted(k for k in ALL_REFERENCED if k not in consolidated)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(consolidated, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(consolidated)} model params to: {OUTPUT_FILE}")
    print(f"\nFound ({len(found)}): {', '.join(found)}")
    if missing:
        print(f"\nMissing ({len(missing)}): {', '.join(missing)}")
        print("  -> Ensemble configs referencing these models will still fail at runtime.")


if __name__ == "__main__":
    main()
