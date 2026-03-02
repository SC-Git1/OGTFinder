"""
Runner: train all reg_* models and generate combined W&B plots.

Usage:
    python reg_all_models.py <train_data_path> <test_data_path> [--wandb]

The runner:
- Trains each reg_*.py script sequentially with shared W&B group.
- Fetches W&B histories for the group and saves combined train/val/test plots
  for loss, R², and MAE to ./outputs/combined.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt

from utils import init_wandb_run, log_wandb_images, WANDB_AVAILABLE, TQDM_AVAILABLE

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT_DIR = os.path.dirname(__file__)
COMBINED_DIR = os.path.join(ROOT_DIR, "outputs", "combined")

METRIC_SPECS = {
    "loss": {
        "train": "loss",
        "val": "val_loss",
        "test": "test_loss",
        "label": "Loss",
    },
    "r2": {
        "train": "r2",
        "val": "val_r2",
        "test": "test_r2",
        "label": "R²",
    },
    "mae": {
        "train": "mean_absolute_error",
        "val": "val_mean_absolute_error",
        "test": "test_mean_absolute_error",
        "label": "MAE",
    },
}


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_model_name(name: str) -> str:
    model = name.strip()
    if model.endswith(".py"):
        model = model[:-3]
    if model.startswith("reg_"):
        model = model[4:]
    return model


def discover_model_scripts(models: list[str] | None, skip_models: list[str] | None) -> list[str]:
    scripts = sorted(glob.glob(os.path.join(ROOT_DIR, "reg_*.py")))
    scripts = [path for path in scripts if os.path.basename(path) != "reg_all_models.py"]

    if not scripts:
        return []

    requested = {normalize_model_name(m) for m in (models or [])}
    skipped = {normalize_model_name(m) for m in (skip_models or [])}

    selected = []
    for path in scripts:
        model = normalize_model_name(os.path.basename(path))
        if requested and model not in requested:
            continue
        if model in skipped:
            continue
        selected.append(path)

    return selected


def ensure_wandb_login():
    if not WANDB_AVAILABLE or wandb is None:
        return
    api_key = os.getenv("KEY_WB_API")
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception as exc:  # pragma: no cover
            logging.warning("wandb login failed: %s", exc)


def run_model(script_path: str, train_path: str, test_path: str, args) -> None:
    model_name = normalize_model_name(os.path.basename(script_path))
    cmd = [sys.executable, script_path, train_path, test_path]

    # Add weights_path if provided
    if args.weights_path:
        cmd += ["--weights_path", args.weights_path]

    # Add experiment_name if provided
    if args.experiment_name:
        cmd += ["--experiment_name", args.experiment_name]

    if args.wandb:
        cmd += [
            "--wandb",
            "--wandb_group",
            args.wandb_group,
            "--wandb_name",
            model_name,
        ]
        if args.wandb_tags:
            cmd += ["--wandb_tags", args.wandb_tags]

    env = os.environ.copy()
    if args.wandb:
        env.setdefault("WANDB_PROJECT", "ogtfinder-dl")

    logging.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def fetch_group_histories(project: str, group: str, entity: str | None) -> dict[str, "pandas.DataFrame"]:
    if not WANDB_AVAILABLE or wandb is None:
        logging.warning("wandb not available; skipping combined plots")
        return {}

    ensure_wandb_login()
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    metric_keys = set()
    for spec in METRIC_SPECS.values():
        metric_keys.update([spec["train"], spec["val"], spec["test"]])

    runs = api.runs(project_path, filters={"group": group})
    histories: dict[str, "pandas.DataFrame"] = {}

    for run in runs:
        history = run.history(keys=list(metric_keys), samples=10000, pandas=True)
        if history is None or history.empty:
            continue

        name = run.name or run.id
        if name in histories:
            name = f"{name}-{run.id[:6]}"
        histories[name] = history

    return histories


def plot_combined_metric(histories: dict, metric: str, output_dir: str, group: str) -> str | None:
    if not histories:
        return None

    spec = METRIC_SPECS[metric]
    train_key = spec["train"]
    val_key = spec["val"]
    test_key = spec["test"]
    label = spec["label"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    splits = [(train_key, "Train"), (val_key, "Validation"), (test_key, "Test")]

    for model_name, df in histories.items():
        if "_step" not in df.columns:
            continue
        for ax, (key, split) in zip(axes, splits):
            if key not in df.columns:
                continue
            series = df[["_step", key]].dropna()
            if series.empty:
                continue
            ax.plot(series["_step"], series[key], label=model_name, linewidth=1.2)
            ax.set_title(split)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)

    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, name in zip(h, l):
            if name not in labels:
                handles.append(handle)
                labels.append(name)

    fig.suptitle(f"{label} (Train/Val/Test) — Group: {group}")
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"combined_{metric}_train_val_test.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_combined_metrics(histories: dict, output_dir: str, group: str) -> dict[str, str]:
    paths = {}
    for metric in METRIC_SPECS.keys():
        path = plot_combined_metric(histories, metric, output_dir, group)
        if path:
            paths[f"combined_{metric}"] = path
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Run all reg_* models sequentially with shared W&B group"
    )
    parser.add_argument("train_path", type=str, help="Path to training data file")
    parser.add_argument("test_path", type=str, help="Path to test data file")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to sample weights JSON file")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for the experiment")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to run")
    parser.add_argument("--skip_models", type=str, default=None, help="Comma-separated model names to skip")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity for API queries")
    parser.add_argument("--skip_combined", action="store_true", help="Skip combined plots")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    models = parse_csv(args.models)
    skip_models = parse_csv(args.skip_models)

    if args.wandb and not args.wandb_group:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        args.wandb_group = f"ogt_all_{timestamp}"

    scripts = discover_model_scripts(models, skip_models)
    if not scripts:
        logging.warning("No model scripts found to run")
        return

    # Progress bar for overall model training
    if tqdm is not None:
        model_pbar = tqdm(scripts, desc="Training Models", unit="model")
    else:
        model_pbar = scripts

    for script_path in model_pbar:
        model_name = normalize_model_name(os.path.basename(script_path))
        if tqdm is not None and hasattr(model_pbar, 'set_postfix_str'):
            model_pbar.set_postfix_str(f"Current: {model_name}")
        run_model(script_path, args.train_path, args.test_path, args)

    if args.wandb and not args.skip_combined:
        project = os.getenv("WANDB_PROJECT", "ogtfinder-dl")
        group = args.wandb_group or os.getenv("WANDB_GROUP")
        if not group:
            logging.warning("No W&B group set; skipping combined plots")
            return

        histories = fetch_group_histories(project, group, args.wandb_entity or os.getenv("WANDB_ENTITY"))
        if not histories:
            logging.warning("No W&B histories found for group %s", group)
            return

        combined_paths = plot_combined_metrics(histories, COMBINED_DIR, group)
        if combined_paths:
            tags = parse_csv(args.wandb_tags) or []
            if "combined" not in tags:
                tags.append("combined")
            combined_run = init_wandb_run(
                enabled=True,
                project=project,
                group=group,
                name="combined_plots",
                tags=tags,
                output_dir=COMBINED_DIR,
                job_type="combined"
            )
            log_wandb_images(combined_run, combined_paths)
            if combined_run:
                combined_run.finish()


if __name__ == "__main__":
    main()
