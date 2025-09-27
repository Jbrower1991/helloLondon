#!/usr/bin/env python3
"""Orchestrate uniform vs. PBit sampler ablations.

This launcher builds a matrix of training runs that compare the
stock uniform sampler against the variance-aware PBit sampler while
sweeping a reusable set of configuration knobs.

Example usage (dry run):
    python run_sampler_ablation.py --dry-run --trainer regular

To actually execute the runs, omit ``--dry-run``. Additional training
arguments can be forwarded after ``--``::
    python run_sampler_ablation.py --trainer slm -- --max_steps 2000
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "04_training"


@dataclass
class VariantDefinition:
    """Description of a sampler variant used in the ablation."""

    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    description: str | None = None


DEFAULT_PBIT_BASE: Dict[str, Any] = {
    "random_seed": 1337,
}

DEFAULT_PBIT_VARIANTS: Sequence[VariantDefinition] = (
    VariantDefinition(
        name="pbit_default",
        overrides={},
        description="Baseline PBit configuration using constructor defaults.",
    ),
    VariantDefinition(
        name="pbit_diversity_high",
        overrides={"diversity_strength": 0.85},
        description="Favor diverse batches by weighting diversity more heavily.",
    ),
    VariantDefinition(
        name="pbit_diversity_low",
        overrides={"diversity_strength": 0.15},
        description="Prioritise high-score windows with minimal diversity pressure.",
    ),
    VariantDefinition(
        name="pbit_shortlist_compact",
        overrides={"shortlist_cap": 512, "shortlist_seed_size": 96},
        description="Keep a smaller shortlist to refresh frequently.",
    ),
    VariantDefinition(
        name="pbit_shortlist_broad",
        overrides={"shortlist_cap": 2048, "shortlist_seed_size": 256},
        description="Expand shortlist breadth for larger candidate pools.",
    ),
    VariantDefinition(
        name="pbit_temperature_cool",
        overrides={"temperature": 0.85},
        description="Sharper sampling over shortlist scores.",
    ),
    VariantDefinition(
        name="pbit_temperature_warm",
        overrides={"temperature": 1.25},
        description="Flatter sampling distribution over shortlist scores.",
    ),
    VariantDefinition(
        name="pbit_refresh_fast",
        overrides={"heavy_refresh_interval": 20},
        description="Refresh shortlist aggressively to measure bookkeeping cost.",
    ),
)


def load_jsonish(value: Optional[str]) -> Any:
    """Parse a JSON string or file path if provided."""

    if not value:
        return None
    potential_path = Path(value)
    if potential_path.exists():
        with open(potential_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def slugify(value: Any) -> str:
    """Convert a value into a filesystem-friendly slug."""

    if isinstance(value, float):
        if value.is_integer():
            value = int(value)
        else:
            value = ("%0.3f" % value).rstrip("0").rstrip(".")
    text = str(value)
    text = text.replace("-", "neg")
    allowed = []
    for ch in text:
        if ch.isalnum():
            allowed.append(ch.lower())
        else:
            allowed.append("_")
    slug = "".join(allowed)
    slug = "_".join(filter(None, slug.split("_")))
    return slug or "value"


def build_variants_from_grid(grid: Dict[str, Iterable[Any]]) -> List[VariantDefinition]:
    """Expand a parameter grid into explicit variant definitions."""

    if not grid:
        return []
    keys = sorted(grid.keys())
    values = [list(grid[key]) for key in keys]
    combos = []
    for combination in product(*values):
        overrides = {key: value for key, value in zip(keys, combination)}
        name_parts = ["pbit"] + [f"{key}-{slugify(value)}" for key, value in overrides.items()]
        name = "_".join(name_parts)
        description = "Auto-generated from grid sweep."
        combos.append(VariantDefinition(name=name, overrides=overrides, description=description))
    return combos


def ensure_no_conflicts(train_args: Sequence[str]) -> None:
    forbidden = {"--sampler_type", "--sampler_kwargs", "--output_dir"}
    for arg in train_args:
        if arg in forbidden:
            raise ValueError(
                f"Argument '{arg}' should be provided via the ablation launcher instead of --train-args"
            )


def make_command(
    python: str,
    training_script: Path,
    base_args: Sequence[str],
    sampler_type: str,
    sampler_kwargs: Optional[Dict[str, Any]],
    output_dir: Path,
) -> List[str]:
    cmd: List[str] = [python, str(training_script)]
    cmd.extend(base_args)
    cmd.extend(["--output_dir", str(output_dir)])
    cmd.extend(["--sampler_type", sampler_type])
    if sampler_kwargs:
        cmd.extend(["--sampler_kwargs", json.dumps(sampler_kwargs)])
    return cmd


def write_commands_script(commands: Sequence[Sequence[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\n")
        handle.write("set -e\n\n")
        for command in commands:
            quoted = " ".join(shlex.quote(part) for part in command)
            handle.write(f"{quoted}\n")
    os.chmod(path, 0o755)


def serialise_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, VariantDefinition):
        return {
            "name": value.name,
            "overrides": value.overrides,
            "description": value.description,
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a uniform vs. PBit sampler ablation sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples
            --------
            Dry-run the regular model sweep:
                python run_sampler_ablation.py --dry-run

            Execute the SLM sweep and limit to three variants:
                python run_sampler_ablation.py --trainer slm --max-variants 3

            Provide a custom grid via JSON file:
                python run_sampler_ablation.py --pbit-grid custom_grid.json

            Forward additional training args after ``--``:
                python run_sampler_ablation.py -- --max_steps 1000 --eval_steps 200
            """
        ),
    )
    parser.add_argument(
        "--trainer",
        choices=("regular", "slm"),
        default="regular",
        help="Select which training script to invoke.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for launches.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "ablations"),
        help="Root directory where ablation run outputs are stored.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for the ablation sweep (defaults to timestamp).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the training data directory passed to the trainer.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Override the tokenizer directory passed to the trainer.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint path forwarded via --resume_from_checkpoint.",
    )
    parser.add_argument(
        "--include-uniform",
        action="store_true",
        default=True,
        help="Include the uniform baseline run (enabled by default).",
    )
    parser.add_argument(
        "--no-uniform",
        action="store_false",
        dest="include_uniform",
        help="Disable the uniform baseline run.",
    )
    parser.add_argument(
        "--pbit-base-kwargs",
        type=str,
        default=None,
        help="JSON string or file containing base kwargs applied to all PBit variants.",
    )
    parser.add_argument(
        "--pbit-variants",
        type=str,
        default=None,
        help="JSON string or file describing explicit PBit variants (list of objects).",
    )
    parser.add_argument(
        "--pbit-grid",
        type=str,
        default=None,
        help="JSON string or file describing a parameter grid for auto-generated variants.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Limit the number of PBit variants considered from the configured list.",
    )
    parser.add_argument(
        "--sampler-seeds",
        type=int,
        nargs="*",
        default=[1337],
        help="List of seeds forwarded to PBit via its random_seed knob.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep immediately if a run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands without executing them.",
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to the training script.",
    )

    args = parser.parse_args()

    training_script = TRAINING_DIR / ("train_model.py" if args.trainer == "regular" else "train_model_slm.py")
    if not training_script.exists():
        raise FileNotFoundError(f"Could not locate training script: {training_script}")

    base_args: List[str] = []
    if args.data_dir:
        base_args.extend(["--data_dir", args.data_dir])
    if args.tokenizer_dir:
        base_args.extend(["--tokenizer_dir", args.tokenizer_dir])
    if args.resume_from:
        base_args.extend(["--resume_from_checkpoint", args.resume_from])

    train_args = args.train_args or []
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    ensure_no_conflicts(train_args)
    base_args.extend(train_args)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_name = args.run_name or datetime.now(timezone.utc).strftime("sampler_ablation_%Y%m%d-%H%M%S")
    sweep_dir = output_root / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pbit_base_kwargs = DEFAULT_PBIT_BASE.copy()
    loaded_base = load_jsonish(args.pbit_base_kwargs)
    if loaded_base:
        if not isinstance(loaded_base, dict):
            raise TypeError("--pbit-base-kwargs must define a JSON object")
        pbit_base_kwargs.update(loaded_base)

    if args.pbit_variants:
        loaded_variants = load_jsonish(args.pbit_variants)
        if not isinstance(loaded_variants, list):
            raise TypeError("--pbit-variants must be a JSON list")
        variant_defs: List[VariantDefinition] = []
        for entry in loaded_variants:
            if not isinstance(entry, dict):
                raise TypeError("Each variant definition must be a JSON object")
            name = entry.get("name")
            overrides = entry.get("overrides", {})
            description = entry.get("description")
            if not name:
                raise ValueError("Variant definitions must include a 'name'")
            if not isinstance(overrides, dict):
                raise TypeError("Variant overrides must be a JSON object")
            variant_defs.append(VariantDefinition(name=name, overrides=overrides, description=description))
    else:
        grid = load_jsonish(args.pbit_grid)
        if grid is not None and not isinstance(grid, dict):
            raise TypeError("--pbit-grid must be a JSON object mapping parameter -> list of values")
        if grid:
            variant_defs = build_variants_from_grid(grid)
        else:
            variant_defs = list(DEFAULT_PBIT_VARIANTS)

    if args.max_variants is not None:
        variant_defs = variant_defs[: args.max_variants]

    commands: List[List[str]] = []
    runs_manifest: List[Dict[str, Any]] = []

    def register_run(name: str, sampler_type: str, sampler_kwargs: Optional[Dict[str, Any]]) -> Path:
        run_dir = sweep_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        command = make_command(
            python=args.python,
            training_script=training_script,
            base_args=base_args,
            sampler_type=sampler_type,
            sampler_kwargs=sampler_kwargs,
            output_dir=checkpoints_dir,
        )
        commands.append(command)
        runs_manifest.append(
            {
                "name": name,
                "sampler_type": sampler_type,
                "sampler_kwargs": sampler_kwargs or {},
                "output_dir": checkpoints_dir,
                "log_file": run_dir / "train.log",
                "command": command,
                "status": "pending" if not args.dry_run else "skipped (dry-run)",
            }
        )
        if sampler_kwargs:
            with open(run_dir / "sampler_kwargs.json", "w", encoding="utf-8") as handle:
                json.dump(sampler_kwargs, handle, indent=2)
        return run_dir

    if args.include_uniform:
        register_run(name="uniform_baseline", sampler_type="uniform", sampler_kwargs=None)

    for variant in variant_defs:
        for seed in args.sampler_seeds:
            variant_kwargs = dict(pbit_base_kwargs)
            variant_kwargs.update(variant.overrides)
            variant_kwargs.setdefault("random_seed", seed)
            seed_suffix = f"seed{seed}" if len(args.sampler_seeds) > 1 else None
            run_name_parts = [variant.name]
            if seed_suffix:
                run_name_parts.append(seed_suffix)
            run_name = "--".join(run_name_parts)
            register_run(name=run_name, sampler_type="pbit", sampler_kwargs=variant_kwargs)

    commands_path = sweep_dir / "commands.sh"
    write_commands_script(commands, commands_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trainer": args.trainer,
        "python": args.python,
        "training_script": training_script,
        "output_root": output_root,
        "sweep_dir": sweep_dir,
        "include_uniform": args.include_uniform,
        "pbit_base_kwargs": pbit_base_kwargs,
        "pbit_variants": variant_defs,
        "sampler_seeds": args.sampler_seeds,
        "dry_run": args.dry_run,
        "runs": runs_manifest,
    }

    manifest_path = sweep_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=serialise_for_json)

    print(f"Ablation sweep manifest created at {manifest_path}")
    print(f"Commands script available at {commands_path}")

    if args.dry_run:
        for command in commands:
            print("DRY-RUN:", " ".join(shlex.quote(part) for part in command))
        return

    for run in runs_manifest:
        command = run["command"]
        run_dir = sweep_dir / run["name"]
        log_path = Path(run["log_file"])
        print(f"\n▶ Running {run['name']}...")
        start_time = time.time()
        with open(log_path, "w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                command,
                cwd=training_script.parent,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - start_time
        run["status"] = "succeeded" if process.returncode == 0 else f"failed ({process.returncode})"
        run["duration_seconds"] = round(elapsed, 2)
        run["returncode"] = process.returncode
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=serialise_for_json)
        if process.returncode != 0:
            print(f"❌ Run {run['name']} failed (exit code {process.returncode}). See {log_path} for details.")
            if args.stop_on_error:
                break
        else:
            print(f"✅ Completed {run['name']} in {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
