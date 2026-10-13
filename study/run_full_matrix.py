"""Run four fresh native training processes inside one managed GPU job."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


def all_finite(value):
    """Reject invalid tensors and invalid scalar measurements."""
    if isinstance(value, dict):
        return value.get("finite", True) is True and all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def phase_configs(base):
    """Build the baseline and selected-candidate diagnostic and timing configurations."""
    if base.get("selected_candidate") != "precast_linear_conv" or not base.get("allow_candidate"):
        raise ValueError("The selected material candidate must be precast_linear_conv")
    root = Path(base["out"])
    rows = []
    for phase in ("diagnostic", "timing"):
        for variant in ("baseline", "precast_linear_conv"):
            child = dict(base)
            child.update(
                variant=variant,
                phase=phase,
                out=str(root / f"{phase}-{variant}"),
                stop_after_steps=1 if phase == "diagnostic" else 12,
                warmup_steps=0 if phase == "diagnostic" else 3,
            )
            rows.append(child)
    return rows


def main():
    """Execute separate processes with a deadline and durable logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = json.loads(args.config.read_text())
    phases = phase_configs(base)
    if args.dry_run:
        print(json.dumps({"phases": phases, "scope": "Dry plan only. No training or launch."}, indent=2))
        return
    root = Path(base["out"])
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve().parent / "validate_native_train.py"
    deadline = time.monotonic() + base.get("total_timeout_seconds", 900)
    import torch

    report = {
        "config": base,
        "runs": [],
        "scope": "One GPU job; four fresh native processes; no asset download",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": sys.version,
        },
    }
    status = 0
    for config in phases:
        name = config["phase"] + "-" + config["variant"]
        config_path, log_path = root / (name + ".json"), root / (name + ".log")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            report["deadline_reached"] = True
            status = 124
            break
        print(f"FULL_NATIVE_RUN_BEGIN={name}", flush=True)
        start = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(script), "--config", str(config_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return_code = 124
        row = {
            "phase": config["phase"],
            "variant": config["variant"],
            "return_code": return_code,
            "wall_seconds": time.monotonic() - start,
            "log": str(log_path),
            "results": str(Path(config["out"]) / "results.json"),
        }
        report["runs"].append(row)
        (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"FULL_NATIVE_RUN_END={name} return_code={return_code}", flush=True)
        if return_code:
            print("FULL_NATIVE_FAILURE_LOG_BEGIN", flush=True)
            print("\n".join(log_path.read_text().splitlines()[-80:]), flush=True)
            print("FULL_NATIVE_FAILURE_LOG_END", flush=True)
            status = return_code
            break
    if status == 0 and len(report["runs"]) == 4:
        baseline = json.loads((root / "diagnostic-baseline" / "results.json").read_text())
        candidate = json.loads((root / "diagnostic-precast_linear_conv" / "results.json").read_text())
        report["teacher_targets_exact"] = baseline["teacher_targets"] == candidate["teacher_targets"]
        left = json.loads((root / "diagnostic-baseline" / "diagnostic-tensors.json").read_text())
        right = json.loads((root / "diagnostic-precast_linear_conv" / "diagnostic-tensors.json").read_text())
        report["diagnostic_exact"] = {key: left[key] == right[key] for key in left}
        report["diagnostic_all_finite"] = all(
            all_finite(value)
            for value in (left, right, baseline["teacher_targets"], candidate["teacher_targets"])
        )
        report["diagnostic_counts"] = {key: len(value) for key, value in left.items()}
        report["timing"] = {}
        report["losses_all_finite"] = True
        for variant in ("baseline", "precast_linear_conv"):
            data = json.loads((root / f"timing-{variant}" / "results.json").read_text())
            rows = [row for row in data["steps"] if not row["warmup"]]
            report["losses_all_finite"] &= all_finite(data["steps"])
            report["timing"][variant] = {
                "measured_steps": len(rows),
                "median_update_ms": statistics.median(row["update_policy_wall_ms"] for row in rows),
                "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
            }
        if not report["diagnostic_all_finite"] or not report["losses_all_finite"]:
            status = 2
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FULL_NATIVE_JOB_RESULTS_JSON_BEGIN")
    print(json.dumps(report, indent=2))
    print("FULL_NATIVE_JOB_RESULTS_JSON_END")
    sys.exit(status)


if __name__ == "__main__":
    main()
