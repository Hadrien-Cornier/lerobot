"""Run fresh native numerical control processes inside one managed GPU job."""

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
    """Compare repeatability and change only the cuDNN benchmark flag in fixed controls."""
    if base.get("selected_candidate") != "precast_linear_conv" or not base.get("allow_candidate"):
        raise ValueError("The exact selected candidate requires parent authorization")
    root = Path(base["out"])
    phases = [
        ("default-baseline-a", "baseline", None, False),
        ("default-baseline-b", "baseline", None, False),
        ("default-candidate", "precast_linear_conv", None, False),
        ("fixed-baseline", "baseline", False, True),
        ("fixed-candidate", "precast_linear_conv", False, True),
    ]
    phases += [
        ("fixed-timing-baseline", "baseline", False, False),
        ("fixed-timing-candidate", "precast_linear_conv", False, False),
    ]
    phase_names = base.get("phase_names")
    if phase_names is not None:
        known_names = {name for name, *_ in phases}
        if not isinstance(phase_names, list) or any(not isinstance(name, str) for name in phase_names):
            raise ValueError("phase_names must be a list of phase names")
        if len(phase_names) != len(set(phase_names)):
            raise ValueError("phase_names cannot contain duplicates")
        unknown_names = set(phase_names) - known_names
        if unknown_names:
            raise ValueError(f"Unknown phase names: {sorted(unknown_names)}")
        required_names = {"fixed-baseline", "fixed-candidate"}
        if not required_names.issubset(phase_names):
            raise ValueError("phase_names must include fixed-baseline and fixed-candidate")
        requested_names = set(phase_names)
        phases = [phase for phase in phases if phase[0] in requested_names]
    rows = []
    for name, variant, benchmark_override, student_hashes in phases:
        timing = name.startswith("fixed-timing")
        child = dict(base)
        child.update(
            control_name=name,
            variant=variant,
            phase="timing" if timing else "diagnostic",
            out=str(root / name),
            stop_after_steps=12 if timing else 1,
            warmup_steps=3 if timing else 0,
            save_teacher_tensors=True,
            capture_student_hashes=student_hashes,
            cudnn_deterministic=base.get("cudnn_deterministic", False),
            cudnn_benchmark_override=benchmark_override,
        )
        rows.append(child)
    return rows


def compare_target_files(left_path, right_path):
    """Measure tensor differences instead of inferring error magnitude from hashes."""
    import torch

    left = torch.load(left_path, weights_only=True, map_location="cpu")
    right = torch.load(right_path, weights_only=True, map_location="cpu")
    if left.keys() != right.keys():
        raise ValueError("Raw teacher target keys differ")
    rows = {}
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Raw teacher target shape or dtype differs: {key}")
        delta = (a.float() - b.float()).abs()
        rows[key] = {
            "exact": torch.equal(a, b),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "changed_elements": int((a != b).sum()),
            "elements": a.numel(),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "relative_l2": float(
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
            ),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        }
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
    deadline = time.monotonic() + base.get("total_timeout_seconds", 1500)
    import torch

    report = {
        "config": base,
        "runs": [],
        "scope": f"One GPU job; {len(phases)} fresh native processes; no asset download",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": sys.version,
        },
    }
    status = 0
    for config in phases:
        name = config["control_name"]
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
            "control_name": name,
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
    if status == 0 and len(report["runs"]) == len(phases):
        data = {row["control_name"]: json.loads((Path(row["results"])).read_text()) for row in report["runs"]}
        report["backend_flags"] = {name: value["backend_flags"] for name, value in data.items()}
        reference_flags = next(iter(report["backend_flags"].values()))
        other_keys = set(reference_flags) - {"cudnn_benchmark"}
        expected_cudnn_deterministic = base.get("cudnn_deterministic", False)
        expected_deterministic_algorithms = base.get("deterministic_algorithms", False)
        expected_cublas_workspace_config = base.get("cublas_workspace_config")
        report["backend_flags_match_expected"] = all(
            flags["cudnn_benchmark"] is (not name.startswith("fixed"))
            and flags["cudnn_deterministic"] is expected_cudnn_deterministic
            and flags["deterministic_algorithms"] is expected_deterministic_algorithms
            and flags["cublas_workspace_config"] == expected_cublas_workspace_config
            and all(flags[key] == reference_flags[key] for key in other_keys)
            for name, flags in report["backend_flags"].items()
        )
        if not report["backend_flags_match_expected"]:
            status = 3
        report["teacher_class_sources"] = {
            name: value["teacher_class_source"] for name, value in data.items()
        }
        report["target_comparisons"] = {}
        pairs = [
            ("default-baseline-a", "default-baseline-b"),
            ("default-baseline-a", "default-candidate"),
            ("default-baseline-b", "default-candidate"),
            ("fixed-baseline", "fixed-candidate"),
        ]
        pairs = [(left, right) for left, right in pairs if left in data and right in data]
        for left_name, right_name in pairs:
            left, right = data[left_name], data[right_name]
            left_capture = json.loads((root / left_name / "diagnostic-tensors.json").read_text())
            right_capture = json.loads((root / right_name / "diagnostic-tensors.json").read_text())
            stats = compare_target_files(
                root / left_name / "teacher-targets-000.pt", root / right_name / "teacher-targets-000.pt"
            )
            report["target_comparisons"][left_name + "_vs_" + right_name] = {
                "input_exact": left_capture["input"] == right_capture["input"],
                "fps_exact": left["teacher_targets"][0]["fps"] == right["teacher_targets"][0]["fps"],
                "targets": stats,
                "loss_left": left["steps"][0]["loss"],
                "loss_right": right["steps"][0]["loss"],
            }
        left = json.loads((root / "fixed-baseline" / "diagnostic-tensors.json").read_text())
        right = json.loads((root / "fixed-candidate" / "diagnostic-tensors.json").read_text())
        required_student_fields = {
            "initial_parameters",
            "clipped_gradients",
            "updated_parameters",
            "adamw_state",
            "optimizer_groups",
        }
        student_fields_present = required_student_fields.issubset(left) and required_student_fields.issubset(
            right
        )
        diagnostic_keys = set(left) | set(right)
        report["fixed_student_diagnostic_complete"] = student_fields_present and set(left) == set(right)
        if not report["fixed_student_diagnostic_complete"]:
            status = 5
        report["inputs_match_expected"] = all(
            row["input_exact"] and row["fps_exact"] for row in report["target_comparisons"].values()
        )
        if not report["inputs_match_expected"]:
            status = 4
        report["all_finite"] = all(
            all_finite(value)
            for value in (left, right, report["target_comparisons"], [x["steps"] for x in data.values()])
        )
        report["fixed_student_diagnostic_exact"] = {
            key: key in left and key in right and left[key] == right[key] for key in diagnostic_keys
        }
        report["fixed_timing"] = {}
        timing_names = {"fixed-timing-baseline", "fixed-timing-candidate"}
        if timing_names.issubset(data):
            for name in sorted(timing_names):
                rows = [row for row in data[name]["steps"] if not row["warmup"]]
                report["fixed_timing"][name] = {
                    "measured_steps": len(rows),
                    "median_update_ms": statistics.median(row["update_policy_wall_ms"] for row in rows),
                    "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
                }
        default_controls = {"default-baseline-a", "default-baseline-b", "default-candidate"}
        if default_controls.issubset(data):
            report["interpretation_limit"] = (
                "The report includes repeated default-backend controls and a benchmark=False pair. "
                "Benchmark dependence remains a hypothesis until the controls resolve it."
            )
        else:
            report["interpretation_limit"] = (
                "The report includes only a fixed benchmark=False pair. "
                "It does not test baseline repeatability or default benchmark behavior."
            )
        if not report["all_finite"]:
            status = 2
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FULL_NATIVE_JOB_RESULTS_JSON_BEGIN")
    print(json.dumps(report, indent=2))
    print("FULL_NATIVE_JOB_RESULTS_JSON_END")
    sys.exit(status)


if __name__ == "__main__":
    main()
