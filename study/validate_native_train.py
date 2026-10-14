"""Proposed full-policy validation through the actual LeRobot training entry point.

This script requires prepared local assets. It does not download or publish assets.
The parent must approve the candidate after a material teacher result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path


class ValidationStopError(Exception):
    """Stop after the requested optimizer updates without a 30,000-step run."""


def file_sha256(path):
    """Compute the SHA-256 of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_record(value):
    """Hash every tensor byte without retaining another full model in memory."""
    import torch

    cpu = value.detach().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(torch.isfinite(cpu).all()),
        "bytes": len(raw),
    }


def verify_local_assets(config):
    """Verify source and prepared assets before any policy construction."""
    repo = Path(config["repo"])
    for relative, expected in config["source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise ValueError(f"Source differs from the pinned PR: {relative}")
    manifest = json.loads(Path(config["asset_verification_manifest"]).read_text())
    roots = {
        "policy": Path(config["checkpoint"]),
        "dataset": Path(config["dataset_root"]),
        "qwen": Path(config["qwen_config"]),
    }
    pins = {
        "policy": ("lerobot/lingbot_vla_v2_base", "5dd4b5299ed85c782a987d58ff3f24f95a95474d"),
        "dataset": ("maximellerbach/omx_multicubes", "f7e700308a7049fd3d33adaf84df6e2ece19676d"),
        "qwen": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    }
    for role, key in (
        ("policy", "model_revision"),
        ("dataset", "dataset_revision"),
        ("qwen", "qwen_revision"),
    ):
        if config[key] != pins[role][1]:
            raise ValueError(f"The configured {role} revision differs from the approved pin")
    listed = {role: set() for role in roots}
    for item in manifest["assets"]:
        role = item.get("role")
        if role in roots:
            repo_id, revision = pins[role]
            if item["repo_id"] != repo_id or item["revision"] != revision:
                raise ValueError(f"Manifest {role} repository or revision differs from the approved pin")
            actual = (roots[role] / item["filename"]).resolve()
            if actual != Path(item["path"]).resolve():
                raise ValueError(
                    f"Configured {role} file is outside the verified manifest: {item['filename']}"
                )
            listed[role].add(item["filename"])
        elif role != "teacher":
            raise ValueError(f"Unknown prepared asset role: {role}")
        if not item.get("sha256") or file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Prepared asset does not match its SHA-256: {item.get('filename')}")
    required = {
        "policy": {
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "image_processor/preprocessor_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/chat_template.jinja",
            "chat_template_tokenizer/tokenizer.json",
            "chat_template_tokenizer/tokenizer_config.json",
            "chat_template_tokenizer/chat_template.jinja",
        },
        "dataset": {
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-000.parquet",
            "videos/observation.images.top/chunk-000/file-000.mp4",
            "videos/observation.images.wrist/chunk-000/file-000.mp4",
        },
        "qwen": {"config.json"},
    }
    for role, files in required.items():
        if not files.issubset(listed[role]):
            raise ValueError(f"Manifest lacks required {role} files: {sorted(files - listed[role])}")
    teacher_config = json.loads(Path(config["teacher_config"]).read_text())
    teacher_entries = [item for item in manifest["assets"] if item.get("role") == "teacher"]
    for name in ("moge", "morgbd", "video", "video_config"):
        asset = teacher_config["assets"][name]
        matches = [
            item
            for item in teacher_entries
            if Path(item["path"]).resolve() == Path(asset["path"]).resolve()
            and all(item[key] == asset[key] for key in ("repo_id", "revision", "filename", "sha256"))
        ]
        if len(matches) != 1:
            raise ValueError(f"Configured teacher is outside the verified manifest: {name}")
    checkpoints = [item for item in manifest["assets"] if item.get("filename") == "model.safetensors"]
    if len(checkpoints) != 1 or checkpoints[0]["revision"] != config["model_revision"]:
        raise ValueError("The manifest does not contain the pinned full policy")
    if checkpoints[0]["sha256"] != "bcd62f119416cd607f5584adceefdf7b74427a62a03cd43251f221620fea6c64":
        raise ValueError("The full checkpoint has the wrong SHA-256")
    return manifest


def cli_arguments(config, teacher_config, out):
    """Use the documented slot mappings and the native training presets."""
    state, action = {}, {}
    for joint, start, end in (("arm.position", 0, 5), ("effector.position", 5, 6)):
        state[f"observation.state.{joint}"] = {
            "origin_keys": [{"observation.state": {"start": start, "end": end}}]
        }
        action[f"action.{joint}"] = {"origin_keys": [{"action": {"start": start, "end": end}}]}
    rename = {
        "observation.images.top": "observation.images.camera_top",
        "observation.images.wrist": "observation.images.camera_wrist_left",
    }
    align = teacher_config["align_params"]
    align["depth"].update(
        moge_path=teacher_config["assets"]["moge"]["path"],
        morgbd_path=teacher_config["assets"]["morgbd"]["path"],
    )
    align["video"].update(
        ckpt_path=teacher_config["assets"]["video"]["path"],
        config_path=teacher_config["assets"]["video_config"]["path"],
    )
    values = {
        "dataset.repo_id": "maximellerbach/omx_multicubes",
        "dataset.root": config["dataset_root"],
        "dataset.revision": config["dataset_revision"],
        "dataset.episodes": "[0]",
        "dataset.video_backend": "pyav",
        "policy.path": config["checkpoint"],
        "policy.tokenizer_path": config["qwen_config"],
        "policy.device": "cuda",
        "policy.dtype": "bfloat16",
        "policy.push_to_hub": "false",
        "policy.state_slots": json.dumps(state),
        "policy.action_slots": json.dumps(action),
        "policy.align_params": json.dumps(align),
        "rename_map": json.dumps(rename),
        "batch_size": config["batch_size"],
        "seed": config["seed"],
        "steps": config["native_training_steps"],
        "num_workers": config["num_workers"],
        "save_checkpoint": "false",
        "wandb.enable": "false",
        "env_eval_freq": 0,
        "log_freq": 1,
        "output_dir": str(out / "train"),
        "policy.gradient_checkpointing": str(config["gradient_checkpointing"]).lower(),
        "cudnn_deterministic": str(config.get("cudnn_deterministic", False)).lower(),
    }
    return [f"--{key}={value}" for key, value in values.items()]


def install_teacher_variant(config, report, out):
    """Load the exact original or committed candidate class in the same module globals."""
    import torch

    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as modeling

    study = Path(__file__).resolve().parent
    candidate = config["variant"] == "precast_linear_conv"
    source = (
        study
        / "source-snapshot"
        / ("modeling_lingbot_vla_v2_candidate.py" if candidate else "modeling_lingbot_vla_v2.py")
    )
    expected = (
        "78242de59473ccbc46c3a310172beff3018fff46561c4370bc15db8cd5fd8096"
        if candidate
        else "046247ec8fd40f100762972f29d75e9370b2d93455ea595f14cfd4d31fbe082d"
    )
    if file_sha256(source) != expected:
        raise ValueError("Teacher class snapshot differs from the approved exact source")
    tree = ast.parse(source.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DistillationTeachers"
    ]
    if len(nodes) != 1:
        raise ValueError("Teacher class snapshot has an unexpected class definition")
    namespace = dict(modeling.__dict__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    teacher_class = namespace["DistillationTeachers"]
    original = teacher_class.targets
    report["teacher_class_source"] = {"sha256": expected, "path": str(source), "variant": config["variant"]}

    @wraps(original)
    @torch.no_grad()
    def targets(self, current, future, fps=None):
        flags = {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        report.setdefault("backend_flags", flags)
        result = original(self, current, future, fps)
        if config["phase"] == "diagnostic":
            row = {
                "fps": fps,
                "targets": {name: tensor_record(value) for name, value in result.items()},
                "backend_flags": flags,
            }
            if config.get("save_teacher_tensors", False):
                index = len(report["teacher_targets"])
                target_path = out / f"teacher-targets-{index:03d}.pt"
                torch.save(
                    {key: value.detach().cpu().contiguous() for key, value in result.items()}, target_path
                )
                row["raw_targets"] = str(target_path)
                input_path = out / f"teacher-inputs-{index:03d}.pt"
                torch.save(
                    {
                        "current": current.detach().cpu().contiguous(),
                        "future": future.detach().cpu().contiguous(),
                        "fps": fps,
                    },
                    input_path,
                )
                row["raw_inputs"] = str(input_path)
            report["teacher_targets"].append(row)
        return result

    teacher_class.targets = targets
    modeling.DistillationTeachers = teacher_class


def install_update_capture(config, report, out):
    """Keep the native forward, backward, gradient clipping, and AdamW update."""
    import torch

    import lerobot.scripts.lerobot_train as train_module

    original = train_module.update_policy

    @wraps(original)
    def update(*args, **kwargs):
        # Native caller passes five positional arguments and accelerator as a keyword.
        _tracker, policy, batch, optimizer, _clip = args[:5]
        accelerator = kwargs["accelerator"]
        base = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        if not report["model"]:
            report["model"] = {
                "parameter_count": sum(p.numel() for p in base.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in base.parameters() if p.requires_grad),
                "parameter_dtypes": sorted({str(p.dtype) for p in base.parameters()}),
                "shape": {
                    key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)
                },
                "teacher_fps": base._teacher_fps,
            }
        diagnostic = config["phase"] == "diagnostic"
        capture_student = diagnostic and config.get("capture_student_hashes", True)
        step_capture = {}
        if diagnostic:
            step_capture["input"] = {
                key: tensor_record(value) for key, value in batch.items() if isinstance(value, torch.Tensor)
            }
        if capture_student:
            native_step = optimizer.step
            named = dict(base.named_parameters())
            by_id = {id(value): name for name, value in named.items()}

            @wraps(native_step)
            def capturing_step(*step_args, **step_kwargs):
                step_capture["initial_parameters"] = {
                    name: tensor_record(value) for name, value in named.items()
                }
                step_capture["clipped_gradients"] = {
                    name: tensor_record(value.grad) for name, value in named.items() if value.grad is not None
                }
                result = native_step(*step_args, **step_kwargs)
                step_capture["updated_parameters"] = {
                    name: tensor_record(value) for name, value in named.items()
                }
                state = optimizer.state
                step_capture["adamw_state"] = {
                    by_id[id(parameter)]: {
                        name: tensor_record(value) if isinstance(value, torch.Tensor) else value
                        for name, value in values.items()
                    }
                    for parameter, values in state.items()
                }
                step_capture["optimizer_groups"] = [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in optimizer.param_groups
                ]
                return result

            optimizer.step = capturing_step
        if config.get("cudnn_benchmark_override") is not None:
            torch.backends.cudnn.benchmark = config["cudnn_benchmark_override"]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            tracker, logs = original(*args, **kwargs)
        finally:
            if capture_student:
                optimizer.step = native_step
        torch.cuda.synchronize()
        row = {
            "step": len(report["steps"]) + 1,
            "loss": tracker.metrics["loss"].val,
            "gradient_norm": tracker.metrics["grad_norm"].val,
            "update_policy_wall_ms": (time.perf_counter() - start) * 1000,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "logs": logs,
            "diagnostic": diagnostic,
            "warmup": len(report["steps"]) < config["warmup_steps"],
        }
        if diagnostic:
            (out / "diagnostic-tensors.json").write_text(json.dumps(step_capture, indent=2) + "\n")
        report["steps"].append(row)
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        if len(report["steps"]) >= config["stop_after_steps"]:
            raise ValidationStopError()
        return tracker, logs

    train_module.update_policy = update


def main():
    """Run a dry plan or the bounded native training process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dry_run:
        print(
            json.dumps(
                config | {"dry_run": True, "scope": "No download, training, or compute launch"}, indent=2
            )
        )
        return
    if config["variant"] not in ("baseline", "precast_linear_conv") or config["phase"] not in (
        "diagnostic",
        "timing",
    ):
        raise ValueError("Invalid validation variant or phase")
    if config["variant"] != "baseline" and not config["allow_candidate"]:
        raise ValueError("The parent must first confirm a material teacher improvement")
    if config["phase"] == "diagnostic" and config["stop_after_steps"] != 1:
        raise ValueError("Run exactly one diagnostic step per variant")
    if not 0 < config["stop_after_steps"] <= 20:
        raise ValueError("The validation run must contain one through twenty steps")
    verify_local_assets(config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if config.get("cublas_workspace_config") is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["cublas_workspace_config"]
    sys.path[:0] = [str(Path(config["repo"]) / "src"), config["repo"]]
    import torch

    torch.use_deterministic_algorithms(config.get("deterministic_algorithms", False), warn_only=False)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < config["minimum_device_memory_gib"] * 1024**3
    ):
        raise ValueError("Full validation requires the configured 80 GB device class")
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=False)
    teacher_config = json.loads(Path(config["teacher_config"]).read_text())
    report = {
        "config": config,
        "model": {},
        "steps": [],
        "teacher_targets": [],
        "timing_scope": (
            "Native update_policy: forward, backward, clipping, AdamW, scheduler and scalar logs; "
            "excludes data loading and preprocessing"
        ),
        "diagnostic_timing_limit": (
            "Tensor hashes synchronize and transfer parameters and optimizer state. "
            "Diagnostic wall time is not a performance result."
        ),
    }
    install_teacher_variant(config, report, out)
    install_update_capture(config, report, out)
    import lerobot.scripts.lerobot_train as train_module

    sys.argv = ["lerobot-train", *cli_arguments(config, teacher_config, out)]
    try:
        train_module.main()
    except ValidationStopError:
        report["bounded_stop"] = True
    finally:
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "results": str(out / "results.json"),
                "steps": len(report["steps"]),
                "variant": config["variant"],
                "phase": config["phase"],
            }
        )
    )


if __name__ == "__main__":
    main()
