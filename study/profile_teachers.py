"""Real-teacher research harness. No production source changes or implicit downloads."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PINNED_HEAD = "f58f7fb9dbb241304df96e967aea0f5b42ff9a4d"
SOURCE_FILES = [
    "src/lerobot/policies/lingbot_vla_v2/modeling_lingbot_vla_v2.py",
    "src/lerobot/policies/lingbot_vla_v2/model_core/depth_teachers.py",
    "src/lerobot/policies/lingbot_vla_v2/model_core/video_teacher.py",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(repo):
    rows = {}
    snapshot = Path(__file__).resolve().parent / "source-snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["source_head"] != PINNED_HEAD:
        raise ValueError("Source snapshot manifest has the wrong PR head")
    for relative in SOURCE_FILES:
        if (repo / ".git").exists():
            expected = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{PINNED_HEAD}:{relative}"]
            )
        else:
            item = manifest["files"][relative]
            expected = (snapshot / item["snapshot"]).read_bytes()
            if hashlib.sha256(expected).hexdigest() != item["sha256"]:
                raise ValueError(f"Source snapshot SHA-256 mismatch: {relative}")
        actual = (repo / relative).read_bytes()
        if actual != expected:
            raise ValueError(f"Teacher source differs from pinned PR head: {relative}")
        rows[relative] = hashlib.sha256(actual).hexdigest()
    return rows


def load_manifest(config):
    """Require local files plus revision and SHA-256 for every teacher asset."""
    paths = {}
    for name in ("moge", "morgbd", "video", "video_config"):
        item = config["assets"][name]
        path = Path(item["path"]).expanduser().resolve()
        if (
            not path.is_file()
            or not item.get("revision")
            or len(item.get("sha256", "")) != 64
        ):
            raise ValueError(f"Incomplete pinned local asset manifest: {name}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"Asset SHA-256 mismatch: {name}")
        paths[name] = str(path)
    return paths


class Recorder:
    """Separate diagnostic timing. Stage synchronization changes the execution schedule."""

    def __init__(self, device, synchronized=True):
        self.device = device
        self.synchronized = synchronized
        self.rows = []

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        if self.synchronized:
            self.sync()
        start, end = None, None
        if self.device.type == "cuda":
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record(torch.cuda.current_stream(self.device))
        tick = time.perf_counter()
        with torch.profiler.record_function(name):
            yield
        if end is not None:
            end.record(torch.cuda.current_stream(self.device))
        if self.synchronized:
            self.sync()
        self.rows.append(
            {
                "stage": name,
                "wall_ms": (time.perf_counter() - tick) * 1000,
                "cuda_event_timeline_ms": start.elapsed_time(end)
                if start is not None and self.synchronized
                else None,
            }
        )


class NoRecording:
    """No CUDA events or profiler labels in the timed candidate path."""

    synchronized = False

    def stage(self, _name):
        return nullcontext()


def recovery_profiled(points, mask, recorder, prefix, size=(64, 64)):
    """Exact pinned recovery equations, with stages around interpolation and transfers."""
    from scipy.optimize import least_squares

    from lerobot.policies.lingbot_vla_v2.model_core.depth_teachers import view_plane_uv

    height, width = points.shape[-3:-1]
    with recorder.stage(prefix + ".uv"):
        uv = view_plane_uv(width, height, width / height, points.dtype, points.device)
    with recorder.stage(prefix + ".points_interpolate"):
        points_lr = F.interpolate(
            points.permute(0, 3, 1, 2), size, mode="nearest"
        ).permute(0, 2, 3, 1)
    with recorder.stage(prefix + ".points_D2H"):
        points_lr = points_lr.cpu().numpy()
    with recorder.stage(prefix + ".uv_interpolate"):
        uv_lr = (
            F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), size, mode="nearest")
            .squeeze(0)
            .permute(1, 2, 0)
        )
    with recorder.stage(prefix + ".uv_D2H"):
        uv_lr = uv_lr.cpu().numpy()
    with recorder.stage(prefix + ".mask_interpolate"):
        mask_lr = (
            F.interpolate(mask.float().unsqueeze(1), size, mode="nearest").squeeze(1)
            > 0
        )
    with recorder.stage(prefix + ".mask_D2H"):
        mask_lr = mask_lr.cpu().numpy()

    def residual(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (focal * xy_proj - uv).ravel()

    shifts = []
    with recorder.stage(prefix + ".scipy_batch"):
        for index, (pts, m) in enumerate(zip(points_lr, mask_lr, strict=True)):
            with recorder.stage(prefix + f".scipy_sample_{index}"):
                pts, uv_i = pts[m], uv_lr[m]
                if uv_i.shape[0] < 2:
                    shifts.append(0.0)
                    continue
                uv_i, xy, z = (
                    uv_i.reshape(-1, 2),
                    pts[..., :2].reshape(-1, 2),
                    pts[..., 2].reshape(-1),
                )
                solution = least_squares(
                    partial(residual, uv_i, xy, z), x0=0, ftol=1e-3, method="lm"
                )
                shifts.append(float(solution["x"].squeeze().astype(np.float32)))
    with recorder.stage(prefix + ".shift_H2D"):
        return torch.tensor(shifts, device=points.device, dtype=points.dtype)


@contextmanager
def instrument_recovery(recorder, prefix):
    import lerobot.policies.lingbot_vla_v2.model_core.depth_teachers as module

    original = module.recover_shift
    module.recover_shift = lambda points, mask, size=(64, 64): recovery_profiled(
        points, mask, recorder, prefix, size
    )
    try:
        yield
    finally:
        module.recover_shift = original


@torch.no_grad()
def targets_candidate(teachers, current, future, fps, joint=False, recorder=None):
    """Source-equivalent target construction. Joint mode changes depth batch shape only."""
    from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import video_input

    params, video = teachers.params, teachers.params.get("video", {})
    recorder = recorder or NoRecording()
    with recorder.stage("input_transfer"):
        current = current.to(teachers.device, torch.float32)
        future = future.to(teachers.device, torch.float32)
    out = {}

    def depth(images, name):
        with (
            instrument_recovery(recorder, name + ".recover_shift"),
            recorder.stage(name + ".moge_including_recovery"),
        ):
            result = teachers.moge.depth(images, use_fp16=teachers.mixed)
        with recorder.stage(name + ".nan_to_num"):
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        with recorder.stage(name + ".rgbd_features"):
            return teachers.morgbd.features(images, result, use_fp16=teachers.mixed)

    # Diagnostic stages use the copied recovery only when a Recorder is supplied.
    # Timed candidate calls below use the original teachers.depth implementation.
    depth_fn = (
        depth if recorder.synchronized else lambda images, name: teachers.depth(images)
    )
    with torch.autocast(
        teachers.device.type, dtype=torch.bfloat16, enabled=teachers.mixed
    ):
        use_future = params["depth"].get("use_future_depth", False)
        if joint and use_future:
            with recorder.stage("joint_depth.input_concat"):
                images = torch.cat([current, future])
            result = depth_fn(images, "joint_depth")
            out["depth"], out["future_depth"] = result.split(len(current), dim=0)
        else:
            out["depth"] = depth_fn(current, "current_depth")
            if use_future:
                out["future_depth"] = depth_fn(future, "future_depth")
        if teachers.video is not None:
            with recorder.stage("video_input"):
                size = int(video.get("input_size", 256))
                cur, fut = video_input(current, size), video_input(future, size)
                clip = (
                    [cur.clone(), cur, fut]
                    if video.get("use_warmup_frame", False)
                    else [cur, fut]
                )
                effective_fps = (
                    fps
                    if fps is not None
                    else video.get("effective_fps", video.get("fps"))
                )
                clip_tensor = torch.stack(clip, dim=2).to(
                    teachers.video.cls_token.dtype
                )
            with recorder.stage("dino_video"):
                patches, frame_cls, pooled = teachers.video(clip_tensor, effective_fps)
            with recorder.stage("video_target_select"):
                out["future_video"], out["current_video"] = (
                    patches[:, -1],
                    patches[:, -2],
                )
                out["future_video_cls"] = (
                    frame_cls[:, -1]
                    if video.get("use_current_patch_loss", False)
                    else pooled
                )
    with recorder.stage("target_bfloat16_pack"):
        return {key: value.to(torch.bfloat16) for key, value in out.items()}


def make_dataset(root, height, width, count, fps=30):
    """Generate PNGs through the real LeRobotDataset writer, then read them through its reader."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STATE

    camera = "observation.images.camera"
    path = root / "dataset"
    if path.exists():
        raise ValueError(f"Use a new output directory; dataset already exists: {path}")
    features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint_{i}" for i in range(6)],
        },
        ACTION: {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint_{i}" for i in range(6)],
        },
        camera: {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        "local/teacher-stage-profile",
        fps=fps,
        features=features,
        root=path,
        use_videos=False,
    )
    y, x = np.indices((height, width))
    for i in range(count):
        image = np.stack(
            [
                (x * 5 + i * 11) % 256,
                (y * 7 + i * 17) % 256,
                ((x + y) * 3 + i * 23) % 256,
            ],
            -1,
        ).astype(np.uint8)
        state = np.sin(np.arange(6, dtype=np.float32) + i * 0.2).astype(np.float32)
        dataset.add_frame(
            {
                OBS_STATE: state,
                ACTION: state + 0.1,
                camera: image,
                "task": "pick up the cube",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset("local/teacher-stage-profile", root=path), camera


def load_input_dataset(config, out):
    """Optional local real-video input. Metadata checks precede any sample read."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    external = config.get("input_dataset")
    count = max(config["batch_sizes"])
    if external is None:
        future_offset = config.get("future_offset", 49)
        dataset, camera = make_dataset(
            out,
            config["image_height"],
            config["image_width"],
            count + future_offset,
            config.get("dataset_fps", 30),
        )
        return (
            dataset,
            camera,
            list(range(count)),
            list(range(future_offset, count + future_offset)),
            {
                "kind": "deterministic_PNG_fixture",
                "future_offset": future_offset,
                "metadata_sha256": {},
                "png_sha256": {
                    str(p.relative_to(out)): sha256_file(p)
                    for p in sorted((out / "dataset").rglob("*.png"))
                },
            },
        )
    root = Path(external["root"]).expanduser().resolve()
    if not external.get("revision") or not external.get("metadata_sha256"):
        raise ValueError(
            "External dataset requires a pinned revision and metadata SHA-256 manifest"
        )
    for relative, expected in external["metadata_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"Dataset metadata SHA-256 mismatch: {relative}")
    current_indices, future_indices = (
        external["current_indices"],
        external["future_indices"],
    )
    if len(current_indices) < count or len(future_indices) < count:
        raise ValueError("Input dataset manifest does not provide enough frame pairs")
    # The parent supplies all required local files. This harness does not download videos.
    from unittest.mock import patch

    def deny_dataset_download(*_args, **_kwargs):
        raise RuntimeError("The harness requires complete local dataset files")

    with (
        patch("lerobot.datasets.dataset_metadata.snapshot_download", deny_dataset_download),
        patch("lerobot.datasets.lerobot_dataset.snapshot_download", deny_dataset_download),
        patch.object(LeRobotDataset, "_download", deny_dataset_download),
    ):
        dataset = LeRobotDataset(
            external["repo_id"], root=root, revision=external["revision"]
        )
    return (
        dataset,
        external["camera"],
        current_indices,
        future_indices,
        {
            "kind": "pinned_local_dataset",
            "manifest": external,
            "media_sha256": {
                str(p.relative_to(root)): sha256_file(p)
                for p in sorted((root / "videos").rglob("*.mp4"))
            },
        },
    )


def tensor_comparison(left, right):
    if left.keys() != right.keys():
        raise AssertionError("Target keys differ")
    rows = {}
    for key in left:
        a, b = left[key], right[key]
        if (
            a.shape != b.shape
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise AssertionError(f"Invalid target: {key}")
        delta = (a.float() - b.float()).abs()
        rows[key] = {
            "shape": list(a.shape),
            "exact": torch.equal(a, b),
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "relative_l2": (
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
            ).item(),
        }
    return rows


def time_call(function, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    tick = time.perf_counter()
    start.record(torch.cuda.current_stream(device))
    output = function()
    end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    wall = (time.perf_counter() - tick) * 1000
    row = {
        "wall_ms": wall,
        "cuda_event_timeline_ms": start.elapsed_time(end),
        "allocated_before_bytes": allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del output
    return row


def tiny_training_control(
    root,
    dataset,
    camera,
    teachers,
    current,
    future,
    fps,
    baseline_targets,
    candidate_targets,
    device,
):
    """Actual policy forward/backward on a two-layer random model. No 6B throughput claim."""
    import pytest
    from torch.utils.data import default_collate

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import TEACHER_IMAGES
    from tests.policies.lingbot_vla_v2.test_lingbot_vla_v2 import (
        _slot_kwargs,
        _tiny_config,
        tiny_backbone,
    )

    patch = pytest.MonkeyPatch()
    assets = tiny_backbone.__wrapped__(root, patch)
    align = copy.deepcopy(teachers.params)
    align["llm"] = {"dim_out": 64, "image_token_size": 2}
    align["num_task_tokens"] = 4
    for name, target_key in (("depth", "depth"), ("video", "future_video")):
        target = baseline_targets[target_key]
        align[name].update(
            {
                "num_layers": 1,
                "num_heads": 2,
                "dim_head": 8,
                "ff_mult": 1,
                "num_backbone_tokens": target.shape[1],
                "dim_out": target.shape[-1],
            }
        )
    cfg = _tiny_config(
        assets,
        device=str(device),
        align_params=align,
        **_slot_kwargs([[0, 5]], [[5, 6]]),
    )
    policy = make_policy(cfg, ds_meta=dataset.meta).to(device)
    processor, _ = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        delta_timestamps=resolve_delta_timestamps(cfg, dataset.meta),
    )
    batch = processor(default_collate([dataset[i] for i in range(len(current))]))
    batch[TEACHER_IMAGES] = torch.stack([current, future], dim=1)
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    original_state = copy.deepcopy(policy.state_dict())

    class FixedTargets:
        def __init__(self, values):
            self.values = values

        def targets(self, *_args, **_kwargs):
            return self.values

    outputs = []
    for values in (baseline_targets, candidate_targets):
        policy.load_state_dict(original_state)
        policy.zero_grad(set_to_none=True)
        policy._teachers = FixedTargets(values)
        torch.manual_seed(444)
        torch.cuda.manual_seed_all(444)
        loss, logs = policy.forward(batch)
        loss.backward()
        gradients = {
            name: param.grad.detach().float().cpu().clone()
            for name, param in policy.named_parameters()
            if param.grad is not None
        }
        outputs.append((loss.detach().float().cpu(), logs, gradients))
    left, right = outputs
    gradient_rows = tensor_comparison(left[2], right[2])
    rows = {
        "scope": (
            "Two-layer random Qwen3-VL plus sparse action expert; supplied teacher targets; "
            "actual LeRobotDataset and processor; no 6B timing claim"
        ),
        "loss_baseline": left[0].item(),
        "loss_candidate": right[0].item(),
        "loss_exact": torch.equal(left[0], right[0]),
        "logs_baseline": left[1],
        "logs_candidate": right[1],
        "gradient_tensor_count": len(gradient_rows),
        "gradient_tensors_exact": sum(row["exact"] for row in gradient_rows.values()),
        "gradient_max_abs": max(row["max_abs"] for row in gradient_rows.values()),
        "gradients": gradient_rows,
    }
    patch.undo()
    del policy, original_state, outputs
    return rows


def self_check():
    from lerobot.policies.lingbot_vla_v2.model_core.depth_teachers import (
        recover_shift,
        view_plane_uv,
    )

    torch.manual_seed(123)
    uv = view_plane_uv(80, 64, 80 / 64, torch.float32, "cpu")
    z = 1 + torch.rand(3, 64, 80, 1)
    xy = uv[None] * (z + 0.3) / 1.2
    points = torch.cat([xy, z], -1)
    mask = torch.ones(3, 64, 80, dtype=torch.bool)
    mask[1] = False
    mask[2] = False
    mask[2, 0, 0] = True
    expected = recover_shift(points, mask)
    recorder = Recorder(torch.device("cpu"))
    actual = recovery_profiled(points, mask, recorder, "recovery")
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    return {
        "scope": "CPU geometry fixture only; no teacher weights or performance result",
        "recovery_exact": True,
        "shifts": actual.tolist(),
        "diagnostic_stages": recorder.rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo / "src"), str(repo)]
    source_hashes = verify_sources(repo)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.self_check:
        result = self_check() | {
            "source_sha256": source_hashes,
            "source_head": PINNED_HEAD,
        }
        (args.out / "self-check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if not args.config:
        parser.error("--config is required for a real teacher run")
    config = json.loads(args.config.read_text())
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "Real performance runs require CUDA; CPU mode is only --self-check"
        )
    torch.cuda.set_device(device)
    paths = load_manifest(config)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    randomizer = random.Random(config["seed"])
    from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
        DistillationTeachers,
    )

    params = copy.deepcopy(config["align_params"])
    params["depth"].update(moge_path=paths["moge"], morgbd_path=paths["morgbd"])
    params["video"].update(ckpt_path=paths["video"], config_path=paths["video_config"])
    # All paths must resolve. Disable any fallback Hub retrieval before teacher construction.
    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as modeling

    def deny_download(*_args, **_kwargs):
        raise RuntimeError("The harness forbids implicit teacher downloads")

    modeling.hf_hub_download = deny_download
    tick = time.perf_counter()
    teachers = DistillationTeachers(params, device)
    torch.cuda.synchronize(device)
    teacher_load_seconds = time.perf_counter() - tick
    dataset, camera, current_indices, future_indices, input_manifest = (
        load_input_dataset(config, args.out)
    )
    supplied_frames = None
    if config.get("fixture_tensor_path"):
        path = Path(config["fixture_tensor_path"])
        if sha256_file(path) != config.get("fixture_tensor_sha256"):
            raise ValueError("Fixture tensor SHA-256 mismatch")
        supplied_frames = torch.load(path, weights_only=True, map_location="cpu")
        for name in ("current", "future"):
            value = supplied_frames[name]
            expected = (
                max(config["batch_sizes"]),
                3,
                config["image_height"],
                config["image_width"],
            )
            if (
                tuple(value.shape) != expected
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid fixture tensor shape or dtype: {name}")
            if value.min() < 0 or value.max() > 1:
                raise ValueError(f"Fixture tensor is outside [0, 1]: {name}")
            torch.testing.assert_close(
                value, (value * 255).round() / 255, atol=1e-7, rtol=0
            )
        input_manifest["teacher_tensor_override"] = {
            "path": str(path),
            "sha256": config["fixture_tensor_sha256"],
            "scope": "Pinned decoded teacher inputs; generated student fixture remains separate",
            "provenance": config.get("fixture_tensor_provenance"),
        }
    result = {
        "config": config,
        "source_head": PINNED_HEAD,
        "source_sha256": source_hashes,
        "asset_sha256": {
            key: value["sha256"] for key, value in config["assets"].items()
        },
        "input_manifest": input_manifest,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "cpu_threads": torch.get_num_threads(),
        },
        "teacher_load_seconds": teacher_load_seconds,
        "batches": {},
        "timing_note": (
            "Uninstrumented synchronized totals are primary. Diagnostic rows overlap and cannot be summed. "
            "CUDA event intervals include idle time during CPU recovery; use profiler kernels for GPU work."
        ),
    }
    for batch_size in config["batch_sizes"]:
        if supplied_frames is None:
            current = torch.stack(
                [dataset[i][camera] for i in current_indices[:batch_size]]
            ).to(device)
            future = torch.stack(
                [dataset[i][camera] for i in future_indices[:batch_size]]
            ).to(device)
        else:
            current = supplied_frames["current"][:batch_size].to(device)
            future = supplied_frames["future"][:batch_size].to(device)
        if config.get("input_dataset"):
            from torchvision.transforms.functional import resize

            current = (
                resize(
                    current,
                    [config["image_height"], config["image_width"]],
                    antialias=True,
                )
                * 255
            ).round().clamp(0, 255) / 255
            future = (
                resize(
                    future,
                    [config["image_height"], config["image_width"]],
                    antialias=True,
                )
                * 255
            ).round().clamp(0, 255) / 255
        fps = config["effective_fps"]
        functions = {"baseline": partial(teachers.targets, current, future, fps)}
        if config.get("mode", "compare") == "compare":
            functions["joint_depth"] = partial(
                targets_candidate, teachers, current, future, fps, joint=True
            )
        # Warm every variant before measurement. Release targets before the next call.
        for _ in range(config["warmup"]):
            order = list(functions)
            randomizer.shuffle(order)
            for name in order:
                value = functions[name]()
                del value
        torch.cuda.synchronize(device)
        rows = []
        for repeat in range(config["repeats"]):
            order = list(functions)
            randomizer.shuffle(order)
            for position, name in enumerate(order):
                rows.append(
                    {
                        "repeat": repeat,
                        "position": position,
                        "variant": name,
                        **time_call(functions[name], device),
                    }
                )
        baseline = functions["baseline"]()
        joint = functions["joint_depth"]() if "joint_depth" in functions else baseline
        copied = targets_candidate(teachers, current, future, fps)
        comparisons = {"source_vs_copied_baseline": tensor_comparison(baseline, copied)}
        if "joint_depth" in functions:
            comparisons["source_vs_joint_depth"] = tensor_comparison(baseline, joint)
        diagnostics = {}
        for name in functions:
            recorder = Recorder(device)
            target = targets_candidate(
                teachers,
                current,
                future,
                fps,
                joint=name == "joint_depth",
                recorder=recorder,
            )
            diagnostics[name] = {
                "stages": recorder.rows,
                "source_parity": tensor_comparison(
                    baseline if name == "baseline" else joint, target
                ),
            }
            del target
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        )
        with profiler:
            with torch.profiler.record_function("production_teacher_targets"):
                value = functions["baseline"]()
                del value
            recorder = Recorder(device, synchronized=True)
            value = targets_candidate(teachers, current, future, fps, recorder=recorder)
            del value
        profiler.export_chrome_trace(str(args.out / f"batch-{batch_size}-trace.json"))
        if config["tiny_training_control"]:
            if config.get("input_dataset"):
                raise ValueError(
                    "Tiny student control uses the generated six-joint fixture. "
                    "Run real-video profiling with tiny_training_control=false."
                )
            training = tiny_training_control(
                args.out / f"tiny-{batch_size}",
                dataset,
                camera,
                teachers,
                current,
                future,
                fps,
                baseline,
                joint,
                device,
            )
        else:
            training = None
        result["batches"][str(batch_size)] = {
            "timings": rows,
            "target_comparisons": comparisons,
            "diagnostics": diagnostics,
            "tiny_training_control": training,
        }
        del baseline, joint, copied, current, future
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "batches": list(result["batches"]),
                "scope": (
                    "Real teachers on pinned decoded video frames; no full-policy performance claim"
                    if supplied_frames is not None
                    else "Real teachers on deterministic PNG fixture; no full-policy performance claim"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
