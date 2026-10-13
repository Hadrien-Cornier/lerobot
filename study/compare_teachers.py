"""Compare bounded changes with pinned real teachers and real video inputs."""

import argparse
import copy
import gc
import json
import math
import random
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from profile_teachers import (
    load_manifest,
    sha256_file,
    targets_candidate,
    tensor_comparison,
    time_call,
    verify_sources,
)


def position_contiguous(self, x, h, w):
    """Keep the source arithmetic. Change only the interpolation input layout."""
    pos = self.pos_embed[:, 1:]
    n = pos.shape[1]
    if x.shape[1] == n and w == h:
        return pos
    m, h0, w0 = int(math.sqrt(n)), h // self.patch_size, w // self.patch_size
    pos = F.interpolate(
        pos.float().reshape(1, m, m, -1).permute(0, 3, 1, 2).contiguous(),
        mode="bicubic",
        antialias=False,
        scale_factor=((h0 + 0.1) / m, (w0 + 0.1) / m),
    )
    return pos.permute(0, 2, 3, 1).flatten(1, 2).to(x.dtype)


@contextmanager
def layout_variant(name):
    """Install the research layout method outside the timed target call."""
    from lerobot.policies.lingbot_vla_v2.model_core.depth_teachers import Dinov2Backbone

    original = Dinov2Backbone.patch_pos_embed
    if name == "position_contiguous":
        Dinov2Backbone.patch_pos_embed = position_contiguous
    try:
        yield
    finally:
        Dinov2Backbone.patch_pos_embed = original


def precast(teachers):
    """Store only autocast Linear and Conv2d weights in their computation dtype."""
    rows = []
    for name, model, dtype in (
        ("moge", teachers.moge, torch.float16),
        ("morgbd", teachers.morgbd, torch.bfloat16),
    ):
        count, before, after = 0, 0, 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                before += sum(p.numel() * p.element_size() for p in module.parameters())
                module.to(dtype=dtype)
                after += sum(p.numel() * p.element_size() for p in module.parameters())
                count += 1
        rows.append(dict(teacher=name, modules=count, before_bytes=before, after_bytes=after))
    return rows


def move(teachers, device):
    """Move teacher modules without a dtype change."""
    for model in (teachers.moge, teachers.morgbd, teachers.video):
        if model is not None:
            model.to(device)


def call_for(name, teachers, current, future, fps):
    """Return the original source call except for the explicit joint-depth variant."""
    if name == "joint_depth":
        return partial(targets_candidate, teachers, current, future, fps, joint=True)
    return partial(teachers.targets, current, future, fps)


def main():
    """Run paired timings, target checks and isolated memory measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo / "src"), str(repo)]
    source_hashes = verify_sources(repo)
    config = json.loads(args.config.read_text())
    paths = load_manifest(config)
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(config["device"])
    assert device.type == "cuda" and torch.cuda.is_available()
    torch.cuda.set_device(device)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    randomizer = random.Random(config["seed"])
    fixture_path = Path(config["fixture_tensor_path"])
    assert sha256_file(fixture_path) == config["fixture_tensor_sha256"]
    fixture = torch.load(fixture_path, weights_only=True, map_location="cpu")
    for value in fixture.values():
        assert value.shape == (4, 3, 224, 224) and value.dtype == torch.float32
        assert torch.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        torch.testing.assert_close(value, (value * 255).round() / 255, rtol=0, atol=1e-7)
    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as modeling

    def deny_download(*_args, **_kwargs):
        raise RuntimeError("All teacher files must be pinned and local")

    modeling.hf_hub_download = deny_download
    params = copy.deepcopy(config["align_params"])
    params["depth"].update(moge_path=paths["moge"], morgbd_path=paths["morgbd"])
    params["video"].update(ckpt_path=paths["video"], config_path=paths["video_config"])
    baseline = modeling.DistillationTeachers(params, device)
    lower = copy.deepcopy(baseline)
    cast_rows = precast(lower)
    variants = {
        "baseline": baseline,
        "position_contiguous": baseline,
        "precast_linear_conv": lower,
        "joint_depth": baseline,
    }
    result = dict(
        study="teacher_layout_weight_comparison",
        source_sha256=source_hashes,
        config=config,
        precast=cast_rows,
        environment=dict(
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device)
        ),
        batches={},
        timing_note="Paired randomized uninstrumented target calls. Diagnostic traces are separate.",
        memory_note=(
            "Primary timings retain two teachers. "
            "Isolated memory measurements retain only one on CUDA."
        ),
    )

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    fps = config["effective_fps"]
    for batch in (1, 4):
        parity, timings = [], []
        for offset in range(4) if batch == 1 else (0,):
            current = fixture["current"][offset : offset + batch].to(device)
            future = fixture["future"][offset : offset + batch].to(device)
            reference = baseline.targets(current, future, fps)
            for name, teachers in variants.items():
                with layout_variant(name):
                    candidate = call_for(name, teachers, current, future, fps)()
                parity.append(
                    dict(offset=offset, variant=name, targets=tensor_comparison(reference, candidate))
                )
                del candidate
            del reference
        for repeat in range(4 + 24):
            offset = repeat % 4 if batch == 1 else 0
            current = fixture["current"][offset : offset + batch].to(device)
            future = fixture["future"][offset : offset + batch].to(device)
            order = list(variants)
            randomizer.shuffle(order)
            for position, name in enumerate(order):
                with layout_variant(name):
                    row = time_call(call_for(name, variants[name], current, future, fps), device)
                if repeat >= 4:
                    timings.append(
                        dict(repeat=repeat - 4, offset=offset, position=position, variant=name, **row)
                    )
        result["batches"][str(batch)] = dict(target_comparisons=parity, timings=timings)
        save()
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        with profiler:
            for name, teachers in variants.items():
                with layout_variant(name), torch.profiler.record_function(name):
                    value = call_for(name, teachers, current, future, fps)()
                    del value
                    torch.cuda.synchronize(device)
        profiler.export_chrome_trace(str(args.out / f"comparison-batch-{batch}-trace.json"))
        del profiler
        del current, future

    # Measure absolute allocated memory with only the active teacher on CUDA.
    move(baseline, "cpu")
    move(lower, "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    result["isolated_memory"] = {}
    for name, teachers in variants.items():
        move(teachers, device)
        rows = []
        for batch in (1, 4):
            current, future = (fixture[k][:batch].to(device) for k in ("current", "future"))
            with layout_variant(name):
                for _ in range(2):
                    value = call_for(name, teachers, current, future, fps)()
                    del value
                row = time_call(call_for(name, teachers, current, future, fps), device)
            rows.append(dict(batch=batch, **row))
            del current, future
        result["isolated_memory"][name] = rows
        move(teachers, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        save()
    print(json.dumps(dict(output=str(args.out), scope="Actual frozen teachers; no full-policy timing claim")))


if __name__ == "__main__":
    main()
