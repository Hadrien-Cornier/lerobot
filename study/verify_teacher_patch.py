"""Compare bounded changes with pinned real teachers and real video inputs."""

import argparse
import ast
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
)


def verify_patch_sources(repo):
    """Require the original snapshot plus only the proposed constructor addition."""
    snapshot = Path(__file__).resolve().parent / "source-snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    old = (
        '        self.moge = load_depth_teacher(MoGe2, _teacher_file(params, "moge")).to(device).eval()\n'
        '        self.morgbd = load_depth_teacher(MoRGBDEncoder, _teacher_file(params, "morgbd"))'
        ".to(device).eval()\n"
    )
    new = (
        "\n"
        "        def load_depth(cls: type[nn.Module], key: str, dtype: torch.dtype) -> nn.Module:\n"
        "            model = load_depth_teacher(cls, _teacher_file(params, key)).eval()\n"
        "            if self.mixed:\n"
        "                for module in model.modules():\n"
        "                    if isinstance(module, (nn.Linear, nn.Conv2d)):\n"
        "                        module.to(dtype=dtype)\n"
        "            return model.to(device)\n"
        "\n"
        '        self.moge = load_depth(MoGe2, "moge", torch.float16)\n'
        '        self.morgbd = load_depth(MoRGBDEncoder, "morgbd", torch.bfloat16)\n'
    )
    rows = {}
    for relative, item in manifest["files"].items():
        original_path = snapshot / item["snapshot"]
        assert sha256_file(original_path) == item["sha256"]
        expected = original_path.read_text()
        if relative.endswith("/modeling_lingbot_vla_v2.py"):
            assert expected.count(old) == 1
            expected = expected.replace(old, new)
        actual = repo / relative
        assert actual.read_text() == expected, relative
        rows[relative] = sha256_file(actual)
    return rows


def original_teacher_class(modeling):
    """Compile only the original class from the hash-verified upstream snapshot."""
    snapshot = Path(__file__).resolve().parent / "source-snapshot" / "modeling_lingbot_vla_v2.py"
    tree = ast.parse(snapshot.read_text())
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "DistillationTeachers"
    )
    namespace = vars(modeling).copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(snapshot), "exec"), namespace)
    return namespace["DistillationTeachers"]


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


def parameter_buffer_parity(baseline, candidate):
    """Check every teacher parameter and buffer against the original GPU model."""
    rows = []
    for name in ("moge", "morgbd", "video"):
        original = getattr(baseline, name)
        actual = getattr(candidate, name)
        if original is None or actual is None:
            assert original is actual, name
            continue
        module_types = {
            module_name: module
            for module_name, module in original.named_modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        }
        converted = {
            f"{module_name}.{field}" if module_name else field
            for module_name in module_types
            for field in ("weight", "bias")
            if getattr(module_types[module_name], field, None) is not None
        }
        if name == "video":
            converted = set()
        original_params = dict(original.named_parameters())
        actual_params = dict(actual.named_parameters())
        assert original_params.keys() == actual_params.keys(), (name, "parameter names")
        cast_count = 0
        max_parameter_diff = 0.0
        for key, expected in original_params.items():
            expected_dtype = expected.dtype
            if key in converted:
                expected_dtype = torch.float16 if name == "moge" else torch.bfloat16
            expected_value = expected.to(dtype=expected_dtype)
            observed = actual_params[key]
            assert observed.dtype == expected_dtype, (name, key, observed.dtype, expected_dtype)
            assert observed.stride() == expected_value.stride(), (name, key, "stride")
            if not torch.equal(observed, expected_value):
                difference = (observed.float() - expected_value.float()).abs().max().item()
                max_parameter_diff = max(max_parameter_diff, difference)
                raise AssertionError((name, key, "parameter value", difference))
            cast_count += key in converted
        original_buffers = dict(original.named_buffers())
        actual_buffers = dict(actual.named_buffers())
        assert original_buffers.keys() == actual_buffers.keys(), (name, "buffer names")
        max_buffer_diff = 0.0
        for key, expected in original_buffers.items():
            observed = actual_buffers[key]
            assert observed.dtype == expected.dtype, (name, key, "buffer dtype")
            assert observed.stride() == expected.stride(), (name, key, "buffer stride")
            if not torch.equal(observed, expected):
                difference = (observed.float() - expected.float()).abs().max().item()
                max_buffer_diff = max(max_buffer_diff, difference)
                raise AssertionError((name, key, "buffer value", difference))
        rows.append(
            dict(
                teacher=name,
                parameters=len(original_params),
                parameters_cast=cast_count,
                buffers=len(original_buffers),
                max_parameter_abs_diff=max_parameter_diff,
                max_buffer_abs_diff=max_buffer_diff,
            )
        )
    return rows


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
    source_hashes = verify_patch_sources(repo)
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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = original_teacher_class(modeling)(params, device)
    torch.cuda.synchronize(device)
    baseline_initialization = dict(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        current_allocated_bytes=torch.cuda.memory_allocated(device),
    )
    move(baseline, "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    lower = modeling.DistillationTeachers(params, device)
    torch.cuda.synchronize(device)
    candidate_initialization = dict(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        current_allocated_bytes=torch.cuda.memory_allocated(device),
    )
    move(baseline, device)
    torch.cuda.synchronize(device)
    parameter_rows = parameter_buffer_parity(baseline, lower)
    cast_rows = [
        {"teacher": name, "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
        for name, model in (("moge", lower.moge), ("morgbd", lower.morgbd))
    ]
    variants = {
        "baseline": baseline,
        "precast_linear_conv": lower,
    }
    result = dict(
        study="actual_source_patch_verification",
        source_sha256=source_hashes,
        config=config,
        precast=cast_rows,
        initialization_memory=dict(
            baseline=baseline_initialization,
            precast_linear_conv=candidate_initialization,
        ),
        parameter_parity=parameter_rows,
        environment=dict(
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device)
        ),
        batches={},
        timing_note=(
            "Paired randomized uninstrumented target calls after "
            "initialization memory and parameter parity checks. "
            "Diagnostic traces are separate."
        ),
        memory_note=(
            "Primary timings retain two teachers. Isolated memory measurements retain only one on CUDA."
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
