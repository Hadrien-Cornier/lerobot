"""Prepare pinned assets and real decoded images for the managed teacher profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

SOURCE_HEAD = "f58f7fb9dbb241304df96e967aea0f5b42ff9a4d"
DATASET_ID = "maximellerbach/omx_multicubes"
DATASET_REVISION = "f7e700308a7049fd3d33adaf84df6e2ece19676d"
VIDEO_FILE = "videos/observation.images.top/chunk-000/file-000.mp4"
CURRENT_INDICES = [30, 150, 300, 600]
FUTURE_INDICES = [value + 49 for value in CURRENT_INDICES]
DINO_CONFIG_SHA256 = "333befe36960b094f2604084802639ab4396c1fa51e51f8d1cb160e29522ac08"
EPISODE_METADATA = {
    "filename": "meta/episodes/chunk-000/file-000.parquet",
    "sha256": "c97599850c94f9f6cdc340d7deddff4a77452dc2792fba6ab490e4634994b2c6",
    "episode_index": 0,
    "length": 807,
    "dataset_from_index": 0,
    "dataset_to_index": 807,
    "video_chunk_index": 0,
    "video_file_index": 0,
    "video_from_timestamp": 0.0,
    "video_to_timestamp": 26.9,
    "verification": "Parent verified the pinned local Parquet row before the managed run.",
}
ASSET_KEYS = {
    "moge": ("Ruicheng/moge-2-vitb-normal", "model.pt"),
    "morgbd": ("robbyant/lingbot-vla-v2-6b", "depth/model.pt"),
    "video": ("robbyant/lingbot-vla-v2-6b", "dino_video/teacher_step_10000.pth"),
    "video_config": ("robbyant/lingbot-vla-v2-6b", "dino_video/config.yaml"),
}
BUCKET_ID = "hadriencornier/jobs-artifacts"


def sha256_file(path):
    """Hash a local file without a large memory copy."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_metadata(study):
    """Select only the four pinned teacher files and one video."""
    assets = json.loads((study / "asset-metadata.json").read_text())
    selected = {}
    for name, (repo_id, filename) in ASSET_KEYS.items():
        item = next(row for row in assets if row["repo_id"] == repo_id and row["filename"] == filename)
        item = dict(item)
        if name == "video_config":
            item["sha256"] = DINO_CONFIG_SHA256
        if len(item["revision"]) != 40 or len(item["sha256"]) != 64:
            raise ValueError(f"Incomplete pinned asset metadata: {name}")
        selected[name] = item
    videos = json.loads((study / "video-metadata.json").read_text())
    video = next(row for row in videos if row["filename"] == VIDEO_FILE)
    return selected, video


def download_verified(item, cache_dir, repo_type="model", repo_id=None, revision=None):
    """Download an immutable revision and verify its size and SHA-256."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=repo_id or item["repo_id"],
            filename=item["filename"],
            revision=revision or item["revision"],
            repo_type=repo_type,
            cache_dir=cache_dir,
        )
    )
    if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
        raise ValueError(f"Pinned download verification failed: {item['filename']}")
    return path.resolve()


def decode_fixture(video_path, video_metadata, destination):
    """Decode the real episode frames and apply the exact image processor resize."""
    import av
    import torch
    from torchvision.transforms.v2.functional import resize

    requested = set(CURRENT_INDICES + FUTURE_INDICES)
    frames, timestamps = {}, {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if float(stream.average_rate) != 30.0:
            raise ValueError(f"Expected a 30 FPS stream, got {stream.average_rate}")
        for index, frame in enumerate(container.decode(stream)):
            if index in requested:
                array = frame.to_ndarray(format="rgb24")
                image = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
                # Match the exact processor: v2 resize, antialias, then round-to-255.
                image = (resize(image, [224, 224], antialias=True) * 255).round().clamp(0, 255) / 255
                frames[index] = image.contiguous()
                timestamps[index] = {
                    "pts": frame.pts,
                    "time_base": str(frame.time_base),
                    "seconds": frame.time,
                }
            if index >= max(requested):
                break
    if set(frames) != requested:
        raise ValueError(f"Video lacks requested frame indices: {sorted(requested - set(frames))}")
    if max(requested) >= EPISODE_METADATA["length"]:
        raise ValueError("Requested frame pair crosses the verified first episode boundary")
    payload = {
        "current": torch.stack([frames[i] for i in CURRENT_INDICES]),
        "future": torch.stack([frames[i] for i in FUTURE_INDICES]),
    }
    torch.save(payload, destination)
    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "video": VIDEO_FILE,
        "video_sha256": video_metadata["sha256"],
        "current_indices": CURRENT_INDICES,
        "future_indices": FUTURE_INDICES,
        "decoded_timestamps": timestamps,
        "episode_metadata": EPISODE_METADATA,
        "preprocessing": (
            "PyAV sequential decode, rgb24, float32 /255, torchvision v2 resize224 antialias=True, "
            "multiply255 round clamp divide255"
        ),
        "scope": (
            "Four real frame pairs from episode zero. "
            "This fixture measures teacher work, not training accuracy."
        ),
        "libraries": {"av": av.__version__, "torch": torch.__version__},
    }
    return provenance


def prepare(study, work, video_only=False):
    """Write the real frame fixture and the filled profile config."""
    selected, video = pinned_metadata(study)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "hub-cache"
    video_path = download_verified(video, cache, "dataset", DATASET_ID, DATASET_REVISION)
    tensor_path = work / "real-frame-pairs.pt"
    provenance = decode_fixture(video_path, video, tensor_path)
    fixture_hash = sha256_file(tensor_path)
    (work / "fixture-provenance.json").write_text(
        json.dumps(provenance | {"tensor_sha256": fixture_hash}, indent=2) + "\n"
    )
    if video_only:
        print(
            json.dumps(
                {
                    "fixture_tensor_path": str(tensor_path),
                    "fixture_tensor_sha256": fixture_hash,
                    "current_indices": CURRENT_INDICES,
                    "future_indices": FUTURE_INDICES,
                    "real_video_verified": True,
                }
            )
        )
        return
    config = json.loads((study / "profile-config.template.json").read_text())
    config.update(
        mode="profile",
        batch_sizes=[1, 4],
        tiny_training_control=False,
        dataset_fps=30,
        effective_fps=0.6122449040412903,
        fixture_tensor_path=str(tensor_path),
        fixture_tensor_sha256=fixture_hash,
        fixture_tensor_provenance=provenance,
    )
    config["align_params"]["video"]["effective_fps"] = config["effective_fps"]
    for name, item in selected.items():
        print(f"Prepare pinned teacher asset: {name}", flush=True)
        path = download_verified(item, cache)
        config["assets"][name] = item | {"path": str(path)}
    (work / "profile-config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        json.dumps(
            {
                "config_path": str(work / "profile-config.json"),
                "teacher_download_bytes": sum(x["size"] for x in selected.values()),
                "video_download_bytes": video["size"],
                "fixture_tensor_sha256": fixture_hash,
            }
        )
    )


def upload_outputs(work):
    """Preserve selected outputs in the existing private bucket."""
    from huggingface_hub import HfApi

    api = HfApi()
    # Use the existing private bucket. Never create or change a bucket.
    info = api.bucket_info(BUCKET_ID)
    if not info.private:
        raise ValueError("Artifact bucket is not private")
    run_id = SOURCE_HEAD[:8] + "-" + datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
    destination = f"hf://buckets/{BUCKET_ID}/lingbot-profile/{run_id}"
    # Exclude downloaded weights, cache state, and raw video. Include results and the trace.
    api.sync_bucket(
        source=str(work),
        dest=destination,
        delete=False,
        exclude=["hub-cache", "hub-cache/**", "venv", "venv/**", "full-assets", "full-assets/**"],
        quiet=True,
    )
    (work / "artifact-destination.txt").write_text(destination + "\n")
    print(f"ARTIFACT_DESTINATION={destination}")


def main():
    """Run preparation, a dry check, or an optional artifact upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    study = Path(__file__).resolve().parent
    work = (args.work_root or study / "runtime").resolve()
    if args.dry_run:
        selected, video = pinned_metadata(study)
        print(
            json.dumps(
                {
                    "source_head": SOURCE_HEAD,
                    "mode": "profile",
                    "work_root": str(work),
                    "teacher_assets": selected,
                    "teacher_download_bytes": sum(x["size"] for x in selected.values()),
                    "video": video,
                    "dataset_id": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "current_indices": CURRENT_INDICES,
                    "future_indices": FUTURE_INDICES,
                    "episode_metadata": EPISODE_METADATA,
                    "bucket": BUCKET_ID,
                    "scope": "Dry check only. No download, upload, package install, or job launch.",
                },
                indent=2,
            )
        )
        return
    if args.upload:
        try:
            upload_outputs(work)
        except Exception as error:
            # Do not print authentication details or URL query strings.
            print(f"ARTIFACT_UPLOAD_FAILED={type(error).__name__}", file=sys.stderr)
            sys.exit(2)
    else:
        prepare(study, work, args.video_only)


if __name__ == "__main__":
    main()
