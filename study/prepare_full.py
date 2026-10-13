"""Prepare verified local assets for the original full-policy training workflow."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare import sha256_file


def git_blob_hash(path):
    """Verify small non-LFS files against their pinned Git object hash."""
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Download pinned files and bind each loaded path to its verification record."""
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    study = Path(__file__).resolve().parent
    work = args.work_root.resolve()
    plan = json.loads((study / "full-asset-plan.json").read_text())
    roots = {
        "lerobot/lingbot_vla_v2_base": ("policy", work / "full-assets" / "base"),
        "maximellerbach/omx_multicubes": ("dataset", work / "full-assets" / "omx"),
        "Qwen/Qwen3-VL-4B-Instruct": ("qwen", work / "full-assets" / "qwen"),
    }
    verified = []
    for item in plan["assets"]:
        print(f"Prepare full-policy asset: {item['repo_id']}/{item['filename']}", flush=True)
        path = Path(
            hf_hub_download(
                repo_id=item["repo_id"],
                repo_type=item["repo_type"],
                revision=item["revision"],
                filename=item["filename"],
                cache_dir=work / "hub-cache",
            )
        ).resolve()
        if path.stat().st_size != item["size"]:
            raise ValueError(f"Asset size mismatch: {item['filename']}")
        actual = sha256_file(path)
        if item.get("sha256"):
            if actual != item["sha256"]:
                raise ValueError(f"Asset SHA-256 mismatch: {item['filename']}")
        elif git_blob_hash(path) != item["git_blob_sha1"]:
            raise ValueError(f"Asset Git blob mismatch: {item['filename']}")
        role, root = roots[item["repo_id"]]
        local = root / item["filename"]
        local.parent.mkdir(parents=True, exist_ok=True)
        local.symlink_to(path)
        verified.append(item | dict(role=role, path=str(local.resolve()), sha256=actual))
    teacher_config = json.loads((work / "profile-config.json").read_text())
    for name, item in teacher_config["assets"].items():
        verified.append(item | dict(role="teacher", name=name, path=str(Path(item["path"]).resolve())))
    manifest = work / "full-policy-assets-verified.json"
    manifest.write_text(json.dumps(dict(source_head=plan["source_head"], assets=verified), indent=2) + "\n")
    config = json.loads((study / "full-policy-config.template.json").read_text())
    config.update(
        repo=str(study.parent),
        checkpoint=str(roots["lerobot/lingbot_vla_v2_base"][1]),
        dataset_root=str(roots["maximellerbach/omx_multicubes"][1]),
        qwen_config=str(roots["Qwen/Qwen3-VL-4B-Instruct"][1]),
        teacher_config=str(work / "profile-config.json"),
        asset_verification_manifest=str(manifest),
        comparison_module=str(study / "compare_teachers.py"),
        comparison_module_sha256=sha256_file(study / "compare_teachers.py"),
        out=str(work / "results"),
        allow_candidate=True,
        num_workers=0,
    )
    (work / "full-policy-runtime-template.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(dict(full_asset_count=len(verified), checkpoint_bytes=25503891644)))


if __name__ == "__main__":
    main()
