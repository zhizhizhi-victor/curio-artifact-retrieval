"""Build versioned release assets without changing frozen experiment files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = "curio-resources-v3-10k"
DEFAULT_REPOSITORY = "zhizhizhi-victor/curio-artifact-retrieval"
MODELS = {
    "siglip": ("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
    "clip": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
    "dino": ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
}
REQUIRED_MODELS = {"siglip", "dino"}
SECRET_PATTERNS = {
    "github_token": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "api_key": rb"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16})",
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "bearer": rb"(?i)bearer\s+[A-Za-z0-9._~+/-]{20,}",
    "credential_url": rb"https?://[^\s/:]{2,}:[^\s/@]{3,}@",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    target = ROOT / path
    if not target.is_file():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_model_snapshot(model_id: str, revision: str) -> Path | None:
    model_dir = "models--" + model_id.replace("/", "--")
    bundled = ROOT / "models" / "huggingface" / "hub" / model_dir / "snapshots" / revision
    if bundled.is_dir():
        return bundled
    env_cache = os.getenv("HF_HUB_CACHE") or (
        str(Path(os.environ["HF_HOME"]) / "hub") if os.getenv("HF_HOME") else None
    )
    candidates = []
    if env_cache:
        candidates.append(Path(env_cache) / model_dir / "snapshots" / revision)
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / model_dir / "snapshots" / revision)
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def collect():
    groups = defaultdict(dict)
    expected = {}

    def add(group, relative, source=None, digest=None):
        source = Path(source) if source else ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = Path(relative).as_posix()
        groups[group][relative] = source
        if digest:
            expected[relative] = digest

    gallery_10k = rows("data/snapshots/collection_10k.jsonl")
    gallery_v3 = rows("data/snapshots/collection_v3_expanded.jsonl")
    seen_items = set()
    for row in gallery_10k + gallery_v3:
        if row["item_id"] in seen_items:
            continue
        seen_items.add(row["item_id"])
        if row["license"] != "CC0":
            raise ValueError(f"Unapproved image license: {row['item_id']}")
        for field, digest in (("image_path", "image_sha256"), ("thumb_path", "thumb_sha256")):
            add("gallery", row[field], digest=row[digest])

    for row in rows("data/snapshots/expansion_10k.jsonl"):
        if row["license"] != "CC0":
            raise ValueError(f"Unapproved expansion_10k license: {row['item_id']}")
        original = f"data/raw/{row['source']}/{row['source_id']}.jpg"
        if (ROOT / original).is_file():
            add("originals-10k", original, digest=row["sha256"])

    refs = rows("data/references/bank.jsonl") + rows("data/references/supplemental.jsonl")
    for row in refs:
        if row["license"] != "CC0":
            raise ValueError(f"Unapproved reference license: {row['reference_id']}")
        add("references", row["image_path"], digest=row["image_sha256"])

    # Preserve source responses, failed acquisitions, licenses, snapshots and audit records.
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet", ".txt", ".md"}:
            add("records", path.relative_to(ROOT))
    for path in sorted((ROOT / "artifacts").rglob("*")):
        if path.is_file() and path.name != "server_pid.txt":
            add("artifacts", path.relative_to(ROOT))
    if (ROOT / "logs").is_dir():
        for path in sorted((ROOT / "logs").rglob("*")):
            if path.is_file() and "github" not in path.relative_to(ROOT / "logs").parts and "dsh" not in path.name:
                add("records", path.relative_to(ROOT))
    for path in sorted(ROOT.glob("data_ingest*.log")):
        add("records", path.relative_to(ROOT))
    if (ROOT / "archive").is_dir():
        for path in sorted((ROOT / "archive").rglob("*")):
            if path.is_file():
                add("history", path.relative_to(ROOT))

    for name, (model_id, revision) in MODELS.items():
        model_dir = "models--" + model_id.replace("/", "--")
        snapshot = _find_model_snapshot(model_id, revision)
        if snapshot is None:
            if name in REQUIRED_MODELS:
                raise FileNotFoundError(f"Required model snapshot missing for {name}: {model_id}@{revision}")
            continue
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                relative = f"models/huggingface/hub/{model_dir}/snapshots/{revision}/{path.relative_to(snapshot).as_posix()}"
                add("models-" + name, relative, source=path)
        if not groups["models-" + name] and name in REQUIRED_MODELS:
            raise FileNotFoundError(snapshot)

    for path in sorted((ROOT / "models/licenses").rglob("*")):
        if path.is_file():
            add("records", path.relative_to(ROOT))
    return groups, expected, len(gallery_10k), len(gallery_v3), len(refs)


def build(output, manifest_path, limit, repository, tag):
    output.mkdir(parents=True, exist_ok=True)
    groups, expected, count_10k, count_v3, count_refs = collect()
    inventory = []
    print("Checking source hashes and text credentials...", flush=True)
    for group, files in sorted(groups.items()):
        for relative, source in sorted(files.items()):
            digest = sha256(source)
            if relative in expected and digest != expected[relative]:
                raise ValueError(f"Frozen source hash mismatch: {relative}")
            if source.suffix.lower() in {".json", ".jsonl", ".csv", ".txt", ".md", ".log", ".yaml", ".yml"}:
                content = source.read_bytes()
                for label, pattern in SECRET_PATTERNS.items():
                    if re.search(pattern, content):
                        raise ValueError(f"Credential pattern {label} in {relative}; no matched value printed")
            inventory.append({"group": group, "path": relative, "bytes": source.stat().st_size, "sha256": digest})
        print(f"Audited {group}: {len(files)} files", flush=True)
    assets = []
    for group, files in sorted(groups.items()):
        chunks = [[]]
        size = 0
        for relative, source in sorted(files.items()):
            if size and size + source.stat().st_size > limit:
                chunks.append([])
                size = 0
            chunks[-1].append((relative, source))
            size += source.stat().st_size
        for number, entries in enumerate(chunks, 1):
            name = f"curio-{group}-{number:02d}.zip"
            path = output / name
            names = [relative for relative, _ in entries]
            # No compression of already compressed images/weights: fast and byte-exact.
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for relative, source in entries:
                    archive.write(source, relative)
            if path.stat().st_size >= 2 * 1024**3:
                raise ValueError(f"Release asset exceeds GitHub limit: {name}")
            assets.append({
                "name": name,
                "group": group,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "files": len(entries),
                "unpacked_bytes": sum(source.stat().st_size for _, source in entries),
            })
            by_name = {relative: name for relative in names}
            for entry in inventory:
                if entry["path"] in by_name and entry["group"] == group:
                    entry["asset"] = name
            print(f"Built {name}: {path.stat().st_size / 1024**2:.1f} MiB", flush=True)
    inventory_path = output / "curio-files.jsonl"
    inventory_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inventory),
        encoding="utf-8",
    )
    assets.append({
        "name": inventory_path.name,
        "group": "inventory",
        "bytes": inventory_path.stat().st_size,
        "sha256": sha256(inventory_path),
        "files": len(inventory),
    })
    manifest = {
        "schema": "curio-resource-release-v1",
        "tag": tag,
        "repository": repository,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "gallery_images": count_10k,
        "v3_expanded_gallery_images": count_v3,
        "frozen_gallery_images": 4692,
        "references": count_refs,
        "models": {
            name: {"model_id": value[0], "revision": value[1]}
            for name, value in MODELS.items()
            if ("models-" + name) in groups
        },
        "default_groups": [
            "records",
            "gallery",
            "references",
            "artifacts",
            "models-siglip",
            "models-dino",
        ],
        "assets": assets,
        "excluded": [
            "credentials, browser authorization screenshots and local conversation snapshots",
            "unapproved or unused candidate image files",
            "virtualenv and node_modules",
            "live server PID; original course Word documents",
        ],
        "history_warning": "history contains invalid v1 splits/results and pre-parser-fix experiments; never cite as final results",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    (output / "resources_manifest.json").write_text(payload, encoding="utf-8")
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{entry['sha256']}  {entry['name']}\n" for entry in assets)
        + f"{sha256(output / 'resources_manifest.json')}  resources_manifest.json\n",
        encoding="utf-8",
    )
    print(
        f"DONE: {len(inventory)} files, {sum(x['bytes'] for x in assets) / 1024**3:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/resources_manifest.json")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--chunk-mib", type=int, default=768)
    args = parser.parse_args()
    out_dir = args.output or (ROOT / "_release" / args.tag)
    build(out_dir, args.manifest, args.chunk_mib * 1024**2, args.repository, args.tag)
