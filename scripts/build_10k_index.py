"""Build the 10,000+ image-only SigLIP index (``artifacts/siglip_10k``).

Reuses the exact 4,692 frozen ``artifacts/siglip_adapted`` image vectors without
re-encoding them, encodes only the newly ingested ``expansion_10k.jsonl`` images
on ``mps`` / ``cuda`` / ``cpu``, writes ``data/snapshots/collection_10k.jsonl`` and
``artifacts/siglip_10k/``, and links ``parent_index_sha256`` so ``/api/search/expanded``
and the Web UI automatically unlock the 10,000+ image gallery while keeping the
4,692 frozen v2 benchmark 100% untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.index.build import build_index  # noqa: E402
from curio.represent.encoders import Encoder  # noqa: E402
from curio.retrieve.engine import SearchService, load_items  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="data/snapshots/collection_v2.jsonl")
    ap.add_argument("--extra", default="data/snapshots/expansion_10k.jsonl")
    ap.add_argument("--base-index", default="artifacts/siglip_adapted")
    ap.add_argument("--combined-out", default="data/snapshots/collection_10k.jsonl")
    ap.add_argument("--index-out", default="artifacts/siglip_10k")
    ap.add_argument("--base-probes", default="artifacts/probes/vision_metadata_adapted")
    ap.add_argument("--probes-out", default="artifacts/probes/siglip_10k")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    base_path = ROOT / args.base
    extra_path = ROOT / args.extra
    if not extra_path.exists():
        raise SystemExit(f"Missing {extra_path}; run scripts/ingest_10k_gallery.py first.")

    base = load_items(base_path)
    extra = load_items(extra_path)
    if not extra:
        raise SystemExit(f"Expansion shard {extra_path} is empty.")
    base_ids = {str(r["item_id"]) for r in base}
    overlap = base_ids.intersection(str(r["item_id"]) for r in extra)
    if overlap:
        raise ValueError(f"Expansion contains frozen gallery IDs: {sorted(overlap)[:5]}")
    if any(r.get("split") != "unassigned" for r in extra):
        raise ValueError("All expansion rows must have split='unassigned'")

    backend = SearchService(base, index_dir=ROOT / args.base_index, encoder_name="siglip", root=ROOT)
    if backend.embeddings is None or len(backend.vector_rows) != len(base):
        raise ValueError("Frozen base image index failed validation")
    if backend.manifest.get("data_sha256") != hashlib.sha256(base_path.read_bytes()).hexdigest():
        raise ValueError("Frozen collection_v2 snapshot SHA256 mismatch")

    for row in extra:
        img_file = ROOT / row["image_path"]
        if not img_file.is_file():
            raise FileNotFoundError(f"Missing expansion image: {img_file}")
        if hashlib.sha256(img_file.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Expansion image SHA256 mismatch: {row['item_id']}")

    raw_base_vecs = np.load(ROOT / args.base_index / "image_embeddings.npy", allow_pickle=False)
    raw_base_ids = json.loads((ROOT / args.base_index / "item_ids.json").read_text(encoding="utf-8"))
    id_to_row = {str(k): idx for idx, k in enumerate(raw_base_ids)}
    ordered_base_vecs = raw_base_vecs[[id_to_row[str(r["item_id"])] for r in base]]

    combined_path = ROOT / args.combined_out
    combined_rows = base + extra
    combined_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined_rows),
        encoding="utf-8",
    )

    encoder = Encoder("siglip", device=args.device, adapter_path=backend.manifest.get("adapter_path"))

    class IncrementalReuseEncoder:
        metadata = encoder.metadata

        def encode_images(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
            new_paths = [str(ROOT / p) if not Path(p).is_absolute() else p for p in paths[len(base) :]]
            print(f"Reusing {len(base)} frozen embeddings; encoding {len(new_paths)} new images on {encoder.device}...", flush=True)
            new_vecs = encoder.encode_images(new_paths, batch_size=batch_size)
            return np.concatenate([ordered_base_vecs, new_vecs], axis=0)

    out_dir = ROOT / args.index_out
    manifest = build_index(
        combined_path,
        out_dir,
        "siglip",
        batch_size=args.batch_size,
        encoder=IncrementalReuseEncoder(),
    )
    manifest.update(
        parent_index_sha256=backend.index_sha256,
        expansion_data_sha256=hashlib.sha256(extra_path.read_bytes()).hexdigest(),
        evaluation_status="Retrieval-only 10k+ expansion; frozen v2 benchmark remains separate",
        original_items=len(base),
        expansion_items=len(extra),
        total_items=len(combined_rows),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_sha = hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest()

    base_probes_dir = ROOT / args.base_probes
    if (base_probes_dir / "report.json").is_file():
        import shutil

        probes_out_dir = ROOT / args.probes_out
        probes_out_dir.mkdir(parents=True, exist_ok=True)
        for p in base_probes_dir.glob("*.npz"):
            shutil.copyfile(p, probes_out_dir / p.name)
        if (base_probes_dir / "ontology.json").is_file():
            shutil.copyfile(base_probes_dir / "ontology.json", probes_out_dir / "ontology.json")
        probe_rep = json.loads((base_probes_dir / "report.json").read_text(encoding="utf-8"))
        probe_rep.update(
            data_sha256=manifest["data_sha256"],
            index_sha256=manifest["index_sha256"],
            index_manifest_sha256=manifest_sha,
            index_version=str(manifest.get("index_version", manifest.get("version"))),
            feature_dimension=int(manifest["dimension"]),
            parent_probe_dir=args.base_probes,
            note="Frozen linear probe heads trained on siglip_adapted v2 train split, bound to siglip_10k embeddings for pure-image inference.",
        )
        (probes_out_dir / "report.json").write_text(json.dumps(probe_rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
