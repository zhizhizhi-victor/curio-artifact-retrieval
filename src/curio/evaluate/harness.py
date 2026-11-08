"""Evaluate configured retrieval systems with query-level confidence intervals."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from curio.evaluate.metrics import evaluate_ranking
from curio.evaluate.stats import bootstrap_ci
from curio.represent.train import paired_retrieval, read_items
from curio.represent.captions import visual_caption
from curio.represent.encoders import Encoder
from curio.represent.validation import validate_object_splits
from curio.retrieve.engine import SearchService


VARIANTS = (
    "baseline",
    "translated",
    "hybrid",
    "adapted",
    "lexical",
    "visual",
    "image_only",
    "subquery_and",
    "compositional",
)


def _probe_files(path):
    """Fingerprint the exact optional probe files, including unavailable paths."""
    if path is None:
        return {}
    directory = Path(path)
    names = {"report.json", "ontology.json"} | {path.name for path in directory.glob("*.npz")}
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in sorted(names)
            if (directory / name).is_file()}


def evaluate(args):
    variants = [variant.strip() for variant in args.variants.split(",")]
    if not variants or any(variant not in VARIANTS for variant in variants) or len(variants) != len(set(variants)):
        raise ValueError(f"Choose distinct retrieval variants from {VARIANTS}")
    languages = [language.strip() for language in args.languages.split(",")]
    if not languages or any(language not in {"en", "zh"} for language in languages) or len(languages) != len(set(languages)):
        raise ValueError("Choose distinct languages from en,zh")
    probes_path = getattr(args, "probes", None)
    probe_weight = getattr(args, "probe_weight", .15)
    if not 0 <= probe_weight <= 1:
        raise ValueError("probe_weight must be between 0 and 1")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {name: Path(getattr(args, name)) for name in ("data", "queries", "qrels")}
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}
    items = read_items(args.data)
    queries = read_items(args.queries)
    qrel_records = read_items(args.qrels)
    if not queries:
        raise ValueError("Evaluation requires at least one query")
    for name, rows in (("queries", queries), ("qrels", qrel_records)):
        qids = [row["qid"] for row in rows]
        if len(qids) != len(set(qids)):
            raise ValueError(f"Duplicate qid in {name}")
    missing_text = [(query["qid"], language) for query in queries for language in languages
                    if not isinstance(query.get(f"text_{language}"), str) or not query[f"text_{language}"].strip()]
    if missing_text:
        raise ValueError(f"Missing query text; sample: {missing_text[:5]}")
    qrels = {r["qid"]: r["grades"] for r in qrel_records}
    for qid, grades in qrels.items():
        if not isinstance(grades, dict) or any(not isinstance(key, str) or not isinstance(grade, int)
                                               or isinstance(grade, bool) or not 0 <= grade <= 3
                                               for key, grade in grades.items()):
            raise ValueError(f"Qrels for {qid} must map string item IDs to integer grades 0..3")
    missing_qids = sorted({query["qid"] for query in queries} - qrels.keys())
    if missing_qids:
        raise ValueError(
            f"Missing qrels for {len(missing_qids)} queries; sample: {missing_qids[:5]}. "
            "Finish human judgments/adjudication or explicitly supply a matching query subset; "
            "evaluation never silently treats an entirely unjudged query as zero relevance."
        )
    gallery_path = getattr(args, "gallery_ids", None)
    catalog_ids = {str(item["item_id"]) for item in items}
    if len(catalog_ids) != len(items):
        raise ValueError("Duplicate item_id in evaluation catalog")
    for query in queries:
        if unknown := set(qrels[query["qid"]]) - catalog_ids:
            raise ValueError(f"Qrels contain IDs outside the evaluation catalog for {query['qid']}; sample: {sorted(unknown)[:5]}")
    allowed = None
    if gallery_path is not None:
        source_paths["gallery_ids"] = Path(gallery_path)
        source_hashes["gallery_ids"] = hashlib.sha256(source_paths["gallery_ids"].read_bytes()).hexdigest()
        raw_ids = json.loads(Path(gallery_path).read_text(encoding="utf-8"))
        if not isinstance(raw_ids, list) or not raw_ids or any(not isinstance(key, str) or not key for key in raw_ids):
            raise ValueError("gallery-ids must contain a nonempty JSON list of string item IDs")
        allowed = set(raw_ids)
        if len(allowed) != len(raw_ids):
            raise ValueError("Duplicate item ID in gallery-ids")
        if unknown := allowed - catalog_ids:
            raise ValueError(f"Unknown gallery IDs; sample: {sorted(unknown)[:5]}")
        for query in queries:
            judged = set(qrels[query["qid"]])
            if judged != allowed:
                raise ValueError(
                    f"Explicit gallery requires exhaustive qrels (including grade 0) for {query['qid']}; "
                    f"missing {len(allowed - judged)} IDs, extra {len(judged - allowed)} IDs"
                )
    gallery_ids = sorted(allowed if allowed is not None else catalog_ids)
    gallery_sha256 = hashlib.sha256(json.dumps(gallery_ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    label_sources = sorted({r.get("label_source", "unspecified") for r in qrel_records})
    index = Path(args.index)
    source_paths["index_manifest"] = index / "manifest.json"
    source_hashes["index_manifest"] = hashlib.sha256(source_paths["index_manifest"].read_bytes()).hexdigest()
    manifest = json.loads(source_paths["index_manifest"].read_text(encoding="utf-8"))
    if manifest.get("data_sha256") != source_hashes["data"]:
        raise ValueError("Index/data mismatch: rebuild before evaluation")
    encoder = Encoder(args.encoder, args.device, adapter_path=manifest.get("adapter_path"))
    # Warmup is separate from steady-state latency.
    encoder.encode_texts(["a porcelain vase"])
    probe_files = _probe_files(probes_path)
    service = SearchService(items, index_dir=index, encoder=encoder, encoder_name=args.encoder,
                            probes_path=probes_path, probe_weight=probe_weight)
    if _probe_files(probes_path) != probe_files:
        raise ValueError("Probe files changed during evaluation initialization; rerun with immutable artifacts")
    health = service.health()
    summary, details, rankings = [], [], []
    for variant in variants:
        for language in languages:
            records = []
            for query in queries:
                search_kwargs = {"allowed_item_ids": allowed} if allowed is not None else {}
                response = service.search(query[f"text_{language}"], top_k=20, variant=variant, **search_kwargs)
                ranking = [r["item_id"] for r in response["results"]]
                scores = evaluate_ranking(ranking, qrels[query["qid"]])
                record = {
                    "qid": query["qid"],
                    "variant": variant,
                    "language": language,
                    "query_type": query.get("type"),
                    **scores,
                    "latency_ms": response["timings"]["total_ms"],
                    "degraded": response["degraded"],
                    "effective_variant": response.get("effective_variant", variant),
                    "retrieval_mode": response.get("retrieval_mode"),
                    "probe_applied": any(row.get("score_breakdown", {}).get("probe") is not None
                                         and row["score_breakdown"].get("probe_weight", 0) > 0 for row in response["results"]),
                    "probe_coverage": response.get("probe_coverage"),
                    "timings": response["timings"],
                    "warnings": response.get("warnings", []),
                }
                records.append(record)
                details.append(record)
                rankings.append(
                    {
                        "qid": query["qid"],
                        "variant": variant,
                        "language": language,
                        "ranking": ranking,
                        "query": query[f"text_{language}"],
                        "results": response["results"],
                        "effective_variant": record["effective_variant"],
                        "degraded": record["degraded"],
                        "probe_coverage": record["probe_coverage"],
                        "warnings": record["warnings"],
                    }
                )
            metrics = {
                key: bootstrap_ci([r[key] for r in records])
                for key in ["ndcg@10", "ap@10", "precision@5", "recall@20", "mrr@10"]
            }
            latency = np.array([r["latency_ms"] for r in records])
            summary.append(
                {
                    "variant": variant,
                    "language": language,
                    "metrics": metrics,
                    "p50_ms": float(np.median(latency)),
                    "p95_ms": float(np.quantile(latency, 0.95)),
                    "degraded_queries": sum(r["degraded"] for r in records),
                    "effective_variants": dict(Counter(r["effective_variant"] for r in records)),
                    "probe_applied_queries": sum(r["probe_applied"] for r in records),
                }
            )
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    if _probe_files(probes_path) != probe_files:
        raise ValueError("Probe files changed during evaluation; rerun with immutable artifacts")
    for name, path in source_paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[name]:
            raise ValueError(f"Evaluation {name} changed during the run; rerun with immutable inputs")
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        git_sha = "uncommitted"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "git_sha": git_sha,
        "model": encoder.metadata,
        "index": manifest,
        "data_sha256": source_hashes["data"],
        "index_manifest_sha256": source_hashes["index_manifest"],
        "service": health,
        "probes": {"configured": probes_path is not None, "path": str(probes_path) if probes_path else None,
                   "weight": probe_weight, "files_sha256": probe_files,
                   "available": health.get("probes_available", False),
                   "tasks": health.get("probe_tasks", []),
                   "load_ms": health.get("probe_load_ms", 0)},
        "gallery_ids": gallery_ids,
        "gallery_sha256": gallery_sha256,
        "gallery_count": len(gallery_ids),
        "gallery_restricted": allowed is not None,
        "qrels_scope": "exhaustive_explicit_gallery" if allowed is not None else "sparse_full_catalog",
        "query_sha256": source_hashes["queries"],
        "qrels_sha256": source_hashes["qrels"],
        "label_source": label_sources[0] if len(label_sources) == 1 else "mixed",
        "label_sources": label_sources,
        "warning": (
            "Silver metadata compliance benchmark; not independent human visual relevance. "
            if any("silver" in source for source in label_sources)
            else "AI visual audit labels are AI-generated image judgments, not independent human ground truth. "
            if any(source.startswith("ai_") for source in label_sources)
            else "Label provenance is read from qrels. "
        ) + (
            "Every item in the explicitly restricted gallery is judged; no items outside it are ranked. "
            if allowed is not None else "Unjudged items count as nonrelevant within the full catalog. "
        ) + "CI resamples semantic queries within each language.",
        "summary": summary,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, records in [("per_query.jsonl", details), ("rankings.jsonl", rankings)]:
        (output / name).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8"
        )
    lines = [
        "# Retrieval benchmark",
        "",
        f"Visual probes configured: {report['probes']['configured']}; available: {report['probes']['available']}; weight: {probe_weight}. visual→hybrid and image_only→dense_rewritten are fallbacks, not evidence of a probe effect.",
        "",
        f"Label source: **{report['label_source']}**. {report['warning']}",
        "",
        f"Gallery: {len(gallery_ids)} items; restricted: {allowed is not None}; SHA256: `{gallery_sha256}`.",
        "",
        "| Variant | Language | nDCG@10 (95% CI) | P@5 | AP@10 | R@20 | MRR@10 | P50 / P95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        m = row["metrics"]
        n = m["ndcg@10"]
        lines.append(
            f"| {row['variant']} | {row['language']} | {n['mean']:.4f} [{n['lower']:.4f}, {n['upper']:.4f}] | {m['precision@5']['mean']:.4f} | {m['ap@10']['mean']:.4f} | {m['recall@20']['mean']:.4f} | {m['mrr@10']['mean']:.4f} | {row['p50_ms']:.1f} / {row['p95_ms']:.1f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def paired_eval(args):
    items = read_items(args.data)
    validate_object_splits(items)
    selected = [x for x in items if x["split"] == args.split]
    if not selected:
        raise ValueError(f"No items in {args.split}")
    encoder = Encoder(args.encoder, args.device, args.adapter)
    annotations_path = getattr(args, "visual_annotations", None)
    if annotations_path:
        from curio.represent.detail_probes import load_annotations

        annotations = load_annotations(annotations_path, items)
        selected = [item for item in selected if item["item_id"] in annotations]
        if not selected:
            raise ValueError(f"No visual annotations in {args.split}")
        captions = [annotations[item["item_id"]]["caption_en"] for item in selected]
    else:
        captions = [visual_caption(x) for x in selected]
    started = time.perf_counter()
    text = encoder.encode_texts(captions)
    images = encoder.encode_images([x["image_path"] for x in selected], args.batch_size)
    metrics = paired_retrieval(images, text, captions)
    report = {
        "model": encoder.metadata,
        "split": args.split,
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "label_source": "held-out AI image-only caption pairs" if annotations_path else "held-out museum metadata-caption pairs",
        "visual_annotations_sha256": hashlib.sha256(Path(annotations_path).read_bytes()).hexdigest() if annotations_path else None,
        "warning": "Exact identical captions treated as multi-positive. " + (
            "Captions are disclosed AI blind-image judgments; not independently human-judged relevance."
            if annotations_path else
            "Captions contain weak metadata supervision; not an independently human-judged relevance test."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    p.add_argument("--index", default="artifacts/siglip")
    p.add_argument("--encoder", default="siglip")
    p.add_argument("--device", default="auto")
    p.add_argument("--queries", default="eval/queries_dev.jsonl")
    p.add_argument("--qrels", default="eval/qrels_dev.jsonl")
    p.add_argument("--gallery-ids", default=None, help="Explicit JSON item-ID list; requires exhaustive qrels including grade 0")
    p.add_argument("--variants", default="baseline,translated,image_only,hybrid,lexical")
    p.add_argument("--probes", default=None, help="Optional visual probe directory; never loaded implicitly")
    p.add_argument("--probe-weight", type=float, default=.15, help="Visual rerank weight in [0,1]; select on dev only")
    p.add_argument("--languages", default="en,zh")
    p.add_argument("--output", default="artifacts/evaluation/siglip_dev")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--visual-annotations", default=None, help="For paired evaluation, use disclosed AI image captions only")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--adapter", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    return p


def main():
    a = build_parser().parse_args()
    paired_eval(a) if a.paired else evaluate(a)


if __name__ == "__main__":
    main()
