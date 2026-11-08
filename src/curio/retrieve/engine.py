from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import numpy as np

from curio.query import parse_query
from curio.query.ontology import ERAS, VOCAB
from curio.query.parser import term_pattern
from .compositional import (
    DEFAULT_LAMBDA_AND,
    DEFAULT_LAMBDA_NEG,
    DEFAULT_LAMBDA_PROBE,
    DEFAULT_POOL_K,
    DEFAULT_TAU,
    calibrated_probe_logscore,
    decompose_visual_query,
    soft_and_zscores,
)
from .lexical import BM25
from .evidence import retrieval_evidence
import re

IMAGE_ONLY_VARIANTS = frozenset({"image_only", "subquery_and", "compositional"})


def searchable_text(item: dict) -> str:
    keys = ("title", "object_name", "classification", "medium", "culture", "period", "description", "object_type", "material", "motifs", "colors", "tags")
    return " ".join(" ".join(str(x) for x in item.get(key, [])) if isinstance(item.get(key), list) else str(item.get(key) or "") for key in keys)


def load_items(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    ids = [str(item["item_id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item_id in catalog")
    return items


class SearchService:
    """Exact image-vector search with optional BM25 / attribute RRF fusion.

    The encoder is lazy loaded on first dense search. A missing index never
    downloads a model. Diagnostic variants may use metadata-only BM25;
    image_only never uses metadata for ranking or as an implicit fallback.
    """

    def __init__(self, items: list[dict], *, index_dir: str | Path | None = None,
                 encoder: Any = None, encoder_name: str = "clip", embeddings: np.ndarray | None = None,
                 item_ids: list[str] | None = None, root: str | Path = ".",
                 probes_path: str | Path | None = None, probes: Any = None,
                 probe_report: dict | None = None, probe_weight: float = .15,
                 index_manifest: dict | None = None):
        if not 0 <= probe_weight <= 1:
            raise ValueError("probe_weight must be between 0 and 1")
        self.items = items
        self.root = Path(root).resolve()
        self.by_id = {str(item["item_id"]): item for item in items}
        self.item_positions = {str(item["item_id"]): index for index, item in enumerate(items)}
        self.encoder = encoder
        self.encoder_name = encoder_name
        self._encoder_lock = threading.Lock()
        self._lock = threading.Lock()
        self.counts = Counter()
        self.latencies: list[float] = []
        self.feedback: list[dict] = []
        self.manifest: dict = dict(index_manifest or {})
        self.index_sha256: str | None = None
        self.index_manifest_sha256: str | None = None
        self.probe_configured = probes_path is not None or probes is not None
        self.probe_weight = probe_weight
        self.probe_warning: str | None = None
        self.probe_report: dict = dict(probe_report or {})
        self.probe_predictions: dict = {}
        self.probe_files_sha256: dict[str, str] = {}
        self.probe_load_ms = 0.0
        self.indexed_item_count = 0
        self.index_warning: str | None = None
        self.dense_error: str | None = None
        self.texts = [searchable_text(item) for item in items]
        self.lexical = BM25(self.texts)
        self.attributes = [self._attributes(item) for item in items]
        self.index_version = "metadata-" + hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
        self.embeddings = None
        self.vector_rows: list[int] = []
        self.catalog_to_embedding_row: dict[int, int] = {}
        try:
            if embeddings is None and index_dir is not None:
                directory = Path(index_dir)
                if (directory / "image_embeddings.npy").exists():
                    embeddings = np.load(directory / "image_embeddings.npy", allow_pickle=False)
                    item_ids = json.loads((directory / "item_ids.json").read_text(encoding="utf-8"))
                    if (directory / "manifest.json").exists():
                        manifest_bytes = (directory / "manifest.json").read_bytes()
                        self.manifest = json.loads(manifest_bytes.decode("utf-8"))
                        self.index_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if embeddings is not None:
                values = np.asarray(embeddings, dtype=np.float32)
                ids = item_ids if item_ids is not None else [str(item["item_id"]) for item in items]
                if values.ndim != 2 or values.shape[0] != len(ids) or len(set(ids)) != len(ids) or not np.isfinite(values).all():
                    raise ValueError("Invalid embedding shape, IDs, or non-finite vectors")
                if any(str(key) not in self.by_id for key in ids):
                    raise ValueError("Index contains IDs missing from the catalog")
                self.index_sha256 = hashlib.sha256(values.tobytes() + json.dumps(ids).encode()).hexdigest()
                if self.manifest.get("index_sha256") and self.manifest["index_sha256"] != self.index_sha256:
                    raise ValueError("Image embeddings or IDs do not match index_sha256")
                declared_encoder = self.manifest.get("encoder")
                if isinstance(declared_encoder, dict):
                    declared_encoder = declared_encoder.get("encoder")
                if declared_encoder and declared_encoder != encoder_name:
                    raise ValueError(f"Index encoder {declared_encoder!r} does not match configured encoder {encoder_name!r}")
                declared_dimension = self.manifest.get("dimension")
                if declared_dimension and int(declared_dimension) != values.shape[1]:
                    raise ValueError("Index dimension does not match its manifest")
                catalog_sha = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if self.manifest.get("catalog_sha256") and self.manifest["catalog_sha256"] != catalog_sha:
                    raise ValueError("Catalog content changed since indexing; rebuild the image index")
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                if np.any(norms <= 0):
                    raise ValueError("Index contains zero vectors")
                lookup = {str(item["item_id"]): idx for idx, item in enumerate(items)}
                self.vector_rows = [lookup[str(key)] for key in ids]
                self.catalog_to_embedding_row = {cat_idx: emb_row for emb_row, cat_idx in enumerate(self.vector_rows)}
                self.indexed_item_count = len(ids)
                self.embeddings = values / norms
                self.index_version = str(self.manifest.get("index_version", self.manifest.get("version", "vectors-" + hashlib.sha256(values.tobytes()).hexdigest()[:12])))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.index_warning = f"图像索引不可用：{exc}"
            self.embeddings = None
            self.catalog_to_embedding_row = {}
        if self.probe_configured:
            probe_started = time.perf_counter()
            try:
                if probes_path is not None:
                    path = Path(probes_path)
                    if not path.is_absolute():
                        path = self.root / path
                    probe_report = json.loads((path / "report.json").read_text(encoding="utf-8"))
                    files = {"report.json", "ontology.json"} | {entry.name for entry in path.glob("*.npz")}
                    self.probe_files_sha256 = {
                        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                        for name in sorted(files) if (path / name).is_file()
                    }
                    if probes is None:
                        from curio.represent.probes import LinearProbes
                        probes = LinearProbes(path)
                self._prepare_probes(probes, probe_report)
                self.probe_report = dict(probe_report or {})
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.probe_warning = f"视觉探针不可用：{exc}。"
                self.probe_predictions = {}
                self.probe_report = {}
                self.probe_files_sha256 = {}
            self.probe_load_ms = (time.perf_counter() - probe_started) * 1000

    def _prepare_probes(self, probes, report):
        if self.embeddings is None or self.indexed_item_count != len(self.items):
            raise ValueError("Probes require complete image index coverage")
        if not report or not self.manifest.get("data_sha256"):
            raise ValueError("Probe report and index data provenance are required")
        for field, expected in (("index_sha256", self.index_sha256), ("data_sha256", self.manifest["data_sha256"]),
                                ("feature_dimension", self.embeddings.shape[1]), ("index_version", self.index_version)):
            if report.get(field) != expected:
                raise ValueError(f"Probe {field} does not match active image index")
        if self.index_manifest_sha256 and report.get("index_manifest_sha256") != self.index_manifest_sha256:
            raise ValueError("Probe index_manifest_sha256 does not match active image index")
        predictions = probes.predict(self.embeddings)
        for task, task_report in report.get("tasks", {}).items():
            if task not in VOCAB or task_report.get("status") != "trained":
                continue
            prediction = predictions.get(task)
            if prediction is None:
                raise ValueError(f"Trained {task} probe did not return predictions")
            classes = prediction["classes"]
            probabilities = np.asarray(prediction["probabilities"], dtype=np.float32)
            if not classes or len(classes) != len(set(classes)) or set(classes) - set(VOCAB[task]):
                raise ValueError(f"Invalid {task} probe class vocabulary")
            if list(classes) != report["tasks"][task].get("classes"):
                raise ValueError(f"Probe {task} classes differ from training report")
            if probabilities.shape != (len(self.vector_rows), len(classes)) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid {task} probability matrix")
            activation = task_report.get("activation", "softmax")
            if activation not in {"softmax", "sigmoid"}:
                raise ValueError(f"Unsupported {task} probe activation")
            if np.any(probabilities < 0) or np.any(probabilities > 1):
                raise ValueError(f"Invalid {task} probe probabilities")
            if activation == "softmax" and not np.allclose(probabilities.sum(1), 1, atol=1e-4):
                raise ValueError(f"Unnormalized {task} probe probabilities")
            ordered = np.empty_like(probabilities)
            ordered[self.vector_rows] = probabilities
            self.probe_predictions[task] = {"classes": list(classes), "probabilities": ordered, "activation": activation}
        if not self.probe_predictions:
            raise ValueError("No trained vocabulary probe is available")

    @staticmethod
    def _probe_constraint_unknown_reason(parsed, task, value, *, negated=False):
        if (task, value) != ("object_type", "lid"):
            return None
        # This single-label class describes an independent lid. It was never
        # trained to detect whether another object has (or lacks) that component.
        text = parsed.positive_text
        component = re.search(
            r"\b(?:covered|lidded|objects?|vessels?|containers?|artifacts?|items?)\b"
            r"|(?:带|帶|有|附|加|配|含|具)[^，。;；]{0,12}盖|器物|器皿|容器|物件|藏品"
            r"|\b(?:with|having|has|have|featuring|includes?|including|bearing)\b[^,;]{0,40}\b(?:lids?|covers?)\b",
            text, re.I,
        )
        independent_noun = re.search(r"\b(?:lids?|covers?)\b|盖子|器盖|盖\s*$", text, re.I)
        if negated or parsed.slots.get("object_type") != ["lid"] or component or not independent_noun:
            return "component_presence_not_trained"
        return None

    def _query_probe_scores(self, parsed, *, include_negations=False, component_safe=False):
        parts, fields = [], []
        for task, prediction in self.probe_predictions.items():
            for value in parsed.slots.get(task, []):
                if component_safe and self._probe_constraint_unknown_reason(parsed, task, value):
                    continue
                if value in prediction["classes"]:
                    probabilities = prediction["probabilities"][:, prediction["classes"].index(value)]
                    parts.append(probabilities)
                    fields.append((task, value, probabilities, False))
        if include_negations:
            seen = set()
            for negative in parsed.negations:
                task, value = negative["slot"], negative["value"]
                prediction = self.probe_predictions.get(task)
                if component_safe and self._probe_constraint_unknown_reason(parsed, task, value, negated=True):
                    continue
                if (task, value) in seen or prediction is None or value not in prediction["classes"]:
                    continue
                seen.add((task, value))
                probabilities = prediction["probabilities"][:, prediction["classes"].index(value)]
                parts.append(1 - probabilities)
                fields.append((task, value, probabilities, True))
        return (np.mean(parts, axis=0) if parts else None), fields

    def _probe_coverage(self, parsed):
        """Expose covered query classes; missing model classes remain unknown."""
        def entry(task, value, *, negated=False):
            prediction = self.probe_predictions.get(task)
            reason = self._probe_constraint_unknown_reason(parsed, task, value, negated=negated)
            if reason:
                return {"slot": task, "value": value, "status": "unknown", "reason": reason}
            return {"slot": task, "value": value,
                    "status": "covered" if prediction is not None and value in prediction["classes"] else "unknown"}
        return {"positive": [entry(task, value) for task, values in parsed.slots.items() for value in values],
                "negations": [entry(task, value, negated=True) for task, value in dict.fromkeys((negative["slot"], negative["value"]) for negative in parsed.negations)]}

    @staticmethod
    def _attributes(item: dict) -> dict[str, set[str]]:
        fields = {
            "object_type": ("object_type", "object_name", "classification", "title"),
            "material": ("material", "medium"),
            "motifs": ("motifs", "tags"),
            "colors": ("colors",),
            "shape": ("shape",),
        }
        result = {}
        for slot, keys in fields.items():
            text = " ".join(" ".join(map(str, item.get(key, []))) if isinstance(item.get(key), list) else str(item.get(key) or "") for key in keys).lower()
            result[slot] = {value for value, aliases in VOCAB[slot].items() if any(re.search(term_pattern(alias), text, re.I) for alias in aliases)}
        return result

    @staticmethod
    def _average_ranks(candidates, values, valid):
        """Return tie-aware one-based ranks without using item IDs."""
        order = sorted((idx for idx in candidates if valid(values[idx])), key=lambda idx: -float(values[idx]))
        ranks = {}
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + 1 + end) / 2
            ranks.update((idx, rank) for idx in order[start:end])
            start = end
        return ranks

    def health(self) -> dict:
        dense = self.embeddings is not None and self.encoder is not False and not self.dense_error
        return {"status": "ok" if dense and self.indexed_item_count == len(self.items) and not self.probe_warning else "degraded", "items": len(self.items), "indexed_images": len(self.vector_rows) if self.embeddings is not None else 0,
                "dense_available": dense, "encoder_loaded": self.encoder is not None and self.encoder is not False,
                "model_version": self.model_version, "index_version": self.index_version,
                "retrieval_mode": "image_embeddings" if dense else "metadata_only",
                "unindexed_items": len(self.items) - self.indexed_item_count,
                "probes_configured": self.probe_configured, "probes_available": bool(self.probe_predictions),
                "probe_tasks": sorted(self.probe_predictions), "probe_weight": self.probe_weight,
                "probe_load_ms": round(self.probe_load_ms, 2),
                "warnings": [warning for warning in (self.index_warning, self.dense_error, self.probe_warning) if warning] or ([] if dense else ["图像索引未就绪；仅提供元数据诊断检索。"])}

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("model_version", self.manifest.get("model", self.encoder_name))) if self.embeddings is not None else "none (metadata-only)"

    def _encode_query_batch(self, texts: list[str], timings: dict | None = None) -> np.ndarray:
        if self.encoder is False or self.embeddings is None:
            raise RuntimeError("Image embeddings unavailable")
        with self._encoder_lock:
            if self.encoder is None:
                loading_started = time.perf_counter()
                try:
                    from curio.represent.encoders import Encoder
                    adapter_path = self.manifest.get("adapter_path") if self.manifest.get("adapted") else None
                    self.encoder = Encoder(self.encoder_name, adapter_path=adapter_path)
                finally:
                    if timings is not None:
                        timings["model_load_ms"] = (time.perf_counter() - loading_started) * 1000
                        timings["cold_start"] = True
                current_metadata = getattr(self.encoder, "metadata", {})
                indexed_metadata = self.manifest.get("encoder_metadata", {})
                for field in ("model_id", "revision", "adapter_hash"):
                    if indexed_metadata.get(field) and current_metadata.get(field) != indexed_metadata[field]:
                        self.encoder = False
                        raise ValueError(f"Loaded encoder {field} does not match index metadata")
            encode_started = time.perf_counter()
            try:
                matrix = np.asarray(self.encoder.encode_texts(texts), dtype=np.float32)
            finally:
                if timings is not None:
                    timings["encode_ms"] += (time.perf_counter() - encode_started) * 1000
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if len(texts) > 1 and matrix.shape[0] == 1:
            matrix = np.repeat(matrix, len(texts), axis=0)
        if matrix.ndim != 2 or matrix.shape != (len(texts), self.embeddings.shape[1]) or not np.isfinite(matrix).all():
            raise ValueError("Query and image embedding dimensions differ")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Encoder returned a zero query vector")
        return matrix / norms

    def _dense(self, text: str, timings: dict | None = None) -> np.ndarray:
        unit_vector = self._encode_query_batch([text], timings)[0]
        retrieve_started = time.perf_counter()
        scores = np.full(len(self.items), -np.inf, dtype=np.float32)
        scores[self.vector_rows] = self.embeddings @ unit_vector
        if timings is not None:
            timings["retrieve_ms"] += (time.perf_counter() - retrieve_started) * 1000
        return scores

    def _filter(self, idx: int, filters: dict) -> bool:
        item, attrs = self.items[idx], self.attributes[idx]
        for slot in ("object_type", "material"):
            if filters.get(slot) and not set(filters[slot]).intersection(attrs[slot]):
                return False
        if filters.get("source") and item.get("source") not in filters["source"]:
            return False
        if filters.get("era"):
            start, end = item.get("year_from"), item.get("year_to")
            if start is None or end is None or end < filters["era"][0] or start > filters["era"][1]:
                return False
        return True

    def _slot_score(self, idx: int, parsed) -> tuple[float, list[dict], int]:
        attrs, item = self.attributes[idx], self.items[idx]
        hits = []
        total = sum(len(values) for values in parsed.slots.values()) + int(parsed.era is not None)
        for slot, values in parsed.slots.items():
            for value in values:
                if value in attrs[slot]:
                    hits.append({"slot": slot, "value": value, "basis": "metadata"})
        if parsed.era and item.get("year_from") is not None and item.get("year_to") is not None:
            if item["year_to"] >= parsed.era[0] and item["year_from"] <= parsed.era[1]:
                hits.append({"slot": "era", "value": parsed.era_label, "basis": "metadata"})
        conflicts = sum(negative["value"] in attrs[negative["slot"]] for negative in parsed.negations)
        return len(hits) / max(total, 1), hits, conflicts

    def search(self, query: str, top_k: int = 12, variant: str = "hybrid", filters: dict | None = None,
               *, allowed_item_ids: set[str] | list[str] | None = None) -> dict:
        allowed = None if allowed_item_ids is None else set(allowed_item_ids)
        if allowed is not None and (unknown := allowed - self.by_id.keys()):
            raise ValueError(f"Unknown allowed_item_ids; sample: {sorted(unknown)[:5]}")
        started = time.perf_counter()
        parsed = parse_query(query)
        parse_ms = (time.perf_counter() - started) * 1000
        warnings = list(parsed.warnings)
        image_only = variant == "image_only"
        compositional_mode = variant in {"subquery_and", "compositional"}
        pure_image_mode = variant in IMAGE_ONLY_VARIANTS
        if pure_image_mode and parsed.negations:
            warnings = [warning for warning in warnings if "元数据冲突" not in warning]
            warnings.append(
                "排除条件仅可由覆盖该属性的图像探针或反向视觉提示提供软概率/相似度支持，并非不存在该属性的证明；不使用元数据冲突降权，缺失模型类别保持未知。"
                if compositional_mode
                else "排除条件仅可由覆盖该属性的图像探针提供软概率支持，并非不存在该属性的证明；不使用元数据冲突降权，缺失模型类别保持未知。"
            )
            parsed.warnings = list(warnings)
        dense = None
        q_vecs = None
        decomp = decompose_visual_query(query, parsed) if compositional_mode else None
        degraded = False
        effective_variant = variant
        load_timings = {"model_load_ms": 0.0, "cold_start": False, "encode_ms": 0.0, "retrieve_ms": 0.0}
        if compositional_mode:
            batch_texts = [decomp["global_prompt"], *decomp["sub_prompts"], *decomp["neg_prompts"]]
            try:
                q_vecs = self._encode_query_batch(batch_texts, load_timings)
                dense_started = time.perf_counter()
                dense = np.full(len(self.items), -np.inf, dtype=np.float32)
                dense[self.vector_rows] = self.embeddings @ q_vecs[0]
                load_timings["retrieve_ms"] += (time.perf_counter() - dense_started) * 1000
                self.dense_error = None
            except Exception as exc:
                degraded = True
                self.dense_error = f"视觉检索不可用，{variant} 不返回元数据替代结果（{type(exc).__name__}）。"
                warnings.append(self.dense_error)
        elif variant != "lexical":
            if variant == "adapted" and not self.manifest.get("adapted", False):
                degraded = True
                warnings.append("适配模型未加载；当前使用基础编码器。")
            try:
                dense = self._dense(parsed.query if variant == "baseline" else parsed.rewritten, load_timings)
                self.dense_error = None
            except Exception as exc:
                degraded = True
                self.dense_error = (f"视觉检索不可用，image_only 不返回元数据替代结果（{type(exc).__name__}）。" if image_only
                                    else f"视觉检索不可用，已使用元数据 BM25 诊断检索（{type(exc).__name__}）。")
                warnings.append(self.dense_error)
        else:
            warnings.append("当前为元数据 BM25 诊断检索，不是视觉检索。")
        if self.index_warning:
            warnings.append(self.index_warning)
        retrieve_started = time.perf_counter()
        lexical = None if pure_image_mode else self.lexical.score(parsed.rewritten if variant != "baseline" else parsed.positive_text)
        candidates = [idx for idx in range(len(self.items))
                      if (allowed is None or str(self.items[idx]["item_id"]) in allowed)
                      and self._filter(idx, filters or {})]
        if dense is not None and self.indexed_item_count < len(self.items):
            indexed = set(self.vector_rows)
            candidates = [idx for idx in candidates if idx in indexed]
            degraded = True
            warnings.append(f"图像索引覆盖 {self.indexed_item_count}/{len(self.items)} 件；未索引藏品未参与本次视觉结果。")
        slot_info = None if pure_image_mode else [self._slot_score(idx, parsed) for idx in range(len(self.items))]
        load_timings["retrieve_ms"] += (time.perf_counter() - retrieve_started) * 1000
        fusion_started = time.perf_counter()
        probe_scores, probe_fields = None, []
        sub_and_map: dict[int, float] = {}
        neg_pen_map: dict[int, float] = {}
        comp_probe_bonus_map: dict[int, float] = {}
        active_probe_weight = 0.0
        probe_coverage = self._probe_coverage(parsed) if variant in {"image_only", "compositional", "visual"} else None
        if variant in {"image_only", "compositional"} and probe_coverage and any(entry.get("reason") == "component_presence_not_trained"
                                  for entries in probe_coverage.values() for entry in entries):
            warning = "object_type=lid 仅表示独立盖子类别，不能判断器物带盖或无盖；未训练盖子部件存在性探针，该部件条件保持未知，不计入探针分数。"
            warnings.append(warning)
            parsed.warnings.append(warning)
        if image_only:
            if dense is None:
                effective_variant = "unavailable"
            elif not self.probe_predictions:
                effective_variant = "dense_rewritten"
                degraded = True
                warnings.append((self.probe_warning or "未加载可用视觉探针。") + " image_only 使用重写查询的图文嵌入排序。")
            else:
                probe_scores, probe_fields = self._query_probe_scores(parsed, include_negations=True, component_safe=True)
                if probe_scores is None:
                    effective_variant = "dense_rewritten"
                    warnings.append("当前查询没有受已训练视觉探针覆盖的属性；使用重写查询的图文嵌入排序。")
        elif variant == "visual":
            if dense is None or not self.probe_predictions:
                effective_variant = "hybrid"
                degraded = True
                warnings.append((self.probe_warning or "未加载可用视觉探针。") + " visual 已降级为 hybrid。")
            else:
                probe_scores, probe_fields = self._query_probe_scores(parsed)
                if probe_scores is None:
                    effective_variant = "hybrid"
                    warnings.append("当前查询没有受已训练视觉探针覆盖的属性；使用 hybrid 排序。")
        fused: dict[int, float] = {idx: 0.0 for idx in candidates}
        ranks = {idx: {} for idx in candidates}
        rrf_denominator = 1.0
        dense_normalized = {}
        if compositional_mode:
            if dense is None or q_vecs is None:
                effective_variant = "unavailable"
                scores = {}
            else:
                valid_candidates = [idx for idx in candidates if np.isfinite(dense[idx]) and idx in self.catalog_to_embedding_row]
                if not valid_candidates:
                    scores = {}
                    if variant == "compositional" and not self.probe_predictions:
                        effective_variant = "subquery_and"
                        degraded = True
                else:
                    emb_rows = [self.catalog_to_embedding_row[idx] for idx in valid_candidates]
                    cand_embs = self.embeddings[emb_rows]
                    s0 = cand_embs @ q_vecs[0]
                    pool_k = min(DEFAULT_POOL_K, len(valid_candidates))
                    pool_idx = np.argsort(-s0)[:pool_k]
                    sd0 = float(s0[pool_idx].std())
                    scale = sd0 if sd0 > 1e-5 else 0.05
                    combined = s0.astype(np.float64).copy()
                    n_sub = len(decomp["sub_prompts"])
                    n_neg = len(decomp["neg_prompts"])
                    if n_sub >= 2:
                        sub_vecs = q_vecs[1 : 1 + n_sub]
                        sub_mat = cand_embs @ sub_vecs.T
                        z_and = soft_and_zscores(sub_mat, pool_idx, tau=DEFAULT_TAU)
                        combined = combined + DEFAULT_LAMBDA_AND * scale * z_and
                        sub_and_map = {idx: float(z_and[pos]) for pos, idx in enumerate(valid_candidates)}
                    if n_neg >= 1:
                        neg_vecs = q_vecs[1 + n_sub : 1 + n_sub + n_neg]
                        neg_mat = cand_embs @ neg_vecs.T
                        pool_neg = neg_mat[pool_idx]
                        z_neg = np.clip(
                            (neg_mat - pool_neg.mean(0, keepdims=True)) / np.maximum(pool_neg.std(0, keepdims=True), 1e-4),
                            0.0,
                            3.5,
                        ).max(axis=1)
                        combined = combined - DEFAULT_LAMBDA_NEG * scale * z_neg
                        neg_pen_map = {idx: float(z_neg[pos]) for pos, idx in enumerate(valid_candidates)}
                    if variant == "compositional":
                        if not self.probe_predictions:
                            effective_variant = "subquery_and"
                            degraded = True
                            warnings.append((self.probe_warning or "未加载可用视觉探针。") + " compositional 已降级为子句软合取排序（subquery_and）。")
                        else:
                            p_log, probe_fields = calibrated_probe_logscore(
                                self.probe_predictions,
                                self.probe_report,
                                parsed,
                                valid_candidates,
                                self._probe_constraint_unknown_reason,
                            )
                            if p_log is None:
                                effective_variant = "subquery_and"
                                warnings.append("当前查询没有受已训练视觉探针覆盖的属性；compositional 使用子句软合取排序（subquery_and）。")
                            else:
                                p_pool = p_log[pool_idx]
                                p_z = np.clip((p_log - p_pool.mean()) / max(float(p_pool.std()), 1e-4), -3.0, 2.5)
                                combined = combined + DEFAULT_LAMBDA_PROBE * scale * p_z
                                active_probe_weight = DEFAULT_LAMBDA_PROBE
                                probe_scores = np.full(len(self.items), np.nan, dtype=np.float32)
                                for pos, idx in enumerate(valid_candidates):
                                    probe_scores[idx] = float(p_z[pos])
                                    comp_probe_bonus_map[idx] = float(DEFAULT_LAMBDA_PROBE * scale * p_z[pos])
                    scores = {idx: float(combined[pos]) for pos, idx in enumerate(valid_candidates)}
        elif image_only:
            if dense is None:
                scores = {}
            elif probe_scores is None:
                scores = {idx: float(dense[idx]) for idx in candidates if np.isfinite(dense[idx])}
            else:
                dense_ranks = self._average_ranks(candidates, dense, np.isfinite)
                dense_normalized = {idx: 61 / (60 + rank) for idx, rank in dense_ranks.items()}
                ranks = {idx: {"dense": rank} for idx, rank in dense_ranks.items()}
                scores = {idx: value + self.probe_weight * float(probe_scores[idx]) for idx, value in dense_normalized.items()}
        elif variant in {"baseline", "translated", "adapted"} and dense is not None:
            scores = {idx: float(dense[idx]) for idx in candidates if np.isfinite(dense[idx])}
        elif variant == "lexical" or dense is None and variant not in {"hybrid", "visual"}:
            scores = {idx: float(lexical[idx]) for idx in candidates if lexical[idx] > 0}
        else:
            channels = [("bm25", lexical, lambda score: score > 0),
                        ("slots", np.array([info[0] for info in slot_info]), lambda score: score > 0)]
            if dense is not None:
                channels.append(("dense", dense, np.isfinite))
            active_channels = 0
            for name, values, valid in channels:
                channel_ranks = self._average_ranks(candidates, values, valid)
                active_channels += bool(channel_ranks)
                for idx, rank in channel_ranks.items():
                    fused[idx] += 1 / (60 + rank)
                    ranks[idx][name] = rank
            rrf_denominator = max(active_channels, 1) / 61
            scores = {idx: value for idx, value in fused.items() if value > 0}
        # The baseline remains the untouched raw-query dense comparator.
        # Pure image variants never touch metadata conflicts; other variants penalize conflicts without inferring absence.
        for idx in scores if variant not in {"baseline", *IMAGE_ONLY_VARIANTS} else []:
            conflicts = slot_info[idx][2]
            if conflicts:
                scores[idx] -= .05 * conflicts if effective_variant in {"hybrid", "visual"} else .25 * conflicts
            if probe_scores is not None:
                scores[idx] = scores[idx] / rrf_denominator + self.probe_weight * float(probe_scores[idx])
        ordered = sorted(scores, key=lambda idx: (-scores[idx], str(self.items[idx]["item_id"])))[:top_k]
        results = []
        for rank, idx in enumerate(ordered, 1):
            item = self.items[idx]
            matched = [] if pure_image_mode else slot_info[idx][1]
            evidence = [{"kind": "metadata", "slot": hit["slot"], "value": hit["value"],
                         "text": f"元数据匹配 {hit['slot']}={hit['value']}；未进行图像可见性核验。"} for hit in matched]
            if not evidence:
                if compositional_mode and dense is not None:
                    if idx in sub_and_map:
                        evidence = [{"kind": "similarity", "text": f"多条件视觉子句软合取（{len(decomp['sub_prompts'])} 子句，Soft-AND z={sub_and_map[idx]:+.2f}）与跨语言全局相似度联合排序；未使用馆藏元数据。"}]
                    else:
                        evidence = [{"kind": "similarity", "text": "按跨语言视觉形态桥接与图文嵌入相似度排序，未核验具体视觉属性；未使用馆藏元数据。"}]
                    if idx in neg_pen_map and neg_pen_map[idx] > 0:
                        evidence.append({"kind": "similarity", "text": f"反向视觉对比提示软抑制 z={neg_pen_map[idx]:.2f}（嵌入相似度软信号，并非属性不存在证明）。"})
                else:
                    evidence = [{"kind": "similarity" if dense is not None else "metadata", "text": "按图文嵌入相似度排序，未核验具体视觉属性。" if dense is not None else "按元数据文本相关性排序，未核验图像内容。"}]
            for task, value, probabilities, negated in probe_fields:
                probability = float(probabilities[idx])
                evidence.append({"kind": "visual_probe", "slot": task, "value": value,
                                 "probability": probability, "negated": negated,
                                 "support_probability": 1 - probability if negated else probability,
                                 "text": (f"弱监督视觉分类预测 {task}={value}，概率 {probability:.1%}；排除条件采用 1−P={1 - probability:.1%} 软支持，不能证明属性不存在，并非人工确认或视觉核验。" if negated
                                          else f"弱监督视觉分类预测 {task}={value}，概率 {probability:.1%}；这是图像嵌入分类器预测，并非人工确认或视觉核验。")})
            key = str(item["item_id"])
            probe_val = float(probe_scores[idx]) if probe_scores is not None and np.isfinite(probe_scores[idx]) else None
            breakdown = {
                "dense": float(dense[idx]) if dense is not None and np.isfinite(dense[idx]) else None,
                "bm25": None if pure_image_mode else round(float(lexical[idx]), 6),
                "slots": None if pure_image_mode else slot_info[idx][0],
                "negative_conflicts": None if pure_image_mode else slot_info[idx][2],
                "rrf_ranks": ranks.get(idx, {}),
                "probe": probe_val,
                "probe_bonus": comp_probe_bonus_map.get(idx, 0.0) if compositional_mode else (self.probe_weight * probe_val if probe_val is not None else 0.0),
                "probe_weight": active_probe_weight if compositional_mode else (self.probe_weight if probe_val is not None else 0.0),
                "rrf_normalized": fused[idx] / rrf_denominator if probe_val is not None and not pure_image_mode else None,
                "dense_rank_normalized": dense_normalized.get(idx),
                "scoring_mode": "image_only" if image_only else (effective_variant if compositional_mode else ("metadata_diagnostic" if dense is None else effective_variant)),
            }
            if compositional_mode:
                breakdown["subquery_soft_and"] = round(sub_and_map[idx], 6) if idx in sub_and_map else None
                breakdown["negative_penalty"] = round(neg_pen_map[idx], 6) if idx in neg_pen_map else None
                breakdown["subclause_count"] = len(decomp["sub_prompts"]) if decomp else 0
            results.append({"item_id": key, "rank": rank, "score": round(scores[idx], 6), "title": item.get("title", ""),
                            "thumb_url": f"/api/images/{key}?thumb=true", "image_url": f"/api/images/{key}",
                            "source": item.get("source"), "source_url": item.get("source_url"),
                            "matched_slots": matched, "evidence": evidence,
                            "score_breakdown": breakdown})
        dense_order = (sorted((idx for idx in candidates if np.isfinite(dense[idx])),
                              key=lambda idx: (-float(dense[idx]), str(self.items[idx]["item_id"])))
                       if dense is not None else [])
        probe_top_support = []
        if dense_order:
            for task, value, probabilities, negated in probe_fields:
                support = 1 - probabilities if negated else probabilities
                probe_top_support.append({
                    "slot": task, "value": value, "negated": negated,
                    "dense_top_support": float(support[dense_order[0]]),
                    "max_support": float(max(support[idx] for idx in dense_order)),
                })
        effective_pw = active_probe_weight if compositional_mode else self.probe_weight
        confidence = retrieval_evidence(
            parsed=parsed, dense_scores=[float(dense[idx]) for idx in dense_order],
            dense_available=dense is not None, results_available=bool(results), degraded=degraded,
            probe_coverage=probe_coverage,
            probe_available=bool(self.probe_predictions) if variant != "subquery_and" else False,
            probe_weight_positive=probe_scores is not None and effective_pw > 0,
            probe_top_support=probe_top_support,
        )
        confidence["scope"] = "dense_recall_before_fusion"
        confidence["dense_top_item_id"] = str(self.items[dense_order[0]]["item_id"]) if dense_order else None
        confidence["final_top_item_id"] = results[0]["item_id"] if results else None
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        elapsed = (time.perf_counter() - started) * 1000
        with self._lock:
            self.counts["searches"] += 1
            self.counts[f"variant_{variant}"] += 1
            self.counts["degraded_searches"] += int(degraded)
            self.counts["cold_start_searches"] += int(load_timings["cold_start"])
            if not load_timings["cold_start"]:
                self.latencies.append(elapsed)
            self.latencies = self.latencies[-1000:]
        if filters and not results:
            warnings.append("显式筛选没有匹配结果；未自动放宽筛选。")
        return {"request_id": str(uuid.uuid4()), "query": query, "parsed": parsed.to_dict(), "variant": variant,
                "effective_variant": effective_variant,
                "probe_coverage": probe_coverage if variant in {"image_only", "compositional"} else None,
                "confidence": confidence,
                "results": results, "timings": {"parse_ms": round(parse_ms, 3), "total_ms": round(elapsed, 3),
                                                "warm_total_ms": round(max(0, elapsed - load_timings["model_load_ms"]), 3),
                                                "fusion_ms": round(fusion_ms, 3),
                                                **{key: round(value, 3) if isinstance(value, float) else value for key, value in load_timings.items()}},
                "model_version": self.model_version, "index_version": self.index_version, "degraded": degraded,
                "retrieval_mode": "image_embeddings" if dense is not None else "unavailable" if pure_image_mode else "metadata_only", "warnings": warnings}

    @staticmethod
    def _level3_visual_query(parsed) -> str:
        """Remove recognized era language before stage-1 rewriting/encoding."""
        text = parsed.query
        if parsed.era is not None:
            aliases = {alias for _, _, era_aliases in ERAS.values() for alias in era_aliases}
            for alias in sorted(aliases, key=len, reverse=True):
                text = re.sub(term_pattern(alias), " ", text, flags=re.I)
            text = re.sub(r"(?<!\d)-?\d{3,4}\s*(?:到|至|[-–—])\s*-?\d{3,4}(?!\d)", " ", text)
            text = re.sub(r"\b(?:dynasty|period)\b", " ", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip(" ,，。;；")

    @staticmethod
    def _level3_catalogue_status(item, attrs, parsed, slot, value, *, negated=False):
        if slot == "era":
            start, end = item.get("year_from"), item.get("year_to")
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not np.isfinite(start) or not np.isfinite(end) or start > end):
                return "unknown", "missing_or_invalid_catalogue_date_range"
            overlap = end >= parsed.era[0] and start <= parsed.era[1]
            return ("satisfied", "catalogue_date_range_overlap") if overlap else ("violated", "disjoint_catalogue_date_range")
        if value in attrs.get(slot, set()):
            return ("violated", "excluded_catalogue_label_present") if negated else ("satisfied", "catalogue_label_match")
        # The alias extractor is neither exhaustive nor an absence detector.
        # An unrecognized field or another label cannot prove contradiction.
        return "unknown", "catalogue_label_absence_not_verified"

    def level3_search(self, query: str, top_k: int = 12, recall_k: int = 100,
                      *, catalogue_weight: float = .2) -> dict:
        """Research-mode two-stage retrieval for mixed visual/catalogue queries.

        Stage 1 is pure image retrieval. Stage 2 only uses catalogue fields for
        historical/object/material constraints and labels that evidence
        separately. This is deliberately not part of the default image_only
        route and does not infer dynasty or authenticity from pixels.
        """
        started = time.perf_counter()
        if top_k < 1 or recall_k < 1:
            raise ValueError("top_k and recall_k must be positive")
        if not np.isfinite(catalogue_weight) or not 0 <= catalogue_weight <= 1:
            raise ValueError("catalogue_weight must be between 0 and 1")
        parsed = parse_query(query)
        visual_query = self._level3_visual_query(parsed)
        recall_k = max(top_k, min(int(recall_k), len(self.items)))
        stage1 = self.search(visual_query, top_k=recall_k, variant="image_only",
                             allowed_item_ids=None if visual_query else set())
        stage2_started = time.perf_counter()
        catalogue_constraints = []
        catalogue_constraints.extend((slot, value, False) for slot in ("object_type", "material")
                                     for value in parsed.slots.get(slot, []))
        catalogue_constraints.extend((entry["slot"], entry["value"], True) for entry in parsed.negations)
        if parsed.era:
            catalogue_constraints.append(("era", parsed.era_label or str(parsed.era), False))
        catalogue_constraints = list(dict.fromkeys(catalogue_constraints))
        has_catalogue_constraints = bool(catalogue_constraints)
        effective_weight = catalogue_weight if has_catalogue_constraints else 0.0
        stage1_values = [result["score"] for result in stage1["results"]]
        visual_ranks = self._average_ranks(range(len(stage1_values)), stage1_values, np.isfinite)
        unverified_constraints = [
            {"slot": slot, "value": value, "status": "unknown", "basis": "unverified_visual_constraint"}
            for slot, values in parsed.slots.items() if slot not in {"object_type", "material"}
            for value in values
        ] + [{"slot": "unsupported", "value": value, "status": "unknown", "basis": "unsupported_constraint"}
             for value in parsed.unsupported]
        ranked = []
        for position, result in enumerate(stage1["results"]):
            item = self.by_id[str(result["item_id"])]
            attrs = self.attributes[self.item_positions[str(result["item_id"])]]
            evidence = []
            matched = 0
            for slot, value, negated in catalogue_constraints:
                status, reason = self._level3_catalogue_status(item, attrs, parsed, slot, value, negated=negated)
                matched += status == "satisfied"
                evidence.append({"slot": slot, "value": value, "negated": negated,
                                 "status": status, "basis": "catalogue", "reason": reason})
            metadata_score = matched / len(catalogue_constraints) if catalogue_constraints else 0.0
            visual_rank_score = (1.0 if len(stage1_values) == 1 else
                                 (len(stage1_values) - visual_ranks[position]) / (len(stage1_values) - 1))
            final_score = (1 - effective_weight) * visual_rank_score + effective_weight * metadata_score
            result = dict(result)
            result["level3_evidence"] = evidence
            result["level3_unverified_constraints"] = unverified_constraints
            result["score"] = round(final_score, 6)
            result["score_breakdown"] = {**result.get("score_breakdown", {}), "level3_visual_rank": round(visual_rank_score, 6),
                                         "level3_catalogue": round(metadata_score, 6), "level3_catalogue_weight": effective_weight}
            ranked.append(result)
        ranked.sort(key=lambda row: (-row["score"], str(row["item_id"])))
        for rank, result in enumerate(ranked[:top_k], 1):
            result["rank"] = rank
        warnings = list(stage1.get("warnings", []))
        warnings.append("三级研究模式分离视觉召回与馆藏字段证据；年代、文化和真实性不由图像确认。")
        warnings.append("组合权重是未经开发集验证的研究原型配置；组合分数和阶段一证据均不是最终结果正确率。")
        if not visual_query:
            warnings.append("移除历史词后没有可检索的视觉描述；未返回任意视觉结果。")
        confidence = retrieval_evidence(
            parsed=parsed, dense_scores=[],
            dense_available=stage1["confidence"]["signal_availability"]["dense"],
            results_available=bool(ranked), degraded=stage1.get("degraded", False),
            probe_coverage=stage1.get("probe_coverage"), probe_available=bool(self.probe_predictions),
        )
        confidence.update({
            "scope": "level3_mixed_ranking", "label": "组合排序未经校准",
            "level": "limited" if ranked else "unknown",
            "basis": "组合排序未经独立标注校准；阶段一诊断单独报告，不能确认最终结果正确或年代真实。",
        })
        confidence["ranking_separation"] = {
            "score_space": "prototype_visual_rank_plus_catalogue", "candidate_count": len(ranked),
            "top1": None, "top2": None, "top1_top2_margin": None,
            "interpretation": "mixed_score_not_calibrated_for_correctness",
        }
        elapsed = (time.perf_counter() - started) * 1000
        timings = {**stage1.get("timings", {}),
                   "stage1_total_ms": stage1["timings"]["total_ms"],
                   "stage2_ms": round((time.perf_counter() - stage2_started) * 1000, 3),
                   "total_ms": round(elapsed, 3),
                   "warm_total_ms": round(max(0, elapsed - stage1["timings"].get("model_load_ms", 0)), 3)}
        return {"request_id": str(uuid.uuid4()), "query": query, "parsed": parsed.to_dict(), "variant": "level3_research",
                "effective_variant": "two_stage_visual_catalogue", "stage1_recall_k": len(stage1["results"]),
                "stage1_query": visual_query, "stage1_parsed": stage1["parsed"],
                "stage1_confidence": stage1["confidence"],
                "fusion_config": {"catalogue_weight": effective_weight, "visual_weight": 1 - effective_weight,
                                  "validation_status": "unvalidated_prototype"},
                "results": ranked[:top_k], "confidence": confidence, "degraded": stage1.get("degraded", False),
                "retrieval_mode": "two_stage_research", "warnings": warnings,
                "timings": timings, "model_version": self.model_version, "index_version": self.index_version}

    def facets(self) -> dict:
        result = {}
        for slot in ("object_type", "material", "motifs", "colors"):
            counts = Counter(value for attrs in self.attributes for value in attrs[slot])
            result[slot] = [{"value": value, "count": count} for value, count in sorted(counts.items())]
        result["source"] = [{"value": value, "count": count} for value, count in sorted(Counter(str(item.get("source", "unknown")) for item in self.items).items())]
        starts = [item["year_from"] for item in self.items if item.get("year_from") is not None]
        ends = [item["year_to"] for item in self.items if item.get("year_to") is not None]
        result["era"] = [min(starts), max(ends)] if starts and ends else None
        return result

    def metrics(self) -> dict:
        with self._lock:
            return {**dict(self.counts), "latency_sample_count": len(self.latencies),
                    "p50_ms": round(float(np.percentile(self.latencies, 50)), 2) if self.latencies else None,
                    "p95_ms": round(float(np.percentile(self.latencies, 95)), 2) if self.latencies else None,
                    "feedback_count": len(self.feedback), "scope": "this server process; last 1000 warm searches; model loading excluded"}
