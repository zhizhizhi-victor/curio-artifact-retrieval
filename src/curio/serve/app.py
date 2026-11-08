from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import threading
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from curio.retrieve import SearchService
from curio.retrieve.engine import load_items
from curio.retrieve.image_search import ImageSearchService, METHODS
from curio.retrieve.references import load_reference_bank
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_type: list[str] = Field(default_factory=list, max_length=30)
    material: list[str] = Field(default_factory=list, max_length=30)
    era: tuple[int, int] | None = None
    source: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_era(self):
        if self.era and self.era[0] > self.era[1]:
            raise ValueError("era start must be <= end")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=12, ge=1, le=50)
    variant: Literal[
        "image_only",
        "compositional",
        "subquery_and",
        "baseline",
        "translated",
        "hybrid",
        "adapted",
        "visual",
        "lexical",
    ] = "image_only"
    filters: Filters = Field(default_factory=Filters)
    stream: Literal[False] = False

    @model_validator(mode="after")
    def nonempty_query(self):
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text")
        return self


class Level3Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=12, ge=1, le=50)
    recall_k: int = Field(default=100, ge=12, le=500)

    @model_validator(mode="after")
    def nonempty_query(self):
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text")
        if self.recall_k < self.top_k:
            raise ValueError("recall_k must be >= top_k")
        return self


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=100)
    item_id: str = Field(min_length=1, max_length=100)
    relevant: bool
    comment: str = Field(default="", max_length=2000)


class ReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    reference_ids: list[str] = Field(default_factory=list, max_length=12)
    top_k: int = Field(default=12, ge=1, le=50)
    limit: int = Field(default=4, ge=1, le=12)
    method: Literal["siglip", "dinov2", "dinov2_patch", "shape", "color", "texture", "siglip_shape", "siglip_color", "siglip_texture", "ensemble"] = "siglip"
    aggregation: Literal["mean", "max", "centroid"] = "mean"

    @model_validator(mode="after")
    def nonempty_query(self):
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text")
        return self


def create_app(*, service: SearchService | None = None, data_path: str | Path | None = None,
               index_dir: str | Path | None = None, encoder=None, root: str | Path | None = None,
               feedback_path: str | Path | None = None, web_dir: str | Path | None = None,
               probes_path: str | Path | None = None, probes=None, probe_report: dict | None = None,
               probe_weight: float | None = None) -> FastAPI:
    root_path = Path(root or ".").resolve()
    if service is None:
        source = Path(data_path or os.getenv("CURIO_DATA", "data/snapshots/collection_v2.jsonl"))
        if not source.is_absolute():
            source = root_path / source
        index = Path(index_dir or os.getenv("CURIO_INDEX", f"artifacts/{os.getenv('CURIO_ENCODER', 'clip')}"))
        if not index.is_absolute():
            index = root_path / index
        service = SearchService(load_items(source), index_dir=index, encoder=encoder,
                                encoder_name=os.getenv("CURIO_ENCODER", "clip"), root=root_path,
                                probes_path=probes_path or os.getenv("CURIO_PROBES") or None,
                                probes=probes, probe_report=probe_report,
                                probe_weight=probe_weight if probe_weight is not None else float(os.getenv("CURIO_PROBE_WEIGHT", ".15")))
    application = FastAPI(title="拾遗 Curio", version="0.1.0", description="中英博物馆文本到图像检索；元数据证据与视觉相似度明确区分。")
    application.state.search_service = service
    image_service = ImageSearchService(service)
    application.state.image_search_service = image_service
    default_exp_data = (
        "data/snapshots/collection_10k.jsonl"
        if (root_path / "data/snapshots/collection_10k.jsonl").exists()
        else "data/snapshots/collection_v3_expanded.jsonl"
    )
    default_exp_idx = (
        "artifacts/siglip_10k"
        if (root_path / "artifacts/siglip_10k/manifest.json").exists()
        else "artifacts/siglip_expanded"
    )
    expanded_path = Path(os.getenv("CURIO_EXPANDED_DATA", default_exp_data))
    if not expanded_path.is_absolute():
        expanded_path = root_path / expanded_path
    expanded_index = Path(os.getenv("CURIO_EXPANDED_INDEX", default_exp_idx))
    if not expanded_index.is_absolute():
        expanded_index = root_path / expanded_index
    expanded_manifest_path = expanded_index / "manifest.json"
    expanded_manifest = json.loads(expanded_manifest_path.read_text(encoding="utf-8")) if expanded_manifest_path.exists() else {}
    expanded_enabled = bool(expanded_path.exists() and service.index_sha256
                            and expanded_manifest.get("parent_index_sha256") == service.index_sha256)
    expanded_items = load_items(expanded_path) if expanded_enabled else []
    public_items = {**service.by_id, **{str(row["item_id"]): row for row in expanded_items}}
    expanded_service = None
    expanded_lock = threading.Lock()
    feedback_target = Path(feedback_path or os.getenv("CURIO_FEEDBACK", "data/feedback.jsonl"))
    if not feedback_target.is_absolute():
        feedback_target = root_path / feedback_target

    @application.get("/api/health")
    def health():
        return {**service.health(), "expanded_catalogue_count": len(expanded_items),
                "reference_count": len(load_reference_bank(root_path))}

    @application.post("/api/search")
    def search(request: SearchRequest):
        return service.search(request.query, request.top_k, request.variant, request.filters.model_dump())

    @application.post("/api/search/expanded")
    def expanded_search(request: SearchRequest):
        nonlocal expanded_service
        if not expanded_enabled:
            raise HTTPException(503, "Expanded gallery unavailable for this base image index")
        with expanded_lock:
            if expanded_service is None:
                if service.encoder is None:
                    service._dense("expanded image index initialization")
                if not service.encoder:
                    raise HTTPException(503, "Matching image encoder unavailable")
                exp_probes = root_path / "artifacts/probes" / expanded_index.name
                use_exp_probes = expanded_index.name.startswith("siglip_10k") and (exp_probes / "report.json").is_file()
                candidate = SearchService(
                    expanded_items,
                    index_dir=expanded_index,
                    encoder_name=service.encoder_name,
                    encoder=service.encoder,
                    root=root_path,
                    probes_path=exp_probes if use_exp_probes else None,
                    probe_weight=service.probe_weight if use_exp_probes else 0,
                )
                if candidate.embeddings is None:
                    raise HTTPException(503, "Expanded image index failed validation")
                candidate._encoder_lock = service._encoder_lock
                expanded_service = candidate
        response = expanded_service.search(request.query, request.top_k, request.variant, request.filters.model_dump())
        response.update(gallery_scope="expanded_v3_research", gallery_count=len(expanded_items))
        response["warnings"].append("扩展图库包含新采集图像；原 v2 评测指标不适用于此图库。")
        return response

    @application.post("/api/search/level3")
    def level3_search(request: Level3Request):
        return service.level3_search(request.query, request.top_k, request.recall_k)

    @application.post("/api/search/reference")
    def reference_search(request: ReferenceRequest):
        try:
            return image_service.search_query(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, "Reference image model or bank unavailable") from exc

    @application.post("/api/search/image")
    async def image_search(request: Request, top_k: int = Query(12, ge=1, le=50), method: str = "siglip"):
        if method not in METHODS:
            raise HTTPException(422, "Unknown image search method")
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > 12 * 1024 * 1024:
                raise HTTPException(413, "Image must be at most 12 MB")
        try:
            with Image.open(BytesIO(content)) as source:
                if source.width * source.height > 24_000_000:
                    raise HTTPException(413, "Image must be at most 24 megapixels")
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, "Invalid image") from exc
        try:
            return await run_in_threadpool(image_service.search, [image], top_k=top_k, method=method)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, "Image model or derived index unavailable; no metadata fallback") from exc

    @application.get("/api/references/{reference_id}/image")
    def reference_image(reference_id: str):
        bank = {row["reference_id"]: row for row in load_reference_bank(image_service.bank_dir)}
        row = bank.get(reference_id)
        if row is None:
            raise HTTPException(404, "Reference not found")
        path = (root_path / row["image_path"]).resolve()
        if not path.is_relative_to(image_service.bank_dir.resolve()) or not path.is_file():
            raise HTTPException(404, "Reference not available")
        return FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})

    @application.get("/api/items/{item_id}")
    def item_detail(item_id: str):
        item = public_items.get(item_id)
        if item is None:
            raise HTTPException(404, "Item not found")
        # Do not expose filesystem locations in public responses.
        public = {key: value for key, value in item.items() if key not in {"image_path", "thumb_path"}}
        return {**public, "image_url": f"/api/images/{item_id}", "thumb_url": f"/api/images/{item_id}?thumb=true"}

    @application.get("/api/images/{item_id}")
    def image(item_id: str, thumb: bool = False):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", item_id) or item_id not in public_items:
            raise HTTPException(404, "Image not found")
        item = public_items[item_id]
        path_value = item.get("thumb_path") if thumb else item.get("image_path")
        path_value = path_value or item.get("image_path")
        if not path_value:
            raise HTTPException(404, "Image not available")
        path = Path(path_value)
        if not path.is_absolute():
            path = service.root / path
        path = path.resolve()
        # Catalog whitelist alone is insufficient if a catalog is tampered with:
        # require the resolved path to remain inside processed image storage.
        allowed_roots = [(service.root / "data" / "processed" / name).resolve() for name in ("images", "thumbs")]
        if not any(path.is_relative_to(allowed) for allowed in allowed_roots):
            raise HTTPException(404, "Image not available")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not path.is_file():
            raise HTTPException(404, "Image not available")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"})

    @application.get("/api/facets")
    def facets():
        return service.facets()

    @application.get("/api/metrics")
    def metrics():
        result = service.metrics()
        indexed = service.manifest.get("encoder_metadata") or {}
        loaded = getattr(service.encoder, "metadata", {}) or {}
        result["runtime_config"] = {
            "default_variant": SearchRequest.model_fields["variant"].default,
            "encoder_name": service.encoder_name, "model_id": indexed.get("model_id"),
            "model_revision": indexed.get("revision"), "adapted": bool(service.manifest.get("adapted")),
            "adapter_hash": indexed.get("adapter_hash") or service.manifest.get("adapter_hash"),
            "data_sha256": service.manifest.get("data_sha256"), "index_sha256": service.index_sha256,
            "index_manifest_sha256": service.index_manifest_sha256, "index_version": service.index_version,
            "probes": {"configured": service.probe_configured, "available": bool(service.probe_predictions),
                       "weight": service.probe_weight if service.probe_configured else None,
                       "tasks": sorted(service.probe_predictions), "files_sha256": dict(service.probe_files_sha256)},
        }
        result["runtime_environment"] = {
            "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "process_id": os.getpid(), "logical_cpu_count": os.cpu_count(),
            "encoder_loaded": service.encoder is not None and service.encoder is not False,
            "encoder_device": loaded.get("device"),
            "loaded_encoder": {field: loaded.get(field) for field in ("model_id", "revision", "adapter_hash")},
        }
        path = Path(os.getenv("CURIO_METRICS", "artifacts/evaluation/metrics.json"))
        if not path.is_absolute():
            path = root_path / path
        try:
            result["evaluation"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            report = result["evaluation"]
            if report is not None:
                probes = report.get("probes", {})
                matches = (
                    service.health()["dense_available"]
                    and bool(service.index_manifest_sha256)
                    and report.get("index_manifest_sha256") == service.index_manifest_sha256
                    and report.get("index", {}).get("index_sha256") == service.index_sha256
                    and bool(probes.get("configured")) == service.probe_configured
                )
                if service.probe_configured:
                    matches = (matches and bool(service.probe_predictions)
                               and probes.get("weight") == service.probe_weight
                               and bool(service.probe_files_sha256)
                               and probes.get("files_sha256") == service.probe_files_sha256)
                if not matches:
                    result["evaluation"] = None
                    result["evaluation_warning"] = "评测报告与当前图像索引或视觉探针配置不一致，未展示该报告的指标。"
        except (OSError, ValueError, AttributeError, TypeError):
            result["evaluation"] = None
        return result

    @application.post("/api/feedback", status_code=201)
    def feedback(request: FeedbackRequest):
        if request.item_id not in public_items:
            raise HTTPException(404, "Item not found")
        record = {**request.model_dump(), "created_at": datetime.now(timezone.utc).isoformat()}
        with service._lock:
            try:
                feedback_target.parent.mkdir(parents=True, exist_ok=True)
                with feedback_target.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                raise HTTPException(503, "Feedback could not be saved") from None
            service.feedback.append(record)
        return {"accepted": True, "persisted": True}

    frontend = Path(web_dir) if web_dir else root_path / "web" / "dist"
    if not frontend.exists():
        frontend = root_path / "web"
    if (frontend / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")
    if (frontend / "index.html").is_file():
        @application.get("/")
        def home():
            return FileResponse(frontend / "index.html")

    return application


app = create_app()
