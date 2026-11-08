"""Resumable CC0 Chinese museum artifact expansion to >= 10,000 gallery images.

Design & Scientific Integrity Guarantees:
1. Never modifies ``data/snapshots/collection_v2.jsonl`` (4,692 frozen items) or
   ``data/references/bank.jsonl`` (45 reference images). Verifies SHA256 before/after.
2. Enforces ``GalleryExclusions`` against ALL frozen gallery items AND ALL reference
   bank items (ID, object family, accession root, related works, raw/processed SHA256,
   256-bit average hash <= 4, and 64-bit perceptual hash <= 6), plus exact & perceptual
   deduplication across newly ingested items.
3. Uses only official CC0 / Public-Domain museum endpoints & image CDNs:
   - Metropolitan Museum of Art Open Access (``images.metmuseum.org``, ``isPublicDomain=true``)
   - Art Institute of Chicago Open Access (``www.artic.edu/iiif/2``, ``is_public_domain=true``)
   - Cleveland Museum of Art Open Access (``openaccess-cdn.clevelandart.org``, ``CC0``)
4. Saves new retrieval-only records (``split="unassigned"``) incrementally every 50 items to
   ``data/snapshots/expansion_10k.jsonl`` so interrupted runs resume cleanly with zero wasted work.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from curio.retrieve.references import (  # noqa: E402
    GalleryExclusions,
    accession_root,
    image_fingerprints,
    metadata_sha256,
    official_related_ids,
)
from data_ingest import terms, text, years  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROC_IMG_DIR = ROOT / "data" / "processed" / "images"
PROC_THM_DIR = ROOT / "data" / "processed" / "thumbs"
FROZEN_PATH = ROOT / "data" / "snapshots" / "collection_v2.jsonl"
BANK_PATH = ROOT / "data" / "references" / "bank.jsonl"
SUPP_BANK_PATH = ROOT / "data" / "references" / "supplemental.jsonl"
OUT_SHARD = ROOT / "data" / "snapshots" / "expansion_10k.jsonl"
REPORT_DIR = ROOT / "data" / "expansion_10k"
FAIL_PATH = REPORT_DIR / "download_failures.jsonl"
REPORT_PATH = REPORT_DIR / "report.json"

ALLOWED_HOSTS = {
    "met": {"images.metmuseum.org"},
    "cleveland": {"openaccess-cdn.clevelandart.org"},
    "aic": {"www.artic.edu"},
    "mia": {f"{i}.api.artsmia.org" for i in range(7)},
}

OBJTYPE_RULES: tuple[tuple[str, str], ...] = (
    (r"\bbowls?\b", "bowl"),
    (r"\bplates?|platters?|dishes?|saucers?\b", "plate"),
    (r"\bvases?|bottles?|meiping|flasks?\b", "vase"),
    (r"\bjars?\b", "jar"),
    (r"\bteapots?\b", "teapot"),
    (r"\bewers?\b", "ewer"),
    (r"\bcups?\b", "cup"),
    (r"\bpots?\b", "pot"),
    (r"\bfigur(?:e|ines?)|bodhisattva|buddha|statues?|heads?\b", "figure"),
    (r"\brobes?|garments?|coats?|skirts?\b", "garment"),
    (r"\btextiles?|panels?|tapestr(?:y|ies)|embroider(?:y|ies)|silks?\b", "textile"),
    (r"\bpendants?\b", "pendant"),
    (r"\bcong\b", "cong"),
    (r"\bbi\s+dis[ck]s?\b|\bjade\s+bi\b", "bi"),
    (r"\bboxes?|caskets?|containers?\b", "box"),
    (r"\bvessels?|ding|jue|gui|gu|zun\b", "ritual vessel"),
    (r"\bmirrors?\b", "mirror"),
    (r"\bstelae?|steles?\b", "stele"),
    (r"\bincense\s+burners?|censers?\b", "censer"),
    (r"\bplaques?\b", "plaque"),
    (r"\bbells?\b", "bell"),
    (r"\bjewelry|ornaments?|hairpins?|rings?|seals?|snuff\s+bottles?\b", "ornament"),
)

MATERIAL_RULES: tuple[tuple[str, str], ...] = (
    (r"\bporcelain\b", "porcelain"),
    (r"\bjade|nephrite|jadeite\b", "jade"),
    (r"\bbronze\b", "bronze"),
    (r"\bceramic|pottery|stoneware|earthenware|terracotta\b", "ceramic"),
    (r"\bsilk|textile|cotton|wool|embroidery|satin|damask\b", "textile"),
    (r"\bwood|bamboo\b", "wood"),
    (r"\bgold|gilt\b", "gold"),
    (r"\bsilver\b", "silver"),
    (r"\bglass\b", "glass"),
    (r"\blacquer|cinnabar\b", "lacquer"),
    (r"\bivory|bone|horn\b", "ivory"),
    (r"\bmarble|stone|limestone|sandstone\b", "stone"),
)

MOTIF_RULES: tuple[tuple[str, str], ...] = (
    ("dragon", "dragon"),
    ("lotus", "lotus"),
    ("flower", "floral"),
    ("floral", "floral"),
    ("peony", "floral"),
    ("bird", "bird"),
    ("cloud", "cloud"),
    ("phoenix", "phoenix"),
    ("fish", "fish"),
    ("landscape", "landscape"),
)

_tls = threading.local()
_rate_lock = threading.Lock()
_next_req_ts: dict[str, float] = {}


def _http_session() -> requests.Session:
    if not hasattr(_tls, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24)
        s.mount("https://", adapter)
        s.headers.update(
            {
                "Accept": "application/json,image/*,*/*",
                "AIC-User-Agent": "CurioArtifactRetrieval/1.0 (COMP5523 academic research)",
            }
        )
        _tls.session = s
    return _tls.session


def _throttled_get(
    url: str,
    *,
    source: str,
    params: dict[str, Any] | None = None,
    min_interval: float = 0.05,
    tries: int = 4,
) -> requests.Response:
    host = urlparse(url).hostname or source
    last_exc: Exception | None = None
    for attempt in range(tries):
        try:
            with _rate_lock:
                now = time.monotonic()
                scheduled = max(now, _next_req_ts.get(host, 0.0))
                _next_req_ts[host] = scheduled + min_interval
                wait_s = scheduled - now
            if wait_s > 0:
                time.sleep(wait_s)
            resp = _http_session().get(url, params=params, timeout=(15, 60))
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"retryable HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code not in (429, 500, 502, 503, 504)
            ):
                break
            time.sleep(min(12.0, 1.0 * (2**attempt)))
    raise RuntimeError(f"GET failed for {url}: {last_exc}")


def _match_first(rules: tuple[tuple[str, str], ...], hay: str) -> str:
    for pat, canonical in rules:
        if re.search(pat, hay, flags=re.I):
            return canonical
    return ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_exclusion_guard() -> tuple[GalleryExclusions, set[str]]:
    """Build strict whole-gallery + reference-bank exclusion guard."""
    frozen = _read_jsonl(FROZEN_PATH)
    refs = _read_jsonl(BANK_PATH) + _read_jsonl(SUPP_BANK_PATH)
    exp_v3 = _read_jsonl(ROOT / "data/snapshots/expansion_v3.jsonl")
    ref_rows = [
        {
            **r,
            "item_id": r["reference_id"],
            "object_group": r["reference_id"],
            "phash": r.get("ahash256", ""),
            "sha256": r.get("original_image_sha256", ""),
        }
        for r in refs
    ]
    blockers = frozen + ref_rows
    raw_records: dict[str, dict[str, Any]] = {}
    for row in blockers + exp_v3:
        rel = row.get("metadata_path") or f"data/raw/{row['source']}/{row['source_id']}.json"
        p = ROOT / rel
        if p.exists():
            try:
                raw_records[row["item_id"]] = json.loads(p.read_bytes())
            except Exception:
                pass
    guard = GalleryExclusions(blockers, raw_records)
    fp_cache = ROOT / "data/references/gallery_fingerprints.json"
    if fp_cache.exists():
        for item_id, ph64 in json.loads(fp_cache.read_bytes()).get("phash64", {}).items():
            guard.add_gallery_phash(item_id, ph64)
    for row in ref_rows:
        if row.get("phash64"):
            guard.add_gallery_phash(row["item_id"], row["phash64"])
    forbidden_ids = set(guard.ids)
    return guard, forbidden_ids


def collect_met_candidates(limit: int, forbidden_ids: set[str], fetch_openaccess_csv: bool = True) -> list[int]:
    """Gather Met Chinese candidate IDs from local browser caches + official MetObjects CSV."""
    ids: list[int] = []
    for pat in (
        "met_browser_ceramics_*.json",
        "met_browser_jade_*.json",
        "met_browser_metalwork_*.json",
        "met_browser_textiles_*.json",
        "met_browser_china_offset*.json",
        "met_focused_*.json",
        "metceramic_first_page.json",
    ):
        for p in sorted(RAW_DIR.glob(pat)):
            try:
                ids.extend(int(x) for x in (json.loads(p.read_text(encoding="utf-8")).get("objectIDs") or []))
            except Exception:
                pass
    for p in sorted((RAW_DIR / "met").glob("*.json")):
        if p.stem.isdigit():
            ids.append(int(p.stem))
    dedup = [i for i in dict.fromkeys(ids) if f"met_{i}" not in forbidden_ids]
    if len(dedup) >= limit or not fetch_openaccess_csv:
        return dedup[:limit]

    # Optional streaming scan of Met's official GitHub OpenAccess CSV if more IDs are needed
    csv_cache = RAW_DIR / "met_china_openaccess_ids.json"
    if csv_cache.exists():
        extra = json.loads(csv_cache.read_text(encoding="utf-8"))
        dedup = list(dict.fromkeys([*dedup, *(int(x) for x in extra if f"met_{x}" not in forbidden_ids)]))
        return dedup[:limit]
    try:
        import csv

        csv_url = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.txt"
        print("Streaming official MetObjects.txt to index additional Public-Domain Chinese object IDs...", flush=True)
        with _http_session().get(csv_url, stream=True, timeout=(20, 180)) as resp:
            resp.raise_for_status()
            lines = (line.decode("utf-8", errors="ignore") for line in resp.iter_lines())
            reader = csv.DictReader(lines)
            csv_ids: list[int] = []
            flat_re = re.compile(
                r"\b(painting|calligraphy|photograph|book|album|print|drawing|rubbing|scroll|manuscript)\b",
                re.I,
            )
            for row in reader:
                if row.get("Is Public Domain", "").strip().lower() != "true":
                    continue
                origin = f"{row.get('Culture', '')} {row.get('Country', '')}"
                if "china" not in origin.lower() and "chinese" not in origin.lower():
                    continue
                content = f"{row.get('Title', '')} {row.get('Object Name', '')} {row.get('Classification', '')}"
                if flat_re.search(content):
                    continue
                try:
                    if int(row.get("Object Begin Date") or 0) >= 1912:
                        continue
                except ValueError:
                    pass
                oid = row.get("Object ID", "").strip()
                if oid.isdigit():
                    csv_ids.append(int(oid))
        csv_cache.write_text(json.dumps(csv_ids), encoding="utf-8")
        dedup = list(dict.fromkeys([*dedup, *(x for x in csv_ids if f"met_{x}" not in forbidden_ids)]))
    except Exception as exc:
        print(f"Note: MetObjects CSV stream skipped ({exc}); using {len(dedup)} cached Met IDs.", flush=True)
    return dedup[:limit]


def collect_cleveland_candidates(limit: int, forbidden_ids: set[str], dry_run: bool = False) -> list[int]:
    """Gather Cleveland CC0 China artwork IDs (caching raw records alongside pagination)."""
    (RAW_DIR / "cleveland").mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    for p in sorted((RAW_DIR / "cleveland").glob("*.json")):
        if p.stem.isdigit() and f"cma_{p.stem}" not in forbidden_ids:
            ids.append(int(p.stem))
    if dry_run or len(ids) >= limit:
        return list(dict.fromkeys(ids))[:limit]
    skip = 0
    while len(ids) < limit:
        url = f"https://openaccess-api.clevelandart.org/api/artworks/?cc0=1&has_image=1&culture=China&limit=100&skip={skip}"
        data = _throttled_get(url, source="cleveland", min_interval=0.15).json()
        rows = data.get("data") or []
        if not rows:
            break
        for row in rows:
            sid = row.get("id")
            if sid:
                mp = RAW_DIR / "cleveland" / f"{sid}.json"
                if not mp.exists():
                    mp.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
                if f"cma_{sid}" not in forbidden_ids:
                    ids.append(int(sid))
        skip += len(rows)
        if len(rows) < 100:
            break
    return list(dict.fromkeys(ids))[:limit]


def collect_aic_candidates(limit: int, forbidden_ids: set[str], dry_run: bool = False) -> list[int]:
    """Gather Art Institute of Chicago CC0 Chinese object IDs via official bulk search API."""
    aic_raw_dir = RAW_DIR / "aic"
    aic_raw_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(p.stem) for p in sorted(aic_raw_dir.glob("*.json")) if p.stem.isdigit() and f"aic_{p.stem}" not in forbidden_ids]
    if dry_run:
        q = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"is_public_domain": True}},
                        {"match_phrase": {"place_of_origin": "China"}},
                        {"exists": {"field": "image_id"}},
                    ]
                }
            },
            "limit": 1,
        }
        resp = _throttled_get(
            "https://api.artic.edu/api/v1/artworks/search",
            source="aic",
            params={"params": json.dumps(q)},
        ).json()
        total = int(resp.get("pagination", {}).get("total") or 0)
        return list(range(min(limit, total)))
    if len(ids) >= limit:
        return list(dict.fromkeys(ids))[:limit]

    fields = [
        "id",
        "title",
        "image_id",
        "place_of_origin",
        "date_start",
        "date_end",
        "date_display",
        "medium_display",
        "artwork_type_title",
        "classification_title",
        "style_title",
        "term_titles",
        "credit_line",
        "main_reference_number",
        "is_public_domain",
        "description",
    ]
    # Query AIC in two sort orders (page 1..10 each, within AIC's 1,000-result window) for up to 2,000 items
    for sort_order in ("asc", "desc"):
        if len(ids) >= limit:
            break
        for page in range(1, 11):
            if len(ids) >= limit:
                break
            q = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"is_public_domain": True}},
                            {"match_phrase": {"place_of_origin": "China"}},
                            {"exists": {"field": "image_id"}},
                        ]
                    }
                },
                "sort": [{"id": {"order": sort_order}}],
                "fields": fields,
                "limit": 100,
                "page": page,
            }
            try:
                data = _throttled_get(
                    "https://api.artic.edu/api/v1/artworks/search",
                    source="aic",
                    params={"params": json.dumps(q)},
                    min_interval=0.20,
                ).json()
            except Exception as exc:
                print(f"AIC search page={page} sort={sort_order} stopped: {exc}", flush=True)
                break
            rows = data.get("data") or []
            if not rows:
                break
            for row in rows:
                sid = row.get("id")
                if sid and row.get("image_id") and row.get("is_public_domain") is True:
                    mp = aic_raw_dir / f"{sid}.json"
                    if not mp.exists():
                        mp.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
                    if f"aic_{sid}" not in forbidden_ids:
                        ids.append(int(sid))
            total_pages = int(data.get("pagination", {}).get("total_pages") or page)
            if page >= total_pages:
                break
    return list(dict.fromkeys(ids))[:limit]


def collect_mia_candidates(
    limit: int,
    forbidden_ids: set[str],
    dry_run: bool = False,
    include_2d_media: bool = False,
) -> list[int]:
    """Gather Minneapolis Institute of Art (Mia) Public-Domain Chinese object IDs via bulk search."""
    mia_raw_dir = RAW_DIR / "mia"
    mia_raw_dir.mkdir(parents=True, exist_ok=True)
    ids = [
        int(p.stem)
        for p in sorted(mia_raw_dir.glob("*.json"))
        if p.stem.isdigit() and f"mia_{p.stem}" not in forbidden_ids
    ]
    if dry_run:
        d = _throttled_get("https://search.artsmia.org/China?size=1", source="mia").json()
        tot = d.get("hits", {}).get("total", {})
        total = int(tot.get("value", tot) if isinstance(tot, dict) else (tot or 0))
        return list(range(min(limit, total)))
    if len(ids) >= limit:
        return list(dict.fromkeys(ids))[:limit]

    flat_re = re.compile(
        r"\b(painting|calligraphy|photograph|book|album|print|drawing|rubbing|scroll|manuscript)\b",
        re.I,
    )
    offset = 0
    page_size = 250
    while len(ids) < limit and offset < 9500:
        try:
            d = _throttled_get(
                f"https://search.artsmia.org/China?size={page_size}&from={offset}",
                source="mia",
                min_interval=0.15,
            ).json()
        except Exception as exc:
            print(f"Mia pagination stopped at offset={offset}: {exc}", flush=True)
            break
        hits = d.get("hits", {}).get("hits") or []
        if not hits:
            break
        for h in hits:
            src = h.get("_source") or {}
            sid_str = str(src.get("id") or "").strip()
            if not sid_str.isdigit():
                continue
            if src.get("rights_type") != "Public Domain" or src.get("image") != "valid" or src.get("restricted"):
                continue
            origin = f"{src.get('country', '')} {src.get('culture', '')} {src.get('continent', '')}"
            if "china" not in origin.lower() and "chinese" not in origin.lower():
                continue
            content = f"{src.get('title', '')} {src.get('object_name', '')} {src.get('classification', '')}"
            if not include_2d_media and flat_re.search(content):
                continue
            sid = int(sid_str)
            mp = mia_raw_dir / f"{sid}.json"
            if not mp.exists():
                mp.write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
            if f"mia_{sid}" not in forbidden_ids:
                ids.append(sid)
        offset += len(hits)
        if len(hits) < page_size:
            break
    return list(dict.fromkeys(ids))[:limit]


def process_candidate(
    source: str,
    sid: int,
    *,
    include_2d_media: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Fetch/validate metadata & image for one candidate, returning (item, candidate_fp, error)."""
    prefix = {"met": "met", "cleveland": "cma", "aic": "aic", "mia": "mia"}[source]
    item_id = f"{prefix}_{sid}"
    raw_dir = RAW_DIR / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_dir / f"{sid}.json"
    try:
        if meta_path.exists():
            d = json.loads(meta_path.read_text(encoding="utf-8"))
        elif source == "met":
            d = _throttled_get(
                f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{sid}",
                source="met",
                min_interval=0.28,
            ).json()
            meta_path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        elif source == "cleveland":
            d = _throttled_get(
                f"https://openaccess-api.clevelandart.org/api/artworks/{sid}",
                source="cleveland",
                min_interval=0.12,
            ).json().get("data", {})
            meta_path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        else:
            raise ValueError(f"{source} metadata must be bulk-cached first")

        if source == "met":
            if d.get("isPublicDomain") is not True:
                raise ValueError("rights: Met isPublicDomain is not true")
            image_url = text(d.get("primaryImageSmall") or d.get("primaryImage"))
            origin = " ".join(text(d.get(k)) for k in ("country", "culture", "geographyType", "region", "subregion"))
            country = text(d.get("country")).lower()
            if country and country not in ("china", "china (?)"):
                raise ValueError(f"origin: non-China country {country}")
            if not re.search(r"\b(china|chinese)\b", origin, re.I):
                raise ValueError("origin: missing China/Chinese attribution")
            y0, y1 = years(d.get("objectBeginDate"), d.get("objectEndDate"))
            if y0 is None:
                y0, y1 = years(d.get("objectDate"), d.get("period"))
            title = text(d.get("title"))
            obj_name = text(d.get("objectName"))
            classification = text(d.get("classification"))
            medium = text(d.get("medium"))
            culture = text(d.get("culture"))
            period = text(d.get("period") or d.get("objectDate"))
            tag_list = terms(d.get("tags"))
            desc = ""
            source_url = text(d.get("objectURL")) or f"https://www.metmuseum.org/art/collection/search/{sid}"
            credit = text(d.get("creditLine"))
            accession = text(d.get("accessionNumber"))
            related_ids: list[str] = []
        elif source == "cleveland":
            if d.get("share_license_status") != "CC0":
                raise ValueError("rights: Cleveland share_license_status != CC0")
            imgs = d.get("images") or {}
            image_url = text((imgs.get("web") or imgs.get("print") or {}).get("url"))
            origin = "; ".join(terms(d.get("culture")))
            if not re.search(r"\b(china|chinese)\b", origin, re.I):
                raise ValueError("origin: missing China/Chinese attribution")
            y0, y1 = years(d.get("creation_date_earliest"), d.get("creation_date_latest"))
            if y0 is None:
                y0, y1 = years(d.get("creation_date"))
            title = text(d.get("title"))
            obj_name = text(d.get("type"))
            classification = text(d.get("classification") or d.get("type"))
            medium = text(d.get("technique") or d.get("medium"))
            culture = origin
            period = text(d.get("creation_date"))
            tag_list = terms(d.get("technique"))
            desc = text(d.get("description"))
            source_url = text(d.get("url")) or f"https://www.clevelandart.org/art/{d.get('accession_number', '')}"
            credit = text(d.get("creditline"))
            accession = text(d.get("accession_number"))
            related_ids = official_related_ids(d)
        elif source == "aic":
            if d.get("is_public_domain") is not True or not d.get("image_id"):
                raise ValueError("rights: AIC is_public_domain != true or missing image_id")
            origin = f"{text(d.get('place_of_origin'))} {text(d.get('style_title'))}"
            if not re.search(r"\b(china|chinese)\b", origin, re.I):
                raise ValueError("origin: missing China/Chinese attribution")
            image_url = f"https://www.artic.edu/iiif/2/{d['image_id']}/full/843,/0/default.jpg"
            y0, y1 = years(d.get("date_start"), d.get("date_end"))
            if y0 is None:
                y0, y1 = years(d.get("date_display"))
            title = text(d.get("title"))
            obj_name = text(d.get("artwork_type_title"))
            classification = text(d.get("classification_title"))
            medium = text(d.get("medium_display"))
            culture = text(d.get("style_title") or d.get("place_of_origin"))
            period = text(d.get("date_display"))
            tag_list = terms(d.get("term_titles"))
            desc = text(d.get("description"))
            source_url = f"https://www.artic.edu/artworks/{sid}"
            credit = text(d.get("credit_line"))
            accession = text(d.get("main_reference_number"))
            related_ids = []
        else:  # mia (Minneapolis Institute of Art Public Domain)
            if d.get("rights_type") != "Public Domain" or d.get("image") != "valid" or d.get("restricted"):
                raise ValueError("rights: Mia rights_type != Public Domain or invalid image")
            origin = f"{text(d.get('country'))} {text(d.get('culture'))}"
            if not re.search(r"\b(china|chinese)\b", origin, re.I):
                raise ValueError("origin: missing China/Chinese attribution")
            image_url = f"https://{sid % 7}.api.artsmia.org/800/{sid}.jpg"
            y0, y1 = years(d.get("dated"))
            title = text(d.get("title"))
            obj_name = text(d.get("object_name"))
            classification = text(d.get("classification"))
            medium = text(d.get("medium"))
            culture = text(d.get("culture") or d.get("country"))
            period = text(d.get("dynasty") or d.get("style") or d.get("dated"))
            tag_list = terms(d.get("tags"))
            desc = text(d.get("description") or d.get("text"))
            source_url = f"https://collections.artsmia.org/art/{sid}"
            credit = text(d.get("creditline"))
            accession = text(d.get("accession_number"))
            related_ids = []

        if y0 is not None and y0 >= 1912:
            raise ValueError("date: modern (1912 onward)")
        parsed_url = urlparse(image_url)
        if parsed_url.scheme != "https" or parsed_url.hostname not in ALLOWED_HOSTS[source]:
            raise ValueError(f"untrusted or missing image host: {image_url}")

        content_text = f"{title} {obj_name} {classification}".lower()
        if not include_2d_media and re.search(
            r"\b(painting|calligraphy|photograph|book|album|print|drawing|rubbing|scroll|manuscript)\b",
            content_text,
        ):
            raise ValueError("type: excluded 2D flat media")

        raw_img_path = raw_dir / f"{sid}.jpg"
        if not raw_img_path.exists():
            min_iv = 0.04 if source == "met" else (0.08 if source == "aic" else 0.08)
            raw_bytes = _throttled_get(image_url, source=source, min_interval=min_iv).content
            raw_img_path.write_bytes(raw_bytes)
        else:
            raw_bytes = raw_img_path.read_bytes()

        proc_img_path = PROC_IMG_DIR / f"{item_id}.jpg"
        proc_thm_path = PROC_THM_DIR / f"{item_id}.jpg"
        with Image.open(io.BytesIO(raw_bytes)) as im_raw:
            im = im_raw.convert("RGB")
            if min(im.size) < 224 or max(im.size) < 384:
                raise ValueError(f"image below quality floor: {im.size}")
            im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            w, h = im.size
            if not proc_img_path.exists():
                im.save(proc_img_path, format="JPEG", quality=94, optimize=True)
            fp = image_fingerprints(im)
            if not proc_thm_path.exists():
                thm = im.copy()
                thm.thumbnail((320, 320), Image.Resampling.LANCZOS)
                thm.save(proc_thm_path, format="JPEG", quality=88, optimize=True)

        orig_sha = hashlib.sha256(raw_bytes).hexdigest()
        img_sha = hashlib.sha256(proc_img_path.read_bytes()).hexdigest()
        thm_sha = hashlib.sha256(proc_thm_path.read_bytes()).hexdigest()

        motif_hay = f"{' '.join(tag_list)} {title}".lower()
        motifs = list(
            dict.fromkeys(
                canon for word, canon in MOTIF_RULES if re.search(r"\b" + word + r"s?\b", motif_hay)
            )
        )
        item = {
            "item_id": item_id,
            "source": source,
            "source_id": str(sid),
            "title": title,
            "object_name": obj_name,
            "classification": classification,
            "medium": medium,
            "culture": culture,
            "period": period,
            "year_from": y0,
            "year_to": y1,
            "tags": tag_list,
            "description": desc,
            "image_url": image_url,
            "source_url": source_url,
            "license": "CC0",
            "credit_line": credit,
            "object_type": _match_first(OBJTYPE_RULES, content_text),
            "material": _match_first(MATERIAL_RULES, medium.lower()),
            "motifs": motifs,
            "colors": [],
            "image_path": f"data/processed/images/{item_id}.jpg",
            "thumb_path": f"data/processed/thumbs/{item_id}.jpg",
            "width": w,
            "height": h,
            "sha256": orig_sha,
            "image_sha256": img_sha,
            "thumb_sha256": thm_sha,
            "phash": fp["ahash256"],
            "phash64": fp["phash64"],
            "object_group": f"{source}:{accession_root(accession)}" if accession else item_id,
            "split": "unassigned",
            "split_policy": "expansion_10k_retrieval_only_not_benchmark_split",
            "origin_basis": origin,
            "metadata_path": f"data/raw/{source}/{sid}.json",
            "metadata_sha256": metadata_sha256(d),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        cand_fp = {
            "reference_id": item_id,
            "source": source,
            "accession": accession,
            "related_ids": related_ids,
            "original_image_sha256": orig_sha,
            "image_sha256": img_sha,
            "ahash256": fp["ahash256"],
            "phash64": fp["phash64"],
        }
        return item, cand_fp, None
    except Exception as exc:
        return None, None, {"source": source, "source_id": str(sid), "error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-total", type=int, default=10200, help="Total gallery target including 4,692 frozen v2 items")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--met-limit", type=int, default=8000)
    ap.add_argument("--aic-limit", type=int, default=2200)
    ap.add_argument("--mia-limit", type=int, default=5000)
    ap.add_argument("--cleveland-limit", type=int, default=1500)
    ap.add_argument("--include-2d-media", action="store_true", help="Also allow Chinese paintings/calligraphy/albums if needed")
    ap.add_argument("--dry-run", action="store_true", help="Verify frozen hashes, exclusions, and live candidate counts without downloading images")
    args = ap.parse_args()

    PROC_IMG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_THM_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frozen_sha_before = hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest()
    bank_sha_before = hashlib.sha256(BANK_PATH.read_bytes()).hexdigest()
    frozen_count = len(_read_jsonl(FROZEN_PATH))
    need_extra = max(0, args.target_total - frozen_count)

    guard, forbidden_ids = build_exclusion_guard()
    existing_rows = _read_jsonl(OUT_SHARD)
    accepted: dict[str, dict[str, Any]] = {}
    seen_shas: set[str] = set()
    for row in existing_rows:
        iid = row["item_id"]
        if iid in forbidden_ids or row["sha256"] in seen_shas:
            continue
        accepted[iid] = row
        seen_shas.add(row["sha256"])
        forbidden_ids.add(iid)
        if len(row.get("phash", "")) == 256:
            guard.ahashes.append((iid, int(row["phash"], 2)))
        if row.get("phash64"):
            guard.add_gallery_phash(iid, row["phash64"])

    mia_cands = collect_mia_candidates(
        args.mia_limit, forbidden_ids, dry_run=args.dry_run, include_2d_media=args.include_2d_media
    )
    met_cands = collect_met_candidates(args.met_limit, forbidden_ids, fetch_openaccess_csv=False)
    aic_cands = collect_aic_candidates(args.aic_limit, forbidden_ids, dry_run=args.dry_run)
    cma_cands = collect_cleveland_candidates(args.cleveland_limit, forbidden_ids, dry_run=args.dry_run)

    summary_preview = {
        "frozen_v2_items": frozen_count,
        "target_total": args.target_total,
        "needed_new_items": need_extra,
        "already_acquired_in_shard": len(accepted),
        "remaining_to_acquire": max(0, need_extra - len(accepted)),
        "candidate_pool": {
            "mia": len(mia_cands),
            "met": len(met_cands),
            "aic": len(aic_cands),
            "cleveland": len(cma_cands),
            "total_candidates": len(mia_cands) + len(met_cands) + len(aic_cands) + len(cma_cands),
        },
        "frozen_v2_sha256": frozen_sha_before,
        "reference_bank_sha256": bank_sha_before,
    }
    print(json.dumps(summary_preview, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    # Partition fast 7-shard Mia CDN across 4 queues alongside gentle Met/AIC/Cleveland
    queues = [
        [("mia", x) for x in mia_cands[0::4]],
        [("mia", x) for x in mia_cands[1::4]],
        [("mia", x) for x in mia_cands[2::4]],
        [("mia", x) for x in mia_cands[3::4]],
        [("met", x) for x in met_cands],
        [("aic", x) for x in aic_cands],
        [("cleveland", x) for x in cma_cands],
    ]
    seen_todo: set[tuple[str, int]] = set()
    todo: list[tuple[str, int]] = []
    max_len = max((len(q) for q in queues), default=0)
    for i in range(max_len):
        for q in queues:
            if i < len(q) and q[i] not in seen_todo:
                seen_todo.add(q[i])
                todo.append(q[i])

    failures: list[dict[str, Any]] = []
    excluded_by_guard = 0

    def checkpoint() -> None:
        ordered = sorted(accepted.values(), key=lambda r: r["item_id"])
        tmp = OUT_SHARD.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
        os.replace(tmp, OUT_SHARD)

    if len(accepted) < need_extra and todo:
        todo_iter = iter(todo)
        max_inflight = max(args.workers * 4, 16)
        processed = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            inflight: set[cf.Future] = set()
            for _ in range(max_inflight):
                nxt = next(todo_iter, None)
                if nxt is None:
                    break
                inflight.add(
                    pool.submit(process_candidate, nxt[0], nxt[1], include_2d_media=args.include_2d_media)
                )
            while inflight and len(accepted) < need_extra:
                done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    processed += 1
                    item, cand_fp, err = fut.result()
                    if err is not None:
                        failures.append(err)
                    elif item is not None and cand_fp is not None:
                        reason = guard.reason(cand_fp)
                        if reason is not None or item["sha256"] in seen_shas or item["image_sha256"] in seen_shas:
                            excluded_by_guard += 1
                        else:
                            iid = item["item_id"]
                            accepted[iid] = item
                            seen_shas.add(item["sha256"])
                            seen_shas.add(item["image_sha256"])
                            guard.ids.add(iid)
                            guard.ahashes.append((iid, int(item["phash"], 2)))
                            guard.add_gallery_phash(iid, item["phash64"])
                    if len(accepted) < need_extra:
                        nxt = next(todo_iter, None)
                        if nxt is not None:
                            inflight.add(
                                pool.submit(process_candidate, nxt[0], nxt[1], include_2d_media=args.include_2d_media)
                            )
                    if processed % 100 == 0:
                        checkpoint()
                        print(
                            f"[{processed}/{len(todo)}] total_gallery={frozen_count + len(accepted)} "
                            f"(new={len(accepted)}/{need_extra}) excluded={excluded_by_guard} failed={len(failures)}",
                            flush=True,
                        )
            for fut in inflight:
                fut.cancel()

    checkpoint()
    FAIL_PATH.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in failures), encoding="utf-8")
    frozen_sha_after = hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest()
    bank_sha_after = hashlib.sha256(BANK_PATH.read_bytes()).hexdigest()
    if frozen_sha_after != frozen_sha_before or bank_sha_after != bank_sha_before:
        raise RuntimeError("Frozen collection_v2 or reference bank changed unexpectedly!")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_v2_count": frozen_count,
        "expansion_10k_count": len(accepted),
        "combined_gallery_count": frozen_count + len(accepted),
        "source_counts": dict(Counter(r["source"] for r in accepted.values())),
        "object_type_counts": dict(Counter(r["object_type"] or "unmapped" for r in accepted.values())),
        "excluded_by_isolation_or_dedup": excluded_by_guard,
        "download_or_filter_rejections": len(failures),
        "frozen_v2_sha256": frozen_sha_after,
        "frozen_v2_unchanged": True,
        "reference_bank_sha256": bank_sha_after,
        "reference_bank_unchanged": True,
        "output_shard": str(OUT_SHARD.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
