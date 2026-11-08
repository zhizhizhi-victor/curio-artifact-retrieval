from __future__ import annotations

from fastapi.testclient import TestClient
import numpy as np
import pytest

from curio.query import parse_query
from curio.retrieve.compositional import (
    calibrated_probe_logscore,
    decompose_visual_query,
    soft_and_zscores,
)
from curio.retrieve.engine import SearchService
from curio.serve.app import create_app


class CompositionalEncoder:
    """Deterministic multi-prompt encoder for testing compositional AND vs partial matches."""

    metadata = {"model_id": "test-compositional", "revision": "1"}

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        # Dimension 4: [vessel_axis, blue_axis, handle_axis, dragon_axis]
        rows = []
        for text in texts:
            low = text.lower()
            vec = np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32)
            if any(k in low for k in ("vase", "bottle", "jar", "瓶", "罐")):
                vec[0] += 1.0
            if any(k in low for k in ("blue", "蓝", "青花")):
                vec[1] += 1.0
            if any(k in low for k in ("handle", "lug", "柄", "双耳", "弧形柄")):
                vec[2] += 1.0
            if any(k in low for k in ("dragon", "龙")):
                vec[3] += 1.0
            rows.append(vec / np.linalg.norm(vec))
        return np.stack(rows, axis=0)


class CompositionalProbes:
    def predict(self, embeddings: np.ndarray) -> dict:
        n = embeddings.shape[0]
        # Row 0: partial match (very blue vase, NO handles, HAS dragon)
        # Row 1: complete match (blue vase WITH handles, NO dragon)
        # Row 2: distractor bowl
        obj_probs = np.tile(np.array([[0.90, 0.10], [0.85, 0.15], [0.10, 0.90]], dtype=np.float32)[:n], (1, 1))
        col_probs = np.array([[0.95, 0.10], [0.82, 0.80], [0.15, 0.85]], dtype=np.float32)[:n]
        shp_probs = np.array([[0.05], [0.92], [0.20]], dtype=np.float32)[:n]
        mot_probs = np.array([[0.88], [0.06], [0.10]], dtype=np.float32)[:n]
        return {
            "object_type": {"classes": ["vase", "bowl"], "probabilities": obj_probs},
            "colors": {"classes": ["blue", "green"], "probabilities": col_probs},
            "shape": {"classes": ["handles"], "probabilities": shp_probs},
            "motifs": {"classes": ["dragon"], "probabilities": mot_probs},
        }


def _catalog_items() -> list[dict]:
    return [
        {
            "item_id": "partial_1",
            "title": "Blue Vase with Curved Handle",  # misleading metadata on purpose
            "object_type": "vase",
            "material": "porcelain",
            "colors": ["blue"],
            "motifs": [],
            "shape": ["handles"],
            "source": "Met Open Access",
        },
        {
            "item_id": "complete_2",
            "title": "Untitled Artifact",
            "object_type": "bowl",
            "material": "bronze",
            "colors": [],
            "motifs": ["dragon"],
            "shape": [],
            "source": "Cleveland Open Access",
        },
        {
            "item_id": "distractor_3",
            "title": "Green Bowl",
            "object_type": "bowl",
            "material": "jade",
            "colors": ["green"],
            "motifs": [],
            "shape": [],
            "source": "Met Open Access",
        },
    ]


def _make_service(*, with_probes: bool = True, permute_ids: bool = False) -> SearchService:
    items = _catalog_items()
    # Embeddings in catalog order:
    # partial_1 has slightly higher global vessel+blue score, zero handle score, high dragon score.
    # complete_2 satisfies vessel + blue + handle simultaneously, zero dragon score.
    raw_embs = np.array(
        [
            [0.82, 0.57, -0.05, 0.35],  # partial_1
            [0.64, 0.52, 0.56, -0.10],  # complete_2
            [0.20, -0.30, 0.10, 0.10],  # distractor_3
        ],
        dtype=np.float32,
    )
    ids = ["partial_1", "complete_2", "distractor_3"]
    if permute_ids:
        order = [2, 0, 1]
        raw_embs = raw_embs[order]
        ids = [ids[i] for i in order]
    manifest = {"data_sha256": "fixture-sha", "index_version": "vectors-fixture"}
    if not with_probes:
        return SearchService(
            items,
            encoder=CompositionalEncoder(),
            embeddings=raw_embs,
            item_ids=ids,
            index_manifest=manifest,
            probe_weight=0.0,
        )
    temp = SearchService(
        items,
        encoder=CompositionalEncoder(),
        embeddings=raw_embs,
        item_ids=ids,
        index_manifest=manifest,
    )
    report = {
        "index_sha256": temp.index_sha256,
        "data_sha256": "fixture-sha",
        "feature_dimension": 4,
        "index_version": temp.index_version,
        "tasks": {
            "object_type": {"status": "trained", "activation": "softmax", "classes": ["vase", "bowl"], "dev_macro_f1": 0.90},
            "colors": {"status": "trained", "activation": "sigmoid", "classes": ["blue", "green"], "dev_macro_f1": 0.80},
            "shape": {"status": "trained", "activation": "sigmoid", "classes": ["handles"], "dev_macro_f1": 0.85},
            "motifs": {"status": "trained", "activation": "sigmoid", "classes": ["dragon"], "dev_macro_f1": 0.65},
        },
    }
    probes = CompositionalProbes()
    if permute_ids:
        # Probe predictions correspond to embedding rows (`order = [2, 0, 1]`).
        orig_predict = probes.predict

        def permuted_predict(embs):
            pred = orig_predict(embs)
            return {
                task: {"classes": val["classes"], "probabilities": val["probabilities"][[2, 0, 1]]}
                for task, val in pred.items()
            }

        probes.predict = permuted_predict
    return SearchService(
        items,
        encoder=CompositionalEncoder(),
        embeddings=raw_embs,
        item_ids=ids,
        index_manifest=manifest,
        probes=probes,
        probe_report=report,
        probe_weight=0.0,
    )


def test_decompose_visual_query_splits_clauses_and_negations_bilingually():
    en_q = "A blue porcelain vase, with a tall curved handle rather than a dragon"
    en_decomp = decompose_visual_query(en_q, parse_query(en_q))
    assert len(en_decomp["sub_prompts"]) >= 2
    assert any("vase" in s and "handle" in s for s in en_decomp["sub_prompts"][1:])
    assert any("dragon" in n for n in en_decomp["neg_prompts"])

    zh_q = "蓝白瓷瓶，带有高弧形柄，而不是龙纹"
    zh_decomp = decompose_visual_query(zh_q, parse_query(zh_q))
    assert "curved handle" in zh_decomp["global_prompt"]
    assert len(zh_decomp["sub_prompts"]) == 2
    assert "vase" in zh_decomp["sub_prompts"][1] and "curved handle" in zh_decomp["sub_prompts"][1]
    assert any("dragon" in n for n in zh_decomp["neg_prompts"])


def test_soft_and_zscores_penalizes_missing_condition_vs_balanced_match():
    # Candidate 0: strong on clause 1, fails clause 2. Candidate 1: satisfies both clauses.
    sub_mat = np.array([[0.90, 0.10], [0.72, 0.72], [0.20, 0.20]], dtype=np.float32)
    scores = soft_and_zscores(sub_mat, np.array([0, 1, 2]), tau=1.5)
    assert scores[1] > scores[0] > scores[2]


def test_calibrated_probe_logscore_handles_conjunction_and_color_disjunction():
    service = _make_service(with_probes=True)
    and_parsed = parse_query("blue green vase with handles")
    and_log, _ = calibrated_probe_logscore(
        service.probe_predictions,
        service.probe_report,
        and_parsed,
        [0, 1, 2],
        service._probe_constraint_unknown_reason,
    )
    # Item 1 has both blue (0.82) and green (0.80) + handles (0.92); Item 0 lacks green (0.10) and handles (0.05).
    assert and_log is not None and and_log[1] > and_log[0]

    or_parsed = parse_query("blue or green vase")
    or_log, _ = calibrated_probe_logscore(
        service.probe_predictions,
        service.probe_report,
        or_parsed,
        [0, 1, 2],
        service._probe_constraint_unknown_reason,
    )
    # Under disjunction ("blue or green"), Item 0 (blue=0.95, vase=0.90) is not penalized for green=0.10!
    assert or_log is not None and or_log[0] > or_log[1]


@pytest.mark.parametrize("permute_ids", [False, True])
@pytest.mark.parametrize("variant", ["subquery_and", "compositional"])
def test_compositional_variants_promote_complete_multi_condition_match_over_partial(variant, permute_ids):
    service = _make_service(with_probes=True, permute_ids=permute_ids)
    query = "A blue vase, with a curved handle, without dragon"
    base = service.search(query, variant="baseline")
    comp = service.search(query, variant=variant)
    # Raw single-vector baseline ranks partial_1 first; compositional/subquery_and promotes complete_2!
    assert base["results"][0]["item_id"] == "partial_1"
    assert comp["results"][0]["item_id"] == "complete_2"
    assert comp["effective_variant"] == variant and not comp["degraded"]


@pytest.mark.parametrize("variant", ["subquery_and", "compositional"])
def test_compositional_variants_never_touch_metadata_and_are_invariant_to_catalog_scrambling(variant, monkeypatch):
    clean = _make_service(with_probes=True)
    scrambled = _make_service(with_probes=True)
    for item in scrambled.items:
        item.update(
            title="Completely unrelated silk robe",
            description="black bronze dragon mirror",
            object_type="garment",
            material="silk",
            colors=["black"],
            motifs=["fish"],
            shape=[],
        )
    scrambled.attributes = [scrambled._attributes(it) for it in scrambled.items]

    def forbidden(*args, **kwargs):
        raise AssertionError(f"{variant} accessed metadata scoring")

    for svc in (clean, scrambled):
        monkeypatch.setattr(svc.lexical, "score", forbidden)
        monkeypatch.setattr(svc, "_slot_score", forbidden)

    query = "蓝白瓷瓶，带有弧形柄，不要龙纹"
    left = clean.search(query, variant=variant)
    right = scrambled.search(query, variant=variant)
    assert [(r["item_id"], r["score"]) for r in left["results"]] == [
        (r["item_id"], r["score"]) for r in right["results"]
    ]
    for row in left["results"]:
        assert row["matched_slots"] == []
        assert row["score_breakdown"]["bm25"] is None
        assert row["score_breakdown"]["slots"] is None
        assert all(entry["kind"] in {"similarity", "visual_probe"} for entry in row["evidence"])


def test_compositional_degrades_gracefully_without_probes_or_embeddings():
    no_probe_svc = _make_service(with_probes=False)
    res = no_probe_svc.search("A blue vase, with a curved handle", variant="compositional")
    assert res["effective_variant"] == "subquery_and" and res["degraded"] is True
    assert res["results"][0]["item_id"] == "complete_2"

    no_emb_svc = SearchService(_catalog_items())
    empty = no_emb_svc.search("A blue vase", variant="compositional")
    assert empty["effective_variant"] == "unavailable"
    assert empty["retrieval_mode"] == "unavailable"
    assert empty["results"] == []


def test_api_accepts_compositional_and_subquery_and_variants():
    service = _make_service(with_probes=True)
    client = TestClient(create_app(service=service))
    for variant in ("subquery_and", "compositional"):
        resp = client.post("/api/search", json={"query": "A blue vase, with a curved handle", "variant": variant})
        assert resp.status_code == 200
        body = resp.json()
        assert body["variant"] == variant
        assert body["results"][0]["item_id"] == "complete_2"
