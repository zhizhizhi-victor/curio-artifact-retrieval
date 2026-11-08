from __future__ import annotations

import torch
import torch.nn as nn

from curio.represent.train_comp_lora import (
    build_bilingual_visual_pairs,
    compositional_contrastive_loss,
    select_vision_lora_targets,
)


def test_bilingual_visual_pairs_strips_nonvisual_and_uses_verified_negatives():
    item = {
        "item_id": "met_99999",
        "title": "Hexagonal Vase with Two Lugs (Ming Dynasty, 16th century)",
        "object_name": "Vase",
        "object_type": "vase",
        "material": "porcelain",
        "medium": "Porcelain painted in underglaze cobalt blue (Jingdezhen ware)",
        "culture": "China, Ming dynasty",
        "period": "1500s",
        "motifs": ["floral"],
        "colors": ["blue", "white"],
    }
    en_views, zh_views, cf_en, cf_zh = build_bilingual_visual_pairs(item)
    joined = " ".join(en_views + zh_views)
    assert "ming" not in joined.lower()
    assert "jingdezhen" not in joined.lower()
    assert "16th" not in joined.lower()
    assert any("瓶" in z for z in zh_views)
    assert any("花卉纹" in z for z in zh_views)
    assert cf_en and cf_zh
    assert "handleless" in cf_en[0].lower()

    # With explicit blind-image annotation, counterfactuals use verified 0-valued attributes
    ann = {
        "caption_en": "A tall green bottle with a narrow neck and smooth sides.",
        "caption_zh": "高形绿釉瓶，细颈，器腹光滑无耳。",
        "object_type": "vase",
        "colors": {"green": 1, "red": 0, "blue": 0},
        "shape": {"handles": 0, "narrow_mouth": 1, "wide_mouth": 0},
        "motifs": {"dragon": 0, "floral": 0},
    }
    ev2, zv2, cfe2, cfz2 = build_bilingual_visual_pairs(item, ann)
    assert ev2[0] == ann["caption_en"]
    assert zv2[0] == ann["caption_zh"]
    assert any("handles" in s for s in cfe2)
    assert any("双耳" in s for s in cfz2)
    assert any("red" in s for s in cfe2)


def test_compositional_contrastive_loss_penalizes_counterfactual_alignment():
    torch.manual_seed(0)
    pos_txt = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    cf_txt = torch.tensor([[0.8, 0.0, 0.6, 0.0], [0.0, 0.8, 0.0, 0.6]])
    pos_mask = torch.eye(2, dtype=torch.bool)

    # Image aligned cleanly with positive and orthogonal to counterfactual component
    img_clean = torch.tensor([[1.0, 0.0, -0.4, 0.0], [0.0, 1.0, 0.0, -0.4]], requires_grad=True)
    # Image confused toward counterfactual attribute
    img_confused = torch.tensor([[0.8, 0.0, 0.6, 0.0], [0.0, 0.8, 0.0, 0.6]], requires_grad=True)

    loss_clean = compositional_contrastive_loss(img_clean, pos_txt, pos_mask, cf_text_features=cf_txt, cf_weight=0.7)
    loss_confused = compositional_contrastive_loss(img_confused, pos_txt, pos_mask, cf_text_features=cf_txt, cf_weight=0.7)
    assert loss_clean < loss_confused
    loss_clean.backward()
    assert img_clean.grad is not None and torch.isfinite(img_clean.grad).all()


def test_select_vision_lora_targets_adapts_pooling_head_only_on_vision():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.out_proj = nn.Linear(8, 8)

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = nn.Linear(8, 8)
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)

    class DummyVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = nn.ModuleDict({"attn": DummyAttn(), "head": DummyHead()})
            self.text_model = nn.ModuleDict({"attn": DummyAttn()})

    m = DummyVLM()
    targets = select_vision_lora_targets(m, include_pooling_head=True)
    assert any("vision_model.head.fc1" in t for t in targets)
    assert any("vision_model.attn.out_proj" in t for t in targets)
    assert all("text_model" not in t for t in targets)
