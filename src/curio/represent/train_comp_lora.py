"""Bilingual Attribute-Contrastive Image-Tower LoRA (Compositional LoRA).

Addresses three limitations of standard English-only random-batch LoRA on pure-image
museum galleries:
1. **Spatial Attention Pooling Adaptation**: Adapts SigLIP-2's Multihead Attention
   Pooling head (``vision_model.head``) alongside full self-attention projections
   (``q_proj, k_proj, v_proj, out_proj``) so small local regions (handles/lugs,
   mouth rims, lids, feet, relief motifs) are preserved during patch aggregation.
2. **Bilingual Multi-View Alignment**: Aligns every ``train`` image simultaneously
   to paired English and Chinese observable visual descriptions (uniting fine-grained
   ``ai_image_only`` ``train`` captions with structured bilingual visual descriptions
   while stripping non-visual dynasty/provenance metadata).
3. **Counterfactual Attribute-Swapped Hard Negatives (NegCLIP-style)**: Augments the
   image-to-text contrastive denominator with single-attribute perturbed descriptions
   (same vessel category/material, swapped local shape/color/motif) plus semantic
   false-negative masking for near-identical in-batch descriptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any

import numpy as np
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from .captions import clean_visual_phrase, visual_caption
from .detail_probes import load_annotations
from .encoders import Encoder
from .train import paired_retrieval, read_items
from .validation import validate_object_splits

ZH_OBJ_NAMES: dict[str, str] = {
    "vase": "瓶",
    "bowl": "碗",
    "plate": "盘",
    "jar": "罐",
    "cup": "杯",
    "teapot": "茶壶",
    "ewer": "执壶",
    "pot": "壶",
    "censer": "香炉",
    "figure": "塑像俑",
    "mirror": "铜镜圆盘",
    "box": "盒",
    "textile": "织物绣片",
    "ritual vessel": "礼器鼎尊",
    "pendant": "佩饰挂件",
    "stele": "浮雕石碑",
    "ornament": "饰件",
    "cong": "琮",
    "bi": "环状玉璧",
    "lid": "器盖",
    "saucer": "碟托",
    "tile": "砖瓦",
    "ring": "指环",
    "jewelry": "首饰",
    "garment": "袍服衣物",
    "fragment": "残片",
    "bottle": "长颈瓶",
}

ZH_MAT_NAMES: dict[str, str] = {
    "porcelain": "瓷质",
    "ceramic": "陶瓷",
    "stoneware": "炻器粗陶",
    "earthenware": "陶质",
    "bronze": "青铜金属",
    "copper": "红铜",
    "iron": "铁质",
    "gold": "金质",
    "silver": "银质",
    "jade": "玉石",
    "silk": "丝绸",
    "textile": "织物",
    "wood": "木雕",
    "glass": "玻璃",
    "lacquer": "漆器",
    "stone": "石雕",
    "ivory": "牙雕",
}

ZH_MOTIF_NAMES: dict[str, str] = {
    "dragon": "龙纹",
    "phoenix": "凤纹",
    "lotus": "莲瓣纹",
    "floral": "花卉纹",
    "cloud": "云纹",
    "bird": "禽鸟纹",
    "fish": "鱼纹",
    "tiger": "虎兽纹",
    "geometric": "几何纹",
    "landscape": "山水纹",
    "leaf": "叶纹",
    "horse": "马纹",
    "lion": "瑞狮纹",
    "butterfly": "蝶纹",
    "figure_scene": "人物纹",
}

ZH_COLOR_NAMES: dict[str, str] = {
    "blue": "蓝色",
    "white": "白色",
    "green": "青绿色",
    "red": "红色",
    "black": "黑色",
    "yellow": "黄色",
    "brown": "褐色",
    "pink": "粉红色",
}

# Fine-grained visual cues observable directly from clean title/medium phrases
OBSERVABLE_CUES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(underglaze\s+(?:cobalt\s+)?blue|blue\s+and\s+white)\b", re.I), "blue and white", "青花蓝白"),
    (re.compile(r"\b(celadon|pale\s+green|grey-green|gray-green|olive\s+green)\b", re.I), "celadon green glaze", "青绿釉"),
    (re.compile(r"\b(polychrome|sancai|three-color|famille\s+rose|overglaze\s+enamel)\b", re.I), "multicolored painted", "多彩彩绘"),
    (re.compile(r"\b(black\s+glaze|burnished\s+black)\b", re.I), "dark black surface", "黑釉深色"),
    (re.compile(r"\b(white\s+glaze|blanc\s+de\s+chine|ivory\s+white)\b", re.I), "monochrome white", "素白釉"),
    (re.compile(r"\b(red\s+glaze|copper\s+red|cinnabar|carved\s+red\s+lacquer)\b", re.I), "deep red", "朱红釉彩"),
    (re.compile(r"\b(yellow\s+glaze|gilt|golden)\b", re.I), "golden yellow", "金黄色"),
    (re.compile(r"\b(two\s+lugs|side\s+lugs|handles?|loop\s+handles|tubular\s+handles|ear\s+handles)\b", re.I), "with side handles", "双耳带柄"),
    (re.compile(r"\b(tripod|three\s+legs|three\s+feet)\b", re.I), "standing on three legs", "三足立器"),
    (re.compile(r"\b(long\s+neck|slender\s+neck|tall\s+neck)\b", re.I), "with a long slender neck", "细长颈"),
    (re.compile(r"\b(wide\s+mouth|flaring\s+rim|broad\s+opening)\b", re.I), "with a wide flaring mouth", "敞口撇口"),
    (re.compile(r"\b(hexagonal)\b", re.I), "hexagonal faceted", "六方形折角"),
    (re.compile(r"\b(octagonal)\b", re.I), "octagonal faceted", "八方形"),
    (re.compile(r"\b(cylindrical|brush\s*pot)\b", re.I), "straight cylindrical", "直筒形"),
    (re.compile(r"\b(pear-shaped)\b", re.I), "pear-shaped", "梨形垂腹"),
    (re.compile(r"\b(double[- ]gourd|gourd-shaped)\b", re.I), "double-gourd shaped", "葫芦形"),
    (re.compile(r"\b(rectangular|plaque|panel)\b", re.I), "rectangular", "长方形"),
    (re.compile(r"\b(openwork|pierced|reticulated)\b", re.I), "openwork pierced", "镂空透雕"),
    (re.compile(r"\b(relief|carved|molded|incised)\b", re.I), "carved surface decoration", "浮雕刻花"),
    (re.compile(r"\b(covered|with\s+cover|with\s+lid|lidded)\b", re.I), "with a fitted lid", "带盖"),
)


def extract_observable_cues(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (en_cues, zh_cues) strictly from cleaned observable text fields."""
    raw = " ".join(
        clean_visual_phrase(str(item.get(k) or ""))
        for k in ("title", "object_name", "medium")
    )
    en_cues: list[str] = []
    zh_cues: list[str] = []
    for pat, en_str, zh_str in OBSERVABLE_CUES:
        if pat.search(raw):
            en_cues.append(en_str)
            zh_cues.append(zh_str)
    return en_cues, zh_cues


def build_bilingual_visual_pairs(
    item: dict[str, Any],
    annotation: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Build (en_views, zh_views, cf_neg_en, cf_neg_zh) for one training item.

    Never reads era, dynasty, date, culture, or museum provenance.
    """
    obj_type = str(item.get("object_type") or "").strip().lower()
    if annotation and annotation.get("object_type"):
        obj_type = str(annotation["object_type"]).strip().lower()
    mat = str(item.get("material") or "").strip().lower()
    motifs = [m for m in item.get("motifs", []) if m in ZH_MOTIF_NAMES]
    colors = [c for c in item.get("colors", []) if c in ZH_COLOR_NAMES]

    en_cues, zh_cues = extract_observable_cues(item)
    obj_en = clean_visual_phrase(item.get("object_name") or obj_type or "artifact").lower() or "artifact"
    if obj_type and obj_type in ZH_OBJ_NAMES:
        obj_en_canon = obj_type
        obj_zh = ZH_OBJ_NAMES[obj_type]
    else:
        obj_en_canon = obj_en
        obj_zh = ZH_MAT_NAMES.get(mat, "馆藏") + "器物"

    mat_zh = ZH_MAT_NAMES.get(mat, "")
    motif_zh_list = [ZH_MOTIF_NAMES[m] for m in motifs[:3]]
    color_zh_list = [ZH_COLOR_NAMES[c] for c in colors[:2]]

    # Construct structured English & Chinese query-style views
    en_mods = list(dict.fromkeys(colors[:2] + en_cues[:2] + ([mat] if mat else [])))
    en_query_view = "a " + " ".join([*en_mods, obj_en_canon]).strip()
    if motifs:
        en_query_view += " decorated with " + " and ".join(motifs[:3])

    zh_prefix_parts = list(dict.fromkeys(color_zh_list + zh_cues[:3] + ([mat_zh] if mat_zh and mat_zh not in obj_zh else [])))
    zh_prefix = "".join(zh_prefix_parts)
    if motif_zh_list:
        zh_motif_str = "、".join(motif_zh_list)
        zh_desc_view = f"{zh_prefix}{obj_zh}，表面饰有{zh_motif_str}。"
        zh_query_view = f"饰有{zh_motif_str}的{zh_prefix}{obj_zh}"
    else:
        zh_desc_view = f"{zh_prefix}{obj_zh}。" if zh_prefix else f"一件{obj_zh}。"
        zh_query_view = f"{zh_prefix}{obj_zh}" if zh_prefix else obj_zh

    base_en = visual_caption(item)
    if annotation:
        ann_en = annotation["caption_en"].strip()
        ann_zh = str(annotation.get("caption_zh") or "").strip() or zh_desc_view
        en_views = [ann_en, en_query_view if len(en_mods) >= 2 else base_en]
        zh_views = [ann_zh, zh_query_view if zh_prefix or motif_zh_list else ann_zh]
    else:
        en_views = [base_en, en_query_view]
        zh_views = [zh_desc_view, zh_query_view]

    # Build single-attribute-swapped counterfactual hard negatives (same category, wrong attribute)
    cf_en: list[str] = []
    cf_zh: list[str] = []

    if annotation:
        sh = annotation.get("shape") or {}
        cl = annotation.get("colors") or {}
        mt = annotation.get("motifs") or {}
        # 1. Verified shape counterfactual
        if sh.get("handles") == 0:
            cf_en.append(f"A {obj_en_canon} with two prominent curved loop handles on the sides.")
            cf_zh.append(f"两侧带有明显弯曲双耳柄的{obj_zh}。")
        elif sh.get("handles") == 1:
            cf_en.append(f"A smooth handleless {obj_en_canon} with plain sides and no lugs.")
            cf_zh.append(f"器侧平整光滑、无耳无柄的{obj_zh}。")
        elif sh.get("wide_mouth") == 1 and sh.get("narrow_mouth") == 0:
            cf_en.append(f"A tall bottle-shaped {obj_en_canon} with a very narrow small mouth and slender neck.")
            cf_zh.append(f"小口细长颈的{obj_zh}。")
        elif sh.get("narrow_mouth") == 1 and sh.get("wide_mouth") == 0:
            cf_en.append(f"A shallow wide-mouthed open {obj_en_canon} without a neck.")
            cf_zh.append(f"浅腹大敞口、无颈的{obj_zh}。")

        # 2. Verified color or motif counterfactual
        absent_colors = [c for c in ("red", "blue", "black", "yellow", "green") if cl.get(c) == 0]
        if absent_colors:
            c_swap = absent_colors[0]
            c_zh = ZH_COLOR_NAMES[c_swap]
            cf_en.append(f"A monochrome {c_swap} {obj_en_canon} with solid {c_swap} surface.")
            cf_zh.append(f"通体纯{c_zh}色表面的{obj_zh}。")
        elif mt.get("dragon") == 0:
            cf_en.append(f"A {obj_en_canon} carved with large coiled dragons among clouds.")
            cf_zh.append(f"雕有盘龙云纹的{obj_zh}。")
    else:
        raw_lower = f"{item.get('title', '')} {item.get('medium', '')}".lower()
        # Handle/lug counterfactual
        if any(w in raw_lower for w in ("handle", "lug", "ewer", "teapot")):
            cf_en.append(f"A plain handleless {mat} {obj_en_canon} without any handles or spout.".replace("  ", " "))
            cf_zh.append(f"无耳无流无柄的素面{mat_zh}{obj_zh}。")
        elif obj_type in {"vase", "jar", "cup", "bowl", "plate"}:
            cf_en.append(f"A {mat} {obj_en_canon} with two tall curved ear handles and three tripod legs.".replace("  ", " "))
            cf_zh.append(f"带高弧形双耳与三足的{mat_zh}{obj_zh}。")

        # Color / motif counterfactual
        if "blue" in raw_lower:
            cf_en.append(f"A monochrome bright red {mat} {obj_en_canon} without any blue pigment.".replace("  ", " "))
            cf_zh.append(f"通体纯红釉、无蓝彩的{mat_zh}{obj_zh}。")
        elif "dragon" in motifs:
            cf_en.append(f"A plain undecorated {mat} {obj_en_canon} with no surface pattern.".replace("  ", " "))
            cf_zh.append(f"通体素面无纹饰的{mat_zh}{obj_zh}。")
        else:
            cf_en.append(f"A blue and white {mat} {obj_en_canon} painted with coiled dragons and human figures.".replace("  ", " "))
            cf_zh.append(f"绘青花盘龙与人物故事纹的{mat_zh}{obj_zh}。")

    return en_views, zh_views, cf_en[:2], cf_zh[:2]


def select_vision_lora_targets(model: torch.nn.Module, include_pooling_head: bool = True) -> list[str]:
    """Select vision self-attention projections + SigLIP-2 spatial pooling head."""
    attn_suffixes = ("q_proj", "k_proj", "v_proj", "out_proj")
    targets: list[str] = []
    for name, module in model.named_modules():
        if "vision_model" not in name or not isinstance(module, torch.nn.Linear):
            continue
        if name.endswith(attn_suffixes):
            targets.append(name)
        elif include_pooling_head and "vision_model.head" in name:
            targets.append(name)
    return targets


def compositional_contrastive_loss(
    image_features: torch.Tensor,
    pos_text_features: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    cf_text_features: torch.Tensor | None = None,
    ignore_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
    cf_weight: float = 0.6,
) -> torch.Tensor:
    """Multi-positive contrastive loss with counterfactual text negatives and false-negative masking."""
    img = F.normalize(image_features.float(), dim=-1)
    txt = F.normalize(pos_text_features.float(), dim=-1)
    logits_pos = (img @ txt.T) / temperature
    pos_f = positive_mask.float()

    # Mask out non-positive in-batch pairs whose text embeddings are near-identical
    masked_logits_i2t = logits_pos
    masked_logits_t2i = logits_pos.T
    if ignore_mask is not None:
        suppress = ignore_mask & (~positive_mask)
        masked_logits_i2t = masked_logits_i2t.masked_fill(suppress, -1e4)
        masked_logits_t2i = masked_logits_t2i.masked_fill(suppress.T, -1e4)

    if cf_text_features is not None and cf_text_features.numel() > 0 and cf_weight > 0:
        cf_txt = F.normalize(cf_text_features.float(), dim=-1)
        logits_cf = (img @ cf_txt.T) / temperature + float(np.log(cf_weight))
        full_i2t = torch.cat([masked_logits_i2t, logits_cf], dim=1)
        log_probs_i2t = F.log_softmax(full_i2t, dim=1)[:, : logits_pos.shape[1]]
    else:
        log_probs_i2t = F.log_softmax(masked_logits_i2t, dim=1)

    forward = -(log_probs_i2t * pos_f).sum(dim=1) / pos_f.sum(dim=1).clamp_min(1.0)
    log_probs_t2i = F.log_softmax(masked_logits_t2i, dim=1)
    backward = -(log_probs_t2i * pos_f.T).sum(dim=1) / pos_f.T.sum(dim=1).clamp_min(1.0)
    return 0.5 * (forward.mean() + backward.mean())


def _predecode_pixel_values(encoder: Encoder, paths: list[str], batch_size: int = 64) -> torch.Tensor:
    """Decode and preprocess deterministic 224x224 pixel_values once in fp16 CPU RAM (~1.04 GB for 3.5k imgs)."""
    chunks: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        imgs: list[Image.Image] = []
        for p in paths[start : start + batch_size]:
            with Image.open(p) as im:
                imgs.append(ImageOps.exif_transpose(im).convert("RGB"))
        pv = encoder.processor(images=imgs, return_tensors="pt")["pixel_values"].to(torch.float16)
        chunks.append(pv)
    return torch.cat(chunks, dim=0)


def train_comp_lora(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    items = read_items(args.data)
    validate_object_splits(items)

    annotations = load_annotations(args.visual_annotations, items) if args.visual_annotations else {}
    train_items_unique = [x for x in items if x["split"] == "train"]
    dev_items = [x for x in items if x["split"] == "dev"]
    dev_annot_items = [x for x in dev_items if x["item_id"] in annotations]
    if args.max_train:
        train_items_unique = train_items_unique[: args.max_train]
    if not train_items_unique or len(dev_items) < 10:
        raise ValueError("Need object-disjoint train and dev items (at least 10 dev)")

    encoder = Encoder(args.encoder, args.device)
    print(f"Loaded {args.encoder} on {encoder.device}; building bilingual views & counterfactuals for {len(train_items_unique)} train items...", flush=True)

    # Pre-build bilingual positive views and counterfactual hard negatives for all unique train items
    en_view0, en_view1 = [], []
    zh_view0, zh_view1 = [], []
    cf_en_all, cf_zh_all = [], []
    cf_owner_idx: list[int] = []
    obj_groups: list[str] = []
    is_annotated: list[bool] = []

    for idx, item in enumerate(train_items_unique):
        ann = annotations.get(item["item_id"])
        ev, zv, cfe, cfz = build_bilingual_visual_pairs(item, ann)
        en_view0.append(ev[0])
        en_view1.append(ev[1] if len(ev) > 1 else ev[0])
        zh_view0.append(zv[0])
        zh_view1.append(zv[1] if len(zv) > 1 else zv[0])
        cf_en_all.append(cfe[0])
        cf_zh_all.append(cfz[0])
        cf_owner_idx.append(idx)
        obj_groups.append(str(item.get("object_group") or item["item_id"]))
        is_annotated.append(ann is not None)

    # Pre-encode all frozen text vectors upfront (zero text-tower overhead during epochs)
    t0 = time.perf_counter()
    t_en0 = torch.from_numpy(encoder.encode_texts(en_view0, batch_size=64))
    t_en1 = torch.from_numpy(encoder.encode_texts(en_view1, batch_size=64))
    t_zh0 = torch.from_numpy(encoder.encode_texts(zh_view0, batch_size=64))
    t_zh1 = torch.from_numpy(encoder.encode_texts(zh_view1, batch_size=64))
    t_cf_en = torch.from_numpy(encoder.encode_texts(cf_en_all, batch_size=64))
    t_cf_zh = torch.from_numpy(encoder.encode_texts(cf_zh_all, batch_size=64))

    # Dev evaluation text vectors (strictly dev split only)
    dev_en_caps = [visual_caption(x) for x in dev_items]
    dev_zh_caps = [build_bilingual_visual_pairs(x, annotations.get(x["item_id"]))[1][0] for x in dev_items]
    dev_en_vec = encoder.encode_texts(dev_en_caps, batch_size=64)
    dev_zh_vec = encoder.encode_texts(dev_zh_caps, batch_size=64)

    dev_ann_en_caps = [annotations[x["item_id"]]["caption_en"] for x in dev_annot_items]
    dev_ann_zh_caps = [annotations[x["item_id"]]["caption_zh"] for x in dev_annot_items]
    dev_ann_en_vec = encoder.encode_texts(dev_ann_en_caps, batch_size=64) if dev_annot_items else None
    dev_ann_zh_vec = encoder.encode_texts(dev_ann_zh_caps, batch_size=64) if dev_annot_items else None
    print(f"Encoded all frozen EN/ZH positive + counterfactual texts in {time.perf_counter() - t0:.1f}s", flush=True)

    # Pre-decode pixel_values once + compute frozen teacher features
    t_img0 = time.perf_counter()
    train_paths = [x["image_path"] for x in train_items_unique]
    dev_paths = [x["image_path"] for x in dev_items]
    dev_annot_idx = [i for i, x in enumerate(dev_items) if x["item_id"] in annotations]

    train_pixels_fp16 = _predecode_pixel_values(encoder, train_paths, batch_size=64)
    dev_pixels_fp16 = _predecode_pixel_values(encoder, dev_paths, batch_size=64)

    def _encode_cached_pixels(pixels_fp16: torch.Tensor, bs: int = 64) -> np.ndarray:
        outs: list[np.ndarray] = []
        with torch.inference_mode():
            for s in range(0, pixels_fp16.shape[0], bs):
                pv = pixels_fp16[s : s + bs].to(device=encoder.device, dtype=torch.float32)
                with encoder.autocast():
                    feats = encoder.model.get_image_features(pixel_values=pv)
                feats = F.normalize(feats.float(), dim=-1)
                outs.append(feats.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    teacher_np = _encode_cached_pixels(train_pixels_fp16, bs=args.batch_size)
    np.save(output / "teacher_train.npy", teacher_np)
    teacher = torch.from_numpy(teacher_np)

    base_dev_feats = _encode_cached_pixels(dev_pixels_fp16, bs=args.batch_size)
    print(f"Cached pixel tensors & teacher embeddings in {time.perf_counter() - t_img0:.1f}s", flush=True)

    def _eval_dev(dev_feats: np.ndarray) -> dict[str, Any]:
        en_568 = paired_retrieval(dev_feats, dev_en_vec, dev_en_caps)
        zh_568 = paired_retrieval(dev_feats, dev_zh_vec, dev_zh_caps)
        res: dict[str, Any] = {
            "dev568_en": en_568,
            "dev568_zh": zh_568,
            "dev568_bi_mrr": 0.5 * (en_568["mrr"] + zh_568["mrr"]),
            "dev568_bi_r5": 0.5 * (en_568["r@5"] + zh_568["r@5"]),
        }
        if dev_annot_items and dev_ann_en_vec is not None and dev_ann_zh_vec is not None:
            ann_feats = dev_feats[dev_annot_idx]
            ann_en = paired_retrieval(ann_feats, dev_ann_en_vec, dev_ann_en_caps)
            ann_zh = paired_retrieval(ann_feats, dev_ann_zh_vec, dev_ann_zh_caps)
            res["annot60_en"] = ann_en
            res["annot60_zh"] = ann_zh
            res["annot60_bi_mrr"] = 0.5 * (ann_en["mrr"] + ann_zh["mrr"])
            res["annot60_bi_r5"] = 0.5 * (ann_en["r@5"] + ann_zh["r@5"])
            res["selection_score"] = 0.55 * res["annot60_bi_mrr"] + 0.45 * res["dev568_bi_mrr"]
        else:
            res["selection_score"] = res["dev568_bi_mrr"]
        return res

    baseline_dev = _eval_dev(base_dev_feats)
    print("Baseline dev metrics:", json.dumps({k: v for k, v in baseline_dev.items() if isinstance(v, float)}, indent=2), flush=True)

    target_modules = select_vision_lora_targets(encoder.model, include_pooling_head=not args.no_head_lora)
    if not target_modules:
        raise RuntimeError("No vision target modules found")
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(encoder.model, lora_cfg)
    encoder.model = model
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Oversample blind-annotated train items (still 100% train split) so rich fine-grained captions get higher signal
    annot_train_indices = [i for i, flag in enumerate(is_annotated) if flag]
    epoch_sample_pool = list(range(len(train_items_unique))) + annot_train_indices * max(0, args.annotated_boost - 1)
    steps_per_epoch = (len(epoch_sample_pool) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min((step + 1) / max(1, total_steps * 0.08), 1.0)
            * 0.5
            * (1.0 + np.cos(np.pi * step / max(1, total_steps)))
        ),
    )

    metadata: dict[str, Any] = {
        "method": "bilingual_attribute_contrastive_lora",
        "config": vars(args),
        "encoder": encoder.metadata,
        "target_modules_count": len(target_modules),
        "adapts_spatial_pooling_head": not args.no_head_lora,
        "trainable_parameters": param_count,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "visual_annotations_sha256": (
            hashlib.sha256(Path(args.visual_annotations).read_bytes()).hexdigest()
            if args.visual_annotations
            else None
        ),
        "train_unique_items": len(train_items_unique),
        "train_annotated_items": len(annot_train_indices),
        "dev_items": len(dev_items),
        "dev_annotated_items": len(dev_annot_items),
        "baseline_dev": baseline_dev,
        "selection_metric": "0.55 * annot60_bi_mrr + 0.45 * dev568_bi_mrr (strictly dev split)",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    history_fh = (output / "history.jsonl").open("w", encoding="utf-8")

    best_score = baseline_dev["selection_score"] - 1e-6
    saved_any = False
    start_ts = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base_model.model.text_model.eval()
        perm = np.random.permutation(epoch_sample_pool)
        epoch_losses: list[float] = []

        for step, b_start in enumerate(range(0, len(perm), args.batch_size)):
            raw_idx = perm[b_start : b_start + args.batch_size]
            # Keep distinct unique indices inside a single batch to avoid duplicate-image triviality
            seen_u: set[int] = set()
            idx_list = [int(i) for i in raw_idx if not (int(i) in seen_u or seen_u.add(int(i)))]
            if len(idx_list) < 4:
                continue
            idx_t = torch.tensor(idx_list, dtype=torch.long)

            pv = train_pixels_fp16[idx_t].to(device=encoder.device, dtype=torch.float32)
            teach_b = teacher[idx_t].to(device=encoder.device)

            # Alternate / mix primary descriptive view and concise query-style view per item
            choose_v1 = torch.rand(len(idx_list)) < 0.35
            en_b = torch.where(choose_v1.unsqueeze(1), t_en1[idx_t], t_en0[idx_t]).to(encoder.device)
            zh_b = torch.where(choose_v1.unsqueeze(1), t_zh1[idx_t], t_zh0[idx_t]).to(encoder.device)
            cf_en_b = t_cf_en[idx_t].to(encoder.device)
            cf_zh_b = t_cf_zh[idx_t].to(encoder.device)

            groups_b = [obj_groups[i] for i in idx_list]
            caps_en_b = [en_view0[i] for i in idx_list]
            pos_mask = torch.tensor(
                [[groups_b[a] == groups_b[b] or caps_en_b[a] == caps_en_b[b] for b in range(len(idx_list))] for a in range(len(idx_list))],
                device=encoder.device,
                dtype=torch.bool,
            )
            # Semantic false-negative mask: if two distinct items have near-identical text vectors (>0.965), don't repel
            with torch.no_grad():
                sim_en = en_b @ en_b.T
                sim_zh = zh_b @ zh_b.T
                ignore_en = sim_en > 0.965
                ignore_zh = sim_zh > 0.965

            optimizer.zero_grad(set_to_none=True)
            with encoder.autocast():
                img_feats = model.base_model.model.get_image_features(pixel_values=pv)
                loss_en = compositional_contrastive_loss(
                    img_feats,
                    en_b,
                    pos_mask,
                    cf_text_features=cf_en_b,
                    ignore_mask=ignore_en,
                    temperature=args.temperature,
                    cf_weight=args.cf_weight,
                )
                loss_zh = compositional_contrastive_loss(
                    img_feats,
                    zh_b,
                    pos_mask,
                    cf_text_features=cf_zh_b,
                    ignore_mask=ignore_zh,
                    temperature=args.temperature,
                    cf_weight=args.cf_weight,
                )
                contrast_loss = (1.0 - args.zh_weight) * loss_en + args.zh_weight * loss_zh
                anchor_loss = (1.0 - F.cosine_similarity(F.normalize(img_feats.float(), dim=-1), teach_b)).mean()
                loss = contrast_loss + args.anchor_weight * anchor_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach()))

            if step % 30 == 0:
                print(
                    json.dumps(
                        {
                            "event": "step",
                            "epoch": epoch,
                            "step": step,
                            "steps_per_epoch": steps_per_epoch,
                            "loss": round(float(loss.detach()), 4),
                            "loss_en": round(float(loss_en.detach()), 4),
                            "loss_zh": round(float(loss_zh.detach()), 4),
                            "elapsed_s": round(time.perf_counter() - start_ts, 1),
                        }
                    ),
                    flush=True,
                )

        model.eval()
        dev_feats = _encode_cached_pixels(dev_pixels_fp16, bs=args.batch_size)
        dev_metrics = _eval_dev(dev_feats)
        score = dev_metrics["selection_score"]
        rec = {
            "epoch": epoch,
            "loss": round(float(np.mean(epoch_losses)), 4),
            "selection_score": round(score, 5),
            "dev568_bi_mrr": round(dev_metrics["dev568_bi_mrr"], 5),
            "annot60_bi_mrr": round(dev_metrics.get("annot60_bi_mrr", 0.0), 5),
            "elapsed_s": round(time.perf_counter() - start_ts, 1),
        }
        history_fh.write(json.dumps({**rec, "dev_full": dev_metrics}, ensure_ascii=False) + "\n")
        history_fh.flush()
        print(json.dumps(rec), flush=True)

        if score > best_score or not saved_any:
            if score > best_score:
                best_score = score
            saved_any = True
            model.save_pretrained(output / "best_adapter")
            metadata["best_epoch"] = epoch
            metadata["best_dev"] = dev_metrics
            metadata["improved_over_baseline"] = score > baseline_dev["selection_score"]
            (output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    history_fh.close()
    return metadata


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    p.add_argument("--visual-annotations", default="eval/visual_labels/annotations.jsonl")
    p.add_argument("--output", default="artifacts/training/siglip_lora_comp")
    p.add_argument("--encoder", choices=["clip", "siglip"], default="siglip")
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=12)
    p.add_argument("--anchor-weight", type=float, default=0.45)
    p.add_argument("--zh-weight", type=float, default=0.50)
    p.add_argument("--cf-weight", type=float, default=0.50, help="Counterfactual attribute-negative weight")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--annotated-boost", type=int, default=3, help="Epoch repeat factor for fine-grained train items")
    p.add_argument("--no-head-lora", action="store_true", help="Disable LoRA on SigLIP spatial pooling head")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=5523)
    p.add_argument("--max-train", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    res = train_comp_lora(args)
    print(json.dumps({"best_epoch": res.get("best_epoch"), "improved": res.get("improved_over_baseline")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
