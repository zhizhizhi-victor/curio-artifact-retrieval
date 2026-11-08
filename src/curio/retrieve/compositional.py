"""Compositional multi-condition visual query decomposition and calibrated gating.

All functions in this module operate strictly in the ``image_only`` regime:
only the input query string, image embeddings, and image-embedding linear probes
are used. Museum catalogue titles, descriptions, tags, and BM25 never participate.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from curio.query import ParsedQuery, parse_query

# General Chinese visual morphology -> English visual appearance glosses.
# Restricted strictly to pixel-visible shapes, structural parts, surface finishes,
# tones, postures, and depicted motifs; never infers dynasty, provenance, or authenticity.
ZH_VISUAL_GLOSSES: tuple[tuple[str, str], ...] = (
    # Silhouettes & overall forms
    ("葫芦形", "double-gourd shaped two stacked bulbs"),
    ("葫芦", "double-gourd"),
    ("梨形腹", "pear-shaped body"),
    ("梨形", "pear-shaped"),
    ("梅瓶", "meiping plum vase high-shouldered"),
    ("玉壶春", "pear-shaped flaring-lip bottle vase"),
    ("蒜头瓶", "garlic-head bulbous-mouth bottle"),
    ("扁瓶", "flattened flask"),
    ("抱月瓶", "moonflask flat round flask"),
    ("笔筒", "cylindrical brush pot"),
    ("圆柱形", "cylindrical"),
    ("圆柱", "cylindrical"),
    ("近球形", "nearly spherical globular"),
    ("球形", "spherical globular"),
    ("圆鼓", "rounded bulbous"),
    ("鼓腹", "bulbous rounded belly"),
    ("直筒", "straight-sided cylindrical"),
    ("较直", "straight-sided"),
    ("四方", "square four-sided"),
    ("六方", "hexagonal six-sided"),
    ("八方", "octagonal eight-sided"),
    ("长方形", "rectangular"),
    ("长方盒", "rectangular box"),
    ("长方", "rectangular"),
    ("圆形", "circular round"),
    ("圆盘", "round disc dish"),
    ("圆镜", "round mirror disc"),
    ("扁平", "flat"),
    ("细长", "long slender narrow elongated"),
    ("狭长", "long narrow strip"),
    ("低矮", "low shallow"),
    ("浅盘", "shallow dish tray"),
    ("高足", "tall stem foot"),
    ("三足", "three-legged tripod"),
    ("四足", "four-legged"),
    ("敞口", "open wide-mouthed"),
    ("撇口", "flaring mouth"),
    ("喇叭口", "flaring trumpet-shaped mouth"),
    ("像喇叭", "flaring trumpet-like"),
    ("喇叭", "trumpet-like flaring"),
    ("细颈", "slender narrow neck"),
    ("长细颈", "long narrow neck"),
    ("颈部特别细长", "exceptionally thin long neck"),
    ("长颈", "long neck"),
    ("短颈", "short neck"),
    ("小口", "small mouth opening"),
    ("开口很小", "small opening at top"),
    ("收窄", "narrowed waist middle"),
    ("下腹", "lower body belly"),
    # Parts & structural details
    ("高弧形柄", "tall curved handle"),
    ("弧形柄", "curved handle"),
    ("短柄", "short handle"),
    ("长柄", "long handle"),
    ("侧柄", "side handle"),
    ("提梁", "overhead arched handle"),
    ("双耳", "paired side lug handles"),
    ("小耳", "small lug ears handles"),
    ("贯耳", "tubular lug handles"),
    ("铺首", "animal-mask ring handles"),
    ("流口", "spout"),
    ("长流", "long spout"),
    ("壶嘴", "spout"),
    ("带盖", "lidded with fitted cover"),
    ("有盖", "lidded with cover"),
    ("盖缝", "horizontal lid seam"),
    ("底座", "pedestal base"),
    ("圈足", "ring foot"),
    ("凸钮", "projecting central boss knob"),
    ("尖刃", "pointed blade tip"),
    ("竖棱", "closely spaced vertical ribs"),
    ("出戟", "projecting flanges"),
    ("装订缝线", "visible binding stitches along edge"),
    ("缝线", "stitching"),
    ("珠穗", "long bead tassels"),
    ("流苏", "hanging tassels"),
    ("悬饰", "hanging pendant ornaments"),
    ("书册状", "closed book-like album"),
    ("书册", "book album"),
    # Surface, texture, tone & color nuances
    ("素净", "mostly plain undecorated"),
    ("少装饰", "mostly undecorated plain"),
    ("光亮", "glossy"),
    ("深红色", "deep-red"),
    ("深红", "deep-red"),
    ("蓝绿色", "blue-green turquoise"),
    ("蓝白色", "blue-and-white"),
    ("蓝白", "blue-and-white"),
    ("红黑", "red and black"),
    ("金黄色", "golden-yellow"),
    ("金黄", "golden-yellow"),
    ("浅色", "pale light-colored"),
    ("深色", "dark"),
    ("粉色", "pink"),
    ("绿袍", "green robes"),
    ("褐色", "brown"),
    ("彩绘", "polychrome painted"),
    ("浮雕牌", "carved relief plaque"),
    ("浮雕", "raised relief carved"),
    ("雕刻纹", "carved relief decoration"),
    ("盘旋雕刻", "deep swirling carved"),
    ("盘旋", "swirling coiling"),
    ("盘曲", "coiling"),
    ("镂空", "openwork pierced cutout"),
    ("开光", "framed decorative panels"),
    ("折线纹", "bold angular zigzag pattern"),
    ("菱形格", "diamond lattice pattern"),
    ("菱形", "diamond pattern"),
    ("几何纹", "repeated geometric pattern"),
    ("环形纹带", "concentric patterned bands"),
    ("边框", "patterned border rim"),
    ("外圈", "outer rim border"),
    # Depicted subjects, figures, animals, scenes
    ("临水建筑", "waterfront buildings architecture"),
    ("楼阁", "buildings pavilions architecture"),
    ("建筑", "buildings architecture"),
    ("小船", "small boats"),
    ("树木", "trees"),
    ("山水", "landscape mountains and water"),
    ("景物", "scenery landscape"),
    ("大朵花卉", "large bold flowers blossoms"),
    ("花卉", "floral flowers"),
    ("叶片", "leaves foliage"),
    ("缠枝", "scrolling floral vines"),
    ("人脸", "detached human face mask"),
    ("人头像", "detached sculpted human head"),
    ("站立人像", "full-length standing human figure"),
    ("站立人物", "standing people figures"),
    ("坐姿人像", "seated human figure"),
    ("坐姿", "seated"),
    ("站立", "standing"),
    ("长袍", "long flowing robes"),
    ("帽状物", "tall headdress hat"),
    ("动物像", "sculpted animal figure"),
    ("小兽", "raised creatures animals"),
    ("卧着", "reclining"),
    ("分叉角", "branching antlers"),
    ("斑点", "pale speckles spots"),
    ("长腿", "long-legged"),
    ("包块", "bulky packs"),
    ("鞍状物", "saddle-like packs on back"),
    ("鸟形", "bird-shaped"),
    ("尾羽", "long sweeping tail feathers"),
    ("有冠", "crest on head"),
)

HEAD_NOUN_EN = re.compile(
    r"\b(bottle|vase|vessel|bowl|dish|plate|disc|plaque|garment|robe|textile|strip|"
    r"face|head|figure|statue|animal|ornament|pendant|box|container|jar|pot|ewer|"
    r"teapot|censer|mirror|cup|bell|stele|book|object)\b",
    re.I,
)

EN_CLAUSE_SPLIT = re.compile(
    r",\s*(?=(?:with|showing|painted|decorated|carved|divided|whose|featuring|wearing|carrying|having|and\s+(?:a|an|one|two|three|several|visible))\b)"
    r"|\s+(?=(?:with|showing|painted\s+with|decorated\s+with|carved\s+with|divided\s+by|whose)\s+(?:a|an|the|one|two|three|several|many|closely|dense|deep|large|small|tall|short|long|broad|narrow|visible|bulky|branching|waterfront|blue|red|green|pink|white|black|dark|pale|bright)\b)"
    r"|\s+and\s+(?=(?:a|an|one|two|three|several|visible)\b)"
    r"|\s+above\s+(?=(?:a|an|the)\b)",
    re.I,
)

# Frozen on visual_dev; never tuned on visual_test.
DEFAULT_TAU = 1.5
DEFAULT_LAMBDA_AND = 0.20
DEFAULT_LAMBDA_NEG = 0.15
DEFAULT_LAMBDA_PROBE = 0.22
DEFAULT_POOL_K = 60


def gloss_zh_clause(text: str) -> str:
    """Map visible Chinese morphological descriptors to concise English visual glosses."""
    hits: list[str] = []
    remaining = text
    for zh_term, en_gloss in ZH_VISUAL_GLOSSES:
        if zh_term in remaining:
            hits.append(en_gloss)
            remaining = remaining.replace(zh_term, " ")
    sub_parsed = parse_query(text)
    for slot in ("object_type", "colors", "shape", "motifs"):
        for val in sub_parsed.slots.get(slot, []):
            word = val.replace("_", " ")
            if not any(word in existing for existing in hits):
                hits.append(word)
    return " ".join(hits).strip()


def extract_negation_phrases(raw_query: str, parsed: ParsedQuery) -> list[str]:
    """Extract negated or contrastive visual phrases for soft embedding repulsion."""
    phrases = [neg["text"] for neg in parsed.negations if neg.get("text")]
    for pat in (
        r"\brather\s+than\s+([^,.;!?]+)",
        r"\binstead\s+of\s+([^,.;!?]+)",
        r"(?:而不是|而非)([^，。；！？]+)",
    ):
        for match in re.finditer(pat, raw_query, flags=re.I):
            span = match.group(1).strip()
            if span:
                phrases.append(span)
    cleaned: list[str] = []
    for phrase in phrases:
        stripped = re.sub(
            r"^(?:不是|不要|排除|不含|没有|沒有|不带|不帶|无|無|非|without|excluding|exclude|not|no)\s*",
            "",
            phrase.strip(),
            flags=re.I,
        ).strip()
        if stripped:
            cleaned.append(stripped)
    return list(dict.fromkeys(cleaned))


def decompose_visual_query(query: str, parsed: ParsedQuery) -> dict[str, Any]:
    """Decompose a natural-language query into global, conjunctive sub-clause, and negative prompts."""
    text = parsed.positive_text
    text = re.sub(r",?\s*\b(?:rather\s+than|instead\s+of)\b[^,.;!?]*", "", text, flags=re.I)
    text = re.sub(r"[，,]?\s*(?:而不是|而非)[^，。；！？]*", "", text)
    text = re.sub(r"[，,]?\s*(?:而|但|却)\s*(?=$|[，,。；;])", "", text)
    neg_phrases = extract_negation_phrases(query, parsed)

    obj_slots = [
        v
        for v in parsed.slots.get("object_type", [])
        if v != "lid" or parsed.slots.get("object_type") == ["lid"]
    ]
    slot_noun = obj_slots[-1].replace("_", " ") if obj_slots else ""

    if parsed.language == "zh":
        clean = re.sub(r"^(?:请帮我找|帮我找|请找|寻找|搜索|查找|找)\s*", "", text).strip(" ，,。；;")
        raw_parts = [
            p.strip()
            for p in re.split(r"[，,；;。]+", clean)
            if p.strip() and p.strip() not in {"而", "但", "却", "且", "并"}
        ]
        full_gloss = gloss_zh_clause(clean)
        global_prompt = f"{clean} . {full_gloss}".strip(" .") if full_gloss else clean

        sub_prompts: list[str] = []
        for idx, part in enumerate(raw_parts):
            gloss = gloss_zh_clause(part)
            if idx == 0:
                sub_prompts.append(f"{part} . {gloss}".strip(" .") if gloss else part)
            else:
                en_part = f"a {slot_noun} with {gloss}".strip() if (slot_noun and gloss) else gloss
                sub_prompts.append(f"{part} . {en_part}".strip(" .") if en_part else part)
        neg_prompts = []
        for neg_txt in neg_phrases:
            neg_gloss = gloss_zh_clause(neg_txt)
            neg_prompts.append(f"{neg_txt} . {neg_gloss}".strip(" .") if neg_gloss else neg_txt)
        return {
            "global_prompt": global_prompt,
            "raw_clauses": raw_parts,
            "sub_prompts": sub_prompts,
            "neg_prompts": neg_prompts,
        }

    clean = re.sub(
        r"^(?:find|search\s+for|show\s+me|looking\s+for)\s+", "", text, flags=re.I
    ).strip(" ,.;")
    global_prompt = parsed.rewritten
    raw_parts = [p.strip(" ,.") for p in EN_CLAUSE_SPLIT.split(clean) if len(p.strip(" ,.")) >= 4]
    head_match = HEAD_NOUN_EN.search(raw_parts[0]) if raw_parts else None
    head_noun = head_match.group(1).lower() if head_match else (slot_noun or "object")

    sub_prompts = []
    for idx, part in enumerate(raw_parts):
        if idx == 0:
            sub_prompts.append(part)
        else:
            part_clean = re.sub(r"^(?:and|with)\s+", "", part, flags=re.I)
            if HEAD_NOUN_EN.search(part_clean):
                sub_prompts.append(part_clean)
            else:
                has_prep = part.lower().startswith(
                    ("with ", "showing ", "painted ", "wearing ", "carrying ", "divided ", "whose ")
                )
                prefix = "" if has_prep else "with "
                sub_prompts.append(f"a {head_noun} {prefix}{part_clean if prefix else part}")
    return {
        "global_prompt": global_prompt,
        "raw_clauses": raw_parts,
        "sub_prompts": sub_prompts,
        "neg_prompts": neg_phrases,
    }


def soft_and_zscores(sub_matrix: np.ndarray, pool_idx: np.ndarray, tau: float = DEFAULT_TAU) -> np.ndarray:
    """Compute candidate-standardized Soft-AND (-1/tau * log mean exp(-tau * z)) across sub-queries."""
    pool_vals = sub_matrix[pool_idx]
    mu = pool_vals.mean(axis=0, keepdims=True)
    sd = np.maximum(pool_vals.std(axis=0, keepdims=True), 1e-4)
    z = np.clip((sub_matrix - mu) / sd, -3.5, 3.5)
    if tau <= 0:
        return z.mean(axis=1)
    scaled = -tau * z
    max_s = scaled.max(axis=1, keepdims=True)
    lse = max_s[:, 0] + np.log(np.mean(np.exp(scaled - max_s), axis=1))
    return -lse / tau


def calibrated_probe_logscore(
    probe_predictions: dict[str, dict[str, Any]],
    probe_report: dict[str, Any],
    parsed: ParsedQuery,
    candidate_indices: list[int],
    unknown_reason_fn: Callable[..., str | None],
) -> tuple[np.ndarray | None, list[tuple[str, str, np.ndarray, bool]]]:
    """Reliability-weighted log-probability conjunction (with OR-group support for color alternatives)."""
    if not probe_predictions or not candidate_indices:
        return None, []
    text_lower = parsed.positive_text.lower()
    disjunctive_colors = bool(re.search(r"\b(?:or|blue-green)\b|或|蓝绿", text_lower))

    group_scores: list[np.ndarray] = []
    group_weights: list[float] = []
    fields: list[tuple[str, str, np.ndarray, bool]] = []

    for task, pred in probe_predictions.items():
        task_rep = (probe_report or {}).get("tasks", {}).get(task, {})
        f1 = float(task_rep.get("dev_macro_f1", 0.75))
        rel = max(0.02, f1 - 0.50)
        vals = [
            v
            for v in parsed.slots.get(task, [])
            if not unknown_reason_fn(parsed, task, v) and v in pred["classes"]
        ]
        if not vals:
            continue
        for v in vals:
            fields.append((task, v, pred["probabilities"][:, pred["classes"].index(v)], False))
        probs = np.stack(
            [pred["probabilities"][candidate_indices, pred["classes"].index(v)] for v in vals],
            axis=1,
        )
        if task == "colors" and disjunctive_colors and len(vals) > 1:
            p_eff = np.clip(probs.max(axis=1), 0.05, 0.999)
            group_scores.append(np.log(p_eff))
            group_weights.append(rel)
        else:
            for col in range(probs.shape[1]):
                p_eff = np.clip(probs[:, col], 0.05, 0.999)
                group_scores.append(np.log(p_eff))
                group_weights.append(rel)

    seen_neg: set[tuple[str, str]] = set()
    for neg in parsed.negations:
        task, val = neg["slot"], neg["value"]
        pred = probe_predictions.get(task)
        if (
            (task, val) in seen_neg
            or pred is None
            or val not in pred["classes"]
            or unknown_reason_fn(parsed, task, val, negated=True)
        ):
            continue
        seen_neg.add((task, val))
        task_rep = (probe_report or {}).get("tasks", {}).get(task, {})
        f1 = float(task_rep.get("dev_macro_f1", 0.75))
        rel = max(0.02, f1 - 0.50)
        full_probs = pred["probabilities"][:, pred["classes"].index(val)]
        fields.append((task, val, full_probs, True))
        p_eff = np.clip(1.0 - full_probs[candidate_indices], 0.05, 0.999)
        group_scores.append(np.log(p_eff))
        group_weights.append(rel)

    if not group_scores:
        return None, []
    weights = np.asarray(group_weights, dtype=np.float32)
    mat = np.stack(group_scores, axis=1)
    return (mat @ weights) / weights.sum(), fields
