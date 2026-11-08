from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from .ontology import ERAS, SPECIALIST_GLOSSARY, VOCAB


def term_pattern(term: str) -> str:
    escaped = re.escape(term)
    # 陶瓷 is the broader ceramic family, not evidence for its porcelain or
    # earthenware subclasses. Preserve bare 瓷/陶 matches outside that word.
    if term == "瓷":
        return r"(?<!陶)瓷"
    if term == "陶":
        return r"陶(?!瓷)"
    # Guard single-character Chinese and polysemous English terms against
    # grammatical similes, verbs, and depicted-scene compounds.
    if term == "像":
        return r"(?<![口端形体状好就图影头頭])像(?![^，。；,\s]{0,8}(?:一样|似的)|尖|喇叭)"
    if term == "盘":
        return r"(?<![算棋])盘(?![旋曲绕繞龙龍长長踞坐])"
    if term == "木":
        return r"(?<![树樹草花林灌乔喬])木"
    if term == "盖":
        return r"(?<![涵覆])盖(?!缝)"
    if term == "covered":
        return r"(?<![a-z])covered(?![a-z])(?!\s+(?:with|in|by)\b)"
    return rf"(?<![a-z]){escaped}(?![a-z])" if term.isascii() else escaped


@dataclass
class ParsedQuery:
    query: str
    language: str
    positive_text: str
    rewritten: str
    slots: dict[str, list[str]] = field(default_factory=dict)
    negations: list[dict] = field(default_factory=list)
    era: list[int] | None = None
    era_label: str | None = None
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    specialist_expansions: list[dict] = field(default_factory=list)
    confidence: float = 1.0
    parser: str = "bilingual-rules-v1"

    def to_dict(self) -> dict:
        return asdict(self)


NEG = r"(?:不是|不要|排除|不含|没有|沒有|不带|不帶|无|無|非|\bwithout\b|\bexcluding\b|\bexclude\b|\bnot\b|\bno\b)"


def parse_query(query: str) -> ParsedQuery:
    """Parse known attributes conservatively; never infer image visibility.

    Negative phrases are removed from *every* retrieval channel, and their
    canonical values are kept exclusively in ``negations`` for penalties.
    Count and spatial phrases are reported as unsupported visual constraints.
    """
    text = query.strip()
    lower = text.lower()
    negatives: list[dict] = []
    negative_spans: list[tuple[int, int]] = []
    slots: dict[str, list[str]] = {key: [] for key in VOCAB}
    # A short phrase after a negator is bounded by punctuation or a clause
    # transition. Only vocabulary matches in that span become exclusions.
    scope_re = re.compile(NEG + r"[^,，。;；!?！？]{0,55}", re.I)
    for scope in scope_re.finditer(lower):
        phrase = re.split(r"(?:而是|但是|但|带有|帶有|with\b|but\b)", scope.group(), maxsplit=1)[0]
        # Chinese 的 terminates an attribute: 没有龙纹的青花瓷瓶.
        phrase = phrase.split("的", 1)[0]
        end = scope.start() + len(phrase)
        negative_spans.append((scope.start(), end))
        for slot, values in VOCAB.items():
            for value, aliases in values.items():
                if any(re.search(term_pattern(alias), phrase, re.I) for alias in aliases):
                    negatives.append({"slot": slot, "value": value, "text": text[scope.start():end]})
    positive = list(text)
    for start, end in negative_spans:
        positive[start:end] = " " * (end - start)
    positive_text = re.sub(r"\s+", " ", "".join(positive)).strip(" ,，。;；")
    excluded = {(entry["slot"], entry["value"]) for entry in negatives}
    for slot, values in VOCAB.items():
        for value, aliases in values.items():
            if (slot, value) not in excluded and any(re.search(term_pattern(alias), positive_text, re.I) for alias in aliases):
                slots[slot].append(value)
    # Avoid broad substring matches duplicating a more specific material.
    if "porcelain" in slots["material"]:
        slots["material"] = [x for x in slots["material"] if x not in {"ceramic", "earthenware"}]
    elif "ceramic" in slots["material"]:
        slots["material"] = [x for x in slots["material"] if x != "earthenware"]
    if "bronze" in slots["material"]:
        slots["material"] = [x for x in slots["material"] if x != "copper"]
    if "stoneware" in slots["material"]:
        slots["material"] = [x for x in slots["material"] if x != "stone"]
    if "silk" in slots["material"]:
        slots["material"] = [x for x in slots["material"] if x != "textile"]
    # Generic Chinese 壶 is contained in 茶壶/执壶/水壶. Keep their
    # specific object type rather than double-counting the same mention.
    if "teapot" in slots["object_type"] or "ewer" in slots["object_type"]:
        specific = [alias for value in ("teapot", "ewer") for alias in VOCAB["object_type"][value]]
        remainder = positive_text
        for alias in sorted(specific, key=len, reverse=True):
            remainder = re.sub(term_pattern(alias), " ", remainder, flags=re.I)
        if not any(re.search(term_pattern(alias), remainder, re.I) for alias in VOCAB["object_type"]["pot"]):
            slots["object_type"] = [value for value in slots["object_type"] if value != "pot"]
    # Robes worn by depicted figures (穿...袍) describe attire, not a garment artifact.
    if "garment" in slots["object_type"]:
        attire_stripped = re.sub(r"(?:身穿|穿着|穿)[^,，。;；\s]{0,8}袍", " ", positive_text)
        if not any(re.search(term_pattern(alias), attire_stripped, re.I) for alias in VOCAB["object_type"]["garment"]):
            slots["object_type"] = [value for value in slots["object_type"] if value != "garment"]
    era = None
    era_label = None
    for label, (start, end, aliases) in ERAS.items():
        if any(re.search(term_pattern(alias), positive_text, re.I) for alias in aliases):
            era, era_label = [start, end], label
            break
    years = re.search(r"(?<!\d)(-?\d{3,4})\s*(?:到|至|[-–—])\s*(-?\d{3,4})(?!\d)", positive_text)
    if years:
        start, end = map(int, years.groups())
        if start <= end:
            era, era_label = [start, end], years.group()
    unsupported = []
    if re.search(r"(?:[一二三四五六七八九十两兩双雙\d]+[件只对對足耳柄]|\b(?:two|three|four|pair|double)\b)", lower):
        unsupported.append("count")
    if re.search(r"腹部|口沿|两侧|兩側|环绕|環繞|左侧|右侧|\b(?:around|rim|left|right|sides|belly)\b", lower):
        unsupported.append("spatial")
    expansion = [value.replace("_", " ") for values in slots.values() for value in values]
    specialist_expansions = [{"rule": key, "text": rule["expansion"], "basis": "project_manual_rule"}
                            for key, rule in SPECIALIST_GLOSSARY.items()
                            if any(re.search(term_pattern(alias), positive_text, re.I) for alias in rule["aliases"])]
    expansion.extend(rule["text"] for rule in specialist_expansions)
    if era_label:
        expansion.append(f"{era_label} dynasty")
    rewritten = " ".join([positive_text] + expansion).strip()
    warnings = []
    if unsupported:
        warnings.append("数量/空间条件尚未经过视觉核验；仅作查询提示，不声称满足。")
    if negatives:
        warnings.append("排除条件仅依据元数据冲突降权；缺失标签不能证明图像不存在该属性。")
    if specialist_expansions:
        warnings.append("专业词采用项目手工视觉释义扩展，未作权威术语核验；扩展不构成可见属性确认。")
    if "转心" in text or "轉心" in text:
        warnings.append("转心结构无法凭单张图像确认；保留原词，未核验内部旋转结构，也未作确定性视觉展开。")
    if "流心" in text:
        warnings.append("“流心”在本项目词表中未核验；原词保留，不自动改写为其他器物术语。")
    return ParsedQuery(text, "zh" if re.search(r"[\u3400-\u9fff]", text) else "en", positive_text, rewritten, slots, negatives, era, era_label, unsupported, warnings,
                       specialist_expansions=specialist_expansions)
