import pytest

from curio.query import parse_query


def test_chinese_compound_query_extracts_slots_and_era():
    parsed = parse_query("明代青花瓷瓶，上面有龙纹")
    assert parsed.language == "zh"
    assert parsed.slots["object_type"] == ["vase"]
    assert parsed.slots["material"] == ["porcelain"]
    assert set(parsed.slots["colors"]) == {"blue", "white"}
    assert parsed.slots["motifs"] == ["dragon"]
    assert parsed.era == [1368, 1644]
    assert parsed.positive_text == "明代青花瓷瓶，上面有龙纹"


def test_negated_slots_never_become_positive_slots():
    parsed = parse_query("明代瓷瓶，不要龙纹")
    assert parsed.slots["motifs"] == []
    assert any(n["slot"] == "motifs" and n["value"] == "dragon" for n in parsed.negations)
    assert "龙纹" not in parsed.positive_text


def test_english_query_and_unsupported_visual_constraints():
    parsed = parse_query("A Ming porcelain vase with two handles and a dragon around the belly")
    assert parsed.slots["object_type"] == ["vase"]
    assert parsed.slots["material"] == ["porcelain"]
    assert "dragon" in parsed.slots["motifs"]
    assert parsed.era_label == "Ming"
    assert set(parsed.unsupported) == {"count", "spatial"}


@pytest.mark.parametrize("query,object_type,material", [
    ("玉琮", "cong", "jade"), ("a jade cong", "cong", "jade"),
    ("玉璧", "bi", "jade"), ("a jade bi disk", "bi", "jade"),
    ("玉佩", "pendant", "jade"), ("jade pendant", "pendant", "jade"),
    ("石碑", "stele", "stone"), ("stone stele", "stele", "stone"),
    ("玉饰", "ornament", "jade"), ("jade ornament", "ornament", "jade"),
    ("陶壶", "pot", "earthenware"), ("earthenware pot", "pot", "earthenware"),
])
def test_catalog_object_types_keep_jade_shapes_distinct(query, object_type, material):
    parsed = parse_query(query)
    assert parsed.slots["object_type"] == [object_type]
    assert parsed.slots["material"] == [material]
    assert "vase" not in parsed.slots["object_type"]


def test_generic_pot_alias_does_not_duplicate_specific_chinese_ewer_or_teapot():
    assert parse_query("青铜执壶").slots["object_type"] == ["ewer"]
    assert parse_query("瓷茶壶").slots["object_type"] == ["teapot"]
    assert set(parse_query("a pot and a teapot").slots["object_type"]) == {"pot", "teapot"}


def test_material_only_jade_query_never_infers_a_vase_or_ritual_shape():
    for query in ("玉器", "a jade object"):
        parsed = parse_query(query)
        assert parsed.slots["material"] == ["jade"]
        assert parsed.slots["object_type"] == []


@pytest.mark.parametrize("query", ["textile", "cotton fabric", "wool textile", "纺织品", "织物材质"])
def test_textile_is_valid_as_both_object_type_and_material(query):
    parsed = parse_query(query)
    assert parsed.slots["object_type"] == ["textile"]
    assert parsed.slots["material"] == ["textile"]


@pytest.mark.parametrize("query", ["silk textile", "丝绸织物"])
def test_explicit_silk_keeps_specific_material_instead_of_generic_textile(query):
    parsed = parse_query(query)
    assert parsed.slots["object_type"] == ["textile"]
    assert parsed.slots["material"] == ["silk"]


def test_covered_lid_alias_does_not_match_uncovered_or_lidless_words():
    assert parse_query("covered jar").slots["object_type"] == ["jar", "lid"]
    assert parse_query("uncovered jar").slots["object_type"] == ["jar"]
    assert parse_query("lidless jar").slots["object_type"] == ["jar"]


@pytest.mark.parametrize("query,shapes", [
    ("窄口细颈小口的瓶子", ["narrow_mouth"]),
    ("a narrow-necked vase", ["narrow_mouth"]),
    ("敞口宽口的碗", ["wide_mouth"]),
    ("a wide-mouthed bowl", ["wide_mouth"]),
    ("有耳有柄高足器物", ["handles", "footed"]),
    ("a round body with handles", ["handles", "round_body"]),
])
def test_image_shape_aliases_parse_as_query_constraints(query, shapes):
    parsed = parse_query(query)
    assert parsed.slots["shape"] == shapes


def test_catalog_expansion_parses_previously_unmapped_categories_and_attributes():
    parsed = parse_query("ivory ring with leaves and a long neck")
    assert parsed.slots["material"] == ["ivory"]
    assert set(parsed.slots["object_type"]) == {"ring"}
    assert "leaf" in parsed.slots["motifs"]
    assert "long_neck" in parsed.slots["shape"]
    assert "_" not in parsed.rewritten


def test_negated_shape_does_not_receive_positive_probe_support():
    parsed = parse_query("a vase without handles")
    assert parsed.slots["shape"] == []
    assert {entry["value"] for entry in parsed.negations if entry["slot"] == "shape"} == {"handles"}


@pytest.mark.parametrize("query,expansion", [
    ("粉彩", "porcelain with soft multicolored enamel decoration"),
    ("famille rose", "porcelain with soft multicolored enamel decoration"),
    ("镂空", "openwork pierced decoration"),
    ("描金", "gilt golden painted decoration"),
])
def test_specialist_visual_paraphrases_are_auditable_and_do_not_inject_hard_slots(query, expansion):
    parsed = parse_query(query)
    assert expansion in parsed.rewritten
    assert all(values == [] for values in parsed.slots.values())
    assert parsed.era is None and parsed.era_label is None
    assert parsed.specialist_expansions[0]["basis"] == "project_manual_rule"
    assert "dynasty" not in parsed.rewritten.lower() and "museum" not in parsed.rewritten.lower()
    assert any("未作权威术语核验" in warning for warning in parsed.warnings)


def test_blue_and_white_keeps_existing_slots_and_appends_plain_visual_text():
    parsed = parse_query("青花")
    assert set(parsed.slots["colors"]) == {"blue", "white"}
    assert "blue and white porcelain decoration" in parsed.rewritten


def test_unverified_specialist_structure_and_unknown_terms_preserve_original_words():
    for query in ("转心瓶", "流心瓶"):
        parsed = parse_query(query)
        assert parsed.query == query and query in parsed.rewritten
        assert parsed.specialist_expansions == []
        assert "openwork" not in parsed.rewritten and "rotating" not in parsed.rewritten
        assert any("未核验" in warning or "无法凭单张" in warning for warning in parsed.warnings)
    assert "转心" not in parse_query("流心瓶").rewritten


def test_english_negation_does_not_match_the_suffix_of_knot():
    parsed = parse_query("a knot ornament with blue decoration")
    assert parsed.negations == []
    assert "ornament" in parsed.slots["object_type"]
    assert parsed.slots["colors"] == ["blue"]
    assert parsed.positive_text == "a knot ornament with blue decoration"


@pytest.mark.parametrize("query,material", [
    ("陶瓷", "ceramic"), ("陶瓷花瓶", "ceramic"),
    ("唐代陶瓷杯", "ceramic"), ("陶器", "earthenware"),
    ("彩陶罐", "earthenware"), ("瓷器", "porcelain"),
    ("青瓷瓶", "porcelain"), ("青花瓷碗", "porcelain"),
])
def test_ceramic_umbrella_does_not_infer_material_subclass(query, material):
    assert parse_query(query).slots["material"] == [material]


def test_negated_ceramic_umbrella_keeps_only_the_requested_exclusion():
    parsed = parse_query("玉器，不要陶瓷")
    assert parsed.slots["material"] == ["jade"]
    assert {entry["value"] for entry in parsed.negations if entry["slot"] == "material"} == {"ceramic"}


@pytest.mark.parametrize("query,forbidden_slot,forbidden_value", [
    ("细长窄条形物件，一端像尖刃，另一端有圆环", "object_type", "figure"),
    ("瓶口像喇叭一样张开，颈部细长", "object_type", "figure"),
    ("深红色圆形小盒，表面满是盘旋雕刻纹", "object_type", "plate"),
    ("圆盘中央绘有临水建筑、树木与小船", "material", "wood"),
    ("长颈瓷瓶，瓶腹画有穿绿袍的人物", "object_type", "garment"),
    ("a round red box covered with deep swirling carved relief", "object_type", "lid"),
])
def test_compound_and_grammatical_boundaries_prevent_false_positive_slots(query, forbidden_slot, forbidden_value):
    parsed = parse_query(query)
    assert forbidden_value not in parsed.slots[forbidden_slot]


def test_true_statue_plate_wood_and_garment_queries_still_match():
    assert "figure" in parse_query("站立人像").slots["object_type"]
    assert "plate" in parse_query("青花圆盘").slots["object_type"]
    assert "wood" in parse_query("木雕盒").slots["material"]
    assert "garment" in parse_query("刺绣长袍").slots["object_type"]
