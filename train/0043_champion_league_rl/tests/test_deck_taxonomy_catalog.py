import hashlib
import importlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]
REPORT = REPOSITORY / "experiments/0043_champion_league_rl/deck_taxonomy"
DECK_ASSETS = PROJECT / "assets/decks"
assets = importlib.import_module("train.0043_champion_league_rl.assets")
generator = importlib.import_module(
    "train.0043_champion_league_rl.generate_deck_taxonomy_catalog"
)


def test_062_name_is_exact_list_corrected_without_identity_change():
    registry = assets.AssetRegistry.load(PROJECT)
    deck = next(row for row in registry.decks if row.deck_id == "062")
    assert deck.name == deck.archetype == "Mega Starmie ex / Mega Froslass ex"
    assert deck.content_sha256 == "654f54a66c951ab2b6f4a714286552a614ecb94e84752b47bf03770c4cd7c933"
    assert deck.file_sha256 == "5807f82f5472c15f6148c66ff81ad912fb80e0cb1f94b9f896b7b7c93248b04a"
    assert deck.source["source_manifest"]["archetype"] == "Mega Starmie ex / Dusknoir"
    cards = tuple(int(value) for value in (PROJECT / deck.deck_path).read_text().splitlines())
    assert len(cards) == 60
    assert not {131, 132, 133}.intersection(cards)
    assert {860, 861, 1030, 1031}.issubset(cards)


def test_catalog_has_67_complete_chinese_text_profiles_and_html_details():
    manifest = json.loads((DECK_ASSETS / "manifest.json").read_text())
    assert manifest["deck_count"] == 67
    assert len(manifest["text_profiles"]) == 67
    assert len(manifest["detail_pages"]) == 67
    assert manifest["semantic_role"].startswith("Own Deck Strategy Archetype")
    html = (DECK_ASSETS / "index.html").read_text()
    assert "Policy-0809 CUDA-2048" in html
    assert "representative-art" in html
    assert "<img" in html
    for index, row in enumerate(manifest["text_profiles"], start=1):
        deck_id = f"{index:03d}"
        assert row["deck_id"] == deck_id
        path = DECK_ASSETS / row["path"]
        text = path.read_text()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert f"0043 DECK {deck_id}" in text
        assert "card_total: 60" in text
        assert "TOTAL | 60" in text
        assert "not opponent Meta classifier" in text
        assert f"definitions/{deck_id}/index.html" in html
        detail = DECK_ASSETS / "definitions" / deck_id / "index.html"
        detail_html = detail.read_text()
        assert hashlib.sha256(detail.read_bytes()).hexdigest() == manifest["detail_pages"][index - 1]["sha256"]
        assert "card-grid" in detail_html
        assert "data-preview" in detail_html
        assert "Meta 分类（Own Archetype V2）" in detail_html
        assert "分类依据" in detail_html
        assert "完整 60 张卡组构成" in detail_html
        assert "为什么这样分类" in detail_html
        assert "deck.csv" in detail_html
        assert row["exact_deck_sha256"] in detail_html
    assert "Mega Starmie ex / Mega Froslass ex" in html
    assert "多龙巴鲁托 ex / 黑夜魔灵干扰控制" in html
    assert "多龙巴鲁托 ex / 愿增猿干扰控制" in html
    assert "主动自爆会让出奖赏" in html


def test_catalog_regeneration_is_deterministic():
    before = {
        path.relative_to(DECK_ASSETS).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DECK_ASSETS.rglob("*") if path.is_file()
    }
    assert generator.generate()["status"] == "PASS"
    after = {
        path.relative_to(DECK_ASSETS).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DECK_ASSETS.rglob("*") if path.is_file()
    }
    assert after == before
