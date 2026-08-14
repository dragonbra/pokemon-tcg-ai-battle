"""Generate the self-contained, image-rich 0044 exact-deck asset browser."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
from typing import Any

from .assets import AssetRegistry, canonical_deck_sha256, sha256_file
from .own_archetype import OwnArchetypeVocabulary


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "assets/decks"
LEGACY_REPORT_ROOT = REPOSITORY_ROOT / "experiments/0044_g2_dragapult_policy_option_lora/deck_taxonomy"
CARD_DATA = REPOSITORY_ROOT / "data/official/EN_Card_Data.csv"
IMAGE_SETS = {
    "ASC": "me2pt5", "BLK": "zsv10pt5", "DRI": "sv10", "JTG": "sv9",
    "MEG": "me1", "PAL": "sv2", "PFL": "me2", "POR": "me3",
    "PRE": "sv8pt5", "SCR": "sv7", "SFA": "sv6pt5", "SSP": "sv8",
    "SVE": "sve", "SVI": "sv1", "SVP": "svp", "TEF": "sv5",
    "TWM": "sv6", "WHT": "rsv10pt5",
}
SCRYDEX_SETS = {"me2pt5", "me3"}
ARCHETYPE_CN = {
    "dragapult_ex": ("以 Dragapult ex 为唯一结构性 Stage 2 主轴，通过 Phantom Dive 同时压低前场与后场血线。", "主要决策是多龙进化时钟、火/超能量分配，以及伤害指示物如何为后续多奖赏击倒铺路。"),
    "mega_lopunny_ex": ("以 Mega Lopunny ex 的换位增伤为核心，主动区与备战区的移动本身就是伤害资源。", "需要围绕换位次数、出场时点与击倒阈值安排动作顺序。"),
    "marnies_grimmsnarl_ex": ("以 Marnie's Grimmsnarl ex 的 Stage 2 展开和恶能量加速为主体，辅以 Froslass/Munkidori 调整伤害。", "进化时机直接决定整套牌的能量经济和接力速度。"),
    "alakazam": ("以 Alakazam 的手牌规模伤害为胜利路线，并用 Dudunsparce 循环补充手牌。", "手牌、牌库预算与进化时钟必须联合规划，不能只追求即时抽牌。"),
    "mega_lucario_ex": ("以 Mega Lucario ex 快速进攻，并把弃牌区能量加速到后场打手。", "攻击轮转、弃牌准备和后场附能共同决定节奏。"),
    "mega_kangaskhan_ex": ("以 Mega Kangaskhan ex 的前场抽牌/攻击为主，搭配耐久副线。", "前场归属、无色能量配置和下一只打手准备是主要判断。"),
    "festival_lead": ("依赖 Festival Grounds 与满备战区，使 Dipplin 建立连续攻击压力。", "场地时机和铺场完整度直接改变奖赏交换速度。"),
    "crustle_barbaracle_wall": ("以 Crustle/Barbaracle 的单奖赏墙和抗性条件拖慢对手，而非普通抢攻。", "对局免疫条件、资源封锁与慢速奖赏交换优先于纯伤害。"),
    "meganium_grass_evolution": ("同时维护多条草系进化线，利用 Meganium 与草系场地加速展开。", "备战区位置、进化先后和可移动草能量是共享资源。"),
    "team_rockets_mewtwo_ex": ("围绕 Team Rocket Pokémon 数量与后场能量牺牲满足 Mewtwo ex 的攻击条件。", "火箭成员数量和后场能量不是附属资源，而是攻击的硬前提。"),
    "teal_mask_ogerpon_heros_cape": ("以多只 Teal Mask Ogerpon ex 反复 Teal Dance，加速能量并同步过牌。", "每次附能的对象与顺序同时影响伤害、抽牌和后续接力。"),
    "cynthias_garchomp_ex": ("围绕 Cynthia 标签检索、进化和斗能量攻击轮转展开。", "标签卡检索、进化线和攻击轮转构成稳定的统一计划。"),
    "mega_starmie_ex": ("以 Mega Starmie ex 的快速 Stage 1 铺开前后场伤害，并由 Mega Froslass 施加手牌压力。", "快速进化与前场/后场伤害去向决定最佳动作顺序。"),
    "archaludon_ex": ("先把金属能量送入弃牌区，再通过 Archaludon ex 回收加速并形成耐久进攻。", "攻击前的弃牌准备决定 Assemble Alloy 是否能及时启动。"),
    "other": ("未知、尚未建模或证据不足的回退类别。", "只用于未知 exact deck；当前正式牌池没有卡组被长期留在这里。"),
    "dragapult_dusknoir": ("在 Dragapult ex 的分散伤害主轴上，额外维护 Duskull→Dusclops→Dusknoir 自爆伤害线。", "第二条进化线占用备战位并改变进化资源；主动自爆会让出奖赏，但能重写击倒阈值和奖赏竞速，所以与普通多龙系统性不同。"),
    "dragapult_blaziken": ("同时维护 Dragapult 与 Blaziken 两条 Stage 2，并利用弃牌区能量回收。", "进化资源、备战位和持续能量回收使其资源规划不同于普通多龙。"),
    "mega_starmie_dusknoir": ("把 Starmie 的铺伤路线与 Dusknoir 主动自爆伤害结合。", "第二进化线、自爆让奖与伤害阈值会改变备战区及击倒次序。"),
    "slowking_toolbox": ("通过牌库顶控制让 Slowking 复制多种非 Rule Box Pokémon 的攻击。", "核心是安排牌库顶和选择被复制的攻击，而非 Kangaskhan 式正面交换。"),
    "area_zero_tera_toolbox": ("利用 Area Zero 扩展备战区，组织多属性 Tera 打手与跨场能量转移。", "八格备战区、Tera 保留和跨目标能量路由共同定义决策。"),
    "pecharunt_area_zero_poison": ("围绕 Pecharunt 的中毒增幅和 Area Zero 扩展场面建立胜利路线。", "Pecharunt 的前场时机、检查阶段毒伤和八格备战区配置会改变击倒节奏。"),
    "teal_mask_ogerpon_toolbox": ("实际主轴是 Ogerpon 能量引擎，少量其他攻击手只负责特定对局。", "检索和堆叠能量长期占主导，薄 Lopunny 线不足以把它归为 Lopunny 主轴。"),
    "mega_gardevoir_leafeon": ("以 Mega Gardevoir ex 的全场超能量配置配合 Leafeon 的伤害/治疗路线。", "Grand Tree 进化、后场能量分配和分布式能量伤害是结构性决策。"),
    "erikas_vileplume_ex": ("通过 Erika 进化体系进行全场治疗，并用睡眠/中毒持续施压。", "多进化线、团队治疗和特殊状态节奏值得独立策略先验。"),
    "ns_zoroark_ex": ("用 N's Zoroark ex 的弃牌抽牌引擎，复制后场不同 N's Pokémon 的攻击。", "后场攻击库如何构建、每回合复制哪种攻击会稳定改变动作选择。"),
    "chandelure_hand_control": ("围绕双方手牌规模调整 Chandelure 伤害，并用 Sylveon 和控制场地防守。", "需要同时管理双方手牌、Rare Candy、锁场组件和非标准伤害曲线。"),
    "crustle_great_tusk_mill": ("用 Crustle 筑墙，同时让 Great Tusk 以弃牌库作为替代胜利条件。", "Supporter 和攻击的价值以推进 deck-out 为准，而不是普通奖赏竞速。"),
    "hydrapple_meganium": ("通过 Hydrapple/Meganium 加速草能量，使全场能量规模转化为伤害与治疗。", "全场能量、治疗附能和进化加速取代 Festival 双击逻辑。"),
    "mega_venusaur_meganium": ("同时维护 Mega Venusaur 与 Meganium 两条 Stage 2，并让草能量在场上移动。", "进化线投入与可移动草能量是主资源；薄 Kangaskhan 线只是副选项。"),
}
STAGE_CN = {
    "Basic Pokémon": "基础宝可梦", "Stage 1 Pokémon": "1 阶宝可梦",
    "Stage 2 Pokémon": "2 阶宝可梦", "Item": "物品",
    "Supporter": "支援者", "Stadium": "竞技场", "Pokémon Tool": "宝可梦道具",
    "Technical Machine": "招式学习器", "Basic Energy": "基本能量", "Special Energy": "特殊能量",
}
ARCHETYPE_NAME_CN = {
    "dragapult_ex": "多龙巴鲁托 ex 铺伤",
    "mega_lopunny_ex": "超级长耳兔 ex 换位进攻",
    "marnies_grimmsnarl_ex": "玛俐的长毛巨魔 ex / 雪妖女",
    "alakazam": "胡地 / 土龙节节",
    "mega_lucario_ex": "超级路卡利欧 ex 斗能量轮转",
    "mega_kangaskhan_ex": "超级袋兽 ex 前场续航",
    "festival_lead": "庆典主轴 / 裹蜜虫",
    "crustle_barbaracle_wall": "岩殿居蟹 / 龟足巨铠单奖赏墙",
    "meganium_grass_evolution": "大竺葵草系多进化线",
    "team_rockets_mewtwo_ex": "火箭队的超梦 ex",
    "teal_mask_ogerpon_heros_cape": "碧草面具厄诡椪 ex / 英雄斗篷",
    "cynthias_garchomp_ex": "竹兰的烈咬陆鲨 ex 进化轴",
    "mega_starmie_ex": "超级宝石海星 ex / 超级雪妖女 ex",
    "archaludon_ex": "铝钢桥龙 ex 金属能量加速",
    "other": "其他 / 尚未建模",
    "dragapult_dusknoir": "多龙巴鲁托 ex / 黑夜魔灵自爆",
    "dragapult_blaziken": "多龙巴鲁托 ex / 火焰鸡能量回收",
    "mega_starmie_dusknoir": "超级宝石海星 ex / 黑夜魔灵自爆",
    "slowking_toolbox": "呆呆王牌库顶攻击工具箱",
    "area_zero_tera_toolbox": "零之大空洞太晶工具箱",
    "pecharunt_area_zero_poison": "桃歹郎 / 零之大空洞中毒轴",
    "teal_mask_ogerpon_toolbox": "碧草面具厄诡椪能量工具箱",
    "mega_gardevoir_leafeon": "超级沙奈朵 ex / 叶伊布能量轴",
    "erikas_vileplume_ex": "莉佳的霸王花 ex 治疗控制",
    "ns_zoroark_ex": "N 的索罗亚克 ex 复制攻击",
    "chandelure_hand_control": "水晶灯火灵手牌控制",
    "crustle_great_tusk_mill": "岩殿居蟹 / 雄伟牙牌库破坏",
    "hydrapple_meganium": "蜜集大蛇 ex / 大竺葵草能量轴",
    "mega_venusaur_meganium": "超级妙蛙花 ex / 大竺葵",
}
DECK_NAME_CN_OVERRIDES = {
    "066": "多龙巴鲁托 ex / 黑夜魔灵干扰控制",
    "067": "多龙巴鲁托 ex / 愿增猿干扰控制",
    "068": "胡地 / 土龙节节训练家控制",
    "069": "胡地 / 土龙节节首领控制",
}


def _official_cards() -> dict[int, dict[str, str]]:
    with CARD_DATA.open(newline="", encoding="utf-8-sig") as handle:
        rows: dict[int, dict[str, str]] = {}
        for row in csv.DictReader(handle):
            rows.setdefault(int(row["Card ID"]), row)
    if not rows:
        raise RuntimeError("official card data is empty")
    return rows


def _category(stage: str) -> str:
    if "Energy" in stage:
        return "Energy"
    if stage in {"Item", "Supporter", "Stadium", "Pokémon Tool", "Technical Machine"}:
        return "Trainer"
    return "Pokémon"


def _image_urls(card: dict[str, str]) -> tuple[str | None, str | None]:
    set_id = IMAGE_SETS.get(card.get("Expansion", ""))
    number = card.get("Collection No.", "").strip()
    if not set_id or not number:
        return None, None
    if set_id in SCRYDEX_SETS:
        base = f"https://images.scrydex.com/pokemon/{set_id}-{number}"
        return f"{base}/small", f"{base}/large"
    base = f"https://images.pokemontcg.io/{set_id}/{number}"
    return f"{base}.png", f"{base}_hires.png"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _profile(deck, mapping, archetype, official: dict[int, dict[str, str]]) -> dict[str, Any]:
    path = PROJECT_ROOT / deck.deck_path
    exact_cards = tuple(int(value) for value in path.read_text().splitlines())
    counts = Counter(exact_cards)
    card_rows = []
    category_counts = Counter()
    evolution_lines: dict[str, list[str]] = defaultdict(list)
    key_trainers: list[str] = []
    energies: list[str] = []
    for card_id, count in sorted(counts.items()):
        card = official.get(card_id)
        if card is None:
            raise RuntimeError(f"deck {deck.deck_id} references unknown official card {card_id}")
        stage = card["Stage (Pokémon)/Type (Energy and Trainer)"]
        category = _category(stage)
        thumb, large = _image_urls(card)
        category_counts[category] += count
        row = {
            "card_id": card_id, "count": count, "name": card["Card Name"],
            "category": category, "stage_or_type": stage,
            "previous_stage": card.get("Previous stage", ""),
            "expansion": card.get("Expansion", ""),
            "collection_number": card.get("Collection No.", ""),
            "image_url": thumb, "large_image_url": large,
        }
        card_rows.append(row)
        if category == "Pokémon":
            previous = card.get("Previous stage", "")
            label = f"{count}× {card['Card Name']} [{card_id}] ({stage})"
            evolution_lines[previous if previous and previous != "n/a" else "Basic / root"].append(label)
        elif category == "Trainer" and stage in {"Supporter", "Stadium"}:
            key_trainers.append(f"{count}× {card['Card Name']} [{card_id}] ({stage})")
        elif category == "Energy":
            energies.append(f"{count}× {card['Card Name']} [{card_id}]")
    if sum(row["count"] for row in card_rows) != 60:
        raise RuntimeError(f"deck {deck.deck_id} profile does not total 60")
    source_manifest = deck.source.get("source_manifest", {})
    preferred = tuple(int(value) for value in source_manifest.get("representative_card_ids", ()))
    pokemon = [row for row in card_rows if row["category"] == "Pokémon"]
    stage_rank = {"Stage 2 Pokémon": 4, "Stage 1 Pokémon": 3, "Basic Pokémon": 2}
    ranked = sorted(
        pokemon,
        key=lambda row: (
            row["card_id"] not in preferred,
            -stage_rank.get(row["stage_or_type"], 1),
            -(" ex" in row["name"]), -row["count"], row["card_id"],
        ),
    )
    representatives = ranked[:2]
    return {
        "deck_id": deck.deck_id, "name": deck.name,
        "exact_deck_sha256": deck.content_sha256, "deck_file_sha256": deck.file_sha256,
        "roles": list(deck.roles), "source": dict(deck.source),
        "upstream_name": source_manifest.get("archetype") or deck.source.get("source_display_name") or deck.name,
        "display_name_correction": deck.source.get("display_name_correction"),
        "old_archetype_id": mapping.old_archetype_id,
        "own_archetype_id": mapping.archetype_id,
        "own_archetype_name": archetype.name,
        "own_archetype_display_name": archetype.display_name,
        "decision": mapping.decision, "parent_archetype": archetype.parent_archetype,
        "strategic_axis": mapping.strategic_axis, "definition": archetype.definition,
        "strategic_rationale": archetype.strategic_rationale,
        "tags": list(archetype.tags), "embedding_init_from": archetype.embedding_init_from,
        "category_counts": dict(category_counts), "evolution_lines": dict(evolution_lines),
        "key_trainers": key_trainers, "energies": energies, "cards": card_rows,
        "representative_cards": representatives,
        "name_cn": DECK_NAME_CN_OVERRIDES.get(deck.deck_id, ARCHETYPE_NAME_CN[archetype.name]),
        "definition_cn": ARCHETYPE_CN[archetype.name][0],
        "strategic_rationale_cn": ARCHETYPE_CN[archetype.name][1],
    }


def _text(profile: dict[str, Any], vocabulary: OwnArchetypeVocabulary) -> str:
    lines = [
        f"0044 DECK {profile['deck_id']} — {profile['name']}", "=" * 78, "", "IDENTITY",
        f"deck_id: {profile['deck_id']}", f"display_name: {profile['name']}",
        f"upstream_name: {profile['upstream_name']}",
        f"display_name_correction: {profile['display_name_correction'] or 'none'}",
        f"exact_deck_sha256: {profile['exact_deck_sha256']}",
        f"deck_file_sha256: {profile['deck_file_sha256']}", "card_total: 60",
        f"roles: {', '.join(profile['roles'])}", "", "TAXONOMY",
        "semantic_role: actor-visible Own Deck Strategy Archetype (not opponent Meta classifier)",
        f"taxonomy_version: {vocabulary.taxonomy_version}",
        f"taxonomy_sha256: {vocabulary.taxonomy_sha256}",
        f"mapping_sha256: {vocabulary.mapping_sha256}",
        f"old_own_archetype_id: {profile['old_archetype_id']}",
        f"own_archetype_id: {profile['own_archetype_id']}",
        f"own_archetype_name: {profile['own_archetype_name']}",
        f"own_archetype_display_name: {profile['own_archetype_display_name']}",
        f"decision: {profile['decision']}",
        f"parent_archetype: {profile['parent_archetype'] or 'none'}",
        f"embedding_init_from: {profile['embedding_init_from']}",
        f"tags: {', '.join(profile['tags'])}", "", "STRATEGIC CHARACTERISTICS",
        f"axis: {profile['strategic_axis']}", f"definition: {profile['definition']}",
        f"rationale: {profile['strategic_rationale']}",
        "evidence_boundary: exact-list/card-text audit; not an official-engine strength result",
        "", "COMPOSITION", "category_counts: " + ", ".join(
            f"{key}={profile['category_counts'].get(key, 0)}" for key in ("Pokémon", "Trainer", "Energy")
        ), "evolution_lines:",
    ]
    for previous, values in profile["evolution_lines"].items():
        lines.append(f"  {previous}:")
        lines.extend(f"    - {value}" for value in values)
    lines.append("key_supporters_and_stadiums:")
    lines.extend(f"  - {value}" for value in profile["key_trainers"])
    lines.append("energy:")
    lines.extend(f"  - {value}" for value in profile["energies"])
    lines += ["", "EXACT 60-CARD LIST", "count | card_id | card_name | set/no | category | stage_or_type", "-" * 92]
    for row in profile["cards"]:
        lines.append(
            f"{row['count']:>5} | {row['card_id']:>7} | {row['name']} | "
            f"{row['expansion']} {row['collection_number']} | {row['category']} | {row['stage_or_type']}"
        )
    lines += ["-" * 92, "TOTAL | 60", "", "PROVENANCE", json.dumps(profile["source"], sort_keys=True, ensure_ascii=False), ""]
    return "\n".join(lines)


BASE_CSS = """
:root{--bg:#f3f6f4;--paper:#fff;--ink:#17231f;--muted:#66766f;--line:#d9e3de;--green:#176b4d;--green-dark:#18382d;--amber:#9b6b12}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}
header{padding:28px max(20px,calc((100vw - 1500px)/2));background:var(--green-dark);color:#fff}h1{margin:0;font-size:30px}header p{max-width:1100px;margin:7px 0 0;color:#cfe1da}main{max-width:1500px;margin:auto;padding:20px}
a{color:var(--green);font-weight:700;text-decoration:none}.stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--paper)}.stat{padding:15px 18px;border-right:1px solid var(--line)}.stat:last-child{border:0}.stat b{display:block;font-size:23px}.stat span,small{display:block;color:var(--muted)}
.notice,.panel{margin-top:18px;padding:15px 17px;border:1px solid var(--line);background:var(--paper)}.notice{border-left:4px solid var(--green)}.tools{display:flex;gap:10px;margin:18px 0}input,select{padding:10px 12px;border:1px solid #b9c9c1;border-radius:4px;background:#fff}.tools input{width:min(520px,100%)}
.table{overflow:auto;border:1px solid var(--line);background:#fff}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{background:#e9f0ec;color:#486158;font-size:12px}tr:hover td{background:#f8fbf9}.number{font-size:16px;font-weight:850;color:var(--green)}
.deck{display:flex;align-items:center;gap:12px}.representative-art{display:flex;width:74px;flex:0 0 74px}.representative-art img{width:42px;height:59px;margin-right:-8px;border:1px solid #c9d5cf;border-radius:4px;object-fit:cover;background:#e4ebe7;box-shadow:0 2px 4px #0002}.strong{font-size:15px;font-weight:750}.pill{display:inline-block;padding:2px 7px;border-radius:999px;background:#e5f1eb;color:var(--green);font-size:12px;font-weight:700}.hash{font:11px ui-monospace,monospace;overflow-wrap:anywhere}
.hero{display:grid;grid-template-columns:220px 1fr;gap:22px}.hero-art{display:flex;align-items:flex-start}.hero-art img{width:132px;border-radius:8px;box-shadow:0 8px 20px #10271e33}.hero-art img+img{margin-left:-44px;margin-top:24px}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}.fact{padding:13px;background:#f7faf8;border:1px solid var(--line)}.fact b{display:block;color:var(--muted);font-size:12px}.correction{background:#fff6d9;border-left:4px solid var(--amber);padding:12px}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:14px}.card-tile{position:relative;background:#fff;border:1px solid var(--line);border-radius:7px;padding:10px}.card-thumb{display:block;width:100%;padding:0;border:0;background:#e9efec;cursor:zoom-in;aspect-ratio:2.5/3.5;overflow:hidden;border-radius:5px}.card-thumb img{width:100%;height:100%;display:block;object-fit:contain}.card-tile b{display:block;margin-top:8px}.count{position:absolute;right:5px;top:5px;z-index:2;background:var(--green-dark);color:#fff;padding:4px 8px;border-radius:999px;font-weight:850}.modal{position:fixed;inset:0;z-index:20;display:none;place-items:center;background:#07140fdd;padding:20px}.modal.open{display:grid}.modal img{max-height:92vh;max-width:92vw;border-radius:9px}.modal button{position:absolute;right:20px;top:14px;background:transparent;border:0;color:#fff;font-size:34px;cursor:pointer}.nav{display:flex;justify-content:space-between;gap:12px;margin:18px 0}.raw{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace}
@media(max-width:760px){header{padding:22px 16px}.stats{grid-template-columns:1fr 1fr}main{padding:14px}.hero{grid-template-columns:1fr}.hero-art{justify-content:center}.tools{flex-direction:column}.card-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def _image(row: dict[str, Any], *, detail: bool = False) -> str:
    if not row.get("image_url"):
        return ""
    src = row["image_url"]
    preview = row.get("large_image_url") or src
    attrs = f"src='{escape(src)}' alt='{escape(row['name'])}' loading='lazy'"
    if detail:
        return f"<button class='card-thumb' type='button' data-preview='{escape(preview)}'><img {attrs}></button>"
    return f"<img {attrs}>"


def _representative_art(profile: dict[str, Any], *, hero: bool = False) -> str:
    cls = "hero-art" if hero else "representative-art"
    return f"<span class='{cls}'>" + "".join(_image(row) for row in profile["representative_cards"]) + "</span>"


def _detail_html(profile: dict[str, Any], vocabulary: OwnArchetypeVocabulary, previous_id: str | None, next_id: str | None) -> str:
    cards = "".join(
        f"<article class='card-tile'><span class='count'>×{row['count']}</span>{_image(row, detail=True)}"
        f"<b>{escape(row['name'])}</b><small>{escape(row['expansion'])} {escape(row['collection_number'])} · "
        f"卡牌 ID {row['card_id']}</small><small>{escape(STAGE_CN.get(row['stage_or_type'], row['stage_or_type']))}</small></article>"
        for row in profile["cards"]
    )
    correction = (
        f"<p class='correction'><b>名称纠正</b> {escape(profile['upstream_name'])} → {escape(profile['name'])}。"
        f"{escape(profile['display_name_correction'])}</p>" if profile["display_name_correction"] else ""
    )
    prev_link = f"<a href='../{previous_id}/index.html'>← {previous_id}</a>" if previous_id else "<span></span>"
    next_link = f"<a href='../{next_id}/index.html'>{next_id} →</a>" if next_id else "<span></span>"
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>0044 卡组 {profile['deck_id']} · {escape(profile['name_cn'])}</title><style>{BASE_CSS}</style></head><body>
<header><h1>{profile['deck_id']} · {escape(profile['name_cn'])}</h1><p>英文资产名：{escape(profile['name'])}</p><p>0044 精确 60 张卡组资产 · Meta 分类（Own Archetype V2）· Policy-0809 CUDA-2048 报告视觉语言</p></header><main>
<nav class='nav'><a href='../../index.html'>← 返回 67 套卡组总览</a><span><a href='../../text/{profile['deck_id']}.txt'>纯文本档案</a> · <a href='deck.csv'>原始 deck.csv</a></span></nav>
<section class='panel hero'>{_representative_art(profile, hero=True)}<div><span class='pill'>Own {profile['own_archetype_id']} · {escape(profile['own_archetype_name'])}</span><h2>{escape(profile['own_archetype_display_name'])}</h2><p>{escape(profile['definition_cn'])}</p>{correction}</div></section>
<section class='stats'><div class='stat'><b>60</b><span>卡组总张数</span></div><div class='stat'><b>{profile['category_counts'].get('Pokémon',0)}</b><span>宝可梦</span></div><div class='stat'><b>{profile['category_counts'].get('Trainer',0)}</b><span>训练家</span></div><div class='stat'><b>{profile['category_counts'].get('Energy',0)}</b><span>能量</span></div></section>
<section class='panel'><h2>Meta 分类与分类依据</h2><div class='facts'><div class='fact'><b>Own Archetype V2 分类</b>{profile['own_archetype_id']} · {escape(profile['own_archetype_name'])}</div><div class='fact'><b>审计决定</b>{'沿用既有类别' if profile['decision']=='keep' else '从旧类别拆分'}</div><div class='fact'><b>父类别</b>{escape(profile['parent_archetype'] or '无')}</div><div class='fact'><b>Embedding 初始化</b>继承第 {profile['embedding_init_from']} 行策略先验</div></div><h3>为什么这样分类</h3><p>{escape(profile['strategic_rationale_cn'])}</p><p><strong>战略主轴：</strong>{escape(profile['strategic_axis'])}</p><p><strong>技术标签：</strong>{escape(', '.join(profile['tags']))}</p><p><small>审计边界：结论来自 exact 60-card 与当前卡文分析，不是 official-engine 强度评测；对手 Meta 分类器没有随本页改变。</small></p></section>
<section class='panel'><h2>完整 60 张卡组构成</h2><p>每个卡片格代表一种卡，右上角显示投入张数；点击卡图可打开大图。卡牌名称保留官方英文，类别和说明使用中文。</p><div class='card-grid'>{cards}</div></section>
<section class='panel'><h2>身份与来源</h2><div class='facts'><div class='fact'><b>精确卡组 SHA-256</b><span class='hash'>{profile['exact_deck_sha256']}</span></div><div class='fact'><b>deck.csv 文件 SHA-256</b><span class='hash'>{profile['deck_file_sha256']}</span></div><div class='fact'><b>分类体系版本</b>{vocabulary.taxonomy_version}<br><span class='hash'>{vocabulary.taxonomy_sha256}</span></div><div class='fact'><b>映射 SHA-256</b><span class='hash'>{vocabulary.mapping_sha256}</span></div></div><details><summary>查看完整来源 JSON</summary><div class='raw'>{escape(json.dumps(profile['source'], indent=2, sort_keys=True, ensure_ascii=False))}</div></details></section>
<nav class='nav'>{prev_link}{next_link}</nav></main><div class='modal' id='preview'><button type='button' aria-label='关闭'>×</button><img alt='卡图大图'></div><script>const m=document.getElementById('preview'),mi=m.querySelector('img');document.querySelectorAll('[data-preview]').forEach(b=>b.onclick=()=>{{mi.src=b.dataset.preview;m.classList.add('open')}});m.onclick=e=>{{if(e.target===m||e.target.tagName==='BUTTON')m.classList.remove('open')}};document.addEventListener('keydown',e=>{{if(e.key==='Escape')m.classList.remove('open')}});</script></body></html>\n"""


def _index_html(profiles: list[dict[str, Any]], vocabulary: OwnArchetypeVocabulary) -> str:
    distribution = Counter(row["own_archetype_name"] for row in profiles)
    rows = "".join(
        f"<tr data-search='{escape((p['deck_id']+' '+p['name']+' '+p['name_cn']+' '+p['own_archetype_name']+' '+ARCHETYPE_NAME_CN[p['own_archetype_name']]+' '+' '.join(p['tags'])).lower())}' data-archetype='{escape(p['own_archetype_name'])}'>"
        f"<td class='number'>{p['deck_id']}</td><td><div class='deck'>{_representative_art(p)}<span><a href='definitions/{p['deck_id']}/index.html'>{escape(p['name_cn'])}</a><small>英文资产名：{escape(p['name'])}</small><small>{escape(p['definition_cn'])}</small></span></div></td>"
        f"<td><span class='pill'>{p['own_archetype_id']} · {escape(ARCHETYPE_NAME_CN[p['own_archetype_name']])}</span><small>技术标识：{escape(p['own_archetype_name'])}</small><small>{escape(p['strategic_rationale_cn'])}</small></td>"
        f"<td>{'沿用既有类别' if p['decision']=='keep' else '从旧类别拆分'}</td><td>{p['category_counts'].get('Pokémon',0)} / {p['category_counts'].get('Trainer',0)} / {p['category_counts'].get('Energy',0)}</td>"
        f"<td><a href='text/{p['deck_id']}.txt'>TXT</a> · <a href='definitions/{p['deck_id']}/deck.csv'>CSV</a></td></tr>"
        for p in profiles
    )
    options = "".join(f"<option value='{escape(name)}'>{escape(ARCHETYPE_NAME_CN[name])}（{count} 套）</option>" for name, count in sorted(distribution.items(), key=lambda item: ARCHETYPE_NAME_CN[item[0]]))
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>0044 · 67 Exact Deck Assets</title><style>{BASE_CSS}</style></head><body>
<header><h1>0044 · 001–067 精确卡组资产</h1><p>67 套正式训练卡组；页面参照 Policy-0809 CUDA-2048 正式评测报告的视觉语言，直接从项目内精确卡组注册表、Own Archetype V2 和官方卡牌数据生成。</p></header><main>
<section class='stats'><div class='stat'><b>{len(profiles)}</b><span>精确卡组</span></div><div class='stat'><b>{sum(distribution.values())}</b><span>正式分类映射</span></div><div class='stat'><b>{len(distribution)}</b><span>当前使用的 Own 类别</span></div><div class='stat'><b>55</b><span>历史冻结评测（未改变）</span></div></section>
<p class='notice'><b>分类语义：</b>页面中的“Meta 分类”指 actor-visible <b>Own Deck Strategy Archetype V2</b>；它与冻结、未扩展的 Opponent Meta classifier 严格独立。每套详情都给出 exact 60-card、分类依据和 provenance。</p>
<div class='tools'><input id='search' type='search' placeholder='筛选编号、卡组、Own Archetype 或技术标签'><select id='archetype'><option value=''>全部 Own Archetype</option>{options}</select></div>
<div class='table'><table id='catalog'><thead><tr><th>编号</th><th>卡组 / 详情</th><th>Meta 分类（Own）</th><th>审计决定</th><th>宝可梦 / 训练家 / 能量</th><th>资产</th></tr></thead><tbody>{rows}</tbody></table></div>
<section class='panel'><h2>分类体系版本</h2><p><code>{vocabulary.taxonomy_version}</code></p><p class='hash'>{vocabulary.taxonomy_sha256}</p><p>映射 SHA-256</p><p class='hash'>{vocabulary.mapping_sha256}</p></section></main>
<script>const s=document.getElementById('search'),a=document.getElementById('archetype'),rows=[...document.querySelectorAll('#catalog tbody tr')];function filter(){{const q=s.value.toLowerCase(),v=a.value;rows.forEach(r=>r.hidden=!r.dataset.search.includes(q)||(v&&r.dataset.archetype!==v))}}s.oninput=filter;a.onchange=filter;</script></body></html>\n"""


def generate() -> dict[str, Any]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    for deck in registry.decks:
        deck_path = PROJECT_ROOT / deck.deck_path
        cards = tuple(int(value) for value in deck_path.read_text().splitlines())
        if (
            len(cards) != 60
            or sha256_file(deck_path) != deck.file_sha256
            or canonical_deck_sha256(cards) != deck.content_sha256
        ):
            raise RuntimeError(f"deck asset identity mismatch: {deck.deck_id}")
    vocabulary = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=PROJECT_ROOT)
    official = _official_cards()
    mapping_by_id = {row.deck_id: row for row in vocabulary.mappings}
    classes = {row.archetype_id: row for row in vocabulary.classes}
    profiles = [_profile(deck, mapping_by_id[deck.deck_id], classes[mapping_by_id[deck.deck_id].archetype_id], official) for deck in registry.decks]
    text_rows, detail_rows = [], []
    for index, profile in enumerate(profiles):
        text_path = OUTPUT_ROOT / "text" / f"{profile['deck_id']}.txt"
        _atomic_text(text_path, _text(profile, vocabulary))
        detail_path = OUTPUT_ROOT / "definitions" / profile["deck_id"] / "index.html"
        previous_id = profiles[index - 1]["deck_id"] if index else None
        next_id = profiles[index + 1]["deck_id"] if index + 1 < len(profiles) else None
        _atomic_text(detail_path, _detail_html(profile, vocabulary, previous_id, next_id))
        text_rows.append({"deck_id": profile["deck_id"], "path": text_path.relative_to(OUTPUT_ROOT).as_posix(), "sha256": sha256_file(text_path), "exact_deck_sha256": profile["exact_deck_sha256"], "own_archetype_id": profile["own_archetype_id"]})
        detail_rows.append({"deck_id": profile["deck_id"], "path": detail_path.relative_to(OUTPUT_ROOT).as_posix(), "sha256": sha256_file(detail_path)})
    html_path = OUTPUT_ROOT / "index.html"
    _atomic_text(html_path, _index_html(profiles, vocabulary))
    manifest = {
        "schema_version": "0044_deck_asset_browser_v2", "deck_count": len(registry.decks),
        "exact_card_total_per_deck": 60,
        "semantic_role": "Own Deck Strategy Archetype; independent from opponent Meta classifier",
        "source_deck_registry_sha256": sha256_file(OUTPUT_ROOT / "registry.json"),
        "taxonomy_version": vocabulary.taxonomy_version, "taxonomy_sha256": vocabulary.taxonomy_sha256,
        "mapping_sha256": vocabulary.mapping_sha256, "official_card_data_sha256": sha256_file(CARD_DATA),
        "html": {"path": "index.html", "sha256": sha256_file(html_path)},
        "text_profiles": text_rows, "detail_pages": detail_rows,
        "image_policy": "external official-set thumbnails with explicit alt text and browser fallback",
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    # Preserve the earlier experiment-report entry as a small pointer to the canonical asset browser.
    pointer = "<!doctype html><meta charset='utf-8'><title>0044 Deck Assets moved</title><meta http-equiv='refresh' content='0;url=../../../train/0044_g2_dragapult_policy_option_lora/assets/decks/index.html'><p><a href='../../../train/0044_g2_dragapult_policy_option_lora/assets/decks/index.html'>打开 canonical 0044 Deck Asset Browser</a></p>\n"
    _atomic_text(LEGACY_REPORT_ROOT / "index.html", pointer)
    return {"status": "PASS", "decks": len(profiles), "text_profiles": len(text_rows), "detail_pages": len(detail_rows), "html": str(html_path)}


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
