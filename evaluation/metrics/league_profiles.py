"""Auditable deck-category focus profiles for Frozen League evaluation."""

from __future__ import annotations

from dataclasses import dataclass


COMMON_METRICS = (
    "first_attack_round",
    "attack_continuity",
    "missed_attack_opportunities",
    "prizes_taken",
    "prizes_per_attack",
    "multi_prize_turns",
    "post_ko_attack_gap",
    "max_bench",
    "max_evolved_pokemon",
    "max_attached_energy",
    "deck_cards_consumed",
    "minimum_deck_count",
    "supporter_turn_rate",
    "energy_attach_turn_rate",
    "opponent_attack_denial_rate",
)


@dataclass(frozen=True)
class DeckQualityProfile:
    profile_id: str
    title: str
    deck_ids: tuple[str, ...]
    key_card_ids: tuple[int, ...]
    focus_metrics: tuple[str, ...]
    interpretation: str
    reward_warning: str


def _numbered(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index:03d}" for index in range(1, count + 1))


PROFILES = (
    DeckQualityProfile(
        "alakazam_relay",
        "Alakazam / Dudunsparce",
        _numbered("alakazam_dudunsparce", 3),
        (743, 741, 742, 66),
        ("key_setup_round", "attack_continuity", "post_ko_attack_gap", "minimum_deck_count"),
        "关注 Stage 2 启动、Powerful Hand 提交、Dudunsparce 过桥和主攻手被击倒后的接力。",
        "不能奖励无条件抽牌或尽快攻击；攻击会结束回合，低牌库继续过牌可能直接破坏终局。",
    ),
    DeckQualityProfile(
        "arboliva_meganium_engine",
        "Arboliva ex / Meganium / Teal Mask Ogerpon ex",
        ("arboliva_ex_meganium_001",),
        (404, 710, 96),
        ("key_setup_round", "max_attached_energy", "attack_continuity", "post_ko_attack_gap"),
        "关注双 Stage 2 引擎成形、草能量场面、主攻连续性和被击倒后的再启动。",
        "铺满 Bench 或堆积能量本身不是目标，必须结合攻击与 Prize 转化解释。",
    ),
    DeckQualityProfile(
        "archaludon_cinderace_tank",
        "Archaludon ex / Cinderace",
        ("archaludon_ex_cinderace_001",),
        (190, 666),
        ("key_setup_round", "attack_continuity", "post_ko_attack_gap", "opponent_attack_denial_rate"),
        "关注进化主攻成形、持续攻击、耐久回合与场面恢复。",
        "对局变长不等于打得更好；耐久必须最终转化为 Prize 或胜利。",
    ),
    DeckQualityProfile(
        "crustle_wall_control",
        "Crustle / Cornerstone Mask Ogerpon ex",
        ("crustle_cornerstone_mask_ogerpon_ex_001",),
        (345, 117),
        ("key_setup_round", "opponent_attack_denial_rate", "minimum_deck_count", "prizes_taken"),
        "关注墙体建立、对手有效攻击受阻、长局牌库安全和最终取胜路径。",
        "不能用伤害量或结束速度单独评价控制卡组，也不能奖励无目的拖延。",
    ),
    DeckQualityProfile(
        "cynthia_evolution_engine",
        "Cynthia's Garchomp ex / Roserade",
        _numbered("cynthias_garchomp_ex_roserade", 2),
        (381, 342),
        ("key_setup_round", "max_evolved_pokemon", "attack_continuity", "prizes_per_attack"),
        "关注双进化线成形、攻击链稳定性和每次攻击的 Prize 转化。",
        "进化数量只能作为准备度代理，不能为了完成进化而错过当回合攻击。",
    ),
    DeckQualityProfile(
        "dragapult_spread",
        "Dragapult ex",
        ("dragapult_ex_001",),
        (121, 119, 120, 112),
        ("key_setup_round", "damage_events", "multi_prize_turns", "attack_continuity"),
        "关注 Dragapult 成形、伤害铺设、跨回合 Prize 转化、多 Prize 回合和攻击接力。",
        "未立即拿 Prize 的攻击不一定差；铺伤只有在后续转化与胜率改善中才有意义。",
    ),
    DeckQualityProfile(
        "dragapult_blaziken",
        "Dragapult ex / Blaziken ex",
        ("dragapult_ex_blaziken_ex_001",),
        (121, 326),
        ("key_setup_round", "max_attached_energy", "attack_continuity", "post_ko_attack_gap"),
        "关注双 Stage 2 路线、能量供给、主副攻击手切换和 KO 后接力。",
        "不能奖励同时建立两条进化线；应观察它是否提高攻击连续性。",
    ),
    DeckQualityProfile(
        "dragapult_hammer_control",
        "Dragapult ex / Crushing Hammer",
        ("dragapult_ex_crushing_hammer_001",),
        (121, 1120),
        ("key_setup_round", "damage_events", "opponent_attack_denial_rate", "attack_continuity"),
        "同时观察 Dragapult 铺伤与 Crushing Hammer 后对手失去攻击的回合。",
        "使用 Hammer 不是成果；只有实际改变对手攻击能力时才是有效压制。",
    ),
    DeckQualityProfile(
        "dragapult_dunsparce_relay",
        "Dragapult ex / Dunsparce",
        ("dragapult_ex_dunsparce_001", "dragapult_ex_dunsparce_002"),
        (121, 65, 66, 305, 306),
        ("key_setup_round", "ability_actions", "attack_continuity", "minimum_deck_count"),
        "关注 Dunsparce/Dudunsparce 过牌换位是否帮助 Dragapult 成形且不造成牌库失控。",
        "不能直接奖励 Ability 次数或抽牌数；二者都可能造成无效循环与 deck-out 风险。",
    ),
    DeckQualityProfile(
        "dragapult_dusknoir_combo",
        "Dragapult ex / Dusknoir",
        _numbered("dragapult_ex_dusknoir", 2),
        (121, 131, 132, 133),
        ("key_setup_round", "self_field_discards", "damage_events", "multi_prize_turns"),
        "关注 Dusknoir 自我牺牲后的伤害与 Prize swing，以及 Dragapult 的后续多杀窗口。",
        "绝不能惩罚所有自我 KO；自爆必须按随后产生的局面价值和 Prize 转化评价。",
    ),
    DeckQualityProfile(
        "dragapult_froslass_spread",
        "Dragapult ex / Mega Froslass ex",
        ("dragapult_ex_mega_froslass_ex_001",),
        (121, 104, 861),
        ("key_setup_round", "damage_events", "multi_prize_turns", "post_ko_attack_gap"),
        "关注两条铺伤路线、伤害累计后的多杀转化和主攻手接力。",
        "单纯累计伤害会奖励无法兑现的铺伤，必须同时看 Prize 与终局。",
    ),
    DeckQualityProfile(
        "festival_single_prize_engine",
        "Festival Lead / Dipplin",
        _numbered("festival_lead_dipplin", 3),
        (93, 90),
        ("key_setup_round", "ability_actions", "attack_continuity", "post_ko_attack_gap"),
        "关注 Festival/Dipplin/Thwackey 引擎成形、单奖交换和连续派出攻击手。",
        "Bench 数量和 Ability 次数不是独立目标；应以攻击链与 Prize race 为约束。",
    ),
    DeckQualityProfile(
        "hydrapple_meganium_engine",
        "Hydrapple ex / Meganium / Teal Mask Ogerpon ex",
        ("hydrapple_ex_meganium_001",),
        (150, 710, 96),
        ("key_setup_round", "max_attached_energy", "attack_continuity", "prizes_per_attack"),
        "关注草能量引擎、双 Stage 2 建立速度和能量转化为有效攻击的效率。",
        "堆能量可能只是资源滞留，不能脱离攻击与胜负单独奖励。",
    ),
    DeckQualityProfile(
        "iono_bellibolt_engine",
        "Iono's Bellibolt ex / Kilowattrel",
        ("ionos_bellibolt_ex_kilowattrel_01",),
        (269, 271),
        ("key_setup_round", "ability_actions", "max_attached_energy", "attack_continuity"),
        "关注 Iono 系组件、Ability 链、雷能量准备与持续攻击。",
        "Ability 使用频率只能作为引擎活跃度，不能直接当作正奖励。",
    ),
    DeckQualityProfile(
        "marnie_froslass_spread",
        "Marnie's Grimmsnarl ex / Froslass",
        (*_numbered("marnies_grimmsnarl_ex_froslass", 6), "marnies_grimmsnarl_ex_froslass_limitless"),
        (648, 104),
        ("key_setup_round", "damage_events", "multi_prize_turns", "attack_continuity"),
        "关注 Grimmsnarl 进化、Froslass 辅助铺伤、伤害兑现和攻击连续性。",
        "不能只奖励伤害指示物数量；必须约束为后续 KO、Prize 或胜率提升。",
    ),
    DeckQualityProfile(
        "kangaskhan_crustle_control",
        "Mega Kangaskhan ex / Crustle",
        _numbered("mega_kangaskhan_ex_crustle", 8),
        (756, 345),
        ("key_setup_round", "opponent_attack_denial_rate", "post_ko_attack_gap", "minimum_deck_count"),
        "关注主攻与墙体切换、对手攻击受阻、耐久恢复和长局资源边界。",
        "拖长比赛、治疗或切换本身都不能作为奖励，必须服务于终局胜率。",
    ),
    DeckQualityProfile(
        "lopunny_pivot_tempo",
        "Mega Lopunny ex",
        ("mega_lopunny_ex_001",),
        (849, 848, 66, 174),
        ("first_attack_round", "attack_continuity", "prizes_per_attack", "post_ko_attack_gap"),
        "关注 Mega Lopunny 进化、Dudunsparce/Fan Rotom 引擎、换位后 Gale Thrust 的有效攻击与持续接力。",
        "频繁换位、治疗或抽牌本身都不是奖励；必须服务于 Gale Thrust、Prize 转化和最终胜率。",
    ),
    DeckQualityProfile(
        "lopunny_froslass_tempo",
        "Mega Lopunny ex / Mega Froslass ex",
        (
            "mega_lopunny_ex_mega_froslass_ex_001",
            "mega_lopunny_ex_mega_froslass_ex_002",
        ),
        (849, 861),
        ("key_setup_round", "damage_events", "attack_continuity", "post_ko_attack_gap"),
        "关注 Mega Lopunny 启动、Froslass 辅助伤害、换位与攻击链。",
        "频繁换位不是独立价值；只应解释攻击可达性与 Prize 转化。",
    ),
    DeckQualityProfile(
        "lucario_solrock_energy",
        "Mega Lucario ex / Solrock",
        _numbered("mega_lucario_ex_solrock", 2),
        (678, 675, 676),
        ("key_setup_round", "max_attached_energy", "first_attack_round", "prizes_per_attack"),
        "关注 Riolu 进化、Solrock/Lunatone 资源引擎、早期攻击和单次攻击 Prize 效率。",
        "能量加速和快速进化必须通过有效攻击验证，不能成为自循环奖励。",
    ),
    DeckQualityProfile(
        "lucario_hariyama_energy",
        "Mega Lucario ex / Hariyama",
        ("0024_lucario_hariyama_zero_shot",),
        (678, 674, 673, 676),
        (
            "key_setup_round",
            "max_evolved_pokemon",
            "first_attack_round",
            "prizes_per_attack",
        ),
        "关注 Mega Lucario ex 成形、Hariyama 进化换位、Solrock/Lunatone 资源引擎和攻击 Prize 转化。",
        "进化、换位和能量加速本身都不是成果，必须通过及时攻击、有效 KO 和最终胜率验证。",
    ),
    DeckQualityProfile(
        "ns_zoroark_toolbox",
        "N's Zoroark ex / N's Zekrom",
        ("ns_zoroark_ex_001",),
        (293, 906),
        ("key_setup_round", "ability_actions", "attack_continuity", "prizes_per_attack"),
        "关注 Zoroark 进化、N 系工具箱调用、攻击选择与 Prize 转化。",
        "不能以 N 系卡牌使用次数替代正确攻击路线。",
    ),
    DeckQualityProfile(
        "raging_bolt_energy_burst",
        "Raging Bolt ex / Mega Kangaskhan ex",
        ("raging_bolt_ex_james_cox_henry_chao_001", "raging_bolt_ex_mega_kangaskhan_ex_002", "raging_bolt_ex_mega_kangaskhan_ex_003"),
        (63, 96, 756),
        ("first_attack_round", "max_attached_energy", "prizes_per_attack", "post_ko_attack_gap"),
        "关注能量吞吐、早期爆发、一击拿奖效率和弃能后的重新攻击能力。",
        "弃能不是负行为，堆能也不是正行为；二者都必须按 KO 与后续恢复解释。",
    ),
    DeckQualityProfile(
        "raging_bolt_ogerpon_energy",
        "Raging Bolt ex / Teal Mask Ogerpon ex",
        ("raging_bolt_ogerpon_user_zero_shot",),
        (63, 96, 226),
        (
            "first_attack_round",
            "max_attached_energy",
            "attack_continuity",
            "prizes_per_attack",
        ),
        "关注 Teal Mask Ogerpon ex 能量引擎、Raging Bolt ex 早期爆发、弃能后攻击连续性和 Prize 转化。",
        "附能、抽牌与弃能都不是独立成果，必须通过及时攻击、有效 KO 和最终胜率验证。",
    ),
    DeckQualityProfile(
        "rmy_ogerpon_energy_engine",
        "Rmy Teal Mask Ogerpon ex",
        ("rmy_teal_mask_ogerpon_001",),
        (96, 18),
        (
            "first_attack_round",
            "max_attached_energy",
            "attack_continuity",
            "prizes_per_attack",
        ),
        "关注 Teal Mask Ogerpon ex 的能量引擎、早期攻击、连续输出和 Prize 转化。",
        "抽牌与堆积草能量不能独立获得正向解释，必须服务于攻击链和真实胜负。",
    ),
    DeckQualityProfile(
        "rocket_mewtwo_spidops_engine",
        "Team Rocket's Mewtwo ex / Spidops",
        _numbered("team_rockets_mewtwo_ex_spidops", 5),
        (431, 401),
        ("key_setup_round", "max_bench", "ability_actions", "opponent_attack_denial_rate"),
        "关注 Team Rocket 场面密度、Spidops 引擎、Mewtwo 攻击条件与控制效果。",
        "铺 Team Rocket 宝可梦和使用 Ability 只是条件，不应脱离有效攻击与胜负获得奖励。",
    ),
)


_BY_DECK = {
    deck_id: profile
    for profile in PROFILES
    for deck_id in profile.deck_ids
}


def profile_for_deck(deck_id: str) -> DeckQualityProfile:
    try:
        return _BY_DECK[deck_id]
    except KeyError as exc:
        raise ValueError(f"no League quality profile for deck: {deck_id}") from exc


def all_profiled_decks() -> tuple[str, ...]:
    return tuple(sorted(_BY_DECK))


__all__ = [
    "COMMON_METRICS",
    "DeckQualityProfile",
    "PROFILES",
    "all_profiled_decks",
    "profile_for_deck",
]
