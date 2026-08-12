"""Render the Frozen-002 stability and Nighttime Mine controlled analysis."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

from engine_cuda_2_0.tools import evaluate_policy_0806_cuda as base
from engine_cuda_2_0.tools import evaluate_policy_0806_seed_replication as replication
from engine_cuda_2_0.tools import evaluate_sp_series_cuda as sp


ANALYSIS_ROOT = ROOT / "docs/reports/sp-series/seeded2048_cuda_v2"
FROZEN_CANONICAL_MANIFEST = (
    ROOT
    / "docs/evaluation/combat_mat/policy_0806"
    / "0806_kaggle_top100_plus_v1_cuda_seeded_2048_resident_v2/manifest.json"
)
SP_MANIFEST = ANALYSIS_ROOT / "manifest.json"
PAIRED_RESULT = replication.TEMP_ROOT / "sp04_paired_result.json"


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
        / denominator
    )
    return center - radius, center + radius


def paired_difference_interval(
    jobs: list[dict[str, Any]],
    left_results: list[int],
    right_results: list[int],
    opponent_ids: set[str] | None = None,
) -> dict[str, Any]:
    differences = []
    left_wins = right_wins = left_draws = right_draws = 0
    for job, left, right in zip(jobs, left_results, right_results, strict=True):
        if opponent_ids is not None and job["opponent_id"] not in opponent_ids:
            continue
        focal_player = 0 if job["focal_first"] else 1
        left_win = int(left == focal_player + 1)
        right_win = int(right == focal_player + 1)
        differences.append(left_win - right_win)
        left_wins += left_win
        right_wins += right_win
        left_draws += int(left == 0)
        right_draws += int(right == 0)
    games = len(differences)
    mean = sum(differences) / games
    variance = sum((value - mean) ** 2 for value in differences) / (games - 1)
    radius = 1.96 * math.sqrt(variance / games)
    return {
        "games": games,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "left_draws": left_draws,
        "right_draws": right_draws,
        "left_rate": left_wins / games,
        "right_rate": right_wins / games,
        "difference": mean,
        "difference_low": mean - radius,
        "difference_high": mean + radius,
    }


def _tera_opponents(catalog: Any, candidates: tuple[Any, ...]) -> set[str]:
    tera_card_ids: set[int] = set()
    with (ROOT / "data/official/EN_Card_Data.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["Category"].startswith("Tera("):
                tera_card_ids.add(int(row["Card ID"]))
    by_id = {candidate.name: candidate for candidate in candidates}
    return {
        entry.deck_id
        for entry in catalog.pool.schedule
        if set(by_id[entry.deck_id].deck) & tera_card_ids
    }


def _record_row(label: str, record: dict[str, Any], note: str) -> dict[str, Any]:
    low, high = wilson(record["wins"], record["games"])
    return {
        "label": label,
        "wins": record["wins"],
        "losses": record["losses"],
        "draws": record["draws"],
        "games": record["games"],
        "rate": record["win_rate"],
        "low": low,
        "high": high,
        "note": note,
    }


def main() -> int:
    catalog, frozen_candidates = base._catalog()
    frozen_002 = next(
        candidate
        for candidate in frozen_candidates
        if candidate.package_manifest["frozen_deck_number"] == "002"
    )
    sp_candidates = sp.load_sp_candidates(catalog)
    sp04 = next(candidate for candidate in sp_candidates if candidate.name == "SP04_MAGA")
    schedule = json.loads((replication.TEMP_ROOT / "schedule.json").read_text())
    result_002 = json.loads((replication.TEMP_ROOT / "result.json").read_text())
    result_sp04 = json.loads(PAIRED_RESULT.read_text())
    if not replication._strict_result(result_sp04, replication.TEMP_ROOT / "schedule.json"):
        raise RuntimeError("paired SP04 result failed the strict contract")
    summary_002 = replication._summarize(
        catalog, frozen_002, schedule, result_002
    )
    summary_sp04 = replication._summarize(catalog, sp04, schedule, result_sp04)
    summary_sp04["deck_number"] = "SP04"
    tera_ids = _tera_opponents(catalog, frozen_candidates)
    nontera_ids = set(summary_002["by_opponent"]) - tera_ids
    paired_all = paired_difference_interval(
        schedule["jobs"],
        result_002["determinism"]["game_results"],
        result_sp04["determinism"]["game_results"],
    )
    paired_tera = paired_difference_interval(
        schedule["jobs"],
        result_002["determinism"]["game_results"],
        result_sp04["determinism"]["game_results"],
        tera_ids,
    )
    paired_nontera = paired_difference_interval(
        schedule["jobs"],
        result_002["determinism"]["game_results"],
        result_sp04["determinism"]["game_results"],
        nontera_ids,
    )

    canonical_manifest = json.loads(FROZEN_CANONICAL_MANIFEST.read_text())
    canonical_002 = next(
        record
        for record in canonical_manifest["reports"]
        if record["deck_number"] == "002"
    )
    sp_manifest = json.loads(SP_MANIFEST.read_text())
    table_records = [
        _record_row("Frozen 002 · canonical", canonical_002, "canonical seed；旧结果未记录 50 回合护栏"),
        _record_row("Frozen 002 · 独立复测", summary_002, "新随机 seed 843573811；当前统一护栏"),
    ]
    table_records.extend(
        _record_row(record["deck_id"], record, "独立 Seeded-2048")
        for record in sp_manifest["reports"]
    )
    table_records.append(
        _record_row("SP04 · 同-seed 受控对照", summary_sp04, "与 002 共用逐局 engine/search seed")
    )

    by_name = {candidate.name: candidate for candidate in frozen_candidates}
    matchup_rows = []
    for opponent_id, left in summary_002["by_opponent"].items():
        right = summary_sp04["by_opponent"][opponent_id]
        opponent = by_name[opponent_id]
        matchup_rows.append(
            {
                "number": opponent.package_manifest["frozen_deck_number"],
                "name": opponent.display_name,
                "tera": opponent_id in tera_ids,
                "games": left["games"],
                "002": left,
                "sp04": right,
                "delta_wins": left["wins"] - right["wins"],
            }
        )
    matchup_rows.sort(key=lambda row: (-abs(row["delta_wins"]), row["number"]))

    paired_report = replication.OUTPUT_ROOT / "reports/SP04_paired_control.html"
    base._atomic_text(
        paired_report,
        base._report_html(
            catalog,
            sp04,
            summary_sp04,
            report_label="SP04 · paired control for Frozen-002",
            back_href="../index.html",
            evaluation_seed=replication.REPLICATION_SEED,
        ),
    )
    replication_manifest_path = replication.OUTPUT_ROOT / "manifest.json"
    replication_manifest = json.loads(replication_manifest_path.read_text())
    replication_manifest["paired_control"] = {
        "deck_id": "SP04_MAGA",
        "only_deck_difference": "Frozen-002 Nighttime Mine x2; SP04 Battle Cage x2",
        "same_schedule_sha256": summary_002["schedule_sha256"],
        "report": "reports/SP04_paired_control.html",
        "summary": summary_sp04,
        "paired_overall": paired_all,
        "paired_tera": paired_tera,
        "paired_nontera": paired_nontera,
    }
    base._atomic_json(replication_manifest_path, replication_manifest)

    def pct(value: float) -> str:
        return f"{value:.2%}"

    markdown_rows = "\n".join(
        f"| {row['label']} | {row['wins']}-{row['losses']}-{row['draws']} | "
        f"{pct(row['rate'])} | {pct(row['low'])}–{pct(row['high'])} | {row['note']} |"
        for row in table_records
    )
    matchup_markdown = "\n".join(
        f"| {row['number']} | {row['name']} | {'是' if row['tera'] else '否'} | "
        f"{row['games']} | {row['002']['wins']}-{row['002']['losses']}-{row['002']['draws']} | "
        f"{row['sp04']['wins']}-{row['sp04']['losses']}-{row['sp04']['draws']} | "
        f"{row['delta_wins']:+d} |"
        for row in matchup_rows
    )
    markdown = f"""# Frozen 002 胡地强度复测与归因

## 结论

Frozen 002 的接近 60% 胜率可以复现：canonical 为 {pct(canonical_002['win_rate'])}，独立新 seed 为 {pct(summary_002['win_rate'])}，相差 {pct(abs(canonical_002['win_rate'] - summary_002['win_rate']))}。这支持它稳定强于多数 SP 构筑，而不是单个 epoch/seed 偶然峰值。

最干净的构筑对照是 SP04：两者只有 `Nighttime Mine ×2` 与 `Battle Cage ×2` 的替换。在相同的 2,048 个 engine/search seed、对手与先后手位置下，002 为 {pct(paired_all['left_rate'])}，SP04 为 {pct(paired_all['right_rate'])}，配对净差 {pct(paired_all['difference'])}，95% 区间 {pct(paired_all['difference_low'])}–{pct(paired_all['difference_high'])}。

差异主要来自含 Tera 的对局。Frozen 池 55 套中有 {len(tera_ids)} 套含 Tera，按固定频率占 {paired_tera['games']}/2048（{pct(paired_tera['games']/2048)}）。该子集 002 比 SP04 高 {pct(paired_tera['difference'])}（95% 区间 {pct(paired_tera['difference_low'])}–{pct(paired_tera['difference_high'])}）；非 Tera 子集只高 {pct(paired_nontera['difference'])}（区间 {pct(paired_nontera['difference_low'])}–{pct(paired_nontera['difference_high'])}，跨 0）。因此数据与 Nighttime Mine “Tera 攻击多 1 个无色能量”的机制一致。

这仍然不是对卡牌普适强度的证明。它证明的是：在固定 Policy-0806、Frozen-0806 频率池和当前 official-state CUDA runtime 下，002 的完整 deck-policy system 优于仅将两张 Stadium 换成 Battle Cage 的 SP04。报告没有逐动作 Stadium 上场率，且换牌会改变洗牌和后续策略轨迹，不能声称每一局净胜都由 Stadium 的即时效果直接造成。

## 总表

| 构筑 / 批次 | W-L-D | 胜率 | Wilson 95% CI | 合同备注 |
|---|---:|---:|---:|---|
{markdown_rows}

## 受控分层

| 分层 | 对局 | 002 胜率 | SP04 胜率 | 配对净差 | 95% CI |
|---|---:|---:|---:|---:|---:|
| 全部 | {paired_all['games']} | {pct(paired_all['left_rate'])} | {pct(paired_all['right_rate'])} | {pct(paired_all['difference'])} | {pct(paired_all['difference_low'])}–{pct(paired_all['difference_high'])} |
| 含 Tera 构筑 | {paired_tera['games']} | {pct(paired_tera['left_rate'])} | {pct(paired_tera['right_rate'])} | {pct(paired_tera['difference'])} | {pct(paired_tera['difference_low'])}–{pct(paired_tera['difference_high'])} |
| 不含 Tera 构筑 | {paired_nontera['games']} | {pct(paired_nontera['left_rate'])} | {pct(paired_nontera['right_rate'])} | {pct(paired_nontera['difference'])} | {pct(paired_nontera['difference_low'])}–{pct(paired_nontera['difference_high'])} |

## 为什么 002 看起来特别高

1. **Nighttime Mine 的目标并不少。** 用户直觉中的“环境 Tera 不多”若按构筑种类看是 20/55；按固定评测频率仍有 496/2048 场。约四分之一的权重足以让一个明显的 Tera matchup 增益贡献约 1.5 个总体百分点。
2. **SP04 证明 Stadium 选择贡献约 2–3 点，而非十几点。** 同-seed 对照的 +2.39 点刚刚达到 95% 配对区间下界大于 0；Tera 组贡献了 31 个净胜，约占总净增 49 胜的 63%。
3. **其他 SP 不是单变量对照。** 它们还改变 Alakazam、Dudunsparce、Shaymin、Boss、Xerosic、Enhanced Hammer 或 Wondrous Patch。SP09 相比 SP04 又少 1 张 Alakazam、多 1 张 Battle Cage；其 {pct(next(row['rate'] for row in table_records if row['label']=='SP09_MAGA'))} 不能归因于 Nighttime Mine 缺失。
4. **这是 deck-policy compatibility。** Policy-0806 是固定策略；卡位会改变起手、检索、动作候选和策略是否落在熟悉分布。某张卡的评测收益可能来自模型更会使用它，而不等同于人类最优策略下的卡牌理论强度。
5. **Battle Cage 并非全面劣势。** 它能阻止对手用攻击或 Ability 向 Bench 放置伤害指示物，理论上会对 Dragapult/Dusknoir 类压力有价值。实际逐对手差异并不单调；这正说明两张 Stadium 是 matchup trade-off，而不是一张严格支配另一张。

## 逐对手同-seed 差分

表按净胜变化绝对值排序。小样本行只用于定位贡献，不单独作显著性结论。

| 编号 | 对手 | 含 Tera | 对局 | 002 W-L-D | SP04 W-L-D | 002 净胜变化 |
|---:|---|:---:|---:|---:|---:|---:|
{matchup_markdown}

## 证据边界

- 通用规则与卡牌级事实：Nighttime Mine 和 Battle Cage 的具体文本来自当前 `data/official/EN_Card_Data.csv`；游戏胜负与回合语义以当前 runtime 为准。
- 实验事实：所有新结果使用 strict FP32、256 resident CUDA lanes、50 完整回合平局、同一方同 Ability 第 20 次判负。
- 策略假设：将“含 Tera 卡的构筑”作为 Nighttime Mine 有潜在直接作用的分层；报告没有证明该 Tera 每局都实际进场或攻击。
- CUDA/official CPU parity 尚未形成完整强度替代合同，因此结论限定于当前 CUDA evaluation runtime。
"""
    base._atomic_text(ANALYSIS_ROOT / "002_strength_analysis.md", markdown)

    html_rows = "".join(
        f"<tr><td>{html.escape(row['label'])}</td><td>{row['wins']}-{row['losses']}-{row['draws']}</td>"
        f"<td><b>{pct(row['rate'])}</b></td><td>{pct(row['low'])}–{pct(row['high'])}</td>"
        f"<td>{html.escape(row['note'])}</td></tr>"
        for row in table_records
    )
    matchup_html = "".join(
        f"<tr><td>{row['number']}</td><td>{html.escape(str(row['name']))}</td>"
        f"<td>{'是' if row['tera'] else '否'}</td><td>{row['games']}</td>"
        f"<td>{row['002']['wins']}-{row['002']['losses']}-{row['002']['draws']}</td>"
        f"<td>{row['sp04']['wins']}-{row['sp04']['losses']}-{row['sp04']['draws']}</td>"
        f"<td class={'up' if row['delta_wins'] > 0 else 'down' if row['delta_wins'] < 0 else ''}>"
        f"{row['delta_wins']:+d}</td></tr>"
        for row in matchup_rows
    )
    analysis_html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen 002 胡地强度复测与归因</title><style>
:root{{--bg:#f2f6f4;--paper:#fff;--ink:#172b25;--muted:#60736c;--line:#dbe6e1;--green:#176b4d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.65 system-ui,"PingFang SC",sans-serif}}header{{padding:36px max(22px,calc((100vw - 1160px)/2));background:#173d30;color:white}}h1{{margin:0;font-size:31px}}header p{{color:#cfe5dc}}main{{max-width:1160px;margin:auto;padding:22px}}section{{margin:18px 0;padding:22px;border:1px solid var(--line);border-radius:9px;background:var(--paper)}}.lead{{border-left:5px solid var(--green)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.metric{{padding:14px;border:1px solid var(--line);border-radius:8px}}.metric b{{display:block;font-size:22px;color:var(--green)}}.metric span{{color:var(--muted)}}.table{{overflow:auto}}table{{width:100%;min-width:850px;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#edf4f0}}.up{{color:#087443;font-weight:800}}.down{{color:#a33b3b;font-weight:800}}a{{color:var(--green);font-weight:750}}li{{margin:8px 0}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}}}</style></head><body><header><h1>Frozen 002 胡地强度复测与归因</h1><p>独立 Seeded-2048 复现 + 同-seed SP04 Stadium 受控对照</p></header><main><section class="lead"><h2>结论</h2><p><b>002 接近 60% 的强度可以复现。</b>canonical 为 {pct(canonical_002['win_rate'])}，独立新 seed 为 {pct(summary_002['win_rate'])}，仅差 {pct(abs(canonical_002['win_rate']-summary_002['win_rate']))}。</p><p>与仅把两张 Nighttime Mine 换成 Battle Cage 的 SP04 做同-seed 对照后，002 高 <b>{pct(paired_all['difference'])}</b>（配对 95% CI {pct(paired_all['difference_low'])}–{pct(paired_all['difference_high'])}）。收益主要集中在 Tera 子集：+{pct(paired_tera['difference'])}；非 Tera 子集的 +{pct(paired_nontera['difference'])} 尚不能排除噪音。</p><p>因此，Nighttime Mine 能解释约 2–3 个百分点，不能解释所有 SP 尤其 SP09 的更大落差。</p></section><section><div class="metrics"><div class="metric"><b>{pct(summary_002['win_rate'])}</b><span>002 独立复测</span></div><div class="metric"><b>+{pct(paired_all['difference'])}</b><span>对 SP04 同-seed</span></div><div class="metric"><b>+{pct(paired_tera['difference'])}</b><span>Tera 子集净差</span></div><div class="metric"><b>{paired_tera['games']}/2048</b><span>Tera 权重</span></div></div></section><section><h2>完整结果表</h2><div class="table"><table><thead><tr><th>构筑 / 批次</th><th>W-L-D</th><th>胜率</th><th>Wilson 95% CI</th><th>合同备注</th></tr></thead><tbody>{html_rows}</tbody></table></div></section><section><h2>为什么 002 更高</h2><ol><li>Frozen 池有 20/55 套含 Tera，按频率是 496/2048 场，不是可以忽略的尾部。</li><li>同-seed 纯 Stadium 对照显示总体 +2.39 点、Tera 子集 +6.25 点，方向与 Nighttime Mine 卡文一致；Tera 子集贡献 31 个净胜，占总净增 49 胜的约 63%。</li><li>其他 SP 同时改了多张卡；SP09 相比 SP04 还少 1 张 Alakazam、多 1 张 Battle Cage，其低胜率不能归咎于缺少 Nighttime Mine。</li><li>固定 Policy-0806 的表现是 deck-policy compatibility：换牌会改变起手、检索、候选动作和模型分布，不等于人类最优策略下的卡牌普适排名。</li><li>Battle Cage 对 Dragapult/Dusknoir 类撒伤也有明确价值，所以逐 matchup 会出现正负混合；Nighttime Mine 并非严格支配。</li></ol></section><section><h2>受控分层</h2><div class="table"><table><thead><tr><th>分层</th><th>对局</th><th>002</th><th>SP04</th><th>配对净差</th><th>95% CI</th></tr></thead><tbody><tr><td>全部</td><td>{paired_all['games']}</td><td>{pct(paired_all['left_rate'])}</td><td>{pct(paired_all['right_rate'])}</td><td>+{pct(paired_all['difference'])}</td><td>{pct(paired_all['difference_low'])}–{pct(paired_all['difference_high'])}</td></tr><tr><td>含 Tera 构筑</td><td>{paired_tera['games']}</td><td>{pct(paired_tera['left_rate'])}</td><td>{pct(paired_tera['right_rate'])}</td><td>+{pct(paired_tera['difference'])}</td><td>{pct(paired_tera['difference_low'])}–{pct(paired_tera['difference_high'])}</td></tr><tr><td>不含 Tera</td><td>{paired_nontera['games']}</td><td>{pct(paired_nontera['left_rate'])}</td><td>{pct(paired_nontera['right_rate'])}</td><td>+{pct(paired_nontera['difference'])}</td><td>{pct(paired_nontera['difference_low'])}–{pct(paired_nontera['difference_high'])}</td></tr></tbody></table></div></section><section><h2>逐对手同-seed 差分</h2><p>按净胜变化绝对值排序；小样本行只用于定位，不单独作显著性结论。</p><div class="table"><table><thead><tr><th>编号</th><th>对手</th><th>Tera</th><th>对局</th><th>002 W-L-D</th><th>SP04 W-L-D</th><th>净胜</th></tr></thead><tbody>{matchup_html}</tbody></table></div></section><section><h2>原始报告与证据边界</h2><ul><li><a href="../../../../evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_replication_002/reports/002_alakazam_dudunsparce.html">002 独立复测完整报告</a></li><li><a href="../../../../evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_replication_002/reports/SP04_paired_control.html">SP04 同-seed 完整报告</a></li><li><a href="../../../../evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_resident_v2/reports/002_alakazam_dudunsparce.html">002 canonical 报告</a></li></ul><p>卡文来自当前官方卡牌数据；“含 Tera 构筑”不保证 Tera 每局实际进场。新结果均为 strict FP32、50 完整回合平局和 repeat-forfeit 20。CUDA/official CPU parity 未完成前，结论限定于当前 CUDA evaluation runtime。</p></section></main></body></html>'''
    base._atomic_text(ANALYSIS_ROOT / "002_strength_analysis.html", analysis_html)
    print(ANALYSIS_ROOT / "002_strength_analysis.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
