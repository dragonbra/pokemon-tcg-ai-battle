from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True)
class DiscussionRecord:
    competition: str
    topic_id: str
    title: str
    author: str
    votes: int
    url: str
    raw: dict[str, object]


def collect_discussion_index(
    competition: str,
    destination: Path,
    *,
    pages: int = 1,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[DiscussionRecord, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[DiscussionRecord] = []
    for page in range(1, pages + 1):
        command = ["kaggle", "competitions", "topics", "list", competition, "--page", str(page), "--format", "json"]
        completed = runner(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            continue
        payload = json.loads(completed.stdout or "[]")
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            topic_id = str(item.get("id", item.get("topicId", "")))
            if not topic_id:
                continue
            detail: dict[str, object] = {}
            show_command = [
                "kaggle",
                "competitions",
                "topics",
                "show",
                competition,
                topic_id,
                "--format",
                "json",
                "--page-size",
                "200",
            ]
            try:
                shown = runner(show_command, capture_output=True, text=True, check=False)
                if shown.returncode == 0:
                    parsed = json.loads(shown.stdout or "{}")
                    if isinstance(parsed, Mapping):
                        detail = {str(key): value for key, value in parsed.items()}
            except (OSError, ValueError, json.JSONDecodeError):
                detail = {}
            enriched = {str(key): value for key, value in item.items()}
            enriched["detail"] = detail
            author = str(
                detail.get("authorName")
                or detail.get("author")
                or item.get("authorName")
                or item.get("author")
                or ""
            )
            votes = detail.get("votes", detail.get("voteCount", item.get("voteCount", item.get("votes", 0))))
            record = DiscussionRecord(
                competition,
                topic_id,
                str(item.get("title", topic_id)),
                author,
                int(votes or 0),
                f"https://www.kaggle.com/competitions/{competition}/discussion/{topic_id}",
                enriched,
            )
            records.append(record)
    (destination / f"{competition}--topics.json").write_text(
        json.dumps([record.__dict__ for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = [
        f"# Kaggle Discussions: {competition}",
        "",
        f"- 快照时间：由本次收集命令写入",
        "",
        "| Topic | 标题 | 作者 | Votes | 评论/详情 |",
        "|---:|---|---|---:|---|",
    ]
    markdown.extend(
        f"| {record.topic_id} | [{record.title.replace('|', '/')}]({record.url}) | "
        f"{record.author.replace('|', '/')} | {record.votes} | `{json.dumps(record.raw.get('detail', {}), ensure_ascii=False)[:180]}` |"
        for record in records
    )
    categories: dict[str, list[DiscussionRecord]] = {key: [] for key in ("策略", "卡组", "Rating", "引擎", "提交经验", "其他")}
    for record in records:
        categories[_topic_category(record)].append(record)
    markdown.extend(("", "## 主题初分类", ""))
    for category, category_records in categories.items():
        markdown.append(f"### {category}（{len(category_records)}）")
        markdown.extend(f"- [{record.title}]({record.url})" for record in category_records[:20])
    (destination / f"{competition}--topics.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return tuple(records)


def _topic_category(record: DiscussionRecord) -> str:
    text = (record.title + " " + json.dumps(record.raw, ensure_ascii=False)).casefold()
    for keywords, category in (
        (("rating", "elo", "point", "score"), "Rating"),
        (("engine", "cg", "crash", "runtime", "rule"), "引擎"),
        (("deck", "alakazam", "lucario", "mewtwo", "archaludon", "crustle"), "卡组"),
        (("submit", "submission", "kaggle", "validation", "error"), "提交经验"),
        (("strategy", "agent", "win", "match", "play"), "策略"),
    ):
        if any(keyword in text for keyword in keywords):
            return category
    return "其他"
