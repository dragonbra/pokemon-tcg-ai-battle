from __future__ import annotations

import html
import re
from collections import Counter


def parse_kaggle_archetypes(html_text: str) -> Counter[str]:
    cards = re.findall(r"<article\b[^>]*\bdata-person-card\b[^>]*>", html_text)
    counts: Counter[str] = Counter()
    for card in cards:
        match = re.search(r'\bdata-archetype="([^"]+)"', card)
        if match:
            counts[html.unescape(match.group(1))] += 1
    return counts

