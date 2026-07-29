from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

from .models import DeckShare, FilterAudit, Match, PairingAudit, Tournament


class _LimitlessHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.filters: dict[str, str] = {}
        self.tournaments: list[Tournament] = []
        self.deck_rows: list[list[tuple[str, dict[str, str]]]] = []
        self._filter: tuple[str, str] | None = None
        self._filter_text: list[str] = []
        self._tournament_attrs: dict[str, str] | None = None
        self._tournament_id: int | None = None
        self._deck_table_depth = 0
        self._row: list[tuple[str, dict[str, str]]] | None = None
        self._cell_text: list[str] = []
        self._cell_link: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "li" and values.get("data-key"):
            self._filter = (values["data-key"], values.get("data-value", ""))
            self._filter_text = []
        if tag == "tr" and values.get("data-players"):
            self._tournament_attrs = values
            self._tournament_id = None
        if tag == "table" and "data-table" in values.get("class", ""):
            self._deck_table_depth += 1
        if tag == "tr" and self._deck_table_depth:
            self._row = []
        if tag in {"td", "th"} and self._row is not None:
            self._cell_text = []
            self._cell_link = {}
        if tag == "a":
            href = values.get("href", "")
            match = re.fullmatch(r"/tournaments/(\d+)", href)
            if self._tournament_attrs is not None and match:
                self._tournament_id = int(match.group(1))
            if self._row is not None and href.startswith("/decks/"):
                self._cell_link = {"href": href}

    def handle_data(self, data: str) -> None:
        if self._filter is not None:
            self._filter_text.append(data)
        if self._row is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._filter is not None:
            key, value = self._filter
            self.filters[key] = value or " ".join(self._filter_text).strip()
            self._filter = None
        if tag == "tr" and self._tournament_attrs is not None:
            attrs = self._tournament_attrs
            if self._tournament_id is not None:
                self.tournaments.append(
                    Tournament(
                        tournament_id=self._tournament_id,
                        date=attrs["data-date"],
                        country=attrs["data-country"],
                        name=attrs["data-name"],
                        format=attrs["data-format"],
                        players=int(attrs["data-players"]),
                    )
                )
            self._tournament_attrs = None
        if tag in {"td", "th"} and self._row is not None:
            text = " ".join("".join(self._cell_text).split())
            self._row.append((text, self._cell_link))
        if tag == "tr" and self._row is not None:
            self.deck_rows.append(self._row)
            self._row = None
        if tag == "table" and self._deck_table_depth:
            self._deck_table_depth -= 1


def _parse_html(html: str) -> _LimitlessHTMLParser:
    parser = _LimitlessHTMLParser()
    parser.feed(html)
    return parser


def parse_filter_audit(html: str) -> FilterAudit:
    filters = _parse_html(html).filters
    if filters.get("format", "").lower() != "tef-por":
        raise ValueError("Limitless active filter must contain format=tef-por")
    return FilterAudit(active_filters=filters)


def validate_filter_query(query: str) -> dict[str, str]:
    expected = {
        "time": "all",
        "type": "all",
        "format": "TEF-POR",
        "region": "all",
        "division": "all",
    }
    parsed = {key: values[-1] for key, values in parse_qs(query).items()}
    for key, expected_value in expected.items():
        actual = parsed.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Limitless filter query requires {key}={expected_value}, got {actual!r}"
            )
    return {key: parsed[key] for key in expected}


def parse_labs_event_id(html: str) -> str | None:
    matches = set(
        re.findall(r'https://labs\.limitlesstcg\.com/(\d{4})/standings(?:["?#/]|$)', html)
    )
    if len(matches) > 1:
        raise ValueError(f"tournament page links multiple Labs events: {sorted(matches)}")
    return next(iter(matches), None)


def parse_tournament_index(html: str) -> list[Tournament]:
    return _parse_html(html).tournaments


def parse_deck_distribution(html: str) -> list[DeckShare]:
    result: list[DeckShare] = []
    for row in _parse_html(html).deck_rows:
        if len(row) < 5:
            continue
        href = next((meta["href"] for _, meta in row if "href" in meta), "")
        match = re.fullmatch(r"/decks/(\d+)(?:\?(.+))?", href)
        if not match or not row[0][0].isdigit():
            continue
        query = parse_qs(urlparse(href).query)
        variant = query.get("variant", [None])[0]
        result.append(
            DeckShare(
                rank=int(row[0][0]),
                deck_id=int(match.group(1)),
                variant_id=int(variant) if variant is not None else None,
                name=row[2][0],
                points=int(row[3][0].replace(",", "")),
                share=float(row[4][0].rstrip("%")) / 100,
                url=f"https://limitlesstcg.com{href}",
            )
        )
    return result


def parse_pairings_payload(
    payload: str, tournament_id: str, round_number: int
) -> tuple[list[Match], PairingAudit]:
    document = json.loads(payload)
    if not document.get("ok") or not isinstance(document.get("message"), list):
        raise ValueError("Limitless Labs pairings payload is not successful")
    matches: list[Match] = []
    seen_matches: set[tuple[int, int]] = set()
    byes = unresolved = incomplete = no_result = 0
    for row in document["message"]:
        if not row.get("player2"):
            byes += 1
            continue
        if not row.get("completed"):
            incomplete += 1
            continue
        if not row.get("p1_deck") or not row.get("p2_deck"):
            unresolved += 1
            continue
        raw_winner = int(row.get("winner") or 0)
        if raw_winner == -1:
            no_result += 1
            continue
        player1 = int(row["player1"])
        player2 = int(row["player2"])
        identity = tuple(sorted((player1, player2)))
        if identity in seen_matches:
            raise ValueError(
                f"duplicate pairing in tournament {tournament_id} round {round_number}: {identity}"
            )
        seen_matches.add(identity)
        winner = raw_winner
        if winner not in {0, player1, player2}:
            raise ValueError(
                f"winner {winner} is not a player in tournament {tournament_id} "
                f"round {round_number}: {(player1, player2)}"
            )
        matches.append(
            Match(
                tournament_id=tournament_id,
                round_number=round_number,
                table=row.get("table"),
                player1=player1,
                player2=player2,
                deck1=row["p1_deck"],
                deck1_name=row["p1_deck_name"],
                deck2=row["p2_deck"],
                deck2_name=row["p2_deck_name"],
                winner=winner,
            )
        )
    audit = PairingAudit(
        rows=len(document["message"]),
        accepted=len(matches),
        byes=byes,
        unresolved=unresolved,
        incomplete=incomplete,
        no_result=no_result,
    )
    return matches, audit
