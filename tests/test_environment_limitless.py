import json
import tempfile
import unittest
from pathlib import Path

from data.processed.environment_limitless import parsers as limitless_parsers
from data.processed.environment_limitless.parsers import (
    parse_deck_distribution,
    parse_filter_audit,
    parse_pairings_payload,
    parse_tournament_index,
)
from data.processed.environment_limitless.stats import aggregate_matchups, wilson_interval
from data.processed.environment_limitless.kaggle import parse_kaggle_archetypes
from data.processed.environment_limitless.labs import (
    parse_deck_meta_payload,
    parse_decklist_payload,
    parse_standings_payload,
)
from data.processed.environment_limitless.fetch import fetch_text
from data.processed.environment_limitless import render
from data.processed.environment_limitless.render import render_report
from data.processed.environment_limitless.stats import (
    evidence_grade,
    hhi,
    jensen_shannon_divergence,
)


FILTER_HTML = """
<ul class="selection multi-selection" data-selection>
  <li data-key="format" data-value="tef-por">Format: tef-por</li>
</ul>
<table class="completed-tournaments">
  <tr><th>Date</th><th>Country</th><th>Name</th><th></th><th>Players</th></tr>
  <tr data-date="2026-05-30" data-country="US" data-name="Regional Indianapolis"
      data-format="standard" data-players="1974">
    <td>30 May 26</td><td>US</td><td><a href="/tournaments/559">Indianapolis</a></td>
    <td>Standard</td><td>1974</td>
  </tr>
</table>
"""

DECK_HTML = """
<table class="data-table striped">
  <tr><th>#</th><th></th><th>Deck</th><th>Points</th><th>Share</th></tr>
  <tr><td>1</td><td></td><td><a href="/decks/284?variant=3">Dragapult Dusknoir</a></td>
      <td>1257</td><td>9.65%</td></tr>
</table>
"""


class LimitlessParserTests(unittest.TestCase):
    def test_filter_and_tournament_are_parsed_from_active_selection(self):
        audit = parse_filter_audit(FILTER_HTML)
        events = parse_tournament_index(FILTER_HTML)

        self.assertEqual(audit.active_filters, {"format": "tef-por"})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tournament_id, 559)
        self.assertEqual(events[0].players, 1974)

    def test_filter_parser_rejects_wrong_format(self):
        with self.assertRaisesRegex(ValueError, "tef-por"):
            parse_filter_audit(FILTER_HTML.replace("tef-por", "tef-cri"))

    def test_filter_query_requires_the_complete_requested_scope(self):
        validator = getattr(limitless_parsers, "validate_filter_query", None)
        self.assertIsNotNone(validator)
        self.assertEqual(
            validator(
                "time=all&type=all&format=TEF-POR&region=all&division=all"
            ),
            {
                "time": "all",
                "type": "all",
                "format": "TEF-POR",
                "region": "all",
                "division": "all",
            },
        )
        with self.assertRaisesRegex(ValueError, "division"):
            validator(
                "time=all&type=all&format=TEF-POR&region=all"
            )

    def test_labs_event_id_is_discovered_from_tournament_page(self):
        html = '<a href="https://labs.limitlesstcg.com/0068/standings">Standings</a>'
        parser = getattr(limitless_parsers, "parse_labs_event_id", None)
        self.assertIsNotNone(parser)
        self.assertEqual(parser(html), "0068")
        self.assertIsNone(parser('<a href="/decks">Decks</a>'))

    def test_deck_distribution_preserves_variant_identity(self):
        shares = parse_deck_distribution(DECK_HTML)

        self.assertEqual(len(shares), 1)
        self.assertEqual(shares[0].deck_id, 284)
        self.assertEqual(shares[0].variant_id, 3)
        self.assertEqual(shares[0].points, 1257)
        self.assertAlmostEqual(shares[0].share, 0.0965)


class MatchupTests(unittest.TestCase):
    def test_pairings_exclude_byes_and_keep_draws(self):
        payload = json.dumps(
            {
                "ok": True,
                "message": [
                    {
                        "table": 1,
                        "completed": 1,
                        "player1": 1,
                        "player2": 2,
                        "winner": 2,
                        "p1_deck": "dragapult-dusknoir",
                        "p1_deck_name": "Dragapult Dusknoir",
                        "p2_deck": "alakazam-dudunsparce",
                        "p2_deck_name": "Alakazam Dudunsparce",
                    },
                    {
                        "table": 2,
                        "completed": 1,
                        "player1": 3,
                        "player2": 4,
                        "winner": 0,
                        "p1_deck": "dragapult-dusknoir",
                        "p1_deck_name": "Dragapult Dusknoir",
                        "p2_deck": "alakazam-dudunsparce",
                        "p2_deck_name": "Alakazam Dudunsparce",
                    },
                    {
                        "table": 3,
                        "completed": 1,
                        "player1": 6,
                        "player2": 7,
                        "winner": -1,
                        "p1_deck": "dragapult-ex",
                        "p1_deck_name": "Dragapult",
                        "p2_deck": "festival-lead",
                        "p2_deck_name": "Festival Lead",
                    },
                    {
                        "table": None,
                        "completed": 1,
                        "player1": 5,
                        "player2": None,
                        "winner": 5,
                        "p1_deck": "dragapult-dusknoir",
                        "p1_deck_name": "Dragapult Dusknoir",
                        "p2_deck": None,
                        "p2_deck_name": None,
                    },
                ],
            }
        )

        matches, audit = parse_pairings_payload(payload, "0068", 1)
        self.assertEqual(len(matches), 2)
        self.assertEqual(audit.byes, 1)
        self.assertEqual(audit.unresolved, 0)
        self.assertEqual(getattr(audit, "no_result", None), 1)

        matrix = aggregate_matchups(matches)
        forward = matrix[("dragapult-dusknoir", "alakazam-dudunsparce")]
        reverse = matrix[("alakazam-dudunsparce", "dragapult-dusknoir")]
        self.assertEqual((forward.wins, forward.losses, forward.draws), (0, 1, 1))
        self.assertEqual((reverse.wins, reverse.losses, reverse.draws), (1, 0, 1))
        self.assertAlmostEqual(forward.effective_rate, 0.25)
        self.assertEqual(forward.n, reverse.n)

    def test_pairings_reject_invalid_winner_and_duplicate_match(self):
        row = {
            "table": 1, "completed": 1, "player1": 1, "player2": 2, "winner": 99,
            "p1_deck": "dragapult-ex", "p1_deck_name": "Dragapult",
            "p2_deck": "alakazam-dudunsparce", "p2_deck_name": "Alakazam",
        }
        with self.assertRaisesRegex(ValueError, "winner"):
            parse_pairings_payload(
                json.dumps({"ok": True, "message": [row]}), "0068", 1
            )
        row["winner"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_pairings_payload(
                json.dumps({"ok": True, "message": [row, dict(row)]}), "0068", 1
            )

    def test_wilson_interval_and_evidence_grade_are_conservative(self):
        low, high = wilson_interval(24.0, 30)
        self.assertGreater(low, 0.5)
        self.assertLess(high, 1.0)
        self.assertEqual(evidence_grade(24, 6, 0), "strong_advantage")
        self.assertEqual(evidence_grade(8, 7, 0), "directional")
        self.assertEqual(evidence_grade(6, 4, 0), "insufficient")

    def test_same_deck_match_is_counted_once(self):
        matches, _ = parse_pairings_payload(json.dumps({"ok": True, "message": [{
            "table": 1, "completed": 1, "player1": 1, "player2": 2, "winner": 1,
            "p1_deck": "dragapult-ex", "p1_deck_name": "Dragapult",
            "p2_deck": "dragapult-ex", "p2_deck_name": "Dragapult",
        }]}), "0068", 1)
        cell = aggregate_matchups(matches)[("dragapult-ex", "dragapult-ex")]
        self.assertEqual(cell.n, 1)
        self.assertEqual((cell.decisive, cell.draws), (1, 0))
        self.assertEqual((cell.wins, cell.losses), (0, 0))
        self.assertAlmostEqual(cell.effective_rate, 0.5)

    def test_same_deck_match_preserves_player2_win_and_draw(self):
        rows = [
            {
                "table": 1, "completed": 1, "player1": 1, "player2": 2, "winner": 2,
                "p1_deck": "dragapult-ex", "p1_deck_name": "Dragapult",
                "p2_deck": "dragapult-ex", "p2_deck_name": "Dragapult",
            },
            {
                "table": 2, "completed": 1, "player1": 3, "player2": 4, "winner": 0,
                "p1_deck": "dragapult-ex", "p1_deck_name": "Dragapult",
                "p2_deck": "dragapult-ex", "p2_deck_name": "Dragapult",
            },
        ]
        matches, _ = parse_pairings_payload(
            json.dumps({"ok": True, "message": rows}), "0068", 1
        )
        cell = aggregate_matchups(matches)[("dragapult-ex", "dragapult-ex")]
        self.assertEqual((cell.decisive, cell.draws), (1, 1))
        self.assertEqual((cell.wins, cell.losses), (0, 0))


class EnvironmentComparisonTests(unittest.TestCase):
    def test_kaggle_parser_reads_each_person_card_once(self):
        html_text = """
        <article class="person-card" data-person-card data-archetype="Dragapult ex"></article>
        <article data-person-card class="person-card"
                 data-archetype="Marnie&#x27;s Grimmsnarl ex / Froslass"></article>
        """
        counts = parse_kaggle_archetypes(html_text)
        self.assertEqual(counts["Dragapult ex"], 1)
        self.assertEqual(counts["Marnie's Grimmsnarl ex / Froslass"], 1)

    def test_distribution_metrics_have_known_limits(self):
        self.assertAlmostEqual(hhi([0.5, 0.5]), 0.5)
        self.assertAlmostEqual(jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)
        self.assertAlmostEqual(jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]), 1.0)


class LabsPayloadTests(unittest.TestCase):
    def test_deck_meta_keeps_variant_and_primary_archetype(self):
        payload = json.dumps({"ok": True, "message": [{
            "identifier": "dragapult-dusknoir", "name": "Dragapult Dusknoir",
            "sup_identifier": "dragapult-ex", "sup_name": "Dragapult", "icons": "dragapult dusknoir",
            "players": 12, "day2s": 3, "wins": 40, "losses": 30, "ties": 5,
        }]})
        item = parse_deck_meta_payload(payload)[0]
        self.assertEqual(item["deck_id"], "dragapult-dusknoir")
        self.assertEqual(item["primary_id"], "dragapult-ex")
        self.assertEqual(item["matches"], 75)

    def test_standings_and_decklist_are_auditable(self):
        standings = parse_standings_payload(json.dumps({"ok": True, "message": [{
            "player_id": 2007, "tp_id": 1808, "name": "Justin Newdorf", "country": "US",
            "placement": 3, "points": 38, "wins": 12, "losses": 2, "ties": 2,
            "day2": 1, "topcut": 1, "decklist": 1,
            "deck_id": "dragapult-dusknoir", "deck_name": "Dragapult Dusknoir",
            "icons": "dragapult dusknoir",
        }]}))[0]
        self.assertEqual(standings["labs_player_id"], 1808)
        cards = parse_decklist_payload(json.dumps({"ok": True, "message": {
            "pokemon": [{"count": 4, "name": "Dreepy", "set": "ASC", "number": "158"}],
            "trainer": [{"count": 48, "name": "Trainer", "set": "TST", "number": "1"}],
            "energy": [{"count": 8, "name": "Energy", "set": "MEE", "number": "5"}],
        }}))
        self.assertEqual(sum(card["count"] for card in cards), 60)


class FetchTests(unittest.TestCase):
    def test_fetch_records_hash_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            cache = root / "cache"
            source.write_text('{"ok":true}', encoding="utf-8")
            first = fetch_text(source.as_uri(), cache)
            source.write_text('{"ok":false}', encoding="utf-8")
            second = fetch_text(source.as_uri(), cache)

        self.assertEqual(first.text, '{"ok":true}')
        self.assertEqual(second.text, first.text)
        self.assertEqual(second.sha256, first.sha256)
        self.assertTrue(second.from_cache)


class ReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot_path = (
            Path(__file__).resolve().parents[1]
            / "data/processed/environment_limitless/snapshot.json"
        )
        cls.snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    def test_card_registry_and_archetype_visuals_are_complete(self):
        registry = render._build_card_registry(self.snapshot["representative_decklists"])

        self.assertEqual(registry["dragapult ex"]["set"], "ASC")
        visual = render._archetype_visual("Dragapult Dusknoir", registry)
        self.assertIn("Dragapult ex", visual)
        self.assertIn("Dusknoir", visual)
        self.assertIn("data-card-preview=", visual)

    def test_report_adds_card_visuals_to_analysis_surfaces(self):
        report = render_report(self.snapshot)

        self.assertIn('id="primary-heatmap"', report)
        self.assertGreaterEqual(report.count('class="archetype-visual'), 100)
        self.assertGreaterEqual(report.count('class="inline-card-ref'), 8)
        self.assertGreaterEqual(report.count('data-card-context="narrative"'), 4)
        self.assertIn('data-card-preview=', report)
        self.assertIn('class="matrix-archetype"', report)

    def test_every_exact_deck_card_type_has_an_image_tile(self):
        report = render_report(self.snapshot)
        expected = sum(
            len(deck["cards"])
            for deck in self.snapshot["representative_decklists"]
        )

        self.assertEqual(report.count('class="deck-card-tile"'), expected)
        self.assertIn('data-card-group="pokemon"', report)
        self.assertIn('data-card-group="trainer"', report)
        self.assertIn('data-card-group="energy"', report)

    def test_frozen_snapshot_renders_complete_analysis_contract(self):
        report = render_report(self.snapshot)
        checked_in_report = (
            Path(__file__).resolve().parents[1] / "docs/environment/limitless.html"
        ).read_text(encoding="utf-8")

        for section_id in (
            "summary", "filter-audit", "metagame", "kaggle-gap", "dragapult-core",
            "variants", "players", "matchup-matrix", "dragapult-matchups",
            "counter-evidence", "causes", "training", "methodology",
        ):
            self.assertIn(f'id="{section_id}"', report)
        self.assertEqual(report.count('data-event-row'), 8)
        self.assertIn("41,663", report)
        self.assertIn("10,923", report)
        self.assertIn("Dragapult Dusknoir", report)
        self.assertIn("95% Wilson", report)
        self.assertIn('<details class="deck-detail" open>', report)
        self.assertNotIn('data-n="0">0.0%', report)
        self.assertIn('id="mapping-audit"', report)
        self.assertIn("站点标签，不是逐份卡表重分类", report)
        self.assertIn("data-sortable", report)
        self.assertIn('aria-sort="none"', report)
        self.assertIn('tabindex="0" aria-label=', report)
        self.assertIn("overflow-x:clip", report)
        self.assertIn(".heat[data-n]:not(.self-matchup)", report)
        self.assertEqual(checked_in_report, report)


class FrozenSnapshotAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads((
            Path(__file__).resolve().parents[1]
            / "data/processed/environment_limitless/snapshot.json"
        ).read_text(encoding="utf-8"))

    def test_filter_points_coverage_and_exact_decks(self):
        data = self.snapshot
        self.assertEqual(data["filter_contract"]["active_filters"], {"format": "tef-por"})
        self.assertEqual(
            data["filter_contract"].get("requested_filters"),
            {
                "time": "all",
                "type": "all",
                "format": "TEF-POR",
                "region": "all",
                "division": "all",
            },
        )
        self.assertEqual((data["filter_contract"]["events"], data["filter_contract"]["players"]), (8, 10923))
        self.assertEqual(
            sum(row["points"] for row in data["primary_points_share"]),
            sum(row["points"] for row in data["variant_points_share"]),
        )
        self.assertEqual(
            sum(
                row["accepted_matches"]
                for row in data["coverage"]
                if row["pairings_available"]
            ),
            41663,
        )
        self.assertEqual(data["pairing_summary"]["no_result_rows"], 78)
        self.assertTrue(data["representative_decklists"])
        self.assertTrue(all(row["total_cards"] == 60 for row in data["representative_decklists"]))
        self.assertTrue(
            all("classification_audit" in row for row in data["representative_decklists"])
        )
        self.assertEqual(
            sum(row.get("labs_link_discovered", False) for row in data["coverage"]), 7
        )
        self.assertTrue(
            all(row.get("event_format_verified", False) for row in data["coverage"])
        )
        self.assertTrue(data["comparison"].get("mapping_audit"))
        self.assertTrue(data["comparison"].get("mapping_rules"))

    def test_matchup_matrices_are_mirrored(self):
        for matrix_name in ("matchups_primary", "matchups_variant"):
            matrix = {
                (row["deck_id"], row["opponent_id"]): row
                for row in self.snapshot[matrix_name]
            }
            for (deck, opponent), row in matrix.items():
                self.assertGreater(row["n"], 0)
                if deck == opponent:
                    self.assertTrue(row["self_matchup"])
                    self.assertEqual(row["n"], row["decisive"] + row["ties"])
                    self.assertIsNone(row["effective_win_rate"])
                    self.assertEqual(row["evidence"], "self_matchup")
                    self.assertIsNone(row["wilson_low"])
                    self.assertIsNone(row["wilson_high"])
                    continue
                reverse = matrix[(opponent, deck)]
                self.assertEqual((row["wins"], row["losses"], row["ties"]),
                                 (reverse["losses"], reverse["wins"], reverse["ties"]))
                self.assertEqual(row["n"], reverse["n"])
                self.assertAlmostEqual(row["effective_win_rate"] + reverse["effective_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
