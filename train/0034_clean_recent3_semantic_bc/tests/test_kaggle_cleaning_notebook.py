from __future__ import annotations

import ast
from collections import Counter
import gzip
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = (
    ROOT
    / "experiments"
    / "0034_clean_recent3_semantic_bc"
    / "kaggle"
    / "01_clean_recent3_data"
    / "01_clean_recent3_data.ipynb"
)


class KaggleCleaningNotebookTests(unittest.TestCase):
    def test_streaming_audit_writer_has_runtime_io_dependency(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        source = next(cell for cell in code_cells if "def _append_gzip_rows" in cell)
        tree = ast.parse(source)

        import_io_index = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.Import)
            and any(alias.name == "io" for alias in node.names)
        )
        function_index = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.FunctionDef) and node.name == "_append_gzip_rows"
        )
        self.assertLess(import_io_index, function_index)

        function_node = tree.body[function_index]
        function_module = ast.fix_missing_locations(
            ast.Module(body=[tree.body[import_io_index], function_node], type_ignores=[])
        )
        namespace = {"Path": Path, "gzip": gzip, "json": json}
        exec(compile(function_module, str(NOTEBOOK), "exec"), namespace)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "audit.jsonl.gz"
            namespace["_append_gzip_rows"](output, [{"row": 1}])
            namespace["_append_gzip_rows"](output, [{"row": 2}])
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
        self.assertEqual(rows, [{"row": 1}, {"row": 2}])

    def test_compacted_relation_counts_replace_preclean_acceptance(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "def _actor_relation_counts" in "".join(cell["source"])
        )
        tree = ast.parse(source)
        function_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_actor_relation_counts"
        )
        namespace = {
            "Counter": Counter,
            "RELATION_FIELDS": (
                "card_parent",
                "event_source",
                "event_target",
                "option_source",
                "option_target",
            ),
        }
        module = ast.fix_missing_locations(
            ast.Module(body=[function_node], type_ignores=[])
        )
        exec(compile(module, str(NOTEBOOK), "exec"), namespace)
        counts = namespace["_actor_relation_counts"](
            {
                "card_parent": [0, 1, 1],
                "event_source": [],
                "event_target": [],
                "option_source": [1, 0],
                "option_target": [2, 0],
            }
        )
        self.assertEqual(
            dict(counts),
            {
                "card_parent": 2,
                "event_source": 0,
                "event_target": 0,
                "option_source": 1,
                "option_target": 1,
            },
        )
        compact_index = source.index('actor[key] = []')
        recount_index = source.index(
            "relation_counts.update(_actor_relation_counts(actor))"
        )
        self.assertLess(compact_index, recount_index)
        self.assertIn('"exact_event_relations": 0', source)
        self.assertIn('"relation_counts": {', source)


if __name__ == "__main__":
    unittest.main()
