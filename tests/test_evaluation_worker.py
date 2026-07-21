from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path

from evaluation.packages.loader import SubmissionPackage
from evaluation.runtime.loader import compute_cg_manifest
from evaluation.runner.models import GameRequest, GameResult
from evaluation.runner.worker import run_game


class EvaluationWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_package(
        self,
        name: str,
        deck_card: int,
        *,
        api_value: int = 1,
        fail_on_play: bool = False,
        system_exit_on_play: bool = False,
        delayed_cg_import: bool = False,
        local_helper_value: str | None = None,
        namespace_helper_value: str | None = None,
        import_system_exit: bool = False,
        start_error: bool = False,
        select_system_exit: bool = False,
        finish_error: bool = False,
    ) -> SubmissionPackage:
        package_root = self.root / name
        cg_root = package_root / "cg"
        cg_root.mkdir(parents=True)
        (cg_root / "__init__.py").write_text("", encoding="utf-8")
        (cg_root / "api.py").write_text(f"API_VALUE = {api_value}\n", encoding="utf-8")
        (cg_root / "libcg.so").write_bytes(b"fake native")
        (cg_root / "game.py").write_text(
            self.fake_game_source(
                import_system_exit=import_system_exit,
                start_error=start_error,
                select_system_exit=select_system_exit,
                finish_error=finish_error,
            ),
            encoding="utf-8",
        )
        deck = [deck_card] * 60
        (package_root / "deck.csv").write_text(
            "\n".join(str(card) for card in deck) + "\n",
            encoding="utf-8",
        )
        if local_helper_value is not None:
            (package_root / "local_helper.py").write_text(
                f"VALUE = {local_helper_value!r}\n",
                encoding="utf-8",
            )
        if namespace_helper_value is not None:
            helpers_root = package_root / "helpers"
            helpers_root.mkdir()
            (helpers_root / "util.py").write_text(
                f"VALUE = {namespace_helper_value!r}\n",
                encoding="utf-8",
            )
        (package_root / "main.py").write_text(
            self.fake_agent_source(
                deck,
                fail_on_play=fail_on_play,
                system_exit_on_play=system_exit_on_play,
                delayed_cg_import=delayed_cg_import,
                local_helper_value=local_helper_value,
                namespace_helper_value=namespace_helper_value,
            ),
            encoding="utf-8",
        )
        return SubmissionPackage(
            name=name,
            root=package_root,
            deck=deck,
            entrypoint=package_root / "main.py",
            package_hash=f"{name}-hash",
            deck_hash=f"{name}-deck-hash",
            cg_manifest=compute_cg_manifest(cg_root),
        )

    @staticmethod
    def fake_agent_source(
        deck: list[int],
        *,
        fail_on_play: bool,
        system_exit_on_play: bool,
        delayed_cg_import: bool,
        local_helper_value: str | None,
        namespace_helper_value: str | None,
    ) -> str:
        if fail_on_play:
            failure = (
                "    if observation.get('select') is not None:\n"
                "        raise RuntimeError('candidate boom')\n"
            )
        elif system_exit_on_play:
            failure = (
                "    if observation.get('select') is not None:\n"
                "        raise SystemExit('candidate exit')\n"
            )
        else:
            failure = ""
        delayed_import = (
            "    if observation.get('select') is not None:\n"
            "        from cg.api import API_VALUE\n"
            "        if API_VALUE != 1:\n"
            "            raise RuntimeError('unexpected api value')\n"
            if delayed_cg_import
            else ""
        )
        local_helper_import = (
            "from local_helper import VALUE as IMPORTED_HELPER_VALUE\n"
            if local_helper_value
            else ""
        )
        local_import_path_record = (
            "with (PACKAGE_ROOT / 'import_path.json').open('w', encoding='utf-8') as file:\n"
            "    json.dump(sys.path, file)\n"
            if local_helper_value
            else ""
        )
        local_helper_record = (
            "    from local_helper import VALUE as PLAY_HELPER_VALUE\n"
            "    with (PACKAGE_ROOT / 'helper_values.txt').open('a', encoding='utf-8') as file:\n"
            "        file.write(f'{IMPORTED_HELPER_VALUE}/{PLAY_HELPER_VALUE}\\n')\n"
            if local_helper_value
            else ""
        )
        namespace_helper_import = (
            "from helpers.util import VALUE as IMPORTED_NAMESPACE_HELPER_VALUE\n"
            if namespace_helper_value
            else ""
        )
        namespace_helper_record = (
            "    from helpers.util import VALUE as PLAY_NAMESPACE_HELPER_VALUE\n"
            "    with (PACKAGE_ROOT / 'namespace_helper_values.txt').open('a', encoding='utf-8') as file:\n"
            "        file.write(\n"
            "            f'{IMPORTED_NAMESPACE_HELPER_VALUE}/{PLAY_NAMESPACE_HELPER_VALUE}\\n'\n"
            "        )\n"
            if namespace_helper_value
            else ""
        )
        return (
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "PACKAGE_ROOT = Path(__file__).resolve().parent\n"
            f"{local_import_path_record}"
            f"{local_helper_import}"
            f"{namespace_helper_import}\n"
            "CALLS = 0\n\n"
            "def agent(observation):\n"
            "    global CALLS\n"
            "    CALLS += 1\n"
            "    with (PACKAGE_ROOT / 'agent_calls.txt').open('a', encoding='utf-8') as file:\n"
            "        file.write(str(CALLS) + '\\n')\n"
            "    if observation.get('select') is None:\n"
            f"        return {deck!r}\n"
            f"{delayed_import}"
            f"{local_helper_record}"
            f"{namespace_helper_record}"
            f"{failure}"
            "    return [0]\n"
        )

    @staticmethod
    def fake_game_source(
        *,
        import_system_exit: bool,
        start_error: bool,
        select_system_exit: bool,
        finish_error: bool,
    ) -> str:
        import_failure = "raise SystemExit('game import exit')\n" if import_system_exit else ""
        start_failure = "    raise RuntimeError('battle start boom')\n" if start_error else ""
        select_failure = "    raise SystemExit('battle select exit')\n" if select_system_exit else ""
        finish_failure = "    raise RuntimeError('battle finish boom')\n" if finish_error else ""
        return (
            "import json\n"
            "from pathlib import Path\n\n"
            f"{import_failure}"
            "PACKAGE_ROOT = Path(__file__).resolve().parents[1]\n"
            "STEP = 0\n"
            "DECK0 = []\n"
            "DECK1 = []\n"
            "ACTIONS = []\n\n"
            "def _observation(result=-1):\n"
            "    return {\n"
            "        'current': {'turn': STEP, 'yourIndex': STEP % 2, 'result': result},\n"
            "        'select': None if result != -1 else {'type': 'choose', 'option': [0]},\n"
            "    }\n\n"
            "def battle_start(deck0, deck1):\n"
            "    global STEP, DECK0, DECK1, ACTIONS\n"
            f"{start_failure}"
            "    STEP = 0\n"
            "    DECK0 = list(deck0)\n"
            "    DECK1 = list(deck1)\n"
            "    ACTIONS = []\n"
            "    return _observation(), {'started': True}\n\n"
            "def battle_select(select_list):\n"
            "    global STEP\n"
            f"{select_failure}"
            "    ACTIONS.append(list(select_list))\n"
            "    STEP += 1\n"
            "    if STEP >= 3:\n"
            "        result = 0 if DECK0 and DECK0[0] == 7 else 1\n"
            "        return _observation(result)\n"
            "    return _observation()\n\n"
            "def battle_finish():\n"
            "    path = PACKAGE_ROOT / 'battle_finish_count.txt'\n"
            "    count = int(path.read_text(encoding='utf-8')) if path.exists() else 0\n"
            "    path.write_text(str(count + 1), encoding='utf-8')\n"
            f"{finish_failure}\n"
            "def visualize_data():\n"
            "    return json.dumps([{'frame': STEP, 'actions': ACTIONS}])\n"
        )

    def make_request(
        self,
        *,
        candidate_first: bool,
        candidate: SubmissionPackage | None = None,
        opponent: SubmissionPackage | None = None,
        max_steps: int = 10,
        visualize: bool = False,
    ) -> GameRequest:
        return GameRequest(
            run_id="run-1",
            game_id=f"game-{candidate_first}-{max_steps}-{visualize}",
            candidate=candidate or self.make_package("candidate", 7),
            opponent=opponent or self.make_package("opponent", 8),
            candidate_first=candidate_first,
            max_steps=max_steps,
            visualize=visualize,
        )

    def test_winner_is_normalized_when_candidate_is_player_zero(self) -> None:
        request = self.make_request(candidate_first=True)
        trace_path = self.root / "trace-first.json"

        result = run_game(request, trace_path)

        self.assertEqual(result.winner, 0)
        self.assertEqual(result.candidate_physical_index, 0)
        self.assertEqual(result.status, "finished")
        self.assertEqual(result.error_kind, None)
        self.assertEqual((request.candidate.root / "battle_finish_count.txt").read_text(), "1")

    def test_winner_is_normalized_when_candidate_is_player_one(self) -> None:
        request = self.make_request(candidate_first=False)

        result = run_game(request, self.root / "trace-second.json")

        self.assertEqual(result.winner, 0)
        self.assertEqual(result.candidate_physical_index, 1)
        self.assertEqual(result.status, "finished")
        self.assertEqual((request.candidate.root / "battle_finish_count.txt").read_text(), "1")

    def test_candidate_agent_exception_becomes_candidate_error_result(self) -> None:
        candidate = self.make_package("candidate", 7, fail_on_play=True)
        opponent = self.make_package("opponent", 8)
        request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)

        result = run_game(request, self.root / "trace-error.json")

        self.assertFalse(result.finished)
        self.assertEqual(result.status, "candidate_error")
        self.assertEqual(result.error_kind, "candidate_error")
        self.assertIn("candidate boom", result.error or "")
        self.assertEqual((candidate.root / "battle_finish_count.txt").read_text(), "1")

    def test_step_limit_becomes_unfinished_and_preserves_trace(self) -> None:
        request = self.make_request(candidate_first=True, max_steps=2)
        trace_path = self.root / "trace-limit.json"

        result = run_game(request, trace_path)

        self.assertFalse(result.finished)
        self.assertEqual(result.status, "unfinished")
        self.assertEqual(result.error_kind, "step_limit")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(len(trace["trace"]), 2)
        first_step = trace["trace"][0]
        self.assertIn("observation", first_step)
        self.assertIn("action", first_step)
        self.assertIn("state", first_step)
        self.assertEqual(first_step["state"]["result"], -1)

    def test_exact_step_limit_inspects_terminal_result_after_final_select(self) -> None:
        request = self.make_request(candidate_first=True, max_steps=3)
        trace_path = self.root / "trace-exact-limit.json"

        result = run_game(request, trace_path)

        self.assertTrue(result.finished)
        self.assertEqual(result.status, "finished")
        self.assertEqual(result.winner, 0)
        self.assertEqual(result.steps, 3)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(len(trace["trace"]), 4)

    def test_delayed_cg_import_works_and_parent_cg_state_is_restored(self) -> None:
        parent_cg = types.ModuleType("cg")
        previous_modules = {name: sys.modules.get(name) for name in ("cg", "cg.api")}
        previous_path = list(sys.path)
        previous_cwd = Path.cwd()
        sys.modules["cg"] = parent_cg
        try:
            candidate = self.make_package("candidate", 7, delayed_cg_import=True)
            request = self.make_request(candidate_first=True, candidate=candidate)

            result = run_game(request, self.root / "trace-delayed-cg.json")

            self.assertEqual(result.status, "finished")
            self.assertIs(sys.modules["cg"], parent_cg)
            self.assertEqual(sys.path, previous_path)
            self.assertEqual(Path.cwd(), previous_cwd)
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_system_exit_from_agent_becomes_candidate_error(self) -> None:
        candidate = self.make_package("candidate", 7, system_exit_on_play=True)
        request = self.make_request(candidate_first=True, candidate=candidate)

        result = run_game(request, self.root / "trace-system-exit.json")

        self.assertFalse(result.finished)
        self.assertEqual(result.status, "candidate_error")
        self.assertEqual(result.error_kind, "candidate_error")
        self.assertIn("candidate exit", result.error or "")

    def test_game_system_exit_becomes_game_error(self) -> None:
        candidate = self.make_package("candidate", 7, select_system_exit=True)
        opponent = self.make_package("opponent", 8, select_system_exit=True)
        request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)

        result = run_game(request, self.root / "trace-game-system-exit.json")

        self.assertFalse(result.finished)
        self.assertEqual(result.status, "game_error")
        self.assertEqual(result.error_kind, "game_error")
        self.assertIn("battle select exit", result.error or "")

    def test_cg_mismatch_is_serialized_without_starting_a_game(self) -> None:
        candidate = self.make_package("candidate", 7, api_value=1)
        opponent = self.make_package("opponent", 8, api_value=2)
        trace_path = self.root / "trace-mismatch.json"

        result = run_game(
            self.make_request(candidate_first=True, candidate=candidate, opponent=opponent),
            trace_path,
        )

        self.assertEqual(result.status, "cg_mismatch")
        self.assertEqual(result.error_kind, "cg_mismatch")
        self.assertTrue(trace_path.is_file())
        self.assertNotIn("battle_finish_count.txt", {path.name for path in self.root.rglob("*")})

    def test_load_failure_is_serialized_and_parent_cg_state_is_restored(self) -> None:
        parent_cg = types.ModuleType("cg")
        previous_path = list(sys.path)
        previous_cwd = Path.cwd()
        sys.modules["cg"] = parent_cg
        try:
            candidate = self.make_package("candidate", 7, import_system_exit=True)
            opponent = self.make_package("opponent", 8, import_system_exit=True)
            request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)
            trace_path = self.root / "trace-load-error.json"

            result = run_game(request, trace_path)

            self.assertEqual(result.status, "load_error")
            self.assertEqual(result.error_kind, "load_error")
            self.assertTrue(trace_path.is_file())
            self.assertIs(sys.modules["cg"], parent_cg)
            self.assertEqual(sys.path, previous_path)
            self.assertEqual(Path.cwd(), previous_cwd)
        finally:
            sys.modules.pop("cg", None)

    def test_start_failure_calls_finish_and_is_serialized(self) -> None:
        candidate = self.make_package("candidate", 7, start_error=True)
        opponent = self.make_package("opponent", 8, start_error=True)
        request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)
        trace_path = self.root / "trace-start-error.json"

        result = run_game(request, trace_path)

        self.assertEqual(result.status, "start_error")
        self.assertEqual(result.error_kind, "start_error")
        self.assertEqual((candidate.root / "battle_finish_count.txt").read_text(), "1")
        self.assertTrue(trace_path.is_file())

    def test_finish_failure_overrides_finished_result(self) -> None:
        candidate = self.make_package("candidate", 7, finish_error=True)
        opponent = self.make_package("opponent", 8, finish_error=True)
        request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)

        result = run_game(request, self.root / "trace-finish-error.json")

        self.assertFalse(result.finished)
        self.assertEqual(result.status, "finish_error")
        self.assertEqual(result.error_kind, "finish_error")
        self.assertIn("battle finish boom", result.error or "")
        self.assertEqual((candidate.root / "battle_finish_count.txt").read_text(), "1")

    def test_finish_failure_preserves_candidate_error(self) -> None:
        candidate = self.make_package("candidate", 7, fail_on_play=True, finish_error=True)
        opponent = self.make_package("opponent", 8, finish_error=True)
        request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)

        result = run_game(request, self.root / "trace-candidate-finish-error.json")

        self.assertFalse(result.finished)
        self.assertEqual(result.status, "candidate_error")
        self.assertEqual(result.error_kind, "candidate_error")
        self.assertIn("candidate boom", result.error or "")
        self.assertEqual((candidate.root / "battle_finish_count.txt").read_text(), "1")

    def test_visualize_frames_are_written_before_finish(self) -> None:
        request = self.make_request(candidate_first=True, visualize=True)
        trace_path = self.root / "trace-visualize.json"

        result = run_game(request, trace_path)

        self.assertEqual(result.status, "finished")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(trace["visualize"], [{"frame": 3, "actions": [[0], [0], [0]]}])

    def test_agents_use_their_own_same_named_local_helper_during_load_and_play(self) -> None:
        candidate = self.make_package("candidate", 7, local_helper_value="candidate-helper")
        opponent = self.make_package("opponent", 8, local_helper_value="opponent-helper")
        request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)

        result = run_game(request, self.root / "trace-local-helper.json")

        self.assertEqual(result.status, "finished")
        candidate_import_path = json.loads(
            (candidate.root / "import_path.json").read_text(encoding="utf-8")
        )
        opponent_import_path = json.loads(
            (opponent.root / "import_path.json").read_text(encoding="utf-8")
        )
        candidate_import_roots = [Path(path).resolve() for path in candidate_import_path]
        opponent_import_roots = [Path(path).resolve() for path in opponent_import_path]
        self.assertEqual(candidate_import_roots[0], candidate.root.resolve())
        self.assertNotIn(opponent.root.resolve(), candidate_import_roots)
        self.assertEqual(opponent_import_roots[0], opponent.root.resolve())
        self.assertNotIn(candidate.root.resolve(), opponent_import_roots)
        self.assertEqual(
            (candidate.root / "helper_values.txt").read_text(encoding="utf-8").splitlines(),
            ["candidate-helper/candidate-helper", "candidate-helper/candidate-helper"],
        )
        self.assertEqual(
            (opponent.root / "helper_values.txt").read_text(encoding="utf-8").splitlines(),
            ["opponent-helper/opponent-helper"],
        )

    def test_agents_use_their_own_same_named_namespace_helper_during_load_and_play(self) -> None:
        candidate = self.make_package("candidate", 7, namespace_helper_value="candidate-helper")
        opponent = self.make_package("opponent", 8, namespace_helper_value="opponent-helper")
        request = self.make_request(candidate_first=True, candidate=candidate, opponent=opponent)

        result = run_game(request, self.root / "trace-namespace-helper.json")

        self.assertEqual(result.status, "finished")
        self.assertFalse((candidate.root / "helpers" / "__init__.py").exists())
        self.assertFalse((opponent.root / "helpers" / "__init__.py").exists())
        self.assertEqual(
            (candidate.root / "namespace_helper_values.txt").read_text(encoding="utf-8").splitlines(),
            ["candidate-helper/candidate-helper", "candidate-helper/candidate-helper"],
        )
        self.assertEqual(
            (opponent.root / "namespace_helper_values.txt").read_text(encoding="utf-8").splitlines(),
            ["opponent-helper/opponent-helper"],
        )

    def test_worker_cli_reads_request_and_writes_serialized_result(self) -> None:
        request = self.make_request(candidate_first=False, visualize=True)
        request_path = self.root / "request.json"
        result_path = self.root / "result.json"
        trace_path = self.root / "trace-cli.json"
        request_payload = asdict(request)
        request_payload["candidate"]["root"] = str(request.candidate.root)
        request_payload["candidate"]["entrypoint"] = str(request.candidate.entrypoint)
        request_payload["opponent"]["root"] = str(request.opponent.root)
        request_payload["opponent"]["entrypoint"] = str(request.opponent.entrypoint)
        request_payload["trace_path"] = str(trace_path)
        request_path.write_text(json.dumps(request_payload), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, "-m", "evaluation.runner.worker", str(request_path), str(result_path)],
            cwd=self.root,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = GameResult(**{**payload, "trace_path": Path(payload["trace_path"])})
        self.assertEqual(result.winner, 0)
        self.assertEqual(result.candidate_physical_index, 1)
        self.assertTrue(trace_path.is_file())
        self.assertEqual(payload["trace_path"], str(trace_path))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(trace["result"], payload)

    def test_worker_cli_serializes_error_for_nonpositive_max_steps(self) -> None:
        request = self.make_request(candidate_first=True, max_steps=0)
        request_path = self.root / "request-invalid-max-steps.json"
        result_path = self.root / "result-invalid-max-steps.json"
        trace_path = self.root / "trace-invalid-max-steps.json"
        request_payload = asdict(request)
        request_payload["candidate"]["root"] = str(request.candidate.root)
        request_payload["candidate"]["entrypoint"] = str(request.candidate.entrypoint)
        request_payload["opponent"]["root"] = str(request.opponent.root)
        request_payload["opponent"]["entrypoint"] = str(request.opponent.entrypoint)
        request_payload["trace_path"] = str(trace_path)
        request_path.write_text(json.dumps(request_payload), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, "-m", "evaluation.runner.worker", str(request_path), str(result_path)],
            cwd=self.root,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["finished"])
        self.assertEqual(payload["status"], "worker_error")
        self.assertEqual(payload["error_kind"], "worker_error")
        self.assertEqual(payload["error"], "max_steps must be greater than zero")
        self.assertEqual(payload["steps"], 0)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(trace["trace"], [])
        self.assertEqual(trace["result"], payload)
        self.assertFalse((request.candidate.root / "battle_finish_count.txt").exists())


if __name__ == "__main__":
    unittest.main()
