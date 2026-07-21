from __future__ import annotations

from typing import Any


def terminal_result(trace: dict) -> dict:
    """返回 worker trace 中嵌套的终态对局结果。"""
    value = trace.get("result")
    return value if isinstance(value, dict) else {}


def result_field(trace: dict, name: str, default: Any = None) -> Any:
    """优先读取嵌套终态字段，兼容历史扁平 fixture。"""
    result = terminal_result(trace)
    if name in result:
        return result[name]
    return trace.get(name, default)


def lifecycle_status(trace: dict) -> str:
    """把 worker 终态归一为 finished、unfinished 或 error。"""
    error_kind = str(result_field(trace, "error_kind", "") or "").lower()
    status = str(result_field(trace, "status", "") or "").lower()
    unfinished_values = {"step_limit", "step limit", "unfinished"}
    if error_kind in unfinished_values or status in unfinished_values:
        return "unfinished"
    if error_kind:
        return "error"
    if status and status not in {"finished", "success"}:
        return "error"
    if result_field(trace, "finished") is False:
        return "unfinished"
    return "finished"


def as_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def observation(step: dict) -> dict:
    value = step.get("observation")
    return value if isinstance(value, dict) else {}


def current(step: dict) -> dict:
    value = observation(step).get("current")
    if isinstance(value, dict):
        return value
    return {
        "turn": step.get("turn", 0),
        "yourIndex": step.get("yourIndex", 0),
        "players": step.get("players") or [],
    }


def players(step: dict) -> list[dict]:
    value = current(step).get("players")
    return value if isinstance(value, list) else []


def player_at(step: dict, index: int) -> dict:
    values = players(step)
    if 0 <= index < len(values) and isinstance(values[index], dict):
        return values[index]
    return {}


def cards(player: dict, area: str) -> list[dict]:
    values = player.get(area) or []
    return [value for value in values if isinstance(value, dict)]


def active(player: dict) -> dict | None:
    values = cards(player, "active")
    return values[0] if values else None


def bench(player: dict) -> list[dict]:
    return cards(player, "bench")


def option_list(step: dict) -> list[dict]:
    select = step.get("select")
    if not isinstance(select, dict):
        select = observation(step).get("select")
    if not isinstance(select, dict):
        return []
    values = select.get("options")
    if values is None:
        values = select.get("option")
    return [value for value in values or [] if isinstance(value, dict)]


def action_list(step: dict) -> list[int]:
    values = step.get("action")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, int) and not isinstance(value, bool)]


def selected_options(step: dict) -> list[dict]:
    options = option_list(step)
    return [
        options[index]
        for index in action_list(step)
        if 0 <= index < len(options)
    ]


def option_attack_id(option: dict) -> int | None:
    return as_int(option.get("attackId", option.get("attack_id")))


def selected_attack_id(step: dict) -> int | None:
    for option in selected_options(step):
        value = option_attack_id(option)
        if value is not None:
            return value
    return None


def logs(step: dict) -> list[dict]:
    values = observation(step).get("logs")
    return values if isinstance(values, list) else []


def field_state(step: dict, player_index: int) -> dict[int, dict]:
    state: dict[int, dict] = {}
    player = player_at(step, player_index)
    for pokemon in [active(player), *bench(player)]:
        if not pokemon:
            continue
        serial = as_int(pokemon.get("serial"))
        if serial is not None:
            state[serial] = pokemon
    return state


def hp_loss_serials(step: dict, player_index: int) -> set[int]:
    return {
        serial
        for log in logs(step)
        if isinstance(log, dict)
        and as_int(log.get("type")) == 16
        and as_int(log.get("playerIndex")) == player_index
        and (
            log.get("putDamageCounter") is True
            or (as_int(log.get("value"), 0) or 0) < 0
        )
        for serial in [as_int(log.get("serial"))]
        if serial is not None
    }


def _moved_from_field_to_discard(log: dict, player_index: int, serial: int) -> bool:
    return (
        as_int(log.get("type")) == 6
        and as_int(log.get("playerIndex")) == player_index
        and as_int(log.get("serial")) == serial
        and as_int(log.get("fromArea")) in {4, 5}
        and as_int(log.get("toArea")) == 3
    )


def knockout_is_confirmed(
    step: dict,
    log: dict,
    player_index: int,
    previous_field: dict[int, dict],
) -> bool:
    serial = as_int(log.get("serial"))
    if serial is None:
        return False
    previous = previous_field.get(serial)
    previous_hp = as_int(previous.get("hp")) if previous is not None else None
    if previous_hp is not None and previous_hp <= 0:
        return True
    current_card = field_state(step, player_index).get(serial)
    current_hp = as_int(current_card.get("hp")) if current_card is not None else None
    if current_hp is not None and current_hp <= 0:
        return True
    return serial in hp_loss_serials(step, player_index) and any(
        _moved_from_field_to_discard(candidate, player_index, serial)
        for candidate in logs(step)
        if isinstance(candidate, dict)
    )


def trace_steps(trace: dict) -> list[dict]:
    values = trace.get("trace")
    return values if isinstance(values, list) else []


def result_steps(trace: dict) -> int | None:
    """读取 worker 的 action 选择次数，兼容历史扁平 trace。"""
    result = terminal_result(trace)
    values = (result.get("steps"), trace.get("steps"))
    for value in values:
        count = as_int(value)
        if count is not None and count >= 0:
            return count
    return None


def candidate_index(trace: dict, context_index: int = 0) -> int:
    for field in ("candidate_physical_index", "candidatePhysicalIndex", "alakazamPhysicalIndex"):
        value = as_int(result_field(trace, field))
        if value in (0, 1):
            return value
    return context_index


def normalized_evidence(
    trace: dict,
    step: dict | None,
    *,
    fallback_step: int,
    candidate_physical_index: int = 0,
    default_role: str = "trace",
) -> dict[str, Any]:
    """标准化 worker 和历史 trace 的最小证据定位字段。"""
    raw_step = step if isinstance(step, dict) else {}
    state = raw_step.get("state")
    state = state if isinstance(state, dict) else {}
    state_current = state.get("current")
    state_current = state_current if isinstance(state_current, dict) else state
    observed_current = observation(raw_step).get("current")
    observed_current = observed_current if isinstance(observed_current, dict) else {}
    sources = (observed_current, state_current, raw_step)

    def read(name: str, default: Any = None) -> Any:
        for source in sources:
            if name in source and source[name] is not None:
                return source[name]
        return default

    role = raw_step.get("role")
    if not role:
        role = read("role")
    if not role:
        current_player = as_int(read("yourIndex"))
        candidate = candidate_index(trace, candidate_physical_index)
        if current_player in (0, 1):
            role = "candidate" if current_player == candidate else "opponent"
        else:
            role = default_role

    return {
        "step": read("step", fallback_step),
        "turn": read("turn", 0),
        "role": role,
        "action": raw_step.get("action", []),
    }


def agent_turn_target(trace: dict, candidate_physical_index: int | None = None) -> int | None:
    candidate = candidate_index(trace, candidate_physical_index or 0)
    for step in trace_steps(trace):
        first_player = as_int(current(step).get("firstPlayer"))
        if first_player in (0, 1):
            return 3 if first_player == candidate else 4
    return None


def is_agent_step(step: dict, agent_label: str | None = None) -> bool:
    if agent_label is None:
        return step.get("role") not in {"opponent", "finished"}
    return step.get("role") == agent_label


# 兼容旧 evaluator helper 的内部命名；这些别名仍然只处理传入的 JSON dict。
_observation = observation
_current = current
_players = players
_player_at = player_at
_active = active
_bench = bench
_option_list = option_list
_action_list = action_list
_selected_options = selected_options
_selected_attack_id = selected_attack_id
_field_state = field_state
_knockout_is_confirmed = knockout_is_confirmed
_agent_turn_target = agent_turn_target
_is_agent_step = is_agent_step
