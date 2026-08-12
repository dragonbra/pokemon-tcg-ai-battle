from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ABI_VERSION = 1
RULE_PACK_VERSION = 1
RULE_MAGIC = b"PTCGR001"
MAX_CODEC_OPTIONS = 80
INSTRUCTION_STRUCT = struct.Struct("<BBHiii")
ACTION_STRUCT = struct.Struct("<IHHHHI")
HEADER_STRUCT = struct.Struct("<8sIIII32s")


OPCODES = {
    "NOP": 0,
    "DRAW": 1,
    "DAMAGE_ACTIVE": 2,
    "HEAL_ACTIVE": 3,
    "END_TURN": 4,
    "CHECK_KNOCKOUT": 5,
    "SET_WINNER": 6,
    "EMIT_DECISION": 7,
    "SET_COUNTER": 8,
    "ADD_COUNTER": 9,
    "DELAY_EFFECT": 10,
    "MOVE_CARD": 11,
    "CONDITIONAL_COUNTER_GE": 12,
    "RANDOM_BRANCH": 13,
    "HALT": 255,
}

TARGETS = {
    "ACTOR": 0,
    "OPPONENT": 1,
    "PLAYER_0": 2,
    "PLAYER_1": 3,
}


def _as_enum(value: Any, table: Mapping[str, int], field: str) -> int:
    if isinstance(value, str):
        key = value.strip().upper()
        if key not in table:
            raise ValueError(f"unknown {field}: {value!r}")
        return table[key]
    parsed = int(value)
    if parsed not in table.values():
        raise ValueError(f"unknown numeric {field}: {parsed}")
    return parsed


@dataclass(frozen=True)
class Instruction:
    opcode: int
    target: int = 0
    flags: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Instruction":
        allowed = {"opcode", "target", "flags", "arg0", "arg1", "arg2"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown instruction fields: {sorted(unknown)}")
        if "opcode" not in raw:
            raise ValueError("instruction is missing opcode")
        result = cls(
            opcode=_as_enum(raw["opcode"], OPCODES, "opcode"),
            target=_as_enum(raw.get("target", "ACTOR"), TARGETS, "target"),
            flags=int(raw.get("flags", 0)),
            arg0=int(raw.get("arg0", 0)),
            arg1=int(raw.get("arg1", 0)),
            arg2=int(raw.get("arg2", 0)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not 0 <= self.opcode <= 255:
            raise ValueError("opcode must fit uint8")
        if not 0 <= self.target <= 255:
            raise ValueError("target must fit uint8")
        if not 0 <= self.flags <= 65535:
            raise ValueError("instruction flags must fit uint16")
        for name, value in (("arg0", self.arg0), ("arg1", self.arg1), ("arg2", self.arg2)):
            if not -(2**31) <= value < 2**31:
                raise ValueError(f"{name} must fit int32")

    def pack(self) -> bytes:
        self.validate()
        return INSTRUCTION_STRUCT.pack(
            self.opcode,
            self.target,
            self.flags,
            self.arg0,
            self.arg1,
            self.arg2,
        )


@dataclass(frozen=True)
class Action:
    action_id: int
    name: str
    option_type: int
    card_id: int
    flags: int
    program: tuple[Instruction, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Action":
        allowed = {"action_id", "name", "option_type", "card_id", "flags", "program"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown action fields: {sorted(unknown)}")
        program_raw = raw.get("program")
        if not isinstance(program_raw, list) or not program_raw:
            raise ValueError("action program must be a non-empty list")
        return cls(
            action_id=int(raw["action_id"]),
            name=str(raw.get("name", f"action_{raw['action_id']}")),
            option_type=int(raw.get("option_type", 0)),
            card_id=int(raw.get("card_id", 0)),
            flags=int(raw.get("flags", 0)),
            program=tuple(Instruction.from_dict(item) for item in program_raw),
        )

    def validate(self) -> None:
        if not 0 <= self.action_id <= 65535:
            raise ValueError("action_id must fit uint16")
        if not 0 <= self.option_type <= 65535:
            raise ValueError("option_type must fit uint16")
        if not 0 <= self.card_id < 2**32:
            raise ValueError("card_id must fit uint32")
        if not 0 <= self.flags <= 65535:
            raise ValueError("action flags must fit uint16")
        if not self.program or len(self.program) > 65535:
            raise ValueError("action program length must be in [1, 65535]")
        for instruction in self.program:
            instruction.validate()


@dataclass(frozen=True)
class RulePack:
    name: str
    actions: tuple[Action, ...]
    metadata: Mapping[str, Any]
    abi_version: int = ABI_VERSION
    rule_pack_version: int = RULE_PACK_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RulePack":
        allowed = {"name", "abi_version", "rule_pack_version", "metadata", "actions"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown rule-pack fields: {sorted(unknown)}")
        action_rows = raw.get("actions")
        if not isinstance(action_rows, list):
            raise ValueError("rule pack actions must be a list")
        pack = cls(
            name=str(raw.get("name", "unnamed")),
            actions=tuple(Action.from_dict(item) for item in action_rows),
            metadata=dict(raw.get("metadata") or {}),
            abi_version=int(raw.get("abi_version", ABI_VERSION)),
            rule_pack_version=int(raw.get("rule_pack_version", RULE_PACK_VERSION)),
        )
        pack.validate()
        return pack

    @classmethod
    def load(cls, path: str | Path) -> "RulePack":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("rule pack root must be an object")
        return cls.from_dict(raw)

    def validate(self) -> None:
        if self.abi_version != ABI_VERSION:
            raise ValueError(f"unsupported ABI version: {self.abi_version}")
        if self.rule_pack_version != RULE_PACK_VERSION:
            raise ValueError(f"unsupported rule-pack version: {self.rule_pack_version}")
        if not self.actions:
            raise ValueError("rule pack must define at least one action")
        if len(self.actions) > MAX_CODEC_OPTIONS:
            raise ValueError(f"rule pack has more than {MAX_CODEC_OPTIONS} actions")
        ids = [action.action_id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("action IDs must be unique")
        if ids != list(range(len(ids))):
            raise ValueError("prototype action IDs must be contiguous from zero")
        for action in self.actions:
            action.validate()

    def canonical_json(self) -> bytes:
        payload = {
            "abi_version": self.abi_version,
            "rule_pack_version": self.rule_pack_version,
            "name": self.name,
            "metadata": dict(self.metadata),
            "actions": [
                {
                    "action_id": action.action_id,
                    "name": action.name,
                    "option_type": action.option_type,
                    "card_id": action.card_id,
                    "flags": action.flags,
                    "program": [
                        {
                            "opcode": instruction.opcode,
                            "target": instruction.target,
                            "flags": instruction.flags,
                            "arg0": instruction.arg0,
                            "arg1": instruction.arg1,
                            "arg2": instruction.arg2,
                        }
                        for instruction in action.program
                    ],
                }
                for action in self.actions
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def flatten(self) -> tuple[tuple[Instruction, ...], tuple[tuple[int, int, Action], ...]]:
        instructions: list[Instruction] = []
        descriptors: list[tuple[int, int, Action]] = []
        for action in self.actions:
            offset = len(instructions)
            instructions.extend(action.program)
            descriptors.append((offset, len(action.program), action))
        return tuple(instructions), tuple(descriptors)

    def to_bytes(self) -> bytes:
        self.validate()
        instructions, descriptors = self.flatten()
        digest = hashlib.sha256(self.canonical_json()).digest()
        header = HEADER_STRUCT.pack(
            RULE_MAGIC,
            self.rule_pack_version,
            self.abi_version,
            len(instructions),
            len(descriptors),
            digest,
        )
        instruction_bytes, action_bytes = self.packed_sections()
        return header + instruction_bytes + action_bytes

    def packed_sections(self) -> tuple[bytes, bytes]:
        self.validate()
        instructions, descriptors = self.flatten()
        instruction_bytes = b"".join(instruction.pack() for instruction in instructions)
        action_bytes = b"".join(
            ACTION_STRUCT.pack(
                offset,
                length,
                action.action_id,
                action.option_type,
                action.flags,
                action.card_id,
            )
            for offset, length, action in descriptors
        )
        return instruction_bytes, action_bytes

    def report(self) -> dict[str, Any]:
        instructions, _ = self.flatten()
        opcodes: dict[str, int] = {}
        reverse = {value: key for key, value in OPCODES.items()}
        for instruction in instructions:
            name = reverse.get(instruction.opcode, str(instruction.opcode))
            opcodes[name] = opcodes.get(name, 0) + 1
        return {
            "name": self.name,
            "abi_version": self.abi_version,
            "rule_pack_version": self.rule_pack_version,
            "actions": len(self.actions),
            "instructions": len(instructions),
            "opcodes": dict(sorted(opcodes.items())),
            "canonical_sha256": hashlib.sha256(self.canonical_json()).hexdigest(),
            "binary_bytes": len(self.to_bytes()),
        }
