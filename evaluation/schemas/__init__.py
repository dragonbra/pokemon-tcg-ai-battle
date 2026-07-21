"""评测记录的 JSON schema 校验接口。"""

from .json_schema import (
    validate_case_record,
    validate_game_record,
    validate_manifest,
    validate_metric_payload,
)

__all__ = [
    "validate_case_record",
    "validate_game_record",
    "validate_manifest",
    "validate_metric_payload",
]
