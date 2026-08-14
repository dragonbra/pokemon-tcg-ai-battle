"""Fail-closed loaders for 0044 project-local immutable assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


DECK_REGISTRY_SCHEMA = "0043_deck_registry_v2_numeric_identity"
POLICY_REGISTRY_SCHEMA = "0044_policy_registry_v1"
EVALUATION_REGISTRY_SCHEMA = "0044_evaluation_registry_v1"


class AssetIntegrityError(RuntimeError):
    """An immutable 0044 asset cannot be reconstructed exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise AssetIntegrityError(f"missing asset: {path}") from error
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_deck_sha256(cards: Iterable[int]) -> str:
    values = tuple(cards)
    if len(values) != 60 or any(type(card) is not int or card <= 0 for card in values):
        raise AssetIntegrityError("a registered deck must contain exactly 60 positive card IDs")
    return hashlib.sha256(
        ",".join(str(card) for card in sorted(values)).encode("ascii")
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AssetIntegrityError(f"missing asset: {path}") from error
    except json.JSONDecodeError as error:
        raise AssetIntegrityError(f"invalid JSON asset: {path}") from error
    if not isinstance(value, dict):
        raise AssetIntegrityError(f"JSON asset must be an object: {path}")
    return value


def _safe_path(project_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise AssetIntegrityError(f"asset path must be project-relative: {relative}")
    root = project_root.resolve()
    path = (root / Path(*pure.parts)).resolve()
    if path != root and root not in path.parents:
        raise AssetIntegrityError(f"asset path escapes project root: {relative}")
    return path


@dataclass(frozen=True, slots=True)
class DeckAsset:
    deck_id: str
    name: str
    archetype: str
    deck_path: str
    file_sha256: str
    content_sha256: str
    card_count: int
    roles: tuple[str, ...]
    tags: tuple[str, ...]
    source: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyArtifact:
    purpose: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PolicyAsset:
    policy_id: str
    role: str
    generation: int | None
    frozen: bool
    manifest_path: str
    manifest_sha256: str
    effective_policy_sha256: str
    artifacts: tuple[PolicyArtifact, ...]


@dataclass(frozen=True, slots=True)
class EvaluationEntry:
    deck_id: str
    exact_deck_sha256: str
    games: int


@dataclass(frozen=True, slots=True)
class EvaluationAsset:
    evaluation_id: str
    deck_pool_role: str
    policy_id: str
    manifest_path: str
    manifest_sha256: str
    seed_manifest_path: str
    seed_manifest_sha256: str
    entries: tuple[EvaluationEntry, ...]


@dataclass(frozen=True, slots=True)
class AssetAudit:
    status: str
    deck_count: int
    training_deck_count: int
    evaluation_deck_count: int
    evaluation_games: int
    policy_ids: tuple[str, ...]
    latest_champion_policy_id: str


@dataclass(frozen=True, slots=True)
class AssetRegistry:
    project_root: Path
    decks: tuple[DeckAsset, ...]
    policies: tuple[PolicyAsset, ...]
    evaluations: tuple[EvaluationAsset, ...]
    latest_champion_policy_id: str
    active_policy_ids: tuple[str, ...]

    @classmethod
    def load(cls, project_root: Path) -> "AssetRegistry":
        project_root = Path(project_root)
        deck_payload = _json(project_root / "assets/decks/registry.json")
        policy_payload = _json(project_root / "assets/policies/registry.json")
        evaluation_payload = _json(project_root / "assets/evaluation/registry.json")
        if deck_payload.get("schema_version") != DECK_REGISTRY_SCHEMA:
            raise AssetIntegrityError("unsupported 0044 deck registry schema")
        if policy_payload.get("schema_version") != POLICY_REGISTRY_SCHEMA:
            raise AssetIntegrityError("unsupported 0044 policy registry schema")
        if evaluation_payload.get("schema_version") != EVALUATION_REGISTRY_SCHEMA:
            raise AssetIntegrityError("unsupported 0044 evaluation registry schema")
        decks = tuple(
            DeckAsset(
                deck_id=row["deck_id"], name=row["name"], archetype=row["archetype"],
                deck_path=row["deck_path"], file_sha256=row["file_sha256"],
                content_sha256=row["content_sha256"], card_count=row["card_count"],
                roles=tuple(row["roles"]), tags=tuple(row.get("tags", ())),
                source=row["source"],
            )
            for row in deck_payload.get("decks", ())
        )
        policies = tuple(
            PolicyAsset(
                policy_id=row["policy_id"], role=row["role"],
                generation=row.get("generation"), frozen=row["frozen"],
                manifest_path=row["manifest_path"],
                manifest_sha256=row["manifest_sha256"],
                effective_policy_sha256=row["effective_policy_sha256"],
                artifacts=tuple(PolicyArtifact(**artifact) for artifact in row["artifacts"]),
            )
            for row in policy_payload.get("policies", ())
        )
        evaluations = tuple(
            EvaluationAsset(
                evaluation_id=row["evaluation_id"],
                deck_pool_role=row["deck_pool_role"], policy_id=row["policy_id"],
                manifest_path=row["manifest_path"],
                manifest_sha256=row["manifest_sha256"],
                seed_manifest_path=row["seed_manifest_path"],
                seed_manifest_sha256=row["seed_manifest_sha256"],
                entries=tuple(EvaluationEntry(**entry) for entry in row["entries"]),
            )
            for row in evaluation_payload.get("evaluations", ())
        )
        return cls(
            project_root=project_root,
            decks=decks,
            policies=policies,
            evaluations=evaluations,
            latest_champion_policy_id=policy_payload.get("latest_champion_policy_id", ""),
            active_policy_ids=tuple(policy_payload.get("active_policy_pool", ())),
        )

    def validate_all(self) -> AssetAudit:
        if len(self.decks) != len({deck.deck_id for deck in self.decks}):
            raise AssetIntegrityError("duplicate immutable deck_id")
        if len(self.decks) != len({deck.content_sha256 for deck in self.decks}):
            raise AssetIntegrityError("duplicate immutable deck content")
        expected_ids = tuple(f"{index:03d}" for index in range(1, len(self.decks) + 1))
        if tuple(deck.deck_id for deck in self.decks) != expected_ids:
            raise AssetIntegrityError(
                "0044 deck IDs must be ordered, contiguous zero-padded numeric identities"
            )
        for deck in self.decks:
            if (
                not deck.deck_id.isascii()
                or not deck.deck_id.isdigit()
                or len(deck.deck_id) < 3
                or deck.deck_path != f"assets/decks/definitions/{deck.deck_id}/deck.csv"
            ):
                raise AssetIntegrityError(
                    f"deck identity/path must use the canonical numeric ID: {deck.deck_id}"
                )
            path = _safe_path(self.project_root, deck.deck_path)
            if sha256_file(path) != deck.file_sha256:
                raise AssetIntegrityError(f"deck file SHA-256 mismatch: {deck.deck_id}")
            try:
                cards = tuple(int(row) for row in path.read_text(encoding="utf-8").splitlines())
            except ValueError as error:
                raise AssetIntegrityError(f"invalid card ID in deck: {deck.deck_id}") from error
            if deck.card_count != 60 or canonical_deck_sha256(cards) != deck.content_sha256:
                raise AssetIntegrityError(f"deck content SHA-256 mismatch: {deck.deck_id}")
            if not deck.roles or not set(deck.roles) <= {"training", "evaluation"}:
                raise AssetIntegrityError(f"invalid deck roles: {deck.deck_id}")

        if len(self.policies) != len({policy.policy_id for policy in self.policies}):
            raise AssetIntegrityError("duplicate immutable policy_id")
        policies = {policy.policy_id: policy for policy in self.policies}
        for policy in self.policies:
            if not policy.frozen:
                raise AssetIntegrityError(
                    f"policy assets may contain only immutable admitted policies: {policy.policy_id}"
                )
            manifest_path = _safe_path(self.project_root, policy.manifest_path)
            if sha256_file(manifest_path) != policy.manifest_sha256:
                raise AssetIntegrityError(f"policy manifest SHA-256 mismatch: {policy.policy_id}")
            manifest = _json(manifest_path)
            if (
                manifest.get("policy_id") != policy.policy_id
                or manifest.get("effective_policy_sha256") != policy.effective_policy_sha256
                or manifest.get("frozen") is not policy.frozen
            ):
                raise AssetIntegrityError(f"policy manifest identity mismatch: {policy.policy_id}")
            allowed_roots = {
                "Policy-0809": ("assets/policies/definitions/policy_0809",),
                "Champion-G1": (
                    "assets/policies/definitions/policy_0809",
                    "assets/policies/definitions/champion_g001",
                ),
            }.get(policy.policy_id)
            if policy.policy_id.startswith("Champion-G") and policy.generation is not None:
                allowed_roots = (
                    "assets/policies/definitions/policy_0809",
                    f"assets/policies/definitions/champion_g{policy.generation:03d}",
                )
            if allowed_roots is None:
                allowed_roots = (
                    f"assets/policies/definitions/{policy.policy_id.lower().replace('-', '_')}",
                )
            for artifact in policy.artifacts:
                if not any(artifact.path.startswith(root + "/") for root in allowed_roots):
                    raise AssetIntegrityError(
                        f"policy artifact path must use its semantic policy directory: {policy.policy_id}"
                    )
                path = _safe_path(self.project_root, artifact.path)
                if sha256_file(path) != artifact.sha256:
                    raise AssetIntegrityError(
                        f"policy artifact SHA-256 mismatch: {policy.policy_id}/{artifact.purpose}"
                    )
        champion = policies.get(self.latest_champion_policy_id)
        if champion is None or champion.role != "latest_champion" or not champion.frozen:
            raise AssetIntegrityError("latest champion pointer must reference a frozen champion")
        expected_active = (self.latest_champion_policy_id,)
        if self.active_policy_ids != expected_active:
            raise AssetIntegrityError(
                "0044 active opponent pool must be the singleton latest champion"
            )
        if len(self.active_policy_ids) != len(set(self.active_policy_ids)):
            raise AssetIntegrityError("active opponent pool contains duplicate identities")
        expected_policies = {
            "Policy-0809",
            *(f"Champion-G{generation}" for generation in range(2, champion.generation + 1)),
        }
        if set(policies) != expected_policies:
            raise AssetIntegrityError(
                "0044 policy assets must contain contiguous admitted Champions plus Policy-0809"
            )
        if (
            champion.generation is None
            or champion.policy_id != f"Champion-G{champion.generation}"
            or policies["Policy-0809"].role != "historical_anchor"
            or any(
                policies[f"Champion-G{generation}"].role
                != ("latest_champion" if generation == champion.generation else "champion")
                for generation in range(2, champion.generation + 1)
            )
        ):
            raise AssetIntegrityError("0044 immutable policy roles are invalid")

        deck_by_id = {deck.deck_id: deck for deck in self.decks}
        evaluation_games = 0
        evaluation_decks: set[str] = set()
        for evaluation in self.evaluations:
            if evaluation.policy_id not in policies:
                raise AssetIntegrityError(f"evaluation policy is unregistered: {evaluation.policy_id}")
            for path_text, digest in (
                (evaluation.manifest_path, evaluation.manifest_sha256),
                (evaluation.seed_manifest_path, evaluation.seed_manifest_sha256),
            ):
                path = _safe_path(self.project_root, path_text)
                if sha256_file(path) != digest:
                    raise AssetIntegrityError(
                        f"evaluation asset SHA-256 mismatch: {evaluation.evaluation_id}"
                    )
            if len(evaluation.entries) != len({row.deck_id for row in evaluation.entries}):
                raise AssetIntegrityError(f"duplicate evaluation deck: {evaluation.evaluation_id}")
            for row in evaluation.entries:
                deck = deck_by_id.get(row.deck_id)
                if (
                    deck is None
                    or "evaluation" not in deck.roles
                    or deck.content_sha256 != row.exact_deck_sha256
                    or type(row.games) is not int
                    or row.games < 1
                ):
                    raise AssetIntegrityError(f"evaluation deck mismatch: {row.deck_id}")
                evaluation_decks.add(row.deck_id)
                evaluation_games += row.games
        if self.evaluations:
            raise AssetIntegrityError(
                "0044 V1 does not admit legacy FrozenMeta evaluations; periodic strength "
                "evidence uses the separate Benchmark V1 CUDA-2048 contract"
            )
        training_count = sum("training" in deck.roles for deck in self.decks)
        if len(self.decks) != 67 or training_count != 67:
            raise AssetIntegrityError("0044 Training Deck Pool must contain exact decks 001-067")
        return AssetAudit(
            status="PASS",
            deck_count=len(self.decks),
            training_deck_count=training_count,
            evaluation_deck_count=len(evaluation_decks),
            evaluation_games=evaluation_games,
            policy_ids=tuple(sorted(policies)),
            latest_champion_policy_id=self.latest_champion_policy_id,
        )


__all__ = [
    "AssetAudit", "AssetIntegrityError", "AssetRegistry", "canonical_deck_sha256",
    "canonical_json_sha256", "sha256_file",
]
