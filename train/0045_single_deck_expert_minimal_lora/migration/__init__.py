"""Strict checkpoint migration into the 0045 minimal-specialist schema."""

from .from_0044 import MigrationReport, migrate_0044_checkpoint

__all__ = ["MigrationReport", "migrate_0044_checkpoint"]
