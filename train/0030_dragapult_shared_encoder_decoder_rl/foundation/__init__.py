"""Frozen, project-local 0028 semantic foundation implementation."""

from .contract import FoundationIdentity, load_foundation, verify_foundation

__all__ = ["FoundationIdentity", "load_foundation", "verify_foundation"]
