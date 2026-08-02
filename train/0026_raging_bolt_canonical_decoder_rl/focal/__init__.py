"""Self-contained canonical focal policy frozen from project 0025."""

from .contract import FocalIdentity, load_focal_actor, verify_focal_checkpoint

__all__ = ["FocalIdentity", "load_focal_actor", "verify_focal_checkpoint"]
