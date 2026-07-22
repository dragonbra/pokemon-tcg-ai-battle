"""Kaggle-compatible entrypoint alias.

The competition examples usually load ``main.agent``.  This file keeps a
second conventional entrypoint available for local import-isolation tests.
"""

from main import agent

__all__ = ["agent"]

