"""Exceptions shared across the package."""

from __future__ import annotations


class ConfigError(ValueError):
    """Invalid configuration; the message names the dotted path of the offending key."""
