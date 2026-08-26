"""Exceptions with dedicated command-line error messages."""


class NoMatchesError(Exception):
    """Raised when OpenLigaDB returns an empty match list."""
