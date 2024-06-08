from enum import Enum

class RoundtripSide(Enum):
    """Enumerates the sides of a round-trip."""

    LONG = 'long'
    """The long round-trip."""

    SHORT = 'short'
    """The short round-trip."""
