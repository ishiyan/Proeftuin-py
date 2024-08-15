from enum import Enum

class Periodicity(Enum):
    """Enumerates periodicity of the performance."""

    DAILY = 0
    """The daily periodicity."""

    WEEKLY = 1
    """The weekly periodicity."""

    MONTHLY = 2
    """The monthly periodicity."""

    QUARTERLY = 3
    """The quarterly periodicity."""

    ANNUAL = 4
    """The annual periodicity."""
