from enum import Enum

class RoundtripMatching(Enum):
    """Enumerates algorithms used to match the offsetting order executions in a round-trip."""

    FIFO = 'fifo'
    """The offsetting order executions will be matched in FIFO (First In First Out) order."""

    LIFO = 'lifo'
    """The offsetting order executions will be matched in LIFO (Last In First Out) order."""
