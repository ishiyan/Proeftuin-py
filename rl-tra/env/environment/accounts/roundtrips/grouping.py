from enum import Enum

class RoundtripGrouping(Enum):
    """Enumerates algorithms used to group order executions into round-trips."""

    FILL_TO_FILL = 'fill_to_fill'
    """
    The round-trip defined by
    - (1) an order execution that establishes or increases a position and
    - (2) an offsetting execution that reduces the position size.
    """

    FLAT_TO_FLAT = 'flat_to_flat'
    """
    The round-trip defined by a sequence of order executions, from a flat
    position to a non-zero position which may increase or decrease in
    quantity, and back to a flat position.
    """

    FLAT_TO_REDUCED = 'flat_to_reduced'
    """
    The round-trip defined by a sequence of order executions, from a flat
    position to a non-zero position and an offsetting execution that
    reduces the position size.
    """
