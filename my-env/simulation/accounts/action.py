from enum import Enum

class AccountAction(Enum):
    """Enumerates actions performed on account."""

    CREDIT = 'credit'
    """An action to deposit money to an account."""

    DEBIT = 'debit'
    """An action to withdraw money from an account."""
