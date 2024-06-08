from datetime import datetime

from ..currencies import Currency
from .action import AccountAction

class AccountTransaction:
    """An immutable account transaction.

    Parameters
    ----------
    action : AccountAction, optional
        The action (deposit or a withdrawal) performed on the account.
    time : datetime, optional
        The time of the transaction.
    currency : Currency, optional
        The currency of the transaction.
    amount : float, optional
        The unsigned amount in of this transaction in the given currency.
        The sign is determided by the action.
    conversion_rate : float, optional
        The exchange conversion rate from the transaction currency to the account currency.
    amount_converted : float, optional
        The unsigned amount converted to the account currency.
    note : str, optional
        A free-text note or a description of the transaction.
    """

    def __init__(self,
                 action: AccountAction,
                 time: datetime,
                 currency: Currency,
                 amount: float,
                 conversion_rate: float,
                 amount_converted: float,
                 note: str,
                 ):
        self.action = action
        self.time = time
        self.currency = currency
        self.amount = amount
        self.conversion_rate = conversion_rate
        self.amount_converted = amount_converted
        self.note = note
        super().__setattr__('_is_frozen', True)

    def __setattr__(self, name, value):
        if getattr(self, '_is_frozen', False):
            raise TypeError(f"Can't modify immutable instance")
        super().__setattr__(name, value)

    def __str__(self):
        return f'{self.action} {self.currency} {self.amount} ({self.note}))'

    def __repr__(self):
        attributes = ['action', 'time', 'currency', 'amount',
            'conversion_rate', 'amount_converted', 'note']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        return 'AccountTransaction(' + ', '.join(attr_strings) + ')'
