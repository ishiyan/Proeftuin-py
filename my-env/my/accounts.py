from datetime import datetime
from enum import Enum

from mics import MIC
from currencies import Currency, Currencies
from .currency_converters import CurrencyConverter

class AccountAction(Enum):
    """Enumerates actions performed on account."""

    CREDIT = 'credit'
    """An action to deposit money to an account."""

    DEBIT = 'debit'
    """An action to withdraw money from an account."""


class AccountTransaction:
    """An immutable account transaction.

    Parameters
    ----------
    action : AccountAction, optional
        The action (deposit or a withdrawal) performed on the account,
        by default AccountAction.CREDIT.
    time : datetime, optional
        The time of the transaction, by default datetime.now().
    currency : Currency, optional
        The currency of the transaction, by default Currencies.USD.
    amount : float, optional
        The unsigned amount in of this transaction in the given currency.
        The sign is determided by the action.
    conversion_rate : float, optional
        The exchange conversion rate from the transaction currency to the account currency,
        by default 1.
    amount_converted : float, optional
        The unsigned amount converted to the account currency.
    note : str, optional
        A free-text note or a description of the transaction, by default ''.
    """

    def __init__(self,
                 action: AccountAction = AccountAction.CREDIT,
                 time: datetime = datetime.now(),
                 currency: Currency = Currencies.USD,
                 amount: float = 0,
                 conversion_rate: float = 1,
                 amount_converted: float = 0,
                 note: str = '',
                 ):
        super().__setattr__('action', action)
        super().__setattr__('time', time)
        super().__setattr__('currency', currency)
        super().__setattr__('amount', amount)
        super().__setattr__('conversion_rate', conversion_rate)
        super().__setattr__('amount_converted', amount_converted)
        super().__setattr__('note', note)
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
        return 'Instrument(' + ', '.join(attr_strings) + ')'
