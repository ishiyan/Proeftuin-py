from datetime import datetime
from typing import List, Tuple

from ..currencies import Currency, CurrencyConverter
from ..utils import ScalarSeries
from ..orders import OrderExecution
from .action import AccountAction
from .transaction import AccountTransaction

class Account:
    """A single-entry account holding transactions in home currency.

    Transactions in a foreign currency are converted to the home currency
    without explicit commission.

    Parameters
    ----------
    holder : str
        Account holder name.
    currency : Currency
        Home currency.
    """
    def __init__(self,
                 holder: str,
                 currency: Currency,
                 currency_converter: CurrencyConverter,
                 ):
        self.holder = holder
        self.currency = currency
        self._currency_converter = currency_converter
        self._balance: ScalarSeries = ScalarSeries()
        self._transactions: List[AccountTransaction] = []

    def __str__(self):
        return f'{self.holder} {self.currency}'

    def balance(self) -> float:
        """Returns the current balance as the total of
        all transactions expressed in the home currency.
        """
        return self._balance.current_value()
    
    def balance_history(self) -> List[Tuple[datetime, float]]:
        """Returns the balance history as a time series of the total of
        all transactions expressed in the home currency.
        """
        return self._balance.history()

    def transaction_history(self) -> List[AccountTransaction]:
        """Returns the list of all transactions."""
        return self._transactions.copy()
    
    def add(self, time: datetime, amount: float, currency: Currency, note: str = ''):
        """Deposits or withdraws an amount of money in the specified currency.
        
        The negative amount means debit withdrawal, and the positive amount
        means credit deposit.

        It does not check if the balance becomes negative.

        The amount will be converted into the home currency
        if the indicated currency differs from the home one.

        Parameters
        ----------
        time : datetime
            Transaction time.
        amount : float
            Transaction amount.
        currency : Currency
            Transaction currency.
        note : str, optional
            Transaction note.
        """
        if amount < 0:
            amount = -amount
            action = AccountAction.DEBIT
        elif amount > 0:
            action = AccountAction.CREDIT
        else:
            return

        conv, rate = (self._currency_converter.convert(amount,
            currency, self.currency) if currency != self.currency else (amount, 1.0))
        
        transaction = AccountTransaction(action, time,
            currency, amount, rate, conv, note)

        self._balance.accumulate(time, amount
            if action == AccountAction.CREDIT else -amount)
        self._transactions.append(transaction)

    def execute(self, execution: OrderExecution):
        """Deposits or withdraws an amount of money associated with the order execution.

        The execution amount is converted into the home currency
        if the execution currency differs from the home one.

        Parameters
        ----------
        execution : OrderExecution
            Order execution.
        """
        amount = execution.cash_flow + execution.debt
        if amount < 0:
            amount = -amount
            action = AccountAction.DEBIT
        elif amount > 0:
            action = AccountAction.CREDIT
        else:
            return

        currency = execution.currency
        conv, rate = (self._currency_converter.convert(amount, currency,
            self.currency) if currency != self.currency else (amount, 1.0))

        transaction = AccountTransaction(action, execution.report_time,
            currency, amount, rate, conv, 'order execution')
        
        if action == AccountAction.DEBIT:
            conv = -conv
        self._balance.accumulate(execution.report_time, conv)
        self._transactions.append(transaction)

        if execution.commission == 0.0:
            return

        currency = execution.commission_currency
        amount = execution.commission
        conv, rate = (self._currency_converter.convert(amount, currency,
            self.currency) if currency != self.currency else (amount, 1.0))

        transaction = AccountTransaction(AccountAction.DEBIT, execution.report_time,
            currency, amount, rate, conv, 'order execution commission')
        
        self._balance.accumulate(execution.report_time, -conv)
        self._transactions.append(transaction)
