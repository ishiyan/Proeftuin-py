from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple
from collections import defaultdict
from threading import RLock

from currencies import Currency

class CurrencyConverter(ABC):
    """Converts base currency units to term currency units."""

    @abstractmethod
    def convert(self, amount: float, base: Currency, term: Currency) -> Tuple[float, float]:
        """
        Converts the amount in the base currency to the converted amount in the term currency.

        If either base or term currency is unknown, the converted amount is zero.

        If both currencies are the same, the amounts are also the same.
        """

    @abstractmethod
    def exchange_rate(self, base: Currency, term: Currency) -> float:
        """Returns a direct exchange rate from the base currency to the term currency.

        X units of the base currency are equal to the X * exchange_rate units of the term currency.

        If USD is the base currency and EUR is the term currency, then the currency pair USDEUR
        gives the required rate.

        If either base or term currencies are unknown, the exchange rate is zero.
        If both currencies are the same, the exchange rate is 1.
        """

    @abstractmethod
    def known_base_currencies(self, term: Currency) -> List[Currency]:
        """
        For a given term currency, returns a collection of base currencies with known exchange rates.
        """

    @abstractmethod
    def known_term_currencies(self, base: Currency) -> List[Currency]:
        """
        For a given base currency, returns a collection of term currencies with known exchange rates.
        """

class UpdatableCurrencyConverter(CurrencyConverter):
    def __init__(self):
        self.known_rates = defaultdict(dict)
        self.lock = RLock()

    def convert(self, amount: float, base: Currency, term: Currency) -> Tuple[float, float]:
        rate = self.exchange_rate(base, term)
        converted = amount * rate
        return converted, rate

    def exchange_rate(self, base: Currency, term: Currency) -> float:
        if base == term:
            return 1

        with self.lock:
            return self.knownRates.get(base, {}).get(term, 0)

    def known_base_currencies(self, term: Currency) -> List[Currency]:
        with self.lock:
            return [bc for bc, m in self.knownRates.items() if term in m]

    def known_term_currencies(self, base: Currency) -> List[Currency]:
        with self.lock:
            return list(self.knownRates.get(base, {}).keys())

    def update(self, base: Currency, term: Currency, rate: float):
        if base == term:
            return

        with self.lock:
            if base in self.knownRates:
                self.knownRates[base][term] = rate
            else:
                self.knownRates[base] = {term: rate}
