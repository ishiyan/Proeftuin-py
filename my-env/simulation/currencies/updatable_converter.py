from typing import List, Tuple
from collections import defaultdict

from .currency import Currency
from .converter import CurrencyConverter

class UpdatableCurrencyConverter(CurrencyConverter):
    def __init__(self):
        self.known_rates = defaultdict(dict)

    def convert(self, amount: float, base: Currency, term: Currency) -> Tuple[float, float]:
        rate = self.exchange_rate(base, term)
        converted = amount * rate
        return converted, rate

    def exchange_rate(self, base: Currency, term: Currency) -> float:
        if base == term:
            return 1

        rate = self.known_rates.get(base, {}).get(term, 0)
        if rate == 0:
            raise ValueError(f"Exchange rate not found for base {base} and term {term}")
        return rate

    def known_base_currencies(self, term: Currency) -> List[Currency]:
        return [bc for bc, m in self.known_rates.items() if term in m]

    def known_term_currencies(self, base: Currency) -> List[Currency]:
        return list(self.known_rates.get(base, {}).keys())

    def update(self, base: Currency, term: Currency, rate: float):
        if base == term:
            return

        if base in self.known_rates:
            self.known_rates[base][term] = rate
        else:
            self.known_rates[base] = {term: rate}
