
_currencies = {}

class Currency:
    """Represents a currency.

    Parameters
    ----------
    symbol : str
        The ISO 4217 three-letter alphabetic code for the representation of currencies.
    precision : int
        The number of decimal places to which the currency is traded
        (e.g. BTC=8, AAPL=1).
    display : str, optional
        The symbol to display the currency with (e.g. €, $, £, ¢).
    name : str, optional
        The name of the currency.
    """

    def __init__(self, symbol: str, precision: int = 2, display: str = '', name: str = None) -> None:
        self.symbol = symbol
        self.precision = precision
        self.display = display
        self.name = name
        if symbol in _currencies:
            raise ValueError(f'Currency code {symbol} is already defined')
        _currencies[symbol] = self
    
    def __eq__(self, other) -> bool:
        """Checks if two currencies are equal.

        Parameters
        ----------
        other : `Any`
            The currency being compared.

        Returns
        -------
        bool
            Whether the currencies are equal.
        """
        if not isinstance(other, Currency):
            return False
        if self.symbol != other.symbol:
            return False
        if self.precision != other.precision:
            return False
        if self.display != other.display:
            return False
        return True

    def __ne__(self, other) -> bool:
        """Checks if two currencies are not equal.

        Parameters
        ----------
        other : `Any`
            The instrument being compared.

        Returns
        -------
        bool
            Whether the currencies are not equal.
        """
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.symbol)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        attributes = ['symbol', 'precision', 'display', 'name']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        return 'Instrument(' + ', '.join(attr_strings) + ')'
