from ..exchanges import MIC
from ..currencies import Currency, Currencies
from .status import InstrumentStatus
from .type import InstrumentType

class Instrument:
    """Contains properties of a financial instrument.

    Parameters
    ----------
    symbol : str
        The symbol (ticker) is a mnemonic of the instrument.
    name : str, optional
        Theshort name of the instrument.
    ISIN : str, optional
        The ISO6166 (International Securities Identifying Number) code of the instrument.
    CFI : str, optional
        The ISO 10962 (Classification of Financial Instruments) code of the instrument.
    MIC : MIC, optional
        The ISO 10383 Market Identifier Code where the instrument is traded.
    currency : Currency, optional
        The currency code which the price of the instrument is denominated.
    instrument_type : InstrumentType, optional
        The type of the instrument.
    status : InstrumentStatus, optional
        The status of the instrument.
    price_decimal_places : int, optional
        The number of decimal places in the price of the instrument.
    price_min_increment : float, optional
        The minimum price increment of the instrument.
    price_factor : float, optional
        The positive multiplier by which price must be adjusted to determine
        the true nominal value of the instrument.

        Nominal Value = Quantity * Price * PriceFactor.
    initial_margin : float, optional
        The initial margin required to open a position in the instrument.
    """

    def __init__(self,
                 symbol: str,
                 name: str = '',
                 ISIN: str = None,
                 CFI: str = None,
                 MIC: MIC = None,
                 currency: Currency = Currencies.USD,
                 instrument_type: InstrumentType = InstrumentType.STOCK,
                 status: InstrumentStatus = InstrumentStatus.ACTIVE,
                 price_decimal_places: int = 2,
                 price_min_increment: float = 0.01,
                 price_factor: float = 1,
                 initial_margin: float = 0,
                 ):
        self.symbol = symbol
        self.name = name
        self.ISIN = ISIN
        self.CFI = CFI
        self.MIC = MIC
        self.currency = currency
        self.status = status
        self.instrument_type = instrument_type
        self.price_decimal_places = price_decimal_places
        self.price_min_increment = price_min_increment
        self.price_factor = price_factor
        self.initial_margin = initial_margin

    def __str__(self):
        if self.ISIN is not None and self.MIC is not None:
            return f'{self.symbol} ({self.ISIN}, {self.MIC})'
        return f'{self.symbol}'

    def __repr__(self):
        attributes = ['symbol', 'name', 'ISIN', 'CFI', 'MIC', 'currency', 'status', 
            'instrument_type', 'price_decimal_places', 'price_min_increment', 
            'price_factor', 'initial_margin']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        #attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes if getattr(self, attr) is not None]
        return 'Instrument(' + ', '.join(attr_strings) + ')'
