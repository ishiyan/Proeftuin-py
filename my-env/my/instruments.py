from enum import Enum

from mics import MIC
from currencies import Currency, Currencies

class InstrumentStatus(Enum):
    """Enumerates the states of an instrument."""

    ACTIVE = 'active'
    """Indicates the trading is possible."""

    ACTIVE_CLOSING_ORDERS_ONLY = 'active_closing_orders_only'
    """Indicates the trading is possible but only closing orders are allowed."""

    INACTIVE = 'inactive'
    """
    Indicates an instrument has previously been active and is now no longer traded
    but has not expired yet. It may become active again.
    """

    SUSPENDED = 'suspended'
    """ Indicates an instrument has been temporarily disabled for trading."""

    PENDING_EXPIRY = 'pending_expiry'
    """
    Indicates an instrument is currently still active but will expire after the
    current business day.

    For example, a contract that expires intraday (e.g. at noon time) and is no
    longer traded but will still show up in the current day's order book with
    related statistics.
    """

    EXPIRED = 'expired'
    """
    Indicates an instrument has been expired due to reaching maturity or based
    on contract definitions or exchange rules.
    """

    PENDING_DELETION = 'pending_deletion'
    """
    Indicates an instrument is awaiting deletion from security reference data.
    """

    DELISTED = 'delisted'
    """
    Indicates an instrument has been removed from securities reference data.

    A delisted instrument would not trade on the exchange but it may still be traded
    over-the-counter.

    Delisting rules vary from exchange to exchange, which may include non-compliance
    of capitalization, revenue, consecutive minimum closing price.
    The instrument may become listed again once the instrument is back in compliance.
    """

    KNOCKED_OUT = 'knocked_out'
    """
    Indicates an instrument has breached a predefined price threshold.
    """

    KNOCK_OUT_REVOKED = 'knock_out_revoked'
    """
    Indicates an instrument reinstated, i.e. threshold has not been breached.
    """

class InstrumentType(Enum):
    """Enumerates types of an instrument."""

    STOCK = 'stock'
    """
    A security that denote an ownership in a public company.
    """

    INDEX = 'index'
    """
    Tracks the performance of a group of assets in a standardized way.
    
    Indexes typically measure the performance of a basket of securities
    intended to replicate a certain area of the market.
    """

    INAV = 'inav'
    """
    An intraday indicative net asset value of an ETF or ETV
    based on the market values of its underlying constituents.
    """

    ETF = 'etf'
    """
    An exchange traded fund, a security that tracks a basket of assets.
    """

    ETC = 'etc'
    """
    An exchange traded commodity, a security thet tracks the price of
    a commodity or a commodity bucket.
    """

    FOREX = 'forex'
    """A currency instrument."""

    CRYPTO = 'crypto'
    """A crypto currency instrument."""

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
