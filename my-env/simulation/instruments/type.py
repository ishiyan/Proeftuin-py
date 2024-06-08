from enum import Enum

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
