from enum import Enum

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
