from collections import namedtuple

Trade = namedtuple('Trade', 'datetime operation amount price')
"""Information about a simple trade"""

TradeOI = namedtuple('TradeOI', 'datetime operation amount price open_interest')
"""Information about a trade with additional field open_interest, used in futures markets"""

Candle = namedtuple('Candle', 'time_start time_end open high low close volume')
"""Aggregated information of price movement and volume traded over some period of time"""

Kline = namedtuple('Kline', 'time_start time_end open high low close volume money buy_volume buy_money')
"""Aggregated information of price movement and volume traded over some period of time,
including additional information about at which prices the most buys and sells were concentrated"""
