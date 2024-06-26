from collections import namedtuple

MarketOrder = namedtuple(
    'MarketOrder',
    'account operation amount time_init time_kill',
    defaults=(None,)*5
)

LimitOrder = namedtuple(
    'LimitOrder',
    'account operation amount time_init time_kill price',
    defaults=(None,)*6
)

StopOrder = namedtuple(
    'StopOrder',
    'account operation amount time_init time_kill price',
    defaults=(None,)*6
)

TrailingStopOrder = namedtuple(
    'TrailingStopOrder',
    'account operation amount time_init time_kill trail_delta best_price',
    defaults=(None,)*7
)

TakeProfitOrder = namedtuple(
    'TakeProfitOrder',
    'account operation amount time_init time_kill target_price trail_delta best_price',
    defaults=(None,)*8
)
