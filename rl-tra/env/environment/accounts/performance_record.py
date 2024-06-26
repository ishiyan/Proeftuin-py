from collections import namedtuple

PerformanceRecord = namedtuple(
    'Record',
    'operation amount enter_date enter_price exit_date exit_price result commission notes',
    defaults=(None,) * 9
)
