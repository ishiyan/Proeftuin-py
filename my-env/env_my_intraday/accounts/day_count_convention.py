from enum import Enum
from typing import Union, Any
from numbers import Real
from datetime import date, datetime, timedelta

class DayCountConvention(Enum):
    RAW = 'raw'
    ACTUAL_365 = 'actual/365' # 'actual/365', 'actual_365', 'actual365'
    ACTUAL_360 = 'actual/360' # 'actual/360', 'actual_360', 'actual360'
    THIRTY_360 = '30/360' # '30/360', '30_360'

SECONDS_IN_GREGORIAN_YEAR = 31556952

def duration(date1: Union[Real, datetime, date, Any],
             date2: Union[Real, datetime, date, Any],
             convention: DayCountConvention) -> float:
    if (date1 is None) or (date2 is None):
        return 0
    delta = date1 - date2

    if convention == DayCountConvention.ACTUAL_365:
        if isinstance(delta, timedelta):
            return abs(delta.days / 365)

    elif convention == DayCountConvention.ACTUAL_360:
        if isinstance(delta, timedelta):
            return abs(delta.days / 360)

    elif convention == DayCountConvention.THIRTY_360:
        assert hasattr(date1, 'year') and hasattr(date1, 'month') and hasattr(date1, 'day')
        assert hasattr(date2, 'year') and hasattr(date2, 'month') and hasattr(date2, 'day')
        df = (min(date2.day, 30) + max(0, (30 - date1.day))) / 360
        mf = (date2.month - date1.month - 1) / 12
        yf = (date2.year - date1.year)
        return abs(df + mf + yf)

    else: # RAW
        if isinstance(delta, Real):
            pass
        elif isinstance(delta, timedelta):
            delta = delta.total_seconds()
        else:
            raise ValueError('Invalid date1 or date2')
        return abs(delta) / SECONDS_IN_GREGORIAN_YEAR
