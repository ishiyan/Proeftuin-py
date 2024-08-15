from datetime import datetime

from .conventions import DayCountConvention
from .daycounting import us_30_360, us_30_360_nasd, eur_30_360_plus, eur_30_360
from .daycounting import us_30_360_eom, eur_30_360_model_2, eur_30_360_model_3
from .daycounting import act_365_fixed, act_360, act_act_isda, act_act_afb
from .daycounting import act_act_excel, thirty_365, act_365_nonleap

_convention_map = {
    DayCountConvention.THIRTY_360_US: us_30_360,
    DayCountConvention.THIRTY_360_US_EOM: us_30_360_eom,
    DayCountConvention.THIRTY_360_US_NASD: us_30_360_nasd,
    DayCountConvention.THIRTY_360_EU: eur_30_360,
    DayCountConvention.THIRTY_360_EU_M2: eur_30_360_model_2,
    DayCountConvention.THIRTY_360_EU_M3: eur_30_360_model_3,
    DayCountConvention.THIRTY_360_EU_PLUS: eur_30_360_plus,
    DayCountConvention.THIRTY_365: thirty_365,
    DayCountConvention.ACT_360: act_360,
    DayCountConvention.ACT_365_FIXED: act_365_fixed,
    DayCountConvention.ACT_365_NONLEAP: act_365_nonleap,
    DayCountConvention.ACT_ACT_EXCEL: act_act_excel,
    DayCountConvention.ACT_ACT_ISDA: act_act_isda,
    DayCountConvention.ACT_ACT_AFB: act_act_afb,
}
_convention_map_keys = _convention_map.keys()

_SECONDS_IN_GREGORIAN_YEAR = 31_556_952
_SECONDS_IN_DAY = 60*60*24

def frac(date_time_1: datetime, date_time_2: datetime,
    method: DayCountConvention, day_frac: bool) -> float:
    if date_time_1 > date_time_2:
        date_time_1, date_time_2 = date_time_2, date_time_1

    if method == DayCountConvention.RAW:
        return (date_time_2 - date_time_1).total_seconds() / \
            (_SECONDS_IN_DAY if day_frac else _SECONDS_IN_GREGORIAN_YEAR)

    dt1 = date_time_1.date()
    dt2 = date_time_2.date()

    tm1 = date_time_1.time()
    tm2 = date_time_2.time()
    # Time as a fraction of the day
    tm1 = (tm1.hour * 3600 + tm1.minute * 60 +  tm1.second) / 86400.
    tm2 = (tm2.hour * 3600 + tm2.minute * 60 +  tm2.second) / 86400.

    y1, m1, d1 = dt1.year, dt1.month, dt1.day
    y2, m2, d2 = dt2.year, dt2.month, dt2.day

    if method not in _convention_map_keys:
        raise ValueError(f"Day count convention {method} " \
                f'must be one of: {set(_convention_map_keys)}')
    return _convention_map[method](y1, m1, d1, y2, m2, d2, tm1, tm2,
                                   day_frac=day_frac)

def year_frac(date_time_1: datetime, date_time_2: datetime,
              method: DayCountConvention = DayCountConvention.RAW) -> float:
    return frac(date_time_1, date_time_2, method, False)

def day_frac(date_time_1: datetime, date_time_2: datetime,
              method: DayCountConvention = DayCountConvention.RAW) -> float:
    return frac(date_time_1, date_time_2, method, True)
