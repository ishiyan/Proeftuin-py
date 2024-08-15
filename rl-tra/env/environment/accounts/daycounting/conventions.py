from enum import Enum

# For Excel YEARFRAC function see
# https://support.microsoft.com/en-us/office/yearfrac-function-3844141e-c76d-4143-82b6-208454ddc6a8
#
# Excel YEARFRAC function:
# Basis Optional: The type of day count basis to use.
# 0: US (NASD) 30/360 (default is not set)
# 1: Actual/actual
# 2: Actual/360
# 3: Actual/365
# 4: European 30/360

# Day counting methods are listed in the ISO 20022, see
# https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

class DayCountConvention(Enum):
    RAW = 0
    """
    Take the differencein seconds between two dates and divides
    it by the number of seconds in a Gregorian year (`31556952`).

    This is what has most sense for intraday periods or when
    we are not concerned with the calculation of the interest
    accrual between coupon payment dates.

    Strings: 'raw'
    """

    THIRTY_360_US = 1 # A001 us_30_360 '30/360 us', '30u/360'
    """
    30/360 (ISDA) or 30/360 (American Basic Rule)

    This is `NOT` the same as the `"US (NASD) 30/360"` (basis 0)
    in Excel `YEARFRAC` function.
    
    Use `THIRTY_360_US_EOM` for the closest match.

    Coded as A001 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: '30/360 us', '30u/360'

    Method whereby interest is calculated based on a 30-day month
    and a 360-day year.
    
    Accrued interest to a value date on the last day of a month shall
    be the same as to the 30th calendar day of the same month, except
    for February, and provided that the interest period started on a
    30th or a 31st.
    
    This means that a 31st is assumed to be a 30th if the period started
    on a 30th or a 31st and the 28 Feb (or 29 Feb for a leap year) is
    assumed to be a 28th (or 29th).
    
    It is the most commonly used 30/360 method for US straight and
    convertible bonds.
    """

    THIRTY_360_US_EOM = 2 # us_30_360_eom '30/360 us eom', '30u/360 eom' (almost basis 0)
    """
    30/360 US End-Of-Month

    This is `NOT` the same as the `"US (NASD) 30/360"` (basis 0)
    in Excel `YEARFRAC` function.
    
    Although the results are not completely the same,
    this is the closest match.
    
    This method is not listed in ISO 20022.
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Found it on github (https://github.com/hcnn/d30360m)

    Strings: '30/360 us eom', '30u/360 eom'
    """

    THIRTY_360_US_NASD = 3 # us_30_360_nasd '30/360 us nasd', '30u/360 nasd'
    """
    30/360 NASD

    This is `NOT` the same as the `"US (NASD) 30/360"` (basis 0)
    in Excel `YEARFRAC` function.
    
    Use `THIRTY_360_US_EOM` for the closest match.

    This method is not listed in ISO 20022.
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Found it on github (https://github.com/hcnn/d30360n)

    Strings: '30/360 us nasd', '30u/360 nasd'
    """

    THIRTY_360_EU = 4 # A011 eur_30_360 '30/360 eu' '30e/360' (basis 4)
    """
    30/360 Eurobond Basis or 30/360 ICMA

    This is the same as the `"Eur 30/360"` (basis 4)
    in Excel `YEARFRAC` function.

    Coded as A011 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: '30/360 eu', '30e/360'
    
    Method whereby interest is calculated based on a 30-day month
    and a 360-day year.
    
    Accrued interest to a value date on the last day of a month
    shall be the same as to the 30th calendar day of the same month,
    except for February.
    
    This means that a 31st is assumed to be a 30th and the 28 Feb
    (or 29 Feb for a leap year) is assumed to be a 28th (or 29th).
    
    It is the most commonly used 30/360 method for non-US straight
    and convertible bonds issued before 01/01/1999.
    """

    THIRTY_360_EU_M2 = 5 # A012 eur_30_360_model_2 '30/360 eu2' '30e2/360' COMMENT-OUT
    """
    30E2/360 or Eurobond basis model 2

    This is `NOT` the same as the `"Eur 30/360"` (basis 4)
    in Excel `YEARFRAC` function.
    
    Use `THIRTY_360_EU` if you want excel-compatible results.

    Coded as A012 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: '30/360 eu2', '30e2/360'

    Method whereby interest is calculated based on a 30-day month and
    a 360-day year.
    
    Accrued interest to a value date on the last day of a month shall
    be the same as to the 30th calendar day of the same month, except
    for the last day of February whose day of the month value shall
    be adapted to the value of the first day of the interest period
    if the latter is higher and if the period is one of a regular
    schedule.
    
    This means that a 31st is assumed to be a 30th and the 28th Feb
    of a non-leap year is assumed to be equivalent to a 29th Feb
    when the first day of the interest period is a 29th, or to a 30th
    Feb when the first day of the interest period is a 30th or a 31st.
    
    The 29th Feb of a leap year is assumed to be equivalent to a 30th
    Feb when the first day of the interest period is a 30th or a 31st.

    Similarly, if the coupon period starts on the last day of February,
    it is assumed to produce only one day of interest in February as if
    it was starting on a 30th Feb when the end of the period is a 30th
    or a 31st, or two days of interest in February when the end of the
    period is a 29th, or 3 days of interest in February when it is the
    28th Feb of a non-leap year and the end of the period is before the
    29th.
    """

    THIRTY_360_EU_M3 = 6 # A013 eur_30_360_model_3 '30/360 eu3' '30e3/360' COMMENT-OUT
    """
    30E3/360 or Eurobond basis model 3

    This is `NOT` the same as the `"Eur 30/360"` (basis 4)
    in Excel `YEARFRAC` function.
    
    Use `THIRTY_360_EU` if you want excel-compatible results.

    Coded as A013 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: '30/360 eu3', '30e3/360'
    
    Method whereby interest is calculated based on a 30-day month
    and a 360-day year.
    
    Accrued interest to a value date on the last day of a month
    shall be the same as to the 30th calendar day of the same month.
    
    This means that a 31st is assumed to be a 30th and the 28 Feb
    (or 29 Feb for a leap year) is assumed to be equivalent to a
    30 Feb.
    
    It is a variation of the 30E/360 (or Eurobond basis) method
    where the last day of February is always assumed to be a 30th,
    even if it is the last day of the maturity coupon period.
    """

    THIRTY_360_EU_PLUS = 7 # eur_30_360_plus '30/360 eu+', '30e+/360'
    """
    30E+/360
    
    This is `NOT` the same as the `"Eur 30/360"` (basis 4)
    in Excel `YEARFRAC` function.
    
    Use `THIRTY_360_EU` if you want excel-compatible results.

    This method is not listed in ISO 20022.
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Found it on github (https://github.com/hcnn/d30360p)

    Strings: '30/360 eu+', '30e+/360'
    """

    THIRTY_365 = 8 # A002 thirty_365 '30/365'
    """
    30/365

    There is no related basis in Excel `YEARFRAC` function.

    Coded as A002 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: '30/365'

    Method whereby interest is calculated based on a 30-day month
    in a way similar to the 30/360 (basic rule) and a 365-day year.

    Accrued interest to a value date on the last day of a month shall
    be the same as to the 30th calendar day of the same month, except
    for February.
    
    This means that a 31st is assumed to be a 30th and the 28 Feb (or
    29 Feb for a leap year) is assumed to be a 28th (or 29th).
    """

    ACT_360 = 9 # A004 act_360 'act/360' (basis 2)
    """
    Actual/360

    This is the same as the `"Actual/360"` (basis 2)
    in Excel `YEARFRAC` function.

    Coded as A004 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: 'act/360'

    Method whereby interest is calculated based on the actual
    number of accrued days in the interest period and a 360-day year.
    """

    ACT_365_FIXED = 10 # A005 act_365_fixed 'act/365 fixed' (basis 3)
    """
    Actual/365 Fixed

    This is the same as the `"Actual/365"` (basis 3)
    in Excel `YEARFRAC` function.

    Coded as A005 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: 'act/365 fixed'

    Method whereby interest is calculated based on the actual
    number of accrued days in the interest period and a 365-day year.
    """

    ACT_365_NONLEAP = 11 # A014 act_365_nonleap 'act/365 nonleap' COMMENT-OUT
    """
    Actual/365 Non-Leap

    This is `NOT` the same as the `"Actual/365"` (basis 3)
    in Excel `YEARFRAC` function.
    
    Use `ACT_365_FIXED` if you want excel-compatible results.

    Coded as A014 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: 'act/365 nonleap'

    Method whereby interest is calculated based on the actual
    number of accrued days in the interest period, excluding
    any leap day from the count, and a 365-day year.
    """

    ACT_ACT_EXCEL = 12 # act_act_excel 'act/act excel' (basis 1)
    """
    Excel-compatible Actual/Actual (basis 1) method.

    This method is not listed in ISO 20022.
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Other actual/actual methods from ISO 20022 produce
    different figures compared to Excel.

    Found it on github (https://github.com/AnatolyBuga/yearfrac)
    and verified it with Excel.

    Strings: 'act/act excel'
    """

    ACT_ACT_ISDA = 13 # A008 act_act_isda 'act/act isda'
    """
    Actual/Actual ISDA or Actual/365 ISDA

    This is `NOT` the same as the `"Actual/Actual"` (basis 1)
    in Excel `YEARFRAC` function.
    
    Use `ACT_ACT_EXCEL` if you want excel-compatible results.

    Coded as A008 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: 'act/act isda', 'act/365 isda'
    
    Method whereby interest is calculated based on the actual number
    of accrued days of the interest period that fall on a normal year,
    divided by 365, added to the actual number of days of the interest
    period that fall on a leap year, divided by 366.
    """

    ACT_ACT_AFB = 14 # A010 act_act_afb 'act/act afb' COMMENT-OUT
    """
    Actual/Actual AFB

    This is `NOT` the same as the `"Actual/Actual"` (basis 1)
    in Excel `YEARFRAC` function.
    
    Use `ACT_ACT_EXCEL` if you want excel-compatible results.

    Coded as A010 in ISO 20022
    (https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm)

    Strings: 'act/act afb'

    Method whereby interest is calculated based on the actual
    number of accrued days and a 366-day year (if 29 Feb falls
    in the coupon period) or a 365-day year (if 29 Feb does not
    fall in the coupon period).
    
    If a coupon period is longer than one year, it is split by
    repetitively separating full year sub-periods counting backwards
    from the end of the coupon period (a year backwards from a 28 Feb
    being 29 Feb, if it exists).
    
    The first of the sub-periods starts on the start date of the
    accrued interest period and thus is possibly shorter than a year.
    
    Then the interest computation is operated separately on each
    sub-period and the intermediate results are summed up.
    """

    @classmethod
    def from_string(cls, convention: str) -> 'DayCountConvention':
        convention_map = {
            'raw': DayCountConvention.RAW,
            '30/360 us': DayCountConvention.THIRTY_360_US,
            '30/360 us eom': DayCountConvention.THIRTY_360_US_EOM,
            '30/360 us nasd': DayCountConvention.THIRTY_360_US_NASD,
            '30/360 eu': DayCountConvention.THIRTY_360_EU,
            '30/360 eu2': DayCountConvention.THIRTY_360_EU_M2,
            '30/360 eu3': DayCountConvention.THIRTY_360_EU_M3,
            '30/360 eu+': DayCountConvention.THIRTY_360_EU_PLUS,
            '30/365': DayCountConvention.THIRTY_365,
            'act/360': DayCountConvention.ACT_360,
            'act/365 fixed': DayCountConvention.ACT_365_FIXED,
            'act/365 nonleap': DayCountConvention.ACT_365_NONLEAP,
            'act/act excel': DayCountConvention.ACT_ACT_EXCEL,
            'act/act isda': DayCountConvention.ACT_ACT_ISDA,
            'act/act afb': DayCountConvention.ACT_ACT_AFB,
        }
        if isinstance(convention, str):
            convention = convention.lower()
        if not isinstance(convention, str) or \
            (convention not in convention_map.keys()):
            raise ValueError(f"Day count convention '{convention}' " \
                f'must be one of: {set(convention_map.keys())}')
        return convention_map[convention]
