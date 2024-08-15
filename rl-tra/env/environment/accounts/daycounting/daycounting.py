from datetime import date

# Wikipedis
# https://en.wikipedia.org/wiki/Day_count_convention

# ISDA 2006 Definitions, Section 4.16 page 11
# https://web.archive.org/web/20140913145444/http://www.hsbcnet.com/gbm/attachments/standalone/2006-isda-definitions.pdf

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

# Source code
# https://github.com/devind-team/devind_yearfrac
# https://github.com/hcnn/d30360s
# https://github.com/hcnn/d30360e2
# https://github.com/hcnn/d30360e3
# https://github.com/hcnn/d30360p
# https://github.com/hcnn/d30360u
# https://github.com/hcnn/d30360m
# https://github.com/hcnn/d30360n
# https://github.com/hcnn/d30365
# https://github.com/hcnn/act365n
# https://github.com/hcnn/act365f
# https://github.com/hcnn/act360
# https://github.com/hcnn/act_isda
# https://github.com/hcnn/act_afb
# https://github.com/AnatolyBuga/yearfrac

def is_leap_year(y: int) -> bool:
    return not(y % 4) and (bool(y % 100) or not(y % 400))

def date_to_jd(year: int, month: int, day: int) -> int:
    a = int((14 - month) / 12.)
    y = int(year + 4800 - a)
    m = int(month + (12 * a) - 3)

    jd = day + int(((153 * m) + 2) / 5.0) + (y * 365)
    jd += int(y / 4.) - int(y / 100.) + int(y / 400.) - 32045
    return jd

def jd_to_date(jd: int) -> tuple[int, int, int]:
    a = jd + 32044
    b = int(((4 * a) + 3) / 146097.)
    c = a - int((b * 146097) / 4.)

    d = int(((4 * c) + 3) / 1461.)
    e = c - int((d * 1461) / 4.)
    m = int(((5 * e) + 2) / 153.)
    m2 = int(m / 10)

    day = e + 1 - int(((153 * m) + 2) / 5.)
    month = (m + 3 - (12 * m2))
    year = ((b * 100) + d - 4800 + m2)

    return year, month, day

def eur_30_360(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30360s
    Synonyms:
        - 30/360 ICMA
        - 30/360 Eurobond Basis
        - ISDA-2006
        - 30S/360 Special German
    
    ISO 20022:
        A011
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

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
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    diff_days += 30 if d2 > 30 else d2        
    diff_days -= 30 if d1 > 30 else d1
    return diff_days if frac_days else diff_days / 360.

def eur_30_360_model_2(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30360e2
    Synonyms:
        - 30E2/360
        - Eurobond basis model 2
    
    ISO 20022:
        A012
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

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
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    leap1 = is_leap_year(y1)
    if leap1 and (m2 == 2) and (d2 == 28):
        diff_days += 29 if d1 == 29 else (30 if d1 >= 30 else d2)
    elif leap1 and (m2 == 2) and (d2 == 29):
        diff_days += 30 if d1 >= 30 else d2
    else:
        diff_days += 30 if d2 > 30 else d2
    diff_days -= 30 if d1 > 30 else d1
    return diff_days if frac_days else diff_days / 360.

def eur_30_360_model_3(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30360e3
    Synonyms:
        - 30E3/360
        - Eurobond basis model 3
    
    ISO 20022:
        A013
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

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
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    if (m2 == 2) and (d2 >= 28):
        diff_days += 30
    else:
        diff_days += 30 if d2 > 30 else d2
    if (m1 == 2) and (d1 >= 28):
        diff_days -= 30
    else:
        diff_days -= 30 if d1 > 30 else d1
    return diff_days if frac_days else diff_days / 360.

def eur_30_360_plus(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30360p
    Synonyms:
        - 30E+/360
    """
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    diff_days += 32 if d2 == 31 else d2        
    diff_days -= 30 if d1 > 30 else d1
    return diff_days if frac_days else diff_days / 360.

def us_30_360(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30360u
    Synonyms:
        - 30/360 ISDA
        - 30U/360
        - 30/360 US
        - 30/360 Bond Basis
        - 30/360 U.S. Municipal
        - American Basic Rule
    
    ISO 20022:
        A001
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

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
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    if (d2 == 31) and (d1 >= 30):
        diff_days += 30
    else:
        diff_days += d2
    diff_days -= 30 if d1 > 30 else d1
    return diff_days if frac_days else diff_days / 360.

def us_30_360_eom(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30360m
    Synonyms:
        - 30/360 US EOM
    """
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    rule2 = (m1 == 2) and (d1 >= 28)
    rule3 = rule2 and (m2 == 2) and (d2 >= 28)
    rule4 = (d2 == 31) and (d1 >= 30)
    if rule2:
        diff_days -= 30
    else:
        diff_days -= 30 if d1 > 30 else d1    
    if rule4 or rule3:
        diff_days += 30
    else:
        diff_days += d2
    return diff_days if frac_days else diff_days / 360.

def us_30_360_nasd(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30360n
    Synonyms:
        - 30/360 NASD
    """
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    if d2 == 31:
        diff_days += 32 if d1 < 30 else 30        
    else:
        diff_days += d2
    diff_days -= 30 if d1 > 30 else d1
    return diff_days if frac_days else diff_days / 360.

def thirty_365(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/d30365
    Synonyms:
        - 30/365
    
    ISO 20022:
        A002
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

    Method whereby interest is calculated based on a 30-day month
    in a way similar to the 30/360 (basic rule) and a 365-day year.

    Accrued interest to a value date on the last day of a month shall
    be the same as to the 30th calendar day of the same month, except
    for February.
    
    This means that a 31st is assumed to be a 30th and the 28 Feb (or
    29 Feb for a leap year) is assumed to be a 28th (or 29th).
    """
    diff_days = 360 * (y2 - y1) + 30 * (m2 - m1) + df2 - df1
    if d2 == 31 and d1 >= 30:
        diff_days += 30
    else:
        diff_days += d2
    diff_days -= 30 if d1 > 30 else d1
    return diff_days if frac_days else diff_days / 365.

def act_365_nonleap(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/act365n
    Synonyms:
        - Actual/365NL
        - Actual/365 Non-Leap
    
    ISO 20022:
        A014
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

    Method whereby interest is calculated based on the actual
    number of accrued days in the interest period, excluding
    any leap day from the count, and a 365-day year.
    """
    diff_days = date_to_jd(y2, m2, d2) - date_to_jd(y1, m1, d1) + df2 - df1
    leap_years = 0
    if is_leap_year(y1) and (m1 <= 2):
        leap_years += 1
    if (y1 != y2) and is_leap_year(y2) and (m2 >= 3):
        leap_years += 1        
    if (y1+1) < y2:
        now = y1 + 1
        while now < y2:
            if is_leap_year(now):
                leap_years += 1
            now += 1        
    diff_days -= leap_years        
    return diff_days if frac_days else diff_days / 365.

def act_365_fixed(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/act365f
    Synonyms:
        - Actual/365 Fixed
        - Act/365 Fixed
        - A/365 Fixed
        - A/365F
        - English
    
    ISO 20022:
        A005
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

    Method whereby interest is calculated based on the actual
    number of accrued days in the interest period and a 365-day year.
    """
    diff_days = date_to_jd(y2, m2, d2) - date_to_jd(y1, m1, d1)
    diff_days += df2 - df1
    return diff_days if frac_days else diff_days / 365.

def act_360(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/act360
    Synonyms:
        - Actual/360
        - Act/360
        - A/360
        - French
    
    ISO 20022:
        A004
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

    Method whereby interest is calculated based on the actual
    number of accrued days in the interest period and a 360-day year.
    """
    diff_days = date_to_jd(y2, m2, d2) - date_to_jd(y1, m1, d1)
    diff_days += df2 - df1
    return diff_days if frac_days else diff_days / 360.

def _feb29_between(
    date1: date, y1: int,
    date2: date, y2) -> bool:
    # Check each year in the range
    for y in range(y1, y2 + 1):
        if is_leap_year(y):
            leap_day = date(y, 2, 29)
            if date1 <= leap_day <= date2:
                return True
    return False

def _appears_le_year(y1: int, m1: int, d1: int,
        y2: int, m2: int, d2: int) -> bool:
    # Returns True if date1 and date2 "appear" to be 1 year or less apart.
    # This compares the values of year, month, and day directly to each other.
    # Requires date1 <= date2; returns boolean. Used by basis 1.
    if y1 == y2:
        return True
    if ((y1 + 1) == y2 and
        (m1 > m2 or
        (m1 == m2 and d1 >= d2))):
        return True
    return False

def act_act_excel(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Excel-compatible Actual/Actual (basis 1) method.

    Cannot find it in ISO 20022.

    Found it on github (https://github.com/AnatolyBuga/yearfrac)
    and verified it with Excel.

    Other actual/actual methods from ISO 20022 produce
    different figures compared to Excel.
    """
    date1 = date(y1, m1, d1)
    date2 = date(y2, m2, d2)
    if _appears_le_year(y1, m1, d1, y2, m2, d2):        
        if (y1 == y2 and is_leap_year(y1)):
            year_days = 366. # leap year
        elif (_feb29_between(date1, y1, date2, y2) or
            (m2 == 2 and d2 == 29)):
            year_days = 366. # leap year feb29
        else:
            year_days = 365. # leap year else
        df = (date2 - date1).total_seconds() / 86400.
        return (df + df2 - df1) if frac_days else (df + df2 - df1) / year_days
    else:
        year_days = (date(y2+1, 1, 1) - \
            date(y1, 1, 1)).total_seconds() / 86400.
        avg_year_days = year_days / (y2 - y1 + 1)
        df = (date2 - date1).total_seconds() / 86400.
        return (df + df2 - df1) if frac_days else (df + df2 - df1) / avg_year_days

def act_act_isda(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/act_isda
    Synonyms:
        - Actual/Actual ISDA
        - Act/Act ISDA
        - Actual/365 ISDA
        - Act/365 ISDA
    
    ISO 20022:
        A008
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

    Method whereby interest is calculated based on the actual number
    of accrued days of the interest period that fall on a normal year,
    divided by 365, added to the actual number of days of the interest
    period that fall on a leap year, divided by 366.
    """
    if y1 == y2:
        denom = 366. if is_leap_year(y2) else 365.
        diff_days = date_to_jd(y2, m2, d2) - date_to_jd(y1, m1, d1)
        diff_days += df2 - df1
        return diff_days if frac_days else (diff_days / denom)
    else:
        denom_a = 366. if is_leap_year(y1) else 365.
        diff_a = date_to_jd(y1, 12, 31) - date_to_jd(y1, m1, d1)

        denom_b = 366. if is_leap_year(y2) else 365.
        diff_b = date_to_jd(y2, m2, d2) - date_to_jd(y2, 1, 1)

        if frac_days:
            diff = diff_a - df1 + diff_b + df2
            y1 +=1
            while y1 < y2:
                if is_leap_year(y1):
                    diff += 366
                else:
                    diff += 365
                y1 += 1
            return diff
        else:
            return (diff_a - df1) / denom_a + \
                (diff_b + df2) / denom_b + y2 - y1 - 1

def act_act_afb(y1: int, m1: int, d1: int,
            y2: int, m2: int, d2: int,
            df1: float = 0., df2: float = 0.,
            frac_days: bool = False) -> float:
    """
    Source:
        https://github.com/hcnn/act_afb
    Synonyms:
        - Actual/Actual AFB
        - Actual/Actual FBF
    
    ISO 20022:
        A010
        https://www.iso20022.org/15022/uhb/mt565-16-field-22f.htm

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
    if y1 == y2:
        denom = 366. if (m1 < 3 and is_leap_year(y1)) else 365.
        diff_days = date_to_jd(y2, m2, d2) - date_to_jd(y1, m1, d1)
        diff_days += df2 - df1
        return diff_days if frac_days else (diff_days / denom)
    else:
        denom_a = 366. if (m1 < 3 and is_leap_year(y1)) else 365.
        diff_a = date_to_jd(y1, 12, 31)
        diff_a -= date_to_jd(y1, m1, d1)

        denom_b = 366. if (m2 >= 3 and is_leap_year(y2)) else 365.
        diff_b = date_to_jd(y2, m2, d2)
        diff_b -= date_to_jd(y2, 1, 1)

        if frac_days:
            diff = diff_a - df1 + diff_b + df2
            y1 +=1
            while y1 < y2:
                if is_leap_year(y1):
                    diff += 366
                else:
                    diff += 365
                y1 += 1
            return diff
        else:
            return (diff_a - df1) / denom_a + \
                (diff_b + df2) / denom_b + y2 - y1 - 1
