import unittest

from accounts.daycounting import is_leap_year, jd_to_date, date_to_jd
from accounts.daycounting import us_30_360, us_30_360_nasd, eur_30_360_plus, eur_30_360
from accounts.daycounting import us_30_360_eom, eur_30_360_model_2, eur_30_360_model_3
from accounts.daycounting import act_365_fixed, act_360, act_act_isda, act_act_afb
from accounts.daycounting import act_act_excel, thirty_365, act_365_nonleap

# From 31 Excel verification cases the number of errors is shown below.
# Excel formula is:
# - basis 0 (nasd 30/360): "=YEARFRAC($A1, $B1, 0)"
# - basis 1 (act/act):     "=YEARFRAC($A1, $B1, 1)"
# - basis 2 (act/360):     "=YEARFRAC($A1, $B1, 2)"
# - basis 3 (act/365):     "=YEARFRAC($A1, $B1, 3)"
# - basis 4 (eur 30/360):  "=YEARFRAC($A1, $B1, 4)"
#
# us_30_360_eom   basis=0  3
# us_30_360       basis=0  8
# us_30_360_nasd  basis=0  9
#
# act_act_excel   basis=1  0
# act_act_isda    basis=1 20
# act_act_afb     basis=1 20
#
# act_360         basis=2  0
#
# act_365_fixed   basis=3  0
# act_365_nonleap basis=3 23
#
# eur_30_360         basis=4  0
# eur_30_360_plus    basis=4  4
# eur_30_360_model_2 basis=4  5
# eur_30_360_model_3 basis=4 11

# test_yearfrac_{0,1,3} is taken from
# https://support.microsoft.com/en-us/office/yearfrac-function-3844141e-c76d-4143-82b6-208454ddc6a8

FD2_360 = .2/360 # 0.2 days as a fraction of a 360-day year
FD2_365 = .2/365 # 0.2 days as a fraction of a 365-day year
FD2_366 = .2/366 # 0.2 days as a fraction of a 366-day year

class TestIsLeapYear(unittest.TestCase):
    def test_leap_years(self):
        lyrs = [
            1804, 1808, 1812, 1816, 1820, 1824, 1828, 1832, 1836, 1840, 1844,
            1848, 1852, 1856, 1860, 1864, 1868, 1872, 1876, 1880, 1884, 1888,
            1892, 1896, 1904, 1908, 1912, 1916, 1920, 1924, 1928, 1932, 1936,
            1940, 1944, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980,
            1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024,
            2028, 2032, 2036, 2040, 2044, 2048, 2052, 2056, 2060, 2064, 2068,
            2072, 2076, 2080, 2084, 2088, 2092, 2096, 2104, 2108, 2112, 2116,
            2120, 2124, 2128, 2132, 2136, 2140, 2144, 2148, 2152, 2156, 2160,
            2164, 2168, 2172, 2176, 2180, 2184, 2188, 2192, 2196, 2204, 2208,
            2212, 2216, 2220, 2224, 2228, 2232, 2236, 2240, 2244, 2248, 2252,
            2256, 2260, 2264, 2268, 2272, 2276, 2280, 2284, 2288, 2292, 2296,
            2304, 2308, 2312, 2316, 2320, 2324, 2328, 2332, 2336, 2340, 2344,
            2348, 2352, 2356, 2360, 2364, 2368, 2372, 2376, 2380, 2384, 2388,
            2392, 2396, 2400]
        x = [is_leap_year(y) for y in lyrs]
        t = [True for _ in lyrs]
        self.assertListEqual(x, t, 'Leap years')
    def test_nonleap_years(self):
        yrs = [
            2017, 2018, 2019, 2021, 2022, 2023, 2025, 2026, 2027, 2029, 2030]
        x = [is_leap_year(y) for y in yrs]
        t = [False for _ in yrs]
        self.assertListEqual(x, t, 'Non-leap years')

class TestJulianDay(unittest.TestCase):
    def test_conversion(self):
        # 24-Nov (-4713) 12Uhr
        # 25-Nov (-4713) 12Uhr
        # 11-Feb-2014 12Uhr, 2456700
        # 27-Feb-6700 12Uhr, 4168242
        # 28-Feb-6700 12Uhr, 4168243
        # 01-Mar-6700 12Uhr, 4168244
        # 02-Mar-6700 12Uhr, 4168245
        jds = [
            0, 1, 2456700, 4168242, 4168243, 4168244, 4168245]
        years = [
            -4713, -4713, 2014, 6700, 6700, 6700, 6700]
        months = [
            11, 11, 2, 2, 2, 3, 3]
        days = [
            24, 25, 11, 27, 28, 1, 2]
        for jd, y, m, d in zip(jds, years, months, days):
            #print(f'Date {y:04d}-{m:02d}-{d:02d} <-> JD {jd}')
            y_, m_, d_ = jd_to_date(jd)
            self.assertEqual(y_, y, f'Year: {y_} != {y}')
            self.assertEqual(m_, m, f'Month: {m_} != {m}')
            self.assertEqual(d_, d, f'Day: {d_} != {d}')
            jd_ = date_to_jd(y, m, d)
            self.assertEqual(jd_, jd, f'JD: {jd_} != {jd}')

class TestEur30360(unittest.TestCase):
    def test(self):
        x = eur_30_360(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(eur_30_360(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_360, 13))
        self.assertEqual(round(eur_30_360(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_360, 13))
        self.assertEqual(round(eur_30_360(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_excel_basis_4(self):
        self.assertEqual(round(eur_30_360(1978, 2, 28, 2020, 5, 17), 13),
            round(42.2194444444444000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(eur_30_360(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3777777777778000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(eur_30_360(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(eur_30_360(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(eur_30_360(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0000000000000000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(eur_30_360(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(eur_30_360(2020, 2, 21, 2024, 3, 25), 13),
            round(4.0944444444444400, 13), '2020-02-21 -> 2024-03-25')
        self.assertEqual(round(eur_30_360(2020, 2, 29, 2021, 2, 28), 13),
            round(0.9972222222222220, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(eur_30_360(2020, 1, 31, 2021, 2, 28), 13),
            round(1.0777777777777800, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(eur_30_360(2020, 1, 31, 2021, 3, 31), 13),
            round(1.1666666666666700, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(eur_30_360(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(eur_30_360(2018, 2, 5, 2023, 5, 14), 13),
            round(5.2750000000000000, 13), '2018-02-05 -> 2023-05-14')
        self.assertEqual(round(eur_30_360(2020, 2, 29, 2024, 2, 28), 13),
            round(3.9972222222222200, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(eur_30_360(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4166666666666700, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(eur_30_360(2016, 2, 28, 2016, 10, 30), 13),
            round(0.6722222222222220, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(eur_30_360(2014, 1, 31, 2014, 8, 31), 13),
            round(0.5833333333333330, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(eur_30_360(2014, 2, 28, 2014, 9, 30), 13),
            round(0.5888888888888890, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(eur_30_360(2016, 2, 29, 2016, 6, 15), 13),
            round(0.2944444444444440, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(eur_30_360(2024, 1, 1, 2024, 12, 31), 13),
            round(0.9972222222222220, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(eur_30_360(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0027777777777800, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(eur_30_360(2024, 1, 1, 2024, 2, 29), 13),
            round(0.1611111111111110, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(eur_30_360(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(eur_30_360(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1666666666666670, 13), '2023-01-01 -> 2023-03-01')
        self.assertEqual(round(eur_30_360(2024, 2, 29, 2025, 2, 28), 13),
            round(0.9972222222222220, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(eur_30_360(2024, 1, 1, 2028, 12, 31), 13),
            round(4.9972222222222200, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(eur_30_360(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0000000000000000, 13), '2024-03-01 -> 2025-03-01')
        self.assertEqual(round(eur_30_360(2024, 2, 29, 2025, 3, 1), 13),
            round(1.0055555555555600, 13), '2024-02-29 -> 2025-03-01')
        self.assertEqual(round(eur_30_360(2024, 2, 29, 2028, 2, 28), 13),
            round(3.9972222222222200, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(eur_30_360(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0000000000000000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(eur_30_360(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0000000000000000, 13), '2024-03-01 -> 2028-03-01')

class TestEur30360Model2(unittest.TestCase):
    def test(self):
        x = eur_30_360_model_2(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(eur_30_360_model_2(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360_model_2(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360_model_2(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_360, 13))
        self.assertEqual(round(eur_30_360_model_2(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_360, 13))
        self.assertEqual(round(eur_30_360_model_2(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_excel_basis_4(self):
        self.assertEqual(round(eur_30_360_model_2(1978, 2, 28, 2020, 5, 17), 13),
            round(42.2194444444444000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(eur_30_360_model_2(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3777777777778000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(eur_30_360_model_2(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(eur_30_360_model_2(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(eur_30_360_model_2(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0000000000000000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(eur_30_360_model_2(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(eur_30_360_model_2(2020, 2, 21, 2024, 3, 25), 13),
            round(4.0944444444444400, 13), '2020-02-21 -> 2024-03-25')
        # AssertionError: 1.0 != 0.9972222222222
        #self.assertEqual(round(eur_30_360_model_2(2020, 2, 29, 2021, 2, 28), 13),
        #    round(0.9972222222222220, 13), '2020-02-29 -> 2021-02-28')
        # AssertionError: 1.0833333333333 != 1.0777777777778
        #self.assertEqual(round(eur_30_360_model_2(2020, 1, 31, 2021, 2, 28), 13),
        #    round(1.0777777777777800, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(eur_30_360_model_2(2020, 1, 31, 2021, 3, 31), 13),
            round(1.1666666666666700, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(eur_30_360_model_2(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(eur_30_360_model_2(2018, 2, 5, 2023, 5, 14), 13),
            round(5.2750000000000000, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 4.0 != 3.9972222222222
        #self.assertEqual(round(eur_30_360_model_2(2020, 2, 29, 2024, 2, 28), 13),
        #    round(3.9972222222222200, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(eur_30_360_model_2(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4166666666666700, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(eur_30_360_model_2(2016, 2, 28, 2016, 10, 30), 13),
            round(0.6722222222222220, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(eur_30_360_model_2(2014, 1, 31, 2014, 8, 31), 13),
            round(0.5833333333333330, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(eur_30_360_model_2(2014, 2, 28, 2014, 9, 30), 13),
            round(0.5888888888888890, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(eur_30_360_model_2(2016, 2, 29, 2016, 6, 15), 13),
            round(0.2944444444444440, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(eur_30_360_model_2(2024, 1, 1, 2024, 12, 31), 13),
            round(0.9972222222222220, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(eur_30_360_model_2(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0027777777777800, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(eur_30_360_model_2(2024, 1, 1, 2024, 2, 29), 13),
            round(0.1611111111111110, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(eur_30_360_model_2(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(eur_30_360_model_2(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1666666666666670, 13), '2023-01-01 -> 2023-03-01')
        # AssertionError: 1.0 != 0.9972222222222
        #self.assertEqual(round(eur_30_360_model_2(2024, 2, 29, 2025, 2, 28), 13),
        #    round(0.9972222222222220, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(eur_30_360_model_2(2024, 1, 1, 2028, 12, 31), 13),
            round(4.9972222222222200, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(eur_30_360_model_2(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0000000000000000, 13), '2024-03-01 -> 2025-03-01')
        self.assertEqual(round(eur_30_360_model_2(2024, 2, 29, 2025, 3, 1), 13),
            round(1.0055555555555600, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 4.0 != 3.9972222222222
        #self.assertEqual(round(eur_30_360_model_2(2024, 2, 29, 2028, 2, 28), 13),
        #    round(3.9972222222222200, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(eur_30_360_model_2(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0000000000000000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(eur_30_360_model_2(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0000000000000000, 13), '2024-03-01 -> 2028-03-01')

class TestEur30360Model3(unittest.TestCase):
    def test(self):
        x = eur_30_360_model_3(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(eur_30_360_model_3(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360_model_3(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360_model_3(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_360, 13))
        self.assertEqual(round(eur_30_360_model_3(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_360, 13))
        self.assertEqual(round(eur_30_360_model_3(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_excel_basis_4(self):
        # AssertionError: 42.2138888888889 != 42.2194444444444
        #self.assertEqual(round(eur_30_360_model_3(1978, 2, 28, 2020, 5, 17), 13),
        #    round(42.2194444444444000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(eur_30_360_model_3(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3777777777778000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(eur_30_360_model_3(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(eur_30_360_model_3(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(eur_30_360_model_3(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0000000000000000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(eur_30_360_model_3(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(eur_30_360_model_3(2020, 2, 21, 2024, 3, 25), 13),
            round(4.0944444444444400, 13), '2020-02-21 -> 2024-03-25')
        # AssertionError: 1.0 != 0.9972222222222
        #self.assertEqual(round(eur_30_360_model_3(2020, 2, 29, 2021, 2, 28), 13),
        #    round(0.9972222222222220, 13), '2020-02-29 -> 2021-02-28')
        # AssertionError: 1.0833333333333 != 1.0777777777778
        #self.assertEqual(round(eur_30_360_model_3(2020, 1, 31, 2021, 2, 28), 13),
        #    round(1.0777777777777800, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(eur_30_360_model_3(2020, 1, 31, 2021, 3, 31), 13),
            round(1.1666666666666700, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(eur_30_360_model_3(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(eur_30_360_model_3(2018, 2, 5, 2023, 5, 14), 13),
            round(5.2750000000000000, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 4.0 != 3.9972222222222
        #self.assertEqual(round(eur_30_360_model_3(2020, 2, 29, 2024, 2, 28), 13),
        #    round(3.9972222222222200, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(eur_30_360_model_3(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4166666666666700, 13), '2010-03-31 -> 2015-08-30')
        # AssertionError: 0.6666666666667 != 0.6722222222222
        #self.assertEqual(round(eur_30_360_model_3(2016, 2, 28, 2016, 10, 30), 13),
        #    round(0.6722222222222220, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(eur_30_360_model_3(2014, 1, 31, 2014, 8, 31), 13),
            round(0.5833333333333330, 13), '2014-01-31 -> 2014-08-31')
        # AssertionError: 0.5833333333333 != 0.5888888888889
        #self.assertEqual(round(eur_30_360_model_3(2014, 2, 28, 2014, 9, 30), 13),
        #    round(0.5888888888888890, 13), '2014-02-28 -> 2014-09-30')
        # AssertionError: 0.2916666666667 != 0.2944444444444
        #self.assertEqual(round(eur_30_360_model_3(2016, 2, 29, 2016, 6, 15), 13),
        #    round(0.2944444444444440, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(eur_30_360_model_3(2024, 1, 1, 2024, 12, 31), 13),
            round(0.9972222222222220, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(eur_30_360_model_3(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0027777777777800, 13), '2024-01-01 -> 2025-01-02')
        # AssertionError: 0.1638888888889 != 0.1611111111111
        #self.assertEqual(round(eur_30_360_model_3(2024, 1, 1, 2024, 2, 29), 13),
        #    round(0.1611111111111110, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(eur_30_360_model_3(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(eur_30_360_model_3(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1666666666666670, 13), '2023-01-01 -> 2023-03-01')
        # AssertionError: 1.0 != 0.9972222222222
        #self.assertEqual(round(eur_30_360_model_3(2024, 2, 29, 2025, 2, 28), 13),
        #    round(0.9972222222222220, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(eur_30_360_model_3(2024, 1, 1, 2028, 12, 31), 13),
            round(4.9972222222222200, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(eur_30_360_model_3(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0000000000000000, 13), '2024-03-01 -> 2025-03-01')
        # AssertionError: 1.0027777777778 != 1.0055555555556
        #self.assertEqual(round(eur_30_360_model_3(2024, 2, 29, 2025, 3, 1), 13),
        #    round(1.0055555555555600, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 4.0 != 3.9972222222222
        #self.assertEqual(round(eur_30_360_model_3(2024, 2, 29, 2028, 2, 28), 13),
        #    round(3.9972222222222200, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(eur_30_360_model_3(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0000000000000000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(eur_30_360_model_3(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0000000000000000, 13), '2024-03-01 -> 2028-03-01')

class TestEur30360Plus(unittest.TestCase):
    def test(self):
        x = eur_30_360_plus(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(eur_30_360_plus(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360_plus(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(eur_30_360_plus(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_360, 13))
        self.assertEqual(round(eur_30_360_plus(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_360, 13))
        self.assertEqual(round(eur_30_360_plus(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_excel_basis_4(self):
        self.assertEqual(round(eur_30_360_plus(1978, 2, 28, 2020, 5, 17), 13),
            round(42.2194444444444000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(eur_30_360_plus(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3777777777778000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(eur_30_360_plus(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(eur_30_360_plus(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(eur_30_360_plus(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0000000000000000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(eur_30_360_plus(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(eur_30_360_plus(2020, 2, 21, 2024, 3, 25), 13),
            round(4.0944444444444400, 13), '2020-02-21 -> 2024-03-25')
        self.assertEqual(round(eur_30_360_plus(2020, 2, 29, 2021, 2, 28), 13),
            round(0.9972222222222220, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(eur_30_360_plus(2020, 1, 31, 2021, 2, 28), 13),
            round(1.0777777777777800, 13), '2020-01-31 -> 2021-02-28')
        # AssertionError: 1.1722222222222 != 1.1666666666667
        #self.assertEqual(round(eur_30_360_plus(2020, 1, 31, 2021, 3, 31), 13),
        #    round(1.1666666666666700, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(eur_30_360_plus(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(eur_30_360_plus(2018, 2, 5, 2023, 5, 14), 13),
            round(5.2750000000000000, 13), '2018-02-05 -> 2023-05-14')
        self.assertEqual(round(eur_30_360_plus(2020, 2, 29, 2024, 2, 28), 13),
            round(3.9972222222222200, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(eur_30_360_plus(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4166666666666700, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(eur_30_360_plus(2016, 2, 28, 2016, 10, 30), 13),
            round(0.6722222222222220, 13), '2016-02-28 -> 2016-10-30')
        # AssertionError: 0.5888888888889 != 0.5833333333333
        #self.assertEqual(round(eur_30_360_plus(2014, 1, 31, 2014, 8, 31), 13),
        #    round(0.5833333333333330, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(eur_30_360_plus(2014, 2, 28, 2014, 9, 30), 13),
            round(0.5888888888888890, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(eur_30_360_plus(2016, 2, 29, 2016, 6, 15), 13),
            round(0.2944444444444440, 13), '2016-02-29 -> 2016-06-15')
        # AssertionError: 1.0027777777778 != 0.9972222222222
        #self.assertEqual(round(eur_30_360_plus(2024, 1, 1, 2024, 12, 31), 13),
        #    round(0.9972222222222220, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(eur_30_360_plus(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0027777777777800, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(eur_30_360_plus(2024, 1, 1, 2024, 2, 29), 13),
            round(0.1611111111111110, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(eur_30_360_plus(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(eur_30_360_plus(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1666666666666670, 13), '2023-01-01 -> 2023-03-01')
        self.assertEqual(round(eur_30_360_plus(2024, 2, 29, 2025, 2, 28), 13),
            round(0.9972222222222220, 13), '2024-02-29 -> 2025-02-28')
        # AssertionError: 5.0027777777778 != 4.9972222222222
        #self.assertEqual(round(eur_30_360_plus(2024, 1, 1, 2028, 12, 31), 13),
        #    round(4.9972222222222200, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(eur_30_360_plus(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0000000000000000, 13), '2024-03-01 -> 2025-03-01')
        self.assertEqual(round(eur_30_360_plus(2024, 2, 29, 2025, 3, 1), 13),
            round(1.0055555555555600, 13), '2024-02-29 -> 2025-03-01')
        self.assertEqual(round(eur_30_360_plus(2024, 2, 29, 2028, 2, 28), 13),
            round(3.9972222222222200, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(eur_30_360_plus(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0000000000000000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(eur_30_360_plus(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0000000000000000, 13), '2024-03-01 -> 2028-03-01')

class TestUs30360(unittest.TestCase):
    def test(self):
        x = us_30_360(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(us_30_360(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(us_30_360(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(us_30_360(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_360, 13))
        self.assertEqual(round(us_30_360(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_360, 13))
        self.assertEqual(round(us_30_360(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_yearfrac_0(self):
        x = us_30_360(2012, 1, 1, 2012, 7, 30)
        self.assertEqual(round(x, 8), 0.58055556)
    def test_excel_basis_0(self):
        # AssertionError: 42.2194444444444 != 42.2138888888889
        #self.assertEqual(round(us_30_360(1978, 2, 28, 2020, 5, 17), 13),
        #    round(42.2138888888889000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(us_30_360(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3777777777778000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(us_30_360(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(us_30_360(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(us_30_360(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0000000000000000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(us_30_360(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(us_30_360(2020, 2, 21, 2024, 3, 25), 13),
            round(4.0944444444444400, 13), '2020-02-21 -> 2024-03-25')
        # AssertionError: 0.9972222222222 != 1.0
        #self.assertEqual(round(us_30_360(2020, 2, 29, 2021, 2, 28), 13),
        #    round(1.0000000000000000, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(us_30_360(2020, 1, 31, 2021, 2, 28), 13),
            round(1.0777777777777800, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(us_30_360(2020, 1, 31, 2021, 3, 31), 13),
            round(1.1666666666666700, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(us_30_360(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(us_30_360(2018, 2, 5, 2023, 5, 14), 13),
            round(5.2750000000000000, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 3.9972222222222 != 3.9944444444444
        #self.assertEqual(round(us_30_360(2020, 2, 29, 2024, 2, 28), 13),
        #    round(3.9944444444444400, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(us_30_360(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4166666666666700, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(us_30_360(2016, 2, 28, 2016, 10, 30), 13),
            round(0.6722222222222220, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(us_30_360(2014, 1, 31, 2014, 8, 31), 13),
            round(0.5833333333333330, 13), '2014-01-31 -> 2014-08-31')
        # AssertionError: 0.5888888888889 != 0.5833333333333
        #self.assertEqual(round(us_30_360(2014, 2, 28, 2014, 9, 30), 13),
        #    round(0.5833333333333330, 13), '2014-02-28 -> 2014-09-30')
        # AssertionError: 0.2944444444444 != 0.2916666666667
        #self.assertEqual(round(us_30_360(2016, 2, 29, 2016, 6, 15), 13),
        #    round(0.2916666666666670, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(us_30_360(2024, 1, 1, 2024, 12, 31), 13),
            round(1.0000000000000000, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(us_30_360(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0027777777777800, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(us_30_360(2024, 1, 1, 2024, 2, 29), 13),
            round(0.1611111111111110, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(us_30_360(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(us_30_360(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1666666666666670, 13), '2023-01-01 -> 2023-03-01')
        # AssertionError: 0.9972222222222 != 1.0
        #self.assertEqual(round(us_30_360(2024, 2, 29, 2025, 2, 28), 13),
        #    round(1.0000000000000000, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(us_30_360(2024, 1, 1, 2028, 12, 31), 13),
            round(5.0000000000000000, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(us_30_360(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0000000000000000, 13), '2024-03-01 -> 2025-03-01')
        # AssertionError: 1.0055555555556 != 1.0027777777778
        #self.assertEqual(round(us_30_360(2024, 2, 29, 2025, 3, 1), 13),
        #    round(1.0027777777777800, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 3.9972222222222 != 3.9944444444444
        #self.assertEqual(round(us_30_360(2024, 2, 29, 2028, 2, 28), 13),
        #    round(3.9944444444444400, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(us_30_360(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0000000000000000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(us_30_360(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0000000000000000, 13), '2024-03-01 -> 2028-03-01')

class TestUs30360Eom(unittest.TestCase):
    def test(self):
        x = us_30_360_eom(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(us_30_360_eom(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(us_30_360_eom(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(us_30_360_eom(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_360, 13))
        self.assertEqual(round(us_30_360_eom(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_360, 13))
        self.assertEqual(round(us_30_360_eom(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_yearfrac_0(self):
        x = us_30_360_eom(2012, 1, 1, 2012, 7, 30)
        self.assertEqual(round(x, 8), 0.58055556)
    def test_excel_basis_0(self):
        self.assertEqual(round(us_30_360_eom(1978, 2, 28, 2020, 5, 17), 13),
            round(42.2138888888889000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(us_30_360_eom(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3777777777778000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(us_30_360_eom(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(us_30_360_eom(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(us_30_360_eom(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0000000000000000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(us_30_360_eom(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(us_30_360_eom(2020, 2, 21, 2024, 3, 25), 13),
            round(4.0944444444444400, 13), '2020-02-21 -> 2024-03-25')
        self.assertEqual(round(us_30_360_eom(2020, 2, 29, 2021, 2, 28), 13),
            round(1.0000000000000000, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(us_30_360_eom(2020, 1, 31, 2021, 2, 28), 13),
            round(1.0777777777777800, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(us_30_360_eom(2020, 1, 31, 2021, 3, 31), 13),
            round(1.1666666666666700, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(us_30_360_eom(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(us_30_360_eom(2018, 2, 5, 2023, 5, 14), 13),
            round(5.2750000000000000, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 4.0 != 3.9944444444444
        #self.assertEqual(round(us_30_360_eom(2020, 2, 29, 2024, 2, 28), 13),
        #    round(3.9944444444444400, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(us_30_360_eom(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4166666666666700, 13), '2010-03-31 -> 2015-08-30')
        # AssertionError: 0.6666666666667 != 0.6722222222222
        #self.assertEqual(round(us_30_360_eom(2016, 2, 28, 2016, 10, 30), 13),
        #    round(0.6722222222222220, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(us_30_360_eom(2014, 1, 31, 2014, 8, 31), 13),
            round(0.5833333333333330, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(us_30_360_eom(2014, 2, 28, 2014, 9, 30), 13),
            round(0.5833333333333330, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(us_30_360_eom(2016, 2, 29, 2016, 6, 15), 13),
            round(0.2916666666666670, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(us_30_360_eom(2024, 1, 1, 2024, 12, 31), 13),
            round(1.0000000000000000, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(us_30_360_eom(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0027777777777800, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(us_30_360_eom(2024, 1, 1, 2024, 2, 29), 13),
            round(0.1611111111111110, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(us_30_360_eom(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(us_30_360_eom(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1666666666666670, 13), '2023-01-01 -> 2023-03-01')
        self.assertEqual(round(us_30_360_eom(2024, 2, 29, 2025, 2, 28), 13),
            round(1.0000000000000000, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(us_30_360_eom(2024, 1, 1, 2028, 12, 31), 13),
            round(5.0000000000000000, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(us_30_360_eom(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0000000000000000, 13), '2024-03-01 -> 2025-03-01')
        self.assertEqual(round(us_30_360_eom(2024, 2, 29, 2025, 3, 1), 13),
            round(1.0027777777777800, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 4.0 != 3.9944444444444
        #self.assertEqual(round(us_30_360_eom(2024, 2, 29, 2028, 2, 28), 13),
        #    round(3.9944444444444400, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(us_30_360_eom(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0000000000000000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(us_30_360_eom(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0000000000000000, 13), '2024-03-01 -> 2028-03-01')

class TestUs30360Nasd(unittest.TestCase):
    def test(self):
        x = us_30_360_nasd(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(us_30_360_nasd(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(us_30_360_nasd(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(us_30_360_nasd(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_360, 13))
        self.assertEqual(round(us_30_360_nasd(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_360, 13))
        self.assertEqual(round(us_30_360_nasd(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_yearfrac_0(self):
        x = us_30_360_nasd(2012, 1, 1, 2012, 7, 30)
        self.assertEqual(round(x, 8), 0.58055556)
    def test_excel_basis_0(self):
        # AssertionError: 42.2194444444444 != 42.2138888888889
        #self.assertEqual(round(us_30_360_nasd(1978, 2, 28, 2020, 5, 17), 13),
        #    round(42.2138888888889000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(us_30_360_nasd(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3777777777778000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(us_30_360_nasd(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(us_30_360_nasd(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(us_30_360_nasd(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0000000000000000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(us_30_360_nasd(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(us_30_360_nasd(2020, 2, 21, 2024, 3, 25), 13),
            round(4.0944444444444400, 13), '2020-02-21 -> 2024-03-25')
        # AssertionError: 0.9972222222222 != 1.0
        #self.assertEqual(round(us_30_360_nasd(2020, 2, 29, 2021, 2, 28), 13),
        #    round(1.0000000000000000, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(us_30_360_nasd(2020, 1, 31, 2021, 2, 28), 13),
            round(1.0777777777777800, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(us_30_360_nasd(2020, 1, 31, 2021, 3, 31), 13),
            round(1.1666666666666700, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(us_30_360_nasd(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(us_30_360_nasd(2018, 2, 5, 2023, 5, 14), 13),
            round(5.2750000000000000, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 3.9972222222222 != 3.9944444444444
        #self.assertEqual(round(us_30_360_nasd(2020, 2, 29, 2024, 2, 28), 13),
        #    round(3.9944444444444400, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(us_30_360_nasd(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4166666666666700, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(us_30_360_nasd(2016, 2, 28, 2016, 10, 30), 13),
            round(0.6722222222222220, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(us_30_360_nasd(2014, 1, 31, 2014, 8, 31), 13),
            round(0.5833333333333330, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(us_30_360_eom(2014, 2, 28, 2014, 9, 30), 13),
            round(0.5833333333333330, 13), '2014-02-28 -> 2014-09-30')
        # AssertionError: 0.2944444444444 != 0.2916666666667
        #self.assertEqual(round(us_30_360_nasd(2016, 2, 29, 2016, 6, 15), 13),
        #    round(0.2916666666666670, 13), '2016-02-29 -> 2016-06-15')
        # AssertionError: 1.0027777777778 != 1.0
        #self.assertEqual(round(us_30_360_nasd(2024, 1, 1, 2024, 12, 31), 13),
        #    round(1.0000000000000000, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(us_30_360_nasd(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0027777777777800, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(us_30_360_nasd(2024, 1, 1, 2024, 2, 29), 13),
            round(0.1611111111111110, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(us_30_360_nasd(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(us_30_360_nasd(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1666666666666670, 13), '2023-01-01 -> 2023-03-01')
        # AssertionError: 0.9972222222222 != 1.0
        #self.assertEqual(round(us_30_360_nasd(2024, 2, 29, 2025, 2, 28), 13),
        #    round(1.0000000000000000, 13), '2024-02-29 -> 2025-02-28')
        # AssertionError: 5.0027777777778 != 5.0
        #self.assertEqual(round(us_30_360_nasd(2024, 1, 1, 2028, 12, 31), 13),
        #    round(5.0000000000000000, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(us_30_360_nasd(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0000000000000000, 13), '2024-03-01 -> 2025-03-01')
        # AssertionError: 1.0055555555556 != 1.0027777777778
        #self.assertEqual(round(us_30_360_nasd(2024, 2, 29, 2025, 3, 1), 13),
        #    round(1.0027777777777800, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 3.9972222222222 != 3.9944444444444
        #self.assertEqual(round(us_30_360_nasd(2024, 2, 29, 2028, 2, 28), 13),
        #    round(3.9944444444444400, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(us_30_360_nasd(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0000000000000000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(us_30_360_nasd(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0000000000000000, 13), '2024-03-01 -> 2028-03-01')

class TestThirty365(unittest.TestCase):
    def test(self):
        x = thirty_365(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.20821918)
    def test_time(self):
        self.assertEqual(round(thirty_365(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 0.986301369863)
        self.assertEqual(round(thirty_365(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 0.986301369863)
        self.assertEqual(round(thirty_365(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(0.986301369863 + FD2_365, 13))
        self.assertEqual(round(thirty_365(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(0.986301369863 - FD2_365, 13))
        self.assertEqual(round(thirty_365(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_365, 13))

class TestAct365NonLeap(unittest.TestCase):
    def test(self):
        x = act_365_nonleap(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.20821918)
    def test_time(self):
        self.assertEqual(round(act_365_nonleap(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(act_365_nonleap(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(act_365_nonleap(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_365, 13))
        self.assertEqual(round(act_365_nonleap(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_365, 13))
        self.assertEqual(round(act_365_nonleap(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(-0.0021917808219, 13))
    def test_excel_basis_3(self):
        # AssertionError: 42.213698630137 != 42.2438356164384
        #self.assertEqual(round(act_365_nonleap(1978, 2, 28, 2020, 5, 17), 13),
        #    round(42.2438356164384, 13), '1978-02-28 -> 2020-05-17')
        # AssertionError: 28.3753424657534 != 28.3945205479452
        #self.assertEqual(round(act_365_nonleap(1993, 12, 2, 2022, 4, 18), 13),
        #    round(28.3945205479452, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(act_365_nonleap(2018, 12, 15, 2019, 3, 1), 13),
            round(0.208219178082192, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(act_365_nonleap(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027397260273973, 13), '2018-12-31 -> 2019-01-01')
        # AssertionError: 3.0 != 3.0027397260274 : 1994-06-30
        #self.assertEqual(round(act_365_nonleap(1994, 6, 30, 1997, 6, 30), 13),
        #    round(3.002739726027400, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(act_365_nonleap(1994, 2, 10, 1994, 6, 30), 13),
            round(0.383561643835616, 13), '1994-02-10 -> 1994-06-30')
        # AssertionError: 4.0876712328767 != 4.0931506849315
        #self.assertEqual(round(act_365_nonleap(2020, 2, 21, 2024, 3, 25), 13),
        #    round(4.093150684931510, 13), '2020-02-21 -> 2024-03-25')
        # AssertionError: 0.9972602739726 != 1.0
        #self.assertEqual(round(act_365_nonleap(2020, 2, 29, 2021, 2, 28), 13),
        #    round(1.000000000000000, 13), '2020-02-29 -> 2021-02-28')
        # AssertionError: 1.0767123287671 != 1.0794520547945
        #self.assertEqual(round(act_365_nonleap(2020, 1, 31, 2021, 2, 28), 13),
        #    round(1.079452054794520, 13), '2020-01-31 -> 2021-02-28')
        # AssertionError: 1.1616438356164 != 1.1643835616438
        #self.assertEqual(round(act_365_nonleap(2020, 1, 31, 2021, 3, 31), 13),
        #    round(1.164383561643840, 13), '2020-01-31 -> 2021-03-31')
        # AssertionError: 0.2438356164384 != 0.2465753424658
        #self.assertEqual(round(act_365_nonleap(2020, 1, 31, 2020, 4, 30), 13),
        #    round(0.246575342465753, 13), '2020-01-31 -> 2020-04-30')
        # AssertionError: 5.2684931506849 != 5.2712328767123
        #self.assertEqual(round(act_365_nonleap(2018, 2, 5, 2023, 5, 14), 13),
        #    round(5.271232876712330, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 3.9972602739726 != 4.0
        #self.assertEqual(round(act_365_nonleap(2020, 2, 29, 2024, 2, 28), 13),
        #    round(4.000000000000000, 13), '2020-02-29 -> 2024-02-28')
        # AssertionError: 5.4164383561644 != 5.4191780821918
        #self.assertEqual(round(act_365_nonleap(2010, 3, 31, 2015, 8, 30), 13),
        #    round(5.419178082191780, 13), '2010-03-31 -> 2015-08-30')
        # AssertionError: 0.6684931506849 != 0.6712328767123
        #self.assertEqual(round(act_365_nonleap(2016, 2, 28, 2016, 10, 30), 13),
        #    round(0.671232876712329, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(act_365_nonleap(2014, 1, 31, 2014, 8, 31), 13),
            round(0.580821917808219, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(act_365_nonleap(2014, 2, 28, 2014, 9, 30), 13),
            round(0.586301369863014, 13), '2014-02-28 -> 2014-09-30')
        # AssertionError: 0.2904109589041 != 0.2931506849315
        #self.assertEqual(round(act_365_nonleap(2016, 2, 29, 2016, 6, 15), 13),
        #    round(0.293150684931507, 13), '2016-02-29 -> 2016-06-15')
        # AssertionError: 0.9972602739726 != 1.0
        #self.assertEqual(round(act_365_nonleap(2024, 1, 1, 2024, 12, 31), 13),
        #    round(1.000000000000000, 13), '2024-01-01 -> 2024-12-31')
        # AssertionError: 1.0027397260274 != 1.0054794520548
        #self.assertEqual(round(act_365_nonleap(2024, 1, 1, 2025, 1, 2), 13),
        #    round(1.005479452054790, 13), '2024-01-01 -> 2025-01-02')
        # AssertionError: 0.158904109589 != 0.1616438356164
        #self.assertEqual(round(act_365_nonleap(2024, 1, 1, 2024, 2, 29), 13),
        #    round(0.161643835616438, 13), '2024-01-01 -> 2024-02-29')
        # AssertionError: 0.1616438356164 != 0.1643835616438
        #self.assertEqual(round(act_365_nonleap(2024, 1, 1, 2024, 3, 1), 13),
        #    round(0.164383561643836, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(act_365_nonleap(2023, 1, 1, 2023, 3, 1), 13),
            round(0.161643835616438, 13), '2023-01-01 -> 2023-03-01')
        # AssertionError: 0.9972602739726 != 1.0
        #self.assertEqual(round(act_365_nonleap(2024, 2, 29, 2025, 2, 28), 13),
        #    round(1.000000000000000, 13), '2024-02-29 -> 2025-02-28')
        # AssertionError: 4.9972602739726 != 5.0027397260274
        #self.assertEqual(round(act_365_nonleap(2024, 1, 1, 2028, 12, 31), 13),
        #    round(5.002739726027400, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(act_365_nonleap(2024, 3, 1, 2025, 3, 1), 13),
            round(1.000000000000000, 13), '2024-03-01 -> 2025-03-01')
        # AssertionError: 1.0 != 1.0027397260274
        #self.assertEqual(round(act_365_nonleap(2024, 2, 29, 2025, 3, 1), 13),
        #    round(1.002739726027400, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 3.9972602739726 != 4.0
        #self.assertEqual(round(act_365_nonleap(2024, 2, 29, 2028, 2, 28), 13),
        #    round(4.000000000000000, 13), '2024-02-29 -> 2028-02-28')
        # AssertionError: 4.0 != 4.0027397260274
        #self.assertEqual(round(act_365_nonleap(2024, 2, 29, 2028, 2, 29), 13),
        #    round(4.002739726027400, 13), '2024-02-29 -> 2028-02-29')
        # AssertionError: 4.0 != 4.0027397260274
        #self.assertEqual(round(act_365_nonleap(2024, 3, 1, 2028, 3, 1), 13),
        #    round(4.002739726027400, 13), '2024-03-01 -> 2028-03-01')

class TestAct365Fixed(unittest.TestCase):
    def test(self):
        x = act_365_fixed(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.20821918)
    def test_time(self):
        self.assertEqual(round(act_365_fixed(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(act_365_fixed(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.0027397260274)
        self.assertEqual(round(act_365_fixed(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1.0027397260274 + FD2_365, 13))
        self.assertEqual(round(act_365_fixed(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1.0027397260274 - FD2_365, 13))
        self.assertEqual(round(act_365_fixed(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_365, 13))
    def test_yearfrac_3(self):
        x = act_365_fixed(2012, 1, 1, 2012, 7, 30)
        self.assertEqual(round(x, 8), 0.57808219)
    def test_excel_basis_3(self):
        self.assertEqual(round(act_365_fixed(1978, 2, 28, 2020, 5, 17), 13),
            round(42.2438356164384, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(act_365_fixed(1993, 12, 2, 2022, 4, 18), 13),
            round(28.3945205479452, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(act_365_fixed(2018, 12, 15, 2019, 3, 1), 13),
            round(0.208219178082192, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(act_365_fixed(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027397260273973, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(act_365_fixed(1994, 6, 30, 1997, 6, 30), 13),
            round(3.002739726027400, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(act_365_fixed(1994, 2, 10, 1994, 6, 30), 13),
            round(0.383561643835616, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(act_365_fixed(2020, 2, 21, 2024, 3, 25), 13),
            round(4.093150684931510, 13), '2020-02-21 -> 2024-03-25')
        self.assertEqual(round(act_365_fixed(2020, 2, 29, 2021, 2, 28), 13),
            round(1.000000000000000, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(act_365_fixed(2020, 1, 31, 2021, 2, 28), 13),
            round(1.079452054794520, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(act_365_fixed(2020, 1, 31, 2021, 3, 31), 13),
            round(1.164383561643840, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(act_365_fixed(2020, 1, 31, 2020, 4, 30), 13),
            round(0.246575342465753, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(act_365_fixed(2018, 2, 5, 2023, 5, 14), 13),
            round(5.271232876712330, 13), '2018-02-05 -> 2023-05-14')
        self.assertEqual(round(act_365_fixed(2020, 2, 29, 2024, 2, 28), 13),
            round(4.000000000000000, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(act_365_fixed(2010, 3, 31, 2015, 8, 30), 13),
            round(5.419178082191780, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(act_365_fixed(2016, 2, 28, 2016, 10, 30), 13),
            round(0.671232876712329, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(act_365_fixed(2014, 1, 31, 2014, 8, 31), 13),
            round(0.580821917808219, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(act_365_fixed(2014, 2, 28, 2014, 9, 30), 13),
            round(0.586301369863014, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(act_365_fixed(2016, 2, 29, 2016, 6, 15), 13),
            round(0.293150684931507, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(act_365_fixed(2024, 1, 1, 2024, 12, 31), 13),
            round(1.000000000000000, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(act_365_fixed(2024, 1, 1, 2025, 1, 2), 13),
            round(1.005479452054790, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(act_365_fixed(2024, 1, 1, 2024, 2, 29), 13),
            round(0.161643835616438, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(act_365_fixed(2024, 1, 1, 2024, 3, 1), 13),
            round(0.164383561643836, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(act_365_fixed(2023, 1, 1, 2023, 3, 1), 13),
            round(0.161643835616438, 13), '2023-01-01 -> 2023-03-01')
        self.assertEqual(round(act_365_fixed(2024, 2, 29, 2025, 2, 28), 13),
            round(1.000000000000000, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(act_365_fixed(2024, 1, 1, 2028, 12, 31), 13),
            round(5.002739726027400, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(act_365_fixed(2024, 3, 1, 2025, 3, 1), 13),
            round(1.000000000000000, 13), '2024-03-01 -> 2025-03-01')
        self.assertEqual(round(act_365_fixed(2024, 2, 29, 2025, 3, 1), 13),
            round(1.002739726027400, 13), '2024-02-29 -> 2025-03-01')
        self.assertEqual(round(act_365_fixed(2024, 2, 29, 2028, 2, 28), 13),
            round(4.000000000000000, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(act_365_fixed(2024, 2, 29, 2028, 2, 29), 13),
            round(4.002739726027400, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(act_365_fixed(2024, 3, 1, 2028, 3, 1), 13),
            round(4.002739726027400, 13), '2024-03-01 -> 2028-03-01')

class TestAct360(unittest.TestCase):
    def test(self):
        x = act_360(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.21111111)
    def test_time(self):
        self.assertEqual(round(act_360(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.0138888888889)
        self.assertEqual(round(act_360(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.0166666666667)
        self.assertEqual(round(act_360(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 12),
            round(1.0166666666667 + FD2_360, 12))
        self.assertEqual(round(act_360(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1.0166666666667 - FD2_360, 13))
        self.assertEqual(round(act_360(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_360, 13))
    def test_excel_basis_2(self):
        self.assertEqual(round(act_360(1978, 2, 28, 2020, 5, 17), 13),
            round(42.830555555555600, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(act_360(1993, 12, 2, 2022, 4, 18), 13),
            round(28.788888888888900, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(act_360(2018, 12, 15, 2019, 3, 1), 13),
            round(0.2111111111111110, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(act_360(2018, 12, 31, 2019, 1, 1), 13),
            round(0.0027777777777778, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(act_360(1994, 6, 30, 1997, 6, 30), 13),
            round(3.0444444444444400, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(act_360(1994, 2, 10, 1994, 6, 30), 13),
            round(0.3888888888888890, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(act_360(2020, 2, 21, 2024, 3, 25), 13),
            round(4.1500000000000000, 13), '2020-02-21 -> 2024-03-25')
        self.assertEqual(round(act_360(2020, 2, 29, 2021, 2, 28), 13),
            round(1.0138888888888900, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(act_360(2020, 1, 31, 2021, 2, 28), 13),
            round(1.0944444444444400, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(act_360(2020, 1, 31, 2021, 3, 31), 13),
            round(1.1805555555555600, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(act_360(2020, 1, 31, 2020, 4, 30), 13),
            round(0.2500000000000000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(act_360(2018, 2, 5, 2023, 5, 14), 13),
            round(5.3444444444444400, 13), '2018-02-05 -> 2023-05-14')
        self.assertEqual(round(act_360(2020, 2, 29, 2024, 2, 28), 13),
            round(4.0555555555555600, 13), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(act_360(2010, 3, 31, 2015, 8, 30), 13),
            round(5.4944444444444400, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(act_360(2016, 2, 28, 2016, 10, 30), 13),
            round(0.6805555555555560, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(act_360(2014, 1, 31, 2014, 8, 31), 13),
            round(0.5888888888888890, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(act_360(2014, 2, 28, 2014, 9, 30), 13),
            round(0.5944444444444440, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(act_360(2016, 2, 29, 2016, 6, 15), 13),
            round(0.2972222222222220, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(act_360(2024, 1, 1, 2024, 12, 31), 13),
            round(1.0138888888888900, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(act_360(2024, 1, 1, 2025, 1, 2), 13),
            round(1.0194444444444400, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(act_360(2024, 1, 1, 2024, 2, 29), 13),
            round(0.1638888888888890, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(act_360(2024, 1, 1, 2024, 3, 1), 13),
            round(0.1666666666666670, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(act_360(2023, 1, 1, 2023, 3, 1), 13),
            round(0.1638888888888890, 13), '2023-01-01 -> 2023-03-01')
        self.assertEqual(round(act_360(2024, 2, 29, 2025, 2, 28), 13),
            round(1.0138888888888900, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(act_360(2024, 1, 1, 2028, 12, 31), 13),
            round(5.0722222222222200, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(act_360(2024, 3, 1, 2025, 3, 1), 13),
            round(1.0138888888888900, 13), '2024-03-01 -> 2025-03-01')
        self.assertEqual(round(act_360(2024, 2, 29, 2025, 3, 1), 13),
            round(1.0166666666666700, 13), '2024-02-29 -> 2025-03-01')
        self.assertEqual(round(act_360(2024, 2, 29, 2028, 2, 28), 13),
            round(4.0555555555555600, 13), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(act_360(2024, 2, 29, 2028, 2, 29), 13),
            round(4.0583333333333300, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(act_360(2024, 3, 1, 2028, 3, 1), 13),
            round(4.0583333333333300, 13), '2024-03-01 -> 2028-03-01')

class TestActActExcel(unittest.TestCase):
    def test_yearfrac_1(self):
        x = act_act_excel(2012, 1, 1, 2012, 7, 30)
        self.assertEqual(round(x, 8), 0.57650273)
    def test_time(self):
        self.assertEqual(round(act_act_excel(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(act_act_excel(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 1.)
        self.assertEqual(round(act_act_excel(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(1. + FD2_366, 13))
        self.assertEqual(round(act_act_excel(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(1. - FD2_366, 13))
        self.assertEqual(round(act_act_excel(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_366, 13))
    def test_excel_basis_1(self):
        self.assertEqual(round(act_act_excel(1978, 2, 28, 2020, 5, 17), 13),
            round(42.214249331465700000, 13), '1978-02-28 -> 2020-05-17')
        self.assertEqual(round(act_act_excel(1993, 12, 2, 2022, 4, 18), 13),
            round(28.376380396093800000, 13), '1993-12-02 -> 2022-04-18')
        self.assertEqual(round(act_act_excel(2018, 12, 15, 2019, 3, 1), 13),
            round(0.208219178082192000, 13), '2018-12-15 -> 2019-03-01')
        self.assertEqual(round(act_act_excel(2018, 12, 31, 2019, 1, 1), 13),
            round(0.002739726027397260, 13), '2018-12-31 -> 2019-01-01')
        self.assertEqual(round(act_act_excel(1994, 6, 30, 1997, 6, 30), 13),
            round(3.000684462696780000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(act_act_excel(1994, 2, 10, 1994, 6, 30), 13),
            round(0.383561643835616000, 13), '1994-02-10 -> 1994-06-30')
        self.assertEqual(round(act_act_excel(2020, 2, 21, 2024, 3, 25), 13),
            round(4.088669950738920000, 13), '2020-02-21 -> 2024-03-25')
        self.assertEqual(round(act_act_excel(2020, 2, 29, 2021, 2, 28), 13),
            round(0.997267759562842000, 13), '2020-02-29 -> 2021-02-28')
        self.assertEqual(round(act_act_excel(2020, 1, 31, 2021, 2, 28), 13),
            round(1.077975376196990000, 13), '2020-01-31 -> 2021-02-28')
        self.assertEqual(round(act_act_excel(2020, 1, 31, 2021, 3, 31), 13),
            round(1.162790697674420000, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(act_act_excel(2020, 1, 31, 2020, 4, 30), 13),
            round(0.245901639344262000, 13), '2020-01-31 -> 2020-04-30')
        self.assertEqual(round(act_act_excel(2018, 2, 5, 2023, 5, 14), 13),
            round(5.268827019625740000, 13), '2018-02-05 -> 2023-05-14')
        self.assertEqual(round(act_act_excel(2020, 2, 29, 2024, 2, 28), 12),
            round(3.995621237000550000, 12), '2020-02-29 -> 2024-02-28')
        self.assertEqual(round(act_act_excel(2010, 3, 31, 2015, 8, 30), 13),
            round(5.416704701049750000, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(act_act_excel(2016, 2, 28, 2016, 10, 30), 13),
            round(0.669398907103825000, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(act_act_excel(2014, 1, 31, 2014, 8, 31), 13),
            round(0.580821917808219000, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(act_act_excel(2014, 2, 28, 2014, 9, 30), 13),
            round(0.586301369863014000, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(act_act_excel(2016, 2, 29, 2016, 6, 15), 13),
            round(0.292349726775956000, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(act_act_excel(2024, 1, 1, 2024, 12, 31), 13),
            round(0.997267759562842000, 13), '2024-01-01 -> 2024-12-31')
        self.assertEqual(round(act_act_excel(2024, 1, 1, 2025, 1, 2), 13),
            round(1.004103967168260000, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(act_act_excel(2024, 1, 1, 2024, 2, 29), 13),
            round(0.161202185792350000, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(act_act_excel(2024, 1, 1, 2024, 3, 1), 13),
            round(0.163934426229508000, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(act_act_excel(2023, 1, 1, 2023, 3, 1), 13),
            round(0.161643835616438000, 13), '2023-01-01 -> 2023-03-01')
        self.assertEqual(round(act_act_excel(2024, 2, 29, 2025, 2, 28), 13),
            round(0.997267759562842000, 13), '2024-02-29 -> 2025-02-28')
        self.assertEqual(round(act_act_excel(2024, 1, 1, 2028, 12, 31), 13),
            round(4.997263273125340000, 13), '2024-01-01 -> 2028-12-31')
        self.assertEqual(round(act_act_excel(2024, 3, 1, 2025, 3, 1), 13),
            round(1.000000000000000000, 13), '2024-03-01 -> 2025-03-01')
        self.assertEqual(round(act_act_excel(2024, 2, 29, 2025, 3, 1), 13),
            round(1.001367989056090000, 13), '2024-02-29 -> 2025-03-01')
        self.assertEqual(round(act_act_excel(2024, 2, 29, 2028, 2, 28), 12),
            round(3.995621237000550000, 12), '2024-02-29 -> 2028-02-28')
        self.assertEqual(round(act_act_excel(2024, 2, 29, 2028, 2, 29), 13),
            round(3.998357963875210000, 13), '2024-02-29 -> 2028-02-29')
        self.assertEqual(round(act_act_excel(2024, 3, 1, 2028, 3, 1), 13),
            round(3.998357963875210000, 13), '2024-03-01 -> 2028-03-01')

class TestActActIsda(unittest.TestCase):
    def test(self):
        x = act_act_isda(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.20547945)
        x = act_act_isda(2018, 12, 31, 2019, 1, 1)
        self.assertEqual(round(x, 8), 0)
        x = act_act_isda(1994, 6, 30, 1997, 6, 30)
        self.assertEqual(round(x, 8), 2.99726027)
        x = act_act_isda(1994, 2, 10, 1994, 6, 30)
        self.assertEqual(round(x, 8), round(140. / 365., 8))
    def test_time(self):
        self.assertEqual(round(act_act_isda(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 0.9972602739726)
        self.assertEqual(round(act_act_isda(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 0.997271502358)
        self.assertEqual(round(act_act_isda(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(0.9978186990045, 13))
        self.assertEqual(round(act_act_isda(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(0.9967243057115, 13))
        self.assertEqual(round(act_act_isda(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_366, 13))
    def test_yearfrac_1(self):
        x = act_act_isda(2012, 1, 1, 2012, 7, 30)
        self.assertEqual(round(x, 8), 0.57650273)
    def test_excel_basis_1(self):
        # AssertionError: 42.2126731042743 != 42.2142493314657
        #self.assertEqual(round(act_act_isda(1978, 2, 28, 2020, 5, 17), 13),
        #    round(42.214249331465700000, 13), '1978-02-28 -> 2020-05-17')
        # AssertionError: 28.372602739726 != 28.3763803960938
        #self.assertEqual(round(act_act_isda(1993, 12, 2, 2022, 4, 18), 13),
        #    round(28.376380396093800000, 13), '1993-12-02 -> 2022-04-18')
        # AssertionError: 0.2054794520548 != 0.2082191780822
        #self.assertEqual(round(act_act_isda(2018, 12, 15, 2019, 3, 1), 13),
        #    round(0.208219178082192000, 13), '2018-12-15 -> 2019-03-01')
        # AssertionError: 0.0 != 0.0027397260274
        #self.assertEqual(round(act_act_isda(2018, 12, 31, 2019, 1, 1), 13),
        #    round(0.002739726027397260, 13), '2018-12-31 -> 2019-01-01')
        # AssertionError: 2.9972602739726 != 3.0006844626968
        #self.assertEqual(round(act_act_isda(1994, 6, 30, 1997, 6, 30), 13),
        #    round(3.000684462696780000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(act_act_isda(1994, 2, 10, 1994, 6, 30), 13),
            round(0.383561643835616000, 13), '1994-02-10 -> 1994-06-30')
        # AssertionError: 4.0874316939891 != 4.0886699507389
        #self.assertEqual(round(act_act_isda(2020, 2, 21, 2024, 3, 25), 13),
        #    round(4.088669950738920000, 13), '2020-02-21 -> 2024-03-25')
        # AssertionError: 0.9949696833595 != 0.9972677595628
        #self.assertEqual(round(act_act_isda(2020, 2, 29, 2021, 2, 28), 13),
        #    round(0.997267759562842000, 13), '2020-02-29 -> 2021-02-28')
        # AssertionError: 1.0742046560371 != 1.077975376197
        #self.assertEqual(round(act_act_isda(2020, 1, 31, 2021, 2, 28), 13),
        #    round(1.077975376196990000, 13), '2020-01-31 -> 2021-02-28')
        # AssertionError: 1.1591361628864 != 1.1627906976744
        #self.assertEqual(round(act_act_isda(2020, 1, 31, 2021, 3, 31), 13),
        #    round(1.162790697674420000, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(act_act_isda(2020, 1, 31, 2020, 4, 30), 13),
            round(0.245901639344262000, 13), '2020-01-31 -> 2020-04-30')
        # AssertionError: 5.2657534246575 != 5.2688270196257
        #self.assertEqual(round(act_act_isda(2018, 2, 5, 2023, 5, 14), 13),
        #    round(5.268827019625740000, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 3.9945355191257 != 3.9956212370006
        #self.assertEqual(round(act_act_isda(2020, 2, 29, 2024, 2, 28), 13),
        #    round(3.995621237000550000, 13), '2020-02-29 -> 2024-02-28')
        # AssertionError: 5.413698630137 != 5.4167047010497
        #self.assertEqual(round(act_act_isda(2010, 3, 31, 2015, 8, 30), 13),
        #    round(5.416704701049750000, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(act_act_excel(2016, 2, 28, 2016, 10, 30), 13),
            round(0.669398907103825000, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(act_act_isda(2014, 1, 31, 2014, 8, 31), 13),
            round(0.580821917808219000, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(act_act_isda(2014, 2, 28, 2014, 9, 30), 13),
            round(0.586301369863014000, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(act_act_isda(2016, 2, 29, 2016, 6, 15), 13),
            round(0.292349726775956000, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(act_act_isda(2024, 1, 1, 2024, 12, 31), 13),
            round(0.997267759562842000, 13), '2024-01-01 -> 2024-12-31')
        # AssertionError: 1.0000074855902 != 1.0041039671683
        #self.assertEqual(round(act_act_isda(2024, 1, 1, 2025, 1, 2), 13),
        #    round(1.004103967168260000, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(act_act_isda(2024, 1, 1, 2024, 2, 29), 13),
            round(0.161202185792350000, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(act_act_isda(2024, 1, 1, 2024, 3, 1), 13),
            round(0.163934426229508000, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(act_act_excel(2023, 1, 1, 2023, 3, 1), 13),
            round(0.161643835616438000, 13), '2023-01-01 -> 2023-03-01')
        # AssertionError: 0.9949696833595 != 0.9972677595628
        #self.assertEqual(round(act_act_isda(2024, 2, 29, 2025, 2, 28), 13),
        #    round(0.997267759562842000, 13), '2024-02-29 -> 2025-02-28')
        # AssertionError: 4.9945355191257 != 4.9972632731253
        #self.assertEqual(round(act_act_isda(2024, 1, 1, 2028, 12, 31), 13),
        #    round(4.997263273125340000, 13), '2024-01-01 -> 2028-12-31')
        # AssertionError: 0.9949771689498 != 1.0
        #self.assertEqual(round(act_act_isda(2024, 3, 1, 2025, 3, 1), 13),
        #    round(1.000000000000000000, 13), '2024-03-01 -> 2025-03-01')
        # AssertionError: 0.9977094093869 != 1.0013679890561
        #self.assertEqual(round(act_act_isda(2024, 2, 29, 2025, 3, 1), 13),
        #    round(1.001367989056090000, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 3.9945355191257 != 3.9956212370006
        #self.assertEqual(round(act_act_isda(2024, 2, 29, 2028, 2, 28), 13),
        #    round(3.995621237000550000, 13), '2024-02-29 -> 2028-02-28')
        # AssertionError: 3.9972677595628 != 3.9983579638752
        #self.assertEqual(round(act_act_isda(2024, 2, 29, 2028, 2, 29), 13),
        #    round(3.998357963875210000, 13), '2024-02-29 -> 2028-02-29')
        # AssertionError: 3.9972677595628 != 3.9983579638752
        #self.assertEqual(round(act_act_isda(2024, 3, 1, 2028, 3, 1), 13),
        #    round(3.998357963875210000, 13), '2024-03-01 -> 2028-03-01')

class TestActActAfb(unittest.TestCase):
    def test(self):
        x = act_act_afb(2018, 12, 15, 2019, 3, 1)
        self.assertEqual(round(x, 8), 0.20547945)
        x = act_act_afb(2018, 12, 31, 2019, 1, 1)
        self.assertEqual(round(x, 8), 0)
        x = act_act_afb(1994, 6, 30, 1997, 6, 30)
        self.assertEqual(round(x, 8), 2.99726027)
        x = act_act_afb(1994, 2, 10, 1994, 6, 30)
        self.assertEqual(round(x, 8), round(140. / 365., 8))
    def test_time(self):
        self.assertEqual(round(act_act_afb(
            2021, 1, 1, 2022, 1, 1, 0.5, 0.5), 13), 0.9972602739726)
        self.assertEqual(round(act_act_afb(
            2020, 1, 1, 2021, 1, 1, 0.5, 0.5), 13), 0.997271502358)
        self.assertEqual(round(act_act_afb(
            2020, 1, 1, 2021, 1, 1, 0.4, 0.6), 13),
            round(0.9978186990045, 13))
        self.assertEqual(round(act_act_afb(
            2020, 1, 1, 2021, 1, 1, 0.6, 0.4), 13),
            round(0.9967243057115, 13))
        self.assertEqual(round(act_act_afb(
            2020, 1, 1, 2020, 1, 1, 0.4, 0.6), 13),
            round(FD2_366, 13))
    def test_yearfrac_1(self):
        x = act_act_afb(2012, 1, 1, 2012, 7, 30)
        self.assertEqual(round(x, 8), 0.57650273)
    def test_excel_basis_1(self):
        # AssertionError: 42.2126731042743 != 42.2142493314657
        #self.assertEqual(round(act_act_afb(1978, 2, 28, 2020, 5, 17), 13),
        #    round(42.214249331465700000, 13), '1978-02-28 -> 2020-05-17')
        # AssertionError: 28.372602739726 != 28.3763803960938
        #self.assertEqual(round(act_act_afb(1993, 12, 2, 2022, 4, 18), 13),
        #    round(28.376380396093800000, 13), '1993-12-02 -> 2022-04-18')
        # AssertionError: 0.2054794520548 != 0.2082191780822
        #self.assertEqual(round(act_act_afb(2018, 12, 15, 2019, 3, 1), 13),
        #    round(0.208219178082192000, 13), '2018-12-15 -> 2019-03-01')
        # AssertionError: 0.0 != 0.0027397260274
        #self.assertEqual(round(act_act_afb(2018, 12, 31, 2019, 1, 1), 13),
        #    round(0.002739726027397260, 13), '2018-12-31 -> 2019-01-01')
        # AssertionError: 2.9972602739726 != 3.0006844626968
        #self.assertEqual(round(act_act_afb(1994, 6, 30, 1997, 6, 30), 13),
        #    round(3.000684462696780000, 13), '1994-06-30 -> 1997-06-30')
        self.assertEqual(round(act_act_afb(1994, 2, 10, 1994, 6, 30), 13),
            round(0.383561643835616000, 13), '1994-02-10 -> 1994-06-30')
        # AssertionError: 4.0874316939891 != 4.0886699507389
        #self.assertEqual(round(act_act_afb(2020, 2, 21, 2024, 3, 25), 13),
        #    round(4.088669950738920000, 13), '2020-02-21 -> 2024-03-25')
        # AssertionError: 0.9949696833595 != 0.9972677595628
        #self.assertEqual(round(act_act_afb(2020, 2, 29, 2021, 2, 28), 13),
        #    round(0.997267759562842000, 13), '2020-02-29 -> 2021-02-28')
        # AssertionError: 1.0742046560371 != 1.077975376197
        #self.assertEqual(round(act_act_afb(2020, 1, 31, 2021, 2, 28), 13),
        #    round(1.077975376196990000, 13), '2020-01-31 -> 2021-02-28')
        # AssertionError: 1.1591361628864 != 1.1627906976744
        #self.assertEqual(round(act_act_afb(2020, 1, 31, 2021, 3, 31), 13),
        #    round(1.162790697674420000, 13), '2020-01-31 -> 2021-03-31')
        self.assertEqual(round(act_act_afb(2020, 1, 31, 2020, 4, 30), 13),
            round(0.245901639344262000, 13), '2020-01-31 -> 2020-04-30')
        # AssertionError: 5.2657534246575 != 5.2688270196257
        #self.assertEqual(round(act_act_afb(2018, 2, 5, 2023, 5, 14), 13),
        #    round(5.268827019625740000, 13), '2018-02-05 -> 2023-05-14')
        # AssertionError: 3.9949696833595 != 3.9956212370006
        #self.assertEqual(round(act_act_afb(2020, 2, 29, 2024, 2, 28), 13),
        #    round(3.995621237000550000, 13), '2020-02-29 -> 2024-02-28')
        # AssertionError: 5.413698630137 != 5.4167047010497
        #self.assertEqual(round(act_act_afb(2010, 3, 31, 2015, 8, 30), 13),
        #    round(5.416704701049750000, 13), '2010-03-31 -> 2015-08-30')
        self.assertEqual(round(act_act_afb(2016, 2, 28, 2016, 10, 30), 13),
            round(0.669398907103825000, 13), '2016-02-28 -> 2016-10-30')
        self.assertEqual(round(act_act_afb(2014, 1, 31, 2014, 8, 31), 13),
            round(0.580821917808219000, 13), '2014-01-31 -> 2014-08-31')
        self.assertEqual(round(act_act_afb(2014, 2, 28, 2014, 9, 30), 13),
            round(0.586301369863014000, 13), '2014-02-28 -> 2014-09-30')
        self.assertEqual(round(act_act_afb(2016, 2, 29, 2016, 6, 15), 13),
            round(0.292349726775956000, 13), '2016-02-29 -> 2016-06-15')
        self.assertEqual(round(act_act_afb(2024, 1, 1, 2024, 12, 31), 13),
            round(0.997267759562842000, 13), '2024-01-01 -> 2024-12-31')
        # AssertionError: 1.0000074855902 != 1.0041039671683
        #self.assertEqual(round(act_act_afb(2024, 1, 1, 2025, 1, 2), 13),
        #    round(1.004103967168260000, 13), '2024-01-01 -> 2025-01-02')
        self.assertEqual(round(act_act_afb(2024, 1, 1, 2024, 2, 29), 13),
            round(0.161202185792350000, 13), '2024-01-01 -> 2024-02-29')
        self.assertEqual(round(act_act_afb(2024, 1, 1, 2024, 3, 1), 13),
            round(0.163934426229508000, 13), '2024-01-01 -> 2024-03-01')
        self.assertEqual(round(act_act_afb(2023, 1, 1, 2023, 3, 1), 13),
            round(0.161643835616438000, 13), '2023-01-01 -> 2023-03-01')
        # AssertionError: 0.9949696833595 != 0.9972677595628
        #self.assertEqual(round(act_act_afb(2024, 2, 29, 2025, 2, 28), 13),
        #    round(0.997267759562842000, 13), '2024-02-29 -> 2025-02-28')
        # AssertionError: 4.9945355191257 != 4.9972632731253
        #self.assertEqual(round(act_act_afb(2024, 1, 1, 2028, 12, 31), 13),
        #    round(4.997263273125340000, 13), '2024-01-01 -> 2028-12-31')
        # AssertionError: 0.9972602739726 != 1.0
        #self.assertEqual(round(act_act_afb(2024, 3, 1, 2025, 3, 1), 13),
        #    round(1.000000000000000000, 13), '2024-03-01 -> 2025-03-01')
        # AssertionError: 0.9977094093869 != 1.0013679890561
        #self.assertEqual(round(act_act_afb(2024, 2, 29, 2025, 3, 1), 13),
        #    round(1.001367989056090000, 13), '2024-02-29 -> 2025-03-01')
        # AssertionError: 3.9949696833595 != 3.9956212370006
        #self.assertEqual(round(act_act_afb(2024, 2, 29, 2028, 2, 28), 13),
        #    round(3.995621237000550000, 13), '2024-02-29 -> 2028-02-28')
        # AssertionError: 3.9977094093869 != 3.9983579638752
        #self.assertEqual(round(act_act_afb(2024, 2, 29, 2028, 2, 29), 13),
        #    round(3.998357963875210000, 13), '2024-02-29 -> 2028-02-29')
        # AssertionError: 3.9995508645857 != 3.9983579638752
        #self.assertEqual(round(act_act_afb(2024, 3, 1, 2028, 3, 1), 13),
        #    round(3.998357963875210000, 13), '2024-03-01 -> 2028-03-01')

if __name__ == "__main__":
    unittest.main()
