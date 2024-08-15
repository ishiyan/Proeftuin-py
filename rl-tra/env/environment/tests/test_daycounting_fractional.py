import unittest
from datetime import datetime
from accounts.daycounting import year_frac, day_frac, DayCountConvention

SECONDS_IN_GREGORIAN_YEAR = 31_556_952
SECONDS_IN_LEAP_YEAR = 31_622_400
SECONDS_IN_NONLEAP_YEAR = 31_536_000

class TestYearFrac(unittest.TestCase):

    def test_raw_method(self):
        date_time_1 = datetime(2020, 1, 1, 0, 0, 0)
        date_time_2 = datetime(2021, 1, 1, 0, 0, 0)
        result = year_frac(date_time_1, date_time_2, DayCountConvention.RAW)
        self.assertAlmostEqual(result, SECONDS_IN_LEAP_YEAR/SECONDS_IN_GREGORIAN_YEAR, places=13)

        date_time_1 = datetime(2021, 1, 1, 0, 0, 0)
        date_time_2 = datetime(2022, 1, 1, 0, 0, 0)
        result = year_frac(date_time_1, date_time_2, DayCountConvention.RAW)
        self.assertAlmostEqual(result, SECONDS_IN_NONLEAP_YEAR/SECONDS_IN_GREGORIAN_YEAR, places=13)

    def test_invalid_method(self):
        date_time_1 = datetime(2020, 1, 1, 0, 0, 0)
        date_time_2 = datetime(2021, 1, 1, 0, 0, 0)
        with self.assertRaises(ValueError):
            year_frac(date_time_1, date_time_2, 'INVALID_METHOD')

class TestDayFrac(unittest.TestCase):

    def test_raw_method(self):
        date_time_1 = datetime(2020, 1, 1, 0, 0, 0)
        date_time_2 = datetime(2021, 1, 1, 0, 0, 0)
        result = day_frac(date_time_1, date_time_2, DayCountConvention.RAW)
        self.assertAlmostEqual(result, 366., places=13)

        date_time_1 = datetime(2021, 1, 1, 0, 0, 0)
        date_time_2 = datetime(2022, 1, 1, 0, 0, 0)
        result = day_frac(date_time_1, date_time_2, DayCountConvention.RAW)
        self.assertAlmostEqual(result, 365, places=13)

    def test_invalid_method(self):
        date_time_1 = datetime(2020, 1, 1, 0, 0, 0)
        date_time_2 = datetime(2021, 1, 1, 0, 0, 0)
        with self.assertRaises(ValueError):
            day_frac(date_time_1, date_time_2, 'INVALID_METHOD')

if __name__ == '__main__':
    unittest.main()