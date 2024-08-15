import unittest

from accounts.daycounting import DayCountConvention

class TestDayCountConvention(unittest.TestCase):
    def test_from_string_valid(self):
        self.assertEqual(DayCountConvention.from_string('raw'),
                         DayCountConvention.RAW)
        self.assertEqual(DayCountConvention.from_string('30/360 us'),
                         DayCountConvention.THIRTY_360_US)
        self.assertEqual(DayCountConvention.from_string('30/360 us eom'),
                         DayCountConvention.THIRTY_360_US_EOM)
        self.assertEqual(DayCountConvention.from_string('30/360 us nasd'),
                         DayCountConvention.THIRTY_360_US_NASD)
        self.assertEqual(DayCountConvention.from_string('30/360 eu'),
                         DayCountConvention.THIRTY_360_EU)
        self.assertEqual(DayCountConvention.from_string('30/360 eu2'),
                         DayCountConvention.THIRTY_360_EU_M2)
        self.assertEqual(DayCountConvention.from_string('30/360 eu3'),
                         DayCountConvention.THIRTY_360_EU_M3)
        self.assertEqual(DayCountConvention.from_string('30/360 eu+'),
                         DayCountConvention.THIRTY_360_EU_PLUS)
        self.assertEqual(DayCountConvention.from_string('30/365'),
                         DayCountConvention.THIRTY_365)
        self.assertEqual(DayCountConvention.from_string('act/360'),
                         DayCountConvention.ACT_360)
        self.assertEqual(DayCountConvention.from_string('act/365 fixed'),
                         DayCountConvention.ACT_365_FIXED)
        self.assertEqual(DayCountConvention.from_string('act/365 nonleap'),
                         DayCountConvention.ACT_365_NONLEAP)
        self.assertEqual(DayCountConvention.from_string('act/act excel'),
                         DayCountConvention.ACT_ACT_EXCEL)
        self.assertEqual(DayCountConvention.from_string('act/act isda'),
                         DayCountConvention.ACT_ACT_ISDA)
        self.assertEqual(DayCountConvention.from_string('act/act afb'),
                         DayCountConvention.ACT_ACT_AFB)

    def test_from_string_case_insensitive(self):
        self.assertEqual(DayCountConvention.from_string('Act/Act Excel'), DayCountConvention.ACT_ACT_EXCEL)
        self.assertEqual(DayCountConvention.from_string('ACT/ACT AFB'), DayCountConvention.ACT_ACT_AFB)
        self.assertEqual(DayCountConvention.from_string('act/act ISDA'), DayCountConvention.ACT_ACT_ISDA)

    def test_from_string_invalid(self):
        with self.assertRaises(ValueError) as context:
            DayCountConvention.from_string('invalid convention')
        self.assertIn("Day count convention 'invalid convention' must be one of:",
                      str(context.exception))

    def test_from_string_non_string(self):
        with self.assertRaises(ValueError) as context:
            DayCountConvention.from_string(123)
        self.assertIn("Day count convention '123' must be one of:",
                      str(context.exception))

if __name__ == "__main__":
    unittest.main()
