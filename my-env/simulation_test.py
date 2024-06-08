from datetime import datetime, timedelta
import unittest

from simulation import *

class TestCurrency(unittest.TestCase):
    def setUp(self):
        self.currency_USD = Currencies.USD
        self.currency_EUR = Currencies.EUR

    def test_eq_same_currency(self):
        self.assertTrue(self.currency_USD == self.currency_USD)

    def test_eq_different_currency(self):
        self.assertFalse(self.currency_USD == self.currency_EUR)

    def test_eq_non_currency(self):
        self.assertFalse(self.currency_USD == 'USD')

    def test_cannot_create_same_currency_twice(self):
        with self.assertRaises(ValueError):
            Currency('USD', 2, '$')

class TestCurrencies(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Currencies()

    def test_currency_attributes(self):        
        self.assertEqual(Currencies.XXX.symbol, 'XXX')
        self.assertEqual(Currencies.XXX.precision, 2)
        self.assertEqual(Currencies.XXX.display, '')
        self.assertEqual(Currencies.XXX.name, 'No currency')
                         
        self.assertEqual(Currencies.XAU.symbol, 'XAU')
        self.assertEqual(Currencies.XAU.precision, 5)
        self.assertEqual(Currencies.XAU.display, '')
        self.assertEqual(Currencies.XAU.name, 'Gold (one troy ounce)')
                         
        self.assertEqual(Currencies.USD.symbol, 'USD')
        self.assertEqual(Currencies.USD.precision, 2)
        self.assertEqual(Currencies.USD.display, '$')
        self.assertEqual(Currencies.USD.name, 'US Dollar')

        self.assertEqual(Currencies.USX.symbol, 'USX')
        self.assertEqual(Currencies.USX.precision, 0)
        self.assertEqual(Currencies.USX.display, '¢')
        self.assertEqual(Currencies.USX.name, 'US cent')

class TestUpdatableCurrencyConverter(unittest.TestCase):
    def setUp(self):
        self.converter = UpdatableCurrencyConverter()

    def test_known_base_currencies(self):
        currencies = self.converter.known_base_currencies(Currencies.EUR)
        self.assertEqual(currencies, [])

        self.converter.update(Currencies.USD, Currencies.EUR, 0.85)
        currencies = self.converter.known_base_currencies(Currencies.EUR)
        self.assertEqual(currencies, [Currencies.USD])

        self.converter.update(Currencies.GBP, Currencies.EUR, 0.86)
        currencies = self.converter.known_base_currencies(Currencies.EUR)
        self.assertEqual(currencies, [Currencies.USD, Currencies.GBP])

        currencies = self.converter.known_base_currencies(Currencies.XAG)
        self.assertEqual(currencies, [])

    def test_known_term_currencies(self):
        currencies = self.converter.known_term_currencies(Currencies.USD)
        self.assertEqual(currencies, [])

        self.converter.update(Currencies.USD, Currencies.EUR, 0.85)
        currencies = self.converter.known_term_currencies(Currencies.USD)
        self.assertEqual(currencies, [Currencies.EUR])

        self.converter.update(Currencies.USD, Currencies.GBP, 0.86)
        currencies = self.converter.known_term_currencies(Currencies.USD)
        self.assertEqual(currencies, [Currencies.EUR, Currencies.GBP])

        currencies = self.converter.known_term_currencies(Currencies.XAG)
        self.assertEqual(currencies, [])

    def test_update_same_currency(self):
        self.converter.update(Currencies.USD, Currencies.USD, 1)

        currencies = self.converter.known_base_currencies(Currencies.USD)
        self.assertEqual(currencies, [])

        currencies = self.converter.known_term_currencies(Currencies.USD)
        self.assertEqual(currencies, [])

    def test_update_different_currency(self):
        self.converter.update(Currencies.USD, Currencies.EUR, 0.85)
        rate = self.converter.exchange_rate(Currencies.USD, Currencies.EUR)
        self.assertEqual(rate, 0.85)

        self.converter.update(Currencies.USD, Currencies.EUR, 0.86)
        rate = self.converter.exchange_rate(Currencies.USD, Currencies.EUR)
        self.assertEqual(rate, 0.86)

    def test_exchange_rate(self):
        with self.assertRaises(ValueError):
            self.converter.exchange_rate(Currencies.USD, Currencies.EUR)

        rate = self.converter.exchange_rate(Currencies.USD, Currencies.USD)
        self.assertEqual(rate, 1)

        self.converter.update(Currencies.USD, Currencies.EUR, 0.85)
        rate = self.converter.exchange_rate(Currencies.USD, Currencies.EUR)
        self.assertEqual(rate, 0.85)

    def test_convert(self):
        with self.assertRaises(ValueError):
            self.converter.convert(100, Currencies.USD, Currencies.EUR)

        converted, rate = self.converter.convert(100, Currencies.USD, Currencies.USD)
        self.assertEqual(converted, 100)
        self.assertEqual(rate, 1)

        self.converter.update(Currencies.USD, Currencies.EUR, 0.85)
        converted, rate = self.converter.convert(100, Currencies.USD, Currencies.EUR)
        self.assertEqual(converted, 85)
        self.assertEqual(rate, 0.85)

class TestMIC(unittest.TestCase):
    def test_mic_attributes(self):
        mic_NASD = MICs.NASD
        self.assertEqual(mic_NASD.mic, 'NASD')
        self.assertEqual(mic_NASD.operational_mic, 'XNAS')
        self.assertEqual(mic_NASD.country_code, 'US')

    def test_eq_same_mic(self):
        mic_XNAS = MICs.XNAS
        mic_XNAS_2 = MICs.XNAS
        self.assertTrue(mic_XNAS == mic_XNAS_2)

    def test_eq_different_mic(self):
        mic_XNAS = MICs.XNAS
        mic_XNYS = MICs.XNYS
        self.assertFalse(mic_XNAS == mic_XNYS)

class TestMICs(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            MICs()

    def test_getitem(self):
        mic = MICs['XNAS']
        self.assertIsInstance(mic, MIC)
        self.assertEqual(mic.mic, 'XNAS')

    def test_getitem_invalid_key(self):
        with self.assertRaises(KeyError):
            MICs['INVALID']

class TestInstrument(unittest.TestCase):
    def test_instrument_attributes(self):
        symbol = 'AAPL'
        name = 'Apple Inc.'
        isin = 'US0378331005'
        cfi = 'ESVUFR'
        mic = MICs.XNAS
        currency = Currencies.EUX
        instrument_type = InstrumentType.ETF
        status = InstrumentStatus.INACTIVE
        price_decimal_places = 4
        price_min_increment = 0.001
        price_factor = 100
        initial_margin = 1000.0

        instrument = Instrument(symbol, name, isin, cfi, mic,
            currency, instrument_type, status, price_decimal_places,
            price_min_increment, price_factor, initial_margin)

        self.assertEqual(instrument.symbol, symbol)
        self.assertEqual(instrument.name, name)
        self.assertEqual(instrument.ISIN, isin)
        self.assertEqual(instrument.CFI, cfi)
        self.assertEqual(instrument.MIC, mic)
        self.assertEqual(instrument.currency, currency)
        self.assertEqual(instrument.instrument_type, instrument_type)
        self.assertEqual(instrument.status, status)
        self.assertEqual(instrument.price_decimal_places, price_decimal_places)
        self.assertEqual(instrument.price_min_increment, price_min_increment)
        self.assertEqual(instrument.price_factor, price_factor)
        self.assertEqual(instrument.initial_margin, initial_margin)

class TestOrder(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument('AAPL', 'Apple Inc.', 'US0378331005', 'ESVUFR', 'XNAS')

    def test_order_defaults(self):
        order = Order(self.instrument)
        self.assertEqual(order.instrument, self.instrument)
        self.assertEqual(order.type, OrderType.MARKET)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.time_in_force, OrderTimeInForce.DAY)
        self.assertEqual(order.quantity, 1)
        self.assertIsNone(order.minimum_quantity)
        self.assertIsNone(order.limit_price)
        self.assertIsNone(order.stop_price)
        self.assertIsNone(order.trailing_distance)
        self.assertIsNone(order.creation_time)
        self.assertIsNone(order.expiration_time)
        self.assertEqual(order.note, '')

    def test_order_set_attributes(self):
        t0 = datetime.now()
        t1 = t0 + timedelta(seconds=1)

        order = Order(self.instrument, OrderType.LIMIT, OrderSide.SELL,
            OrderTimeInForce.FILL_OR_KILL, 10, 5, 150.1, 150.2, 150.3,
            t0, t1, 'Test order')

        self.assertEqual(order.type, OrderType.LIMIT)
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.time_in_force, OrderTimeInForce.FILL_OR_KILL)
        self.assertEqual(order.quantity, 10)
        self.assertEqual(order.minimum_quantity, 5)
        self.assertEqual(order.limit_price, 150.1)
        self.assertEqual(order.stop_price, 150.2)
        self.assertEqual(order.trailing_distance, 150.3)
        self.assertEqual(order.creation_time, t0)
        self.assertEqual(order.expiration_time, t1)
        self.assertEqual(order.note, 'Test order')

class TestOrderExecutionReport(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument('AAPL', 'Apple Inc.', 'US0378331005', 'ESVUFR', 'XNAS')
        self.order = Order(self.instrument)
        self.transaction_time = datetime.now()
        self.report = OrderExecutionReport(self.order, self.transaction_time, OrderStatus.NEW)

    def test_report_attributes(self):
        self.assertEqual(self.report.order, self.order)
        self.assertEqual(self.report.transaction_time, self.transaction_time)
        self.assertEqual(self.report.status, OrderStatus.NEW)
        self.assertEqual(self.report.report_type, OrderReportType.PENDING_NEW)
        self.assertEqual(self.report.ID, '')
        self.assertEqual(self.report.note, '')
        self.assertIsNone(self.report.replace_source_order)
        self.assertIsNone(self.report.replace_target_order)
        self.assertEqual(self.report.last_filled_price, 0)
        self.assertEqual(self.report.average_price, 0)
        self.assertEqual(self.report.last_filled_quantity, 0)
        self.assertEqual(self.report.leaves_quantity, 0)
        self.assertEqual(self.report.cumulative_quantity, 0)
        self.assertEqual(self.report.last_fill_commission, 0)
        self.assertEqual(self.report.cumulative_commission, 0)
        self.assertEqual(self.report.commission_currency, Currencies.USD)

class TestOrderExecution(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument('AAPL', 'Apple Inc.', 'US0378331005', 'ESVUFR', 'XNAS')
        self.order = Order(self.instrument)
        self.report = OrderExecutionReport(self.order, OrderStatus.NEW)
        self.currency_converter = UpdatableCurrencyConverter()
        self.execution = OrderExecution(self.report, self.currency_converter)

    def test_execution_attributes(self):
        self.assertEqual(self.execution.report_ID, self.report.ID)
        self.assertEqual(self.execution.report_time, self.report.transaction_time)
        self.assertEqual(self.execution.side, self.order.side)
        
class TestScalarSeries(unittest.TestCase): # TODO: Add proper tests
    def setUp(self):
        self.series = ScalarSeries()
        self.now = datetime.now()
        self.series._series = [(self.now, 1.0), (self.now + timedelta(seconds=1), 2.0)]

    def test_current_value(self):
        self.series._scalar = (self.now, 1.5)
        self.assertEqual(self.series.current_value(), 1.5)

    def test_current_value_no_scalar(self):
        self.series._scalar = None
        self.assertEqual(self.series.current_value(), 0.0)

    def test_at(self):
        self.assertEqual(self.series.at(self.now + timedelta(seconds=0.5)), 1.0)
        self.assertEqual(self.series.at(self.now + timedelta(seconds=1.5)), 2.0)

    def test_at_empty_series(self):
        self.series._series = []
        self.assertEqual(self.series.at(self.now), 0.0)

class TestAccountTransaction(unittest.TestCase):
    def test_init_and_immutability(self):
        action = AccountAction.CREDIT
        time = datetime.now()
        currency = Currencies.USD
        amount = 100.0
        conversion_rate = 1.0
        amount_converted = 100.0
        note = "Test transaction"

        transaction = AccountTransaction(action, time, currency, amount, conversion_rate, amount_converted, note)

        self.assertEqual(transaction.action, action)
        self.assertEqual(transaction.time, time)
        self.assertEqual(transaction.currency, currency)
        self.assertEqual(transaction.amount, amount)
        self.assertEqual(transaction.conversion_rate, conversion_rate)
        self.assertEqual(transaction.amount_converted, amount_converted)
        self.assertEqual(transaction.note, note)

        with self.assertRaises(TypeError):
            transaction.amount = 100

class TestAccount(unittest.TestCase):
    def setUp(self):
        self.converter = UpdatableCurrencyConverter()
        self.account = Account('Test Holder', Currencies.EUR, self.converter)

    def test_account_attributes(self):
        self.assertEqual(self.account.holder, 'Test Holder')
        self.assertEqual(self.account.currency, Currencies.EUR)
        self.assertEqual(self.account.balance(), 0.0)
        self.assertEqual(self.account.balance_history(), [])
        self.assertEqual(self.account.transaction_history(), [])

class TestRoundtrip(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument('AAPL', 'Apple Inc.', 'US0378331005', 'ESVUFR', 'XNAS')
        self.currency_converter = UpdatableCurrencyConverter()

        self.entry_order = Order(self.instrument)
        self.entry_report = OrderExecutionReport(self.entry_order, OrderStatus.FILLED)
        self.entry = OrderExecution(self.entry_report, self.currency_converter)

        self.exit_order = Order(self.instrument)
        self.exit_report = OrderExecutionReport(self.exit_order, OrderStatus.FILLED)
        self.exit = OrderExecution(self.exit_report, self.currency_converter)

        self.entry.price = 100.0
        self.exit.price = 110.0
        self.entry.quantity = 100
        self.exit.quantity = 100
        self.entry.unrealized_price_high = 105.0
        self.entry.unrealized_price_low = 95.0
        self.exit.unrealized_price_high = 115.0
        self.exit.unrealized_price_low = 105.0
        self.entry.report_time = datetime.now()
        self.exit.report_time = self.entry.report_time + timedelta(seconds=10)
        
        self.roundtrip = Roundtrip(self.instrument, self.entry, self.exit, 100)

    def test_roundtrip_attributes(self):
        self.assertEqual(self.roundtrip.instrument, self.instrument)
        self.assertEqual(self.roundtrip.quantity, 100)

        with self.assertRaises(TypeError):
            self.roundtrip.quantity = 100

class TestPnL(unittest.TestCase):
    def setUp(self):
        self.pnl = PnL()

    def test_add(self):
        self.pnl.add(datetime(2022, 1, 1), 1000.0, 200.0, 800.0, 0.0)
        self.assertEqual(self.pnl.amount(), 200.0)
        self.assertEqual(self.pnl.unrealized_amount(), 800.0)
        self.assertEqual(self.pnl.percentage(), 20.0)

        self.pnl.add(datetime(2022, 1, 2), 1000.0, 20.0, 80.0, 4.0)
        self.assertEqual(self.pnl.amount(), 24.0)
        self.assertEqual(self.pnl.unrealized_amount(), 80.0)
        self.assertEqual(self.pnl.percentage(), 2.4)

        h = self.pnl.amount_history()
        t, v = h[0]
        self.assertEqual(t, datetime(2022, 1, 1))
        self.assertEqual(v, 200.0)
        t, v = h[1]
        self.assertEqual(t, datetime(2022, 1, 2))
        self.assertEqual(v, 24.0)

        h = self.pnl.unrealized_amount_history()
        t, v = h[0]
        self.assertEqual(t, datetime(2022, 1, 1))
        self.assertEqual(v, 800.0)
        t, v = h[1]
        self.assertEqual(t, datetime(2022, 1, 2))
        self.assertEqual(v, 80.0)

        h = self.pnl.percentage_history()
        t, v = h[0]
        self.assertEqual(t, datetime(2022, 1, 1))
        self.assertEqual(v, 20.0)
        t, v = h[1]
        self.assertEqual(t, datetime(2022, 1, 2))
        self.assertEqual(v, 2.4)

class TestDrawdown(unittest.TestCase):
    def setUp(self):
        self.drawdown = Drawdown()

    def test_amount(self):
        self.drawdown.add(datetime(2022, 1, 1), -1000.0)
        self.assertEqual(self.drawdown.amount(), -1000.0)

    def test_amount_history(self):
        self.drawdown.add(datetime(2022, 1, 1), -1000.0)
        self.assertEqual(self.drawdown.amount_history(), [(datetime(2022, 1, 1), -1000.0)])

    #def test_percentage(self):
    #    self.drawdown.add(datetime(2022, 1, 1), -1000.0)
    #    self.assertEqual(self.drawdown.percentage(), -10.0)

    #def test_percentage_history(self):
    #    self.drawdown.add(datetime(2022, 1, 1), -1000.0)
    #    self.drawdown.add(datetime(2022, 1, 1), -1000.0)
    #    self.assertEqual(self.drawdown.percentage_history(), [(datetime(2022, 1, 1), -10.0)])

if __name__ == "__main__":
    unittest.main()
