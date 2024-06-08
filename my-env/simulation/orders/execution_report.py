from datetime import datetime

from ..currencies import Currency, Currencies
from .report_type import OrderReportType
from .status import OrderStatus
from .order import Order

class OrderExecutionReport:
    """A report event for an order in a single instrument.

    Parameters
    ----------
    order : Order
        The underlying order for this execution report.

        If there were any successful order replacements, this will be the most recent version.
    transaction_time : datetime
        The date and time when the business represented by this report occurred.
    status : OrderStatus
        The current state of an order as understood by the broker.
    report_type : OrderReportType
        Identifies an action of this report.
    ID : str
        The unique identifier of this report as assigned by the sell-side.
    note : str
        A free-format text that accompany this report.
    replace_source_order : Order, optional
        The replace source order. Filled when report type is Replaced or Replace-rejected.
    replace_target_order : Order, optional
        The replace target order. Filled when report type is Replaced or Replace-rejected.
    last_filled_price : float
        The price (in order instrument's currency) of the last fill.
    average_price : float
        The average price (in order instrument's currency) of all fills.
    last_filled_quantity : float
        the quantity bought or sold on the last fill.
    leaves_quantity : float
        The remaining quantity to be filled.

        If the order status is Canceled, Expired or Rejected (in which case
        the order is no longer active) then this could be 0, otherwise it is
        
        order.quantity - cumulative_quantity.
    cumulative_quantity : float
        The total quantity filled.
    last_fill_commission : float
        The commission (in commission currency) of the last fill.
    cumulative_commission : float
        The total commission (in commission currency) of all fills.
    commission_currency : Currency
        The currency in which the commission is denominated.
    """
    def __init__(self,
                 order: Order,
                 transaction_time: datetime,
                 status: OrderStatus = OrderStatus.NEW,
                 report_type: OrderReportType = OrderReportType.PENDING_NEW,
                 ID: str = '',
                 note: str = '',
                 replace_source_order: Order = None,
                 replace_target_order: Order = None,
                 last_filled_price: float = 0,
                 average_price: float = 0,
                 last_filled_quantity: float = 0,
                 leaves_quantity: float = 0,
                 cumulative_quantity: float = 0,
                 last_fill_commission: float = 0,
                 cumulative_commission: float = 0,
                 commission_currency: Currency = Currencies.USD,
                 ):
        self.order = order
        self.transaction_time = transaction_time
        self.status = status
        self.report_type = report_type
        self.ID = ID
        self.note = note
        self.replace_source_order = replace_source_order
        self.replace_target_order = replace_target_order
        self.last_filled_price = last_filled_price
        self.average_price = average_price
        self.last_filled_quantity = last_filled_quantity
        self.leaves_quantity = leaves_quantity
        self.cumulative_quantity = cumulative_quantity
        self.last_fill_commission = last_fill_commission
        self.cumulative_commission = cumulative_commission
        self.commission_currency = commission_currency
        
    def __str__(self):
        return f'{self.order.instrument.symbol} {self.status} {self.report_type}'

    def __repr__(self):
        attributes = ['order', 'transaction_time', 'status', 'report_type', 'ID',
            'note', 'replace_source_order', 'replace_target_order', 'last_filled_price',
            'average_price', 'last_filled_quantity', 'leaves_quantity', 'cumulative_quantity',
            'last_fill_commission', 'cumulative_commission', 'commission_currency']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        #attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes if getattr(self, attr) is not None]
        return 'OrderSingleExecutionReport(' + ', '.join(attr_strings) + ')'
