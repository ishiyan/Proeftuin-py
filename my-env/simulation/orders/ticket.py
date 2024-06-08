from abc import ABC, abstractmethod
from typing import List

from .execution_report import OrderExecutionReport
from .order import Order
from .status import OrderStatus

class OrderTicket(ABC):
    """A ticket to track a new order in a single instrument.

    Parameters
    ----------
    order : Order
        The underlying order for this ticket.

        If there were any successful order replacements,
        this will be the most recent version.
    client_order_ID : str
        The unique identifier for an order as assigned
        by the buy-side (institution, broker, intermediary etc.).
    order_ID : str
        The unique identifier for an order as assigned by the sell-side.
    status : OrderStatus
        The current state of an order as understood by the broker.
    last_report : OrderExecutionReport
        The last execution report for this order, None if not any.
    reports : List[OrderExecutionReport]
        The list of execution reports in the chronological order.
    """
    def __init__(self,
                 order: Order,
                 client_order_ID: str = '',
                 order_ID: str = '',
                 status: OrderStatus = OrderStatus.ACCEPTED,
                 last_report: OrderExecutionReport = None,
                 reports: List[OrderExecutionReport] = [],
                 ):
        self.order = order
        self.client_order_ID = client_order_ID
        self.order_ID = order_ID
        self.status = status
        self.last_report = last_report
        self.reports = reports

    def __str__(self):
        return f'{self.order.instrument.symbol} {self.status} {self.client_order_ID} {self.order_ID}'

    def __repr__(self):
        attributes = ['order', 'client_order_ID', 'order_ID', 'status',
            'last_report', 'reports']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        #attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes if getattr(self, attr) is not None]
        return 'OrderTicket(' + ', '.join(attr_strings) + ')'

    @abstractmethod
    def cancel(self):
        """Cancels this order

        If order has been already completed (successfully or not), does nothing.

        Produces an execution report on completion.
      """

    @abstractmethod
    def cancel_replace(self, replacement_order: Order):
        """Used to change the parameters of an existing order.

        Allowed modifications to an order include:
        - reducing or increasing order quantity
        - changing a limit order to a market order
        - changing the limit price
        - changing time in force

        Modifications cannot include:
        - changing order side
        - changing series
        - reducing quantity to zero (canceling the order)
        - re-opening a filled order by increasing quantity

        Unchanging attributes to be carried over from the
        original order must be specified in the replacement.

        If the order has been completed (successfully or not), does nothing.

        Produces an execution report on completion.
        """        
