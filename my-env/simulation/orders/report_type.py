from enum import Enum

class OrderReportType(Enum):
    """Enumerates an order report event types."""

    PENDING_NEW = 'pending_new'
    """
    Accepted indicates the order has been received by the broker and is being evaluated.

    The order will proceed to the PendingNew status.
    """

    NEW = 'new'
    """
    New reports a transition to the "new" order status.
    """

    REJECTED = 'rejected'
    """
    Rejected reports a transition to the "rejected" order status.
    """

    PARTIALLY_FILLED = 'partially_filled'
    """
    Partially-filled reports a transition to the "partially filled" order status.
    """

    FILLED = 'filled'
    """
    Filled reports a transition to the "filled" order status.
    """

    EXPIRED = 'expired'
    """
    Expired reports a transition to the "expired" order status.
    """

    PENDING_REPLACE = 'pending_replace'
    """
    Pending-replace reports a transition to the "pending replace" order status.
    """

    REPLACED = 'replaced'
    """
    Replaced reports that an order has been replaced.
    """

    REPLACE_REJECTED = 'replace_rejected'
    """
    Replace-rejected reports that an order replacement has been rejected.
    """

    PENDING_CANCEL = 'pending_cancel'
    """
    Pending-cancel reports a transition to the "pending cancel" order status.
    """

    CANCELED = 'canceled'
    """
    Canceled reports a transition to the "canceled" order status.
    """

    CANCEL_REJECTED = 'cancel_rejected'
    """
    Cancel-rejected reports that an order cancellation has been rejected.
    """

    ORDER_STATUS = 'order_status'
    """
    Order-status reports an order status.
    """
