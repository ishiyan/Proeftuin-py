from enum import Enum

class OrderStatus(Enum):
    """Enumerates the states an order runs through during its lifetime."""

    ACCEPTED = 'accepted'
    """
    Accepted indicates the order has been received by the broker and is being evaluated.

    The order will proceed to the Pending-new status.
    """

    PENDING_NEW = 'pending_new'
    """
    Pending-new indicates the order has been accepted by the broker but not yet acknowledged
    for execution.

    The order will proceed to the either New or the Rejected status.
    """

    NEW = 'new'
    """
    New indicates the order has been acknowledged by the broker and becomes the
    outstanding order with no executions.

    The order can proceed to the Filled, the Partially-filled, the Expired,
    the Pending-cancel, the Pending-replace, or to the Rejected status.
    """

    REJECTED = 'rejected'
    """
    Rejected indicates the order has been rejected by the broker. No executions were done.

    This is a terminal state of an order, no further changes are allowed.
    """

    PARTIALLY_FILLED = 'partially_filled'
    """
    Partially-filled indicates the order has been partially filled and has remaining quantity.

    The order can proceed to the Filled, the Pending-cancel, or to the
    Pending-replace status.
    """

    FILLED = 'filled'
    """
    Filled indicates the order has been completely filled.

    This is a terminal state of an order, no further changes are allowed.
    """

    EXPIRED = 'expired'
    """
    Expired indicates the order (with or without executions) has been canceled
    in broker's system due to time in force instructions.

    The only exceptions are Fill-or-kill and Immediate-or-cancel
    orders that have Canceled as terminal order state.

    This is a terminal state of an order, no further changes are allowed.
    """

    PENDING_REPLACE = 'pending_replace'
    """
    Pending-replace indicates a replace request has been sent to the broker, but the broker
    hasn't replaced the order yet.

    The order will proceed back to the previous status.
    """

    PENDING_CANCEL = 'pending_cancel'
    """
    Pending-cancel indicates a cancel request has been sent to the broker, but
    the broker hasn't canceled the order yet.

    The order will proceed to the either Canceled
    or back to the previous status.
    """

    CANCELED = 'canceled'
    """
    Canceled indicates the order (with or without executions)
    has been canceled by the broker.

    The order may still be partially filled.
    This is a terminal state of an order, no further changes are allowed.
    """
