from typing import Tuple, Sequence, Union, Literal, Optional
from numbers import Real
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from ..providers import Trade
from ..frame import Frame
from .trade_aggregator import TradeAggregator

class IntervalTradeAggregator(TradeAggregator):
    """
    Groups trades into frames based on specified interval and methodology.

    Args:
        method (str):
            Method of how to group the flow of records into frames.
            Possible values: `'time'`, `'tick'`, `'volume'`, `'money'`.
        interval (int):
            Initial threshold of the frame. The units depend on the
            aggregation method.
            - If method is `'time'`, it is the number of seconds.            
            - If method is `'tick'`, it is the number of trades.            
            - If method is `'volume'`, it is the volume units.
            - If method is `'money'`, it is the money units.
            It is used to group trades into frames.
        duration (tuple)
            Tuple with two values: (min_duration, max_duration).
            If frame duration exceeds max_duration, it will be finalized.
            If frame duration exceeds min_duration and frame is full,
            it will be finalized.
            Default: (1, 4*60*60)
    """
    FramingMethods = [
        'time',
        'tick',
        'volume',
        'money']
    
    def __init__(self,
                 method: Literal['time', 'tick', 'volume', 'money'],
                 interval: Real,
                 duration: Tuple[Real, Real] = (1, 4*60*60)):
        super().__init__()
        
        if not isinstance(method, str) or \
            method not in IntervalTradeAggregator.FramingMethods:
            raise ValueError(f"method {method} must be one of ["
                             "'time', 'tick', 'volume', 'money']")

        if not isinstance(interval, Real) or interval <= 0:
            raise ValueError(f'interval {interval} must be apositive real number')

        if not (isinstance(duration, tuple) and len(duration) == 2 and 
            isinstance(duration[0], Real) and isinstance(duration[1], Real)):
            raise ValueError(f'duration {duration} must be a tuple of two real numbers')
        
        self.method = method
        self.interval = interval
        self.duration = duration        
        self.frame = None
        
    @property
    def name(self):
        return f'{self.method}@{self.interval}'

    def reset(self):
        self.frame: Optional[Frame] = None

    def aggregate(self, trades: Sequence[Trade]) -> Optional[Frame]:
        result = None
        trade = trades[-1]
        frame = self.frame

        if (frame is None) or ((self.method == 'time') and \
                               (trade.datetime > frame.time_end)):
            # Initialize new frame.
            result = frame.finalize() if (frame is not None) else None
            frame = Frame(time_start=trade.datetime)
            self.frame = frame
            if self.method == 'time':
                # Round time_start to specified interval and also get time_end.
                time_start, time_end = _get_time_span(trade.datetime, self.interval)
                frame.time_start = time_start
                frame.time_end = time_end

        # Update the frame.
        frame.update(trades)
        
        # Check for the end of the frame.
        if ((frame.duration >= self.duration[0]) and (
                ((self.method == 'tick') and (frame.ticks >= self.interval)) or
                ((self.method == 'volume') and (frame.volume >= self.interval)) or
                ((self.method == 'money') and (frame.money >= self.interval))
            ) or (frame.duration >= self.duration[1])):
            # Reset the frame.
            result = frame.finalize() if (frame is not None) else None
            self.frame = None
            
        return result

    def finish(self) -> Optional[Frame]:
        result = self.frame.finalize()
        self.frame = None
        return result

    def __repr__(self):
        return (f'{self.__class__.__name__}('
            f'method={self.method}, interval={self.interval}, '
            f'duration={self.duration})')

def _get_time_span(dt: Union[datetime, datetime],
                   interval: int) -> Tuple[datetime, datetime]:
    if not isinstance(dt, datetime):
        dt = datetime.fromisoformat(dt)
    # Calculate start of frame.
    if interval <= 24*60*60:
        # Get start of day.
        s = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Get start of year.
        s = dt.replace(month=1, day=1, hour=0, minute=0, second=0,
                       microsecond=0)
    # Calculate start of frame with respect to start time s
    # and the frame interval.
    seconds = ((dt - s).total_seconds() // interval) * interval
    start = s + timedelta(seconds=seconds)
    end = start + timedelta(seconds=interval)
    return start, end
