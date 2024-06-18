from __future__ import annotations
from typing import Tuple, Sequence, Union, Literal, Optional
import math
from numbers import Real
import arrow
import datetime as dt

from ..providers import Trade
from ..frame import Frame
from .trade_aggregator import TradeAggregator

class IntervalTradeAggregator(TradeAggregator):
    '''
    Groups trades into frames based on specified interval and methodology.

    Parameters:
    - method: str
        Method of how to group the flow of records into frames.
        Possible values: 'time', 'tick', 'volume', 'money'.
    - interval: int
        Initial_threshold of the frame in seconds.
        It is used to group trades into frames.
    - duration: tuple
        Tuple with two values: (min_duration, max_duration).
        If frame duration exceeds max_duration, it will be finalized.
        If frame duration exceeds min_duration and frame is full, it will be finalized.
    '''
    FramingMethods = [
        'time',
        'tick',
        'volume',
        'money',
    ]
    
    def __init__(self,
                 method: Literal['time', 'tick', 'volume', 'money'],
                 interval: Real,
                 duration: Tuple[Real, Real] = (1, 4*60*60),
                 **kwargs):
        super().__init__(**kwargs)
        # Method of how to group the flow of records into frames
        # Frame initial_threshold which depends on selected method
        assert isinstance(method, str) and (method in IntervalTradeAggregator.FramingMethods)
        assert isinstance(interval, int) and (interval > 0)
        assert (isinstance(duration, tuple) and (len(duration) == 2) and
                isinstance(duration[0], Real) and isinstance(duration[1], Real))
        
        # Save parameters
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

        if (frame is None) or ((self.method == 'time') and (trade.datetime > frame.time_end)):
            # Init new frame
            result = frame.finalize() if (frame is not None) else None
            frame = Frame(time_start=trade.datetime)
            self.frame = frame
            if self.method == 'time':
                # Round time_start to specified initial_threshold and also get time_end
                time_start, time_end = _get_time_span(trade.datetime, self.interval)
                frame.time_start = time_start
                frame.time_end = time_end

        # Update frame
        frame.update(trades)
        
        # Check for the end of frame
        if ((frame.duration >= self.duration[0]) and (
                ((self.method == 'tick') and (frame.ticks >= self.interval)) or
                ((self.method == 'volume') and (frame.volume >= self.interval)) or
                ((self.method == 'money') and (frame.money >= self.interval))
            ) or (frame.duration >= self.duration[1])):
            # Reset current frame
            result = frame.finalize() if (frame is not None) else None
            self.frame = None
            
        return result

    def finish(self) -> Optional[Frame]:
        result = self.frame.finalize()
        self.frame = None
        return result

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'method={self.method}, interval={self.interval}, duration={self.duration})'
        )

def _get_time_span(datetime: Union[dt.datetime, arrow.Arrow], interval: int) -> Tuple[arrow.Arrow, arrow.Arrow]:
    if not isinstance(datetime, arrow.Arrow):
        datetime = arrow.get(datetime)
    # Calculate start of frame
    if interval <= 24*60*60:
        # Get start of day
        s = datetime.floor('day')
    else:
        # Get start of year
        s = datetime.floor('year')
    # Calculate start of frame with respect to start time s and frame initial_threshold
    seconds = math.floor((datetime.timestamp() - s.timestamp()) / interval) * interval
    return (s + dt.timedelta(seconds=seconds)).span('second', interval)
