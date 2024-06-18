from typing import NamedTuple, Union, Optional, Tuple
from numbers import Real
from datetime import datetime, date, timedelta, timezone
import math
import numpy as np

from .entities import Trade
from .provider import Provider

class SineTradesProvider(Provider):
    """
    Generates a fake stream of trades to move price in a sinusoidal form.

    If you have trained a trading bot, this sine generator would be
    a good test for it.
    
    If your algorithm fails to learn to make profit even on such
    simple data, it will never find any profit in a real market.
    
    With some value combinations (e.g. `mean=0`, `amplitude=100`),
    price will take both positive and negative values. It may be
    useful to test your trading bot and your features for robustness.
    
    Args:
        mean (float):
            Mean value of a sinusoid.
            Default: 100.
        amplitude (float):
            Amplitude of a sinusoid.
            Default: 90.
        frequency (Optional[Union[float, Tuple[float, float]]]):
            Frequency of a sinusoid in Hertz (i.e. 1 / seconds),
            or a range specified by two values, to take a random
            frequency at each episode.

            You must specify either `frequency` or `period`.
            Default: None
        period (Optional[Union[timedelta, Tuple[timedelta, timedelta]]])
            Period of a sinusoid in seconds, or a range specified
            by two values, to take a random period at each episode.
        
            You must specify either `frequency` or `period`.
            Default: None
        SNRdb (float):
            Signal to noise ratio in Db.
            The less this value - the more noise is added to sinusoid.
            Default: 15.
        date_from (Optional[Union[date, datetime]]):
            Specify starting date for simulated trades.
            If None - uses the date a year ago from current date.
            Default: None
        date_to (Optional[Union[date, datetime]]):
            Specify ending date for simulated trades.
            If None - uses the current date.
            Default: None
    """
    def __init__(self,
                 mean: float = 0.0,
                 amplitude: float = 100.0,
                 frequency: Optional[Union[float, Tuple[float, float]]] = None,
                 period: Optional[Union[timedelta, Tuple[timedelta, timedelta]]] = None,
                 SNRdb: float = 15.0,
                 date_from: Optional[Union[date, datetime]] = None,
                 date_to: Optional[Union[date, datetime]] = None):
        super().__init__()
        
        self.mean = mean
        self.amplitude = amplitude
        self.SNRdb = SNRdb
        self.noise_amplitude = math.sqrt(amplitude ** 2 / (math.pow(10, SNRdb/10)))

        if isinstance(frequency, float):
            self.freq1 = self.freq2 = frequency
        elif isinstance(frequency, Tuple):
            assert (len(frequency) == 2) and isinstance(frequency[0], Real) and isinstance(frequency[1], Real)
            self.freq1, self.freq2 = min(*frequency), max(*frequency)
        elif isinstance(period, timedelta):
            self.freq1 = self.freq2 = 1 / period.total_seconds()
        elif isinstance(period, Tuple):
            assert (len(period) == 2) and isinstance(period[0], timedelta) and isinstance(period[1], timedelta)
            freq1, freq2 = (1 / period[0].total_seconds()), (1 / period[1].total_seconds())
            self.freq1, self.freq2 = min(freq1, freq2), max(freq1, freq2)
        else:
            raise ValueError('please specify either frequency or period')
            
        if date_to is None:
            date_to = datetime.now(timezone.utc)
        elif isinstance(date_to, date):
            date_to = datetime.combine(date_to, datetime.min.time(),
                                       tzinfo=timezone.utc)
        if isinstance(date_to, datetime):
            date_to = date_to.astimezone(timezone.utc)
        if not isinstance(date_to, datetime):
            raise ValueError('date_to must be an instance of datetime.datetime: '
                            f'{date_to} of type {type(date_to)}')
        self.date_to = date_to

        if date_from is None:
            date_from = date_to - timedelta(days=365)
        elif isinstance(date_from, date):
            date_from = datetime.combine(date_from, datetime.min.time(),
                                        tzinfo=timezone.utc)
        if isinstance(date_from, datetime):
            date_from = date_from.astimezone(timezone.utc)
        if not isinstance(date_from, datetime):
            raise ValueError('date_from must be an instance of datetime.datetime: '
                            f'{date_from} of type {type(date_from)}')
        self.date_from = date_from
        
        if date_from > date_to:
            raise ValueError(f'date_from {date_from} must be less '
                             f'than or equal to date_to {date_to}')
        
        # Prepare episode variables.
        self._freq: Optional[float] = None
        self._datetime: Optional[datetime] = None
        self._last_price: Optional[float] = None
        self._episode_start_datetime: Optional[datetime] = None
    
    def reset(self,
              episode_start_datetime: Optional[datetime] = None,
              episode_min_duration: Union[None, Real, timedelta] = None,
              rng: Optional[np.random.RandomState] = None) -> datetime:
        # Check episode_min_duration.
        if episode_min_duration is None:
            episode_min_duration = timedelta(seconds=0)
        elif isinstance(episode_min_duration, Real):
            episode_min_duration = timedelta(seconds=float(episode_min_duration))
        elif isinstance(episode_min_duration, timedelta):
            pass
        else:
            raise ValueError('invalid episode_min_duration value '
                f'{episode_min_duration} of type {type(episode_min_duration)}')
        assert episode_min_duration.total_seconds() >= 0

        # Check episode_start_datetime.
        if episode_start_datetime is None:
            # Choose random datetime.
            rt = rng.random() if (rng is not None) else np.random.random()
            start, end = self.date_from.timestamp(), \
                        (self.date_to - episode_min_duration).timestamp()
            episode_start_datetime = start + rt * (end - start)
            episode_start_datetime = datetime.fromtimestamp(episode_start_datetime,
                                                            tz=timezone.utc)
        elif isinstance(episode_start_datetime, datetime):
            pass
        else:
            raise ValueError('invalid episode_start_datetime value: '
                f'{episode_start_datetime} of type {type(episode_start_datetime)}')

        self._episode_start_datetime = episode_start_datetime

        # Generate random frequency.
        r = rng.random() if (rng is not None) else np.random.random()
        self._freq = self.freq1 + r * (self.freq2 - self.freq1)

        self._datetime = episode_start_datetime
        self._last_price = 0
        
        return self._episode_start_datetime
    
    def __next__(self) -> NamedTuple:
        self._datetime += timedelta(seconds=5 * np.random.random())
        t = (self._datetime - self._episode_start_datetime).total_seconds()
        sine = math.sin(2 * math.pi * self._freq * t)
        noise = np.random.randn()
        price = self.amplitude * sine + self.noise_amplitude * noise
        self._last_price = price        
        # Return next trade.
        return Trade(
            datetime=self._datetime,
            operation='S' if (price < self._last_price) else 'B',
            amount=np.random.randint(10) + 1,
            price=price)
    
    def close(self):
        self._freq = None
        self._datetime = None
        self._last_price = None
        self._episode_start_datetime = None

    @property
    def name(self) -> str:
        return 'Sine'

    @property
    def session_start_datetime(self) -> Optional[datetime]:
        return self.date_from.datetime

    @property
    def session_end_datetime(self) -> Optional[datetime]:
        return self.date_to.datetime

    @property
    def episode_start_datetime(self) -> Optional[datetime]:
        return self._episode_start_datetime
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
            f'frequency=({self.freq1}, {self.freq1}), '
            f'mean={self.mean}, amplitude={self.amplitude}, SNRdb={self.SNRdb}, '
            f'date_from={self.date_from}, date_to={self.date_to})')
