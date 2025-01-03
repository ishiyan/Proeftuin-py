from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
import math

import gymnasium as gym

from ..frame import Frame
from .feature import Feature

def price_raw(p0, p1):
    return p0

def price_delta(p0, p1):
    return (p0 - p1) if (p0 is not None) and (p1 is not None) else 0.0

def price_return(p0, p1):
    return (p0 / p1 - 1.0) if (p0 is not None) and (p1 is not None) and (p1 != 0) else 0.0

def price_logreturn(p0, p1):
    assert (p0 is None) or (p0 >= 0)
    assert (p1 is None) or (p1 >= 0)
    return math.log(p0 / p1) if (p0 is not None) and (p1 is not None) and (p1 != 0) else 0.0

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-max-abs-scaler-section
# https://github.com/scikit-learn/scikit-learn/blob/2621573e6/sklearn/preprocessing/_data.py#L291


class PriceScaler(Feature):
    """
    Encodes price values in time-series via different methods

    Accepted methods are:
    - `raw`: price value is not encoded
    - `delta`: `(current price - price N periods ago)`
    - `return`: `(current price / price N periods ago) - 1`
    - `logreturn`: `log(current price / price N periods ago) == log(current price) - log(price N periods ago)`

    N periods is specified via `period` argument and defaults to 1.

    Raw price values are not good for machine learning models,
    as they don't satisfy i.i.d. criteria.

    Using 'delta', 'return' or 'logreturn' is recommended
    as it makes price values more identically distributed.
    """
    
    Methods = {
        'raw': price_raw,
        'delta': price_delta,
        'return': price_return,
        'logreturn': price_logreturn,
    }
    
    def __init__(self,
                 source: Union[str, Sequence[str]] = ('open', 'high', 'low', 'close'),
                 method: str = 'delta',
                 base: Tuple[None, str] = None,
                 period: int = 2,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes `PriceEncoder` feature processor.

        Args:
            source (str or Sequence[str]):
                Names of Frame's attributes to encode.
            method str {'raw', 'delta', 'return', 'logreturn'}:
                Name of method to use for encoding.
            base (str or None):
                Name of Frame's attribute to be used as a previous price value.
                
                Specify `None` to use the same attribute.

                For instance, you may use previous close price as a base value
                to encode highest price of current frame.
            period (int):
                Number of frames to compute change over.
                
                This value is ignored in case of a 'raw' encoding method.

                Otherwise it should be greater or equal to 2.
            write_to str {'frame','state','both'}:
                Destination of where to put computed values.
        """
        super().__init__(period=period, write_to=write_to)
        
        if not isinstance(method, str) or method not in PriceEncoder.Methods:
            raise ValueError(f"method {method} should be one of 'raw', 'delta', 'return', 'logreturn'")
        self.method = method
        
        if not ((base is None) or isinstance(base, str)):
            raise ValueError(f'base {base} should be string or None')
        self.base = base
        
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError(f'source {source} must be a string '
                             'or a sequence of strings')
        
        for name in self.source:
            assert isinstance(name, str)
            if base is None:
                self.names.append(f'{method}_{period}_{name}')
            else:
                self.names.append(f'{method}_{period}_{name}_{base}')
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
        # Initialize _encode_price as a price price_encoding instance function
        # Note: this is not instance method, so it is called without self
        # but the syntax looks the same: processor._encode_price(p0, p1)
        self._encode_price = PriceEncoder.Methods[method]
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        prev_frame = frames[-min(self.period, len(frames))]
        last_frame = frames[-1]
        price1 = None
        if self.base is not None:
            price1 = getattr(prev_frame, self.base) if (prev_frame is not None) else None
        for i, name in enumerate(self.source):
            price0 = getattr(last_frame, name)
            if self.base is None:
                price1 = getattr(prev_frame, name) if (prev_frame is not None) else None
            price = self._encode_price(price0, price1)
            if self.write_to_frame:
                setattr(last_frame, self.names[i], price)
            if self.write_to_state:
                state[self.names[i]] = price
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(source={self.source}, '
            f'method={self.method}, base={self.base},'
            f'period={self.period}, write_to={self.write_to})')
