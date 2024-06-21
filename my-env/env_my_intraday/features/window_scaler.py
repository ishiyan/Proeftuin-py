from typing import Optional, Sequence, Tuple, Union, Literal
from collections import OrderedDict
import math

import gymnasium as gym

from ..frame import Frame
from .feature import Feature

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-max-abs-scaler-section
# https://github.com/scikit-learn/scikit-learn/blob/2621573e6/sklearn/preprocessing/_data.py#L291
# https://en.wikipedia.org/wiki/Feature_scaling
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

class WindowScaler(Feature):
    """
    Scales a window of the historical prices, and copies scaled
    prices to the output.

    Scaling methods are:
    - `raw`: take raw value
    - `minmax`: `z = (x - min) / (max - min)`
    - `standard`: `z = (x - u) / 𝜎`
    - `z-score`: `z = (x - u) / 𝜎`
    - `robust`: `z = (x - median) / IQR`

    The scaling window size is specified via `scaling_period` argument and defaults to 64.
    If the scaling window size is 1, the prices will remain raw.

    The `copy_period` argument  specifies how many frames to copy to the output and defaults to 1.
    To make sense, the copy_period should be less or equal to the window_period.

    If you want just to copy `copy_period` raw prices, set the `method` to `'raw'`.
    This will ignore the `scaling_period` argumant.

    Raw price values are not good for machine learning models,
    as they don't satisfy i.i.d. criteria.

    Using scaling is recommended
    as it makes price values more identically distributed.
    """
    
    def __init__(self,
                 source: Union[str, Sequence[str]] = ('open', 'high', 'low', 'close', 'volume'),
                 method: str = 'minmax',
                 base: Tuple[None, str] = None,
                 scale_period: Optional[int] = 64,
                 copy_period: int = 1,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes the feature processor.

        Args:
            source str or Sequence[str]:
                Names of Frame's attributes to encode.
            method str {'raw', 'minmax', 'standard'}:
                Name of method to use for encoding.
            base str or None:
                Name of Frame's attribute to be used as a previous price value.
                
                Specify `None` to use the same attribute.

                For instance, you may use previous close price as a base value
                to encode highest price of current frame.
            scale_period int or None:
                Number of frames to scale over.
                
                This value is ignored in case of a 'raw' encoding method.

                Otherwise it should be greater or equal to 2 in order for scaling to make sense.
            copy_period int:
                Number of frames to copy to the state.
                
                If the scaling method is not 'raw', the `copy_period`
                should be less or equal to the `scale_period`.

                If the scaling method is 'raw', the `copy_period` should be greater or equal to 2.
            write_to str {'frame','state','both'}:
                Destination of where to put computed values.
        """
        
        if not isinstance(method, str) or method not in [
            'raw', 'minmax', 'z-score', 'robust']:
            raise ValueError(f"method {method} should be one of 'raw', 'minmax', 'z-score', 'robust'")
        self.method = method

        if not isinstance(copy_period, int) or copy_period < 1:
            raise ValueError(f'copy_period {copy_period} must be a positive integer')

        period = copy_period
        if method != 'raw':
            if not isinstance(scale_period, int) or scale_period < 2:
                raise ValueError(f'for method {method}, scale_period {scale_period} must be an integer greater or equal to 2')
            if copy_period > scale_period:
                raise ValueError(f'for method {method}, copy_period {copy_period} must not exceed the scale_period {scale_period}')
            period = max(period, scale_period)
        else:
            scale_period = None
        self.copy_period = copy_period
        self.scale_period = scale_period

        super().__init__(period=period, write_to=write_to)
        
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
        mi = -math.inf
        ma = math.inf
        if method == 'minmax':
            mi = 0.
            ma = 1.
        sca_per = f'({scale_period})' if scale_period is not None else ''
        for name in self.source:
            assert isinstance(name, str)
            if base is None:
                self.names.append(f'{method}{sca_per}_{copy_period}_{name}')
            else:
                self.names.append(f'{method}{sca_per}_{copy_period}_{name}_{base}')
        if write_to in {'state', 'both'}:          
            self.spaces = OrderedDict({name: gym.spaces.Box(mi, ma, shape=(copy_period,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window = frames[-self.period:]
        for i, name in enumerate(self.source):
            if self.method == 'minmax':
                result = self.process_minmax(window, name, self.copy_period)
            elif self.method == 'standard':
                result = self.process_standard(window, name, self.copy_period)
            elif self.method == 'robust':
                result = self.process_robust(window, name, self.copy_period)
            else: # raw
                result = self.process_raw(window, name, self.copy_period)
            if self.copy_period == 1:
                result = result[0]
            if self.write_to_frame:
                setattr(frames[-1], self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result
    
    @staticmethod
    def process_raw(frames: Sequence[Frame], source: str, length: int) -> Sequence[float]:
        return [getattr(frame, source) for frame in frames[-length:]]
    
    @staticmethod
    def process_minmax(frames: Sequence[Frame], source: str, length: int) -> Sequence[float]:
        values = [getattr(frame, source) for frame in frames]
        lowest = min(values)
        highest = max(values)
        delta = highest - lowest
        if delta == 0:
            delta = 1
        return [(x - lowest) / delta for x in values[-length:]]
    
    @staticmethod
    def process_standard(frames: Sequence[Frame], source: str, length: int) -> Sequence[float]:
        values = [getattr(frame, source) for frame in frames]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        standard_deviation = math.sqrt(variance)
        if standard_deviation == 0:
            standard_deviation = 1
        return [(x - mean) / standard_deviation for x in values[-length:]]
    
    @staticmethod
    def process_robust(frames: Sequence[Frame], source: str, length: int) -> Sequence[float]:
        values = [getattr(frame, source) for frame in frames]
        median = sorted(values)[len(values) // 2]
        q1 = sorted(values)[len(values) // 4]
        q3 = sorted(values)[3 * len(values) // 4]
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1
        return [(x - median) / iqr for x in values[-length:]]
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(source={self.source}, '
            f'method={self.method}, base={self.base},'
            f'copy_period={self.copy_period}, scale_period={self.scale_period}, '
            f'write_to={self.write_to})')
