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

class Scale(Feature):
    """
    Scales a window of the historical attributes, and writes last scaled
    value to the output.

    The scaling methods are:
    - `minmax`: `z = (x - min) / (max - min)`
    - `zscore`: `z = (x - u) / 𝜎`
    - `robust`: `z = (x - median) / IQR`

    The scaling window size is specified in the `period` argument and
    defaults to 64.

    Raw price values are not good for machine learning models, as they don't
    satisfy i.i.d. criteria. Using scaling is recommended as it makes price
    values more identically distributed.
    """
    
    def __init__(self,
                 source: Union[str, Sequence[str]],
                 method: str = 'minmax',
                 period: int = 64,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes the feature processor.

        Args:
            source str or Sequence[str]:
                Names of Frame's attributes to encode.
            method str {'minmax', 'zscore', 'robust'}:
                Name of method to use for encoding.

                The scaling methods are:

                `minmax`: `z = (x - min) / (max - min)`
                `zscore`: `z = (x - u) / 𝜎`
                `robust`: `z = (x - median) / IQR`
            period int:
                Number of frames to scale over.
                
                The value must be be greater or equal to 2 in order for scaling
                to make sense.

                Default: 64
            write_to str {'frame','state','both'}:
                Destination of where to put computed values.

                If `frame`, the scaled values will be written to all historical
                frames, from `frames[-1]` till `frames[-period]`.

                If `state`, only the last scaled value will be written to the
                state dictionary.
        """
        
        if not isinstance(method, str) or method not in [
            'minmax', 'zscore', 'robust']:
            raise ValueError(f"method {method} should be one of "
                             "'minmax', 'zscore', 'robust'")
        self.method = method

        if not isinstance(period, int) or period < 2:
            raise ValueError(f'period {period} must be an integer '
                             'greater or equal to 2')

        super().__init__(period=period, write_to=write_to)
        
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
            mi = 0.0
            ma = 1.0
        for name in self.source:
            assert isinstance(name, str)
            self.names.append(f'{method}{period}_{name}')
        if write_to in {'state', 'both'}:          
            self.spaces = OrderedDict({name: gym.spaces.Box(mi, ma, \
                                shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window = frames[-self.period:]
        for i, name in enumerate(self.source):
            if self.method == 'minmax':
                scaled = self.process_minmax(window, name)
            elif self.method == 'zscore':
                scaled = self.process_zscore(window, name)
            else: # self.method == 'robust'
                scaled = self.process_robust(window, name)
            value = scaled[0]
            if self.write_to_frame:
                for fram, val in zip(window, scaled):
                    setattr(fram, self.names[i], val)
            if self.write_to_state:
                state[self.names[i]] = value
    
    @staticmethod
    def process_minmax(frames: Sequence[Frame], source: str) -> Sequence[float]:
        values = [getattr(frame, source) for frame in frames]
        lowest = min(values)
        highest = max(values)
        delta = highest - lowest
        if delta == 0:
            delta = 1
        return [(x - lowest) / delta for x in values]
    
    @staticmethod
    def process_zscore(frames: Sequence[Frame], source: str) -> Sequence[float]:
        values = [getattr(frame, source) for frame in frames]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        standard_deviation = math.sqrt(variance)
        if standard_deviation == 0:
            standard_deviation = 1
        return [(x - mean) / standard_deviation for x in values]
    
    @staticmethod
    def process_robust(frames: Sequence[Frame], source: str) -> Sequence[float]:
        values = [getattr(frame, source) for frame in frames]
        median = sorted(values)[len(values) // 2]
        q1 = sorted(values)[len(values) // 4]
        q3 = sorted(values)[3 * len(values) // 4]
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1
        return [(x - median) / iqr for x in values]
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(source={self.source}, '
            f'method={self.method}, period={self.period}, '
            f'write_to={self.write_to})')
