    
from typing import Tuple

import numpy as np

def _zigzag_open_high_low_close(open, high, low, close, step) -> Tuple[np.ndarray, np.ndarray]:
        open_to_high = np.concatenate((np.arange(open, high, step), (high,)))        
        high_to_low = np.concatenate((np.arange(high - step, low, -step), (low,)))
        low_to_close = np.concatenate((np.arange(low + step, close, step), (close,)))
        prices = np.concatenate((
            open_to_high,
            high_to_low,
            low_to_close
        ))
        orders = np.concatenate((
            np.ones(len(open_to_high), dtype=np.int8),
            -1 * np.ones(len(high_to_low), dtype=np.int8),
            np.ones(len(low_to_close), dtype=np.int8)
        ))
        return prices, orders

p, o = _zigzag_open_high_low_close(100.2, 100.4, 100.1, 100.3, 0.1)
#p, o = _zigzag_open_high_low_close(0.2, 0.4, 0.1, 0.3, 0.1)
for i in range(len(p)):
    print(i, p[i], o[i])
#print(0.3/0.05)
