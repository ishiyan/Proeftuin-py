import bisect
from datetime import datetime
from typing import List, Tuple

class ScalarSeries:
    def __init__(self):
        self._scalar = None
        self._series = []

    def is_empty(self) -> bool:
        return not self._scalar

    def current(self) -> Tuple[datetime, float]:
        return self._scalar
    
    def current_value(self) -> float:
        if self._scalar is not None:
            return self._scalar[1]
        return 0.0

    def at(self, t: datetime) -> float:
        if self._series:
            i = bisect.bisect_right([s[0] for s in self._series], t)
            if i:
                return self._series[i-1][1]
        return 0.0

    def history(self) -> List[Tuple[datetime, float]]:
        return self._series.copy() if self._series else []

    def add(self, t: datetime, v: float):
        if not self._scalar or self._scalar[0] < t:
            s = (t, v)
            self._series.append(s)
            self._scalar = s
        elif self._scalar[0] == t:
            self._scalar = (t, v)
            self._series[-1] = self._scalar
        else:
            i = bisect.bisect_left([s[0] for s in self._series], t)
            if self._series[i][0] == t:
                self._series[i] = (t, v)
            else:
                s = (t, v)
                self._series.insert(i, s)

    def accumulate(self, t: datetime, v: float):
        if not self._scalar:
            s = (t, v)
            self._series.append(s)
            self._scalar = s
        elif self._scalar[0] < t:
            s = (t, v + self._scalar[1])
            self._series.append(s)
            self._scalar = s
        elif self._scalar[0] == t:
            self._scalar = (t, self._scalar[1] + v)
            self._series[-1] = self._scalar
        else:
            i = bisect.bisect_left([s[0] for s in self._series], t)
            if self._series[i][0] == t:
                self._series[i] = (t, self._series[i][1] + v)
                for j in range(i+1, len(self._series)):
                    self._series[j] = (self._series[j][0], self._series[j][1] + v)
            else:
                s = (t, v)
                if i:
                    s = (t, v + self._series[i-1][1])
                self._series.insert(i, s)
                for j in range(i+1, len(self._series)):
                    self._series[j] = (self._series[j][0], self._series[j][1] + v)