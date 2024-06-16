from typing import Sequence, Optional

from ..providers import Trade
from ..frame import Frame

class Processor(object):
    """
    Base abstract class for IntervalProcessor, ImbalanceProcessor and RunProcessor
    """
    def __init__(self, **kwargs):
        pass
    
    def reset(self):
        raise NotImplementedError()

    def process(self, trades: Sequence[Trade]) -> Optional[Frame]:
        raise NotImplementedError()
    
    def finish(self) -> Optional[Frame]:
        raise NotImplementedError()
    
    @property
    def name(self):
        raise NotImplementedError()
