from .environment import Environment
from .dataframe_observer import DataframeObserver
from .any1_renderer import Any1Renderer
from .any1_trader import Any1Trader

class Any1Env(Environment):
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, frame_bound, window_size, render_mode=None):
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.observer = DataframeObserver(df, frame_bound, window_size)
        self.trader = Any1Trader(window_size)
        self.renderer = Any1Renderer(render_mode)

        super().__init__(self.observer, self.trader, self.renderer)
