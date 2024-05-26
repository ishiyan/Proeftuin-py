import numpy as np

from .trading_env import TradingEnv, Actions, Positions

class StocksEnv(TradingEnv):
    """
    * Properties:

    `frame_bound`: A tuple which specifies the start and end of `df`. It is passed in the class' constructor.

    `trade_fee_bid_percent`: A default constant fee percentage for bids.
    For example with `trade_fee_bid_percent=0.01`, you will lose 1% of your money every time you sell your shares.

    `trade_fee_ask_percent`: A default constant fee percentage for asks.
    For example with `trade_fee_ask_percent=0.005`, you will lose 0.5% of your money every time you buy some shares.

    Besides, you can create your own customized environment by extending `TradingEnv` or `StocksEnv`
    with your desired policies for calculating reward, profit, fee, etc.
    """
    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()
        open = self.df.loc[:, 'f_open'].to_numpy()
        high = self.df.loc[:, 'f_high'].to_numpy()
        low = self.df.loc[:, 'f_low'].to_numpy()
        close = self.df.loc[:, 'f_close'].to_numpy()
        volume = self.df.loc[:, 'f_volume'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        open = open[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        high = high[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        low = low[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        close = close[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        volume = volume[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        signal_features = np.column_stack((open, high, low, close, volume))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _process_data_original(self):
        prices = self.df.loc[:, 'close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

        return step_reward

    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
    