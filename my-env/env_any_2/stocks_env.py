from typing import Any
from copy import deepcopy
import numpy as np

from .trading_env import TradingEnv, Actions, Positions

class StocksEnv(TradingEnv):

    def __init__(self,
        df,
        env_id='stocks-v0',
        eps_length=253, # one trading year, episode length
        window_size=20, # associated with the feature length
        plot_save_path='./', # the path to save result image
        plot_freq = 10,
        # the stocks range percentage used by train/test.
        # if one of them is None, train & test set will use all data by default.
        train_range=None,
        test_range=None,
        start_idx=None,
        render_mode=None
        ):
        super().__init__(
            env_id=env_id,
            eps_length=eps_length,
            window_size=window_size,
            plot_save_path=plot_save_path,
            plot_freq=plot_freq,
            train_range=train_range,
            test_range=test_range,
            start_idx=start_idx,
            render_mode=render_mode)

        # ====== load Google stocks data =======
        self.raw_prices = df.loc[:, 'close'].to_numpy()
        EPS = 1e-10
        self.df = deepcopy(df)
        if self.train_range == None or self.test_range == None:
            self.df = self.df.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        else:
            boundary = int(len(self.df) * self.train_range)
            train_data = df[:boundary].copy()
            boundary = int(len(df) * (1 + self.test_range))
            test_data = df[boundary:].copy()
            train_data = train_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
            test_data = test_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
            self.df.loc[train_data.index, train_data.columns] = train_data
            self.df.loc[test_data.index, test_data.columns] = test_data

        # set cost
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    # override
    def _process_data(self):
        '''
        Overview:
            used by env.reset(), process the raw data.
        Arguments:
            - start_idx (int): the start tick; if None, then randomly select.
        Returns:
            - prices: the close.
            - signal_features: feature map
            - feature_dim_len: the dimension length of selected feature
        '''

        # ====== build feature map ========
        all_feature_names = ['f_open', 'f_high', 'f_low', 'f_close', 'f_volume']
        all_feature = {k: self.df.loc[:, k].to_numpy() for k in all_feature_names}
        prices = self.df.loc[:, 'close'].to_numpy()

        # you can select features you want
        selected_feature_name = ['f_open', 'f_high', 'f_low', 'f_close', 'f_volume']
        selected_feature = np.column_stack([all_feature[k] for k in selected_feature_name])
        feature_dim_len = len(selected_feature_name)

        # validate index
        if self.start_idx is None:
            if self.train_range == None or self.test_range == None:
                self.start_idx = np.random.randint(self.window_size - 1, len(self.df) - self.eps_length)
            elif self._env_id[-1] == 'e':
                boundary = int(len(self.df) * (1 + self.test_range))
                assert len(self.df) - self.eps_length > boundary + self.window_size, \
                 "parameter test_range is too large!"
                self.start_idx = np.random.randint(boundary + self.window_size, len(self.df) - self.eps_length)
            else:
                boundary = int(len(self.df) * self.train_range)
                assert boundary - self.eps_length > self.window_size,\
                 "parameter test_range is too small!"
                self.start_idx = np.random.randint(self.window_size, boundary - self.eps_length)
        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self.eps_length - 1

        return prices, selected_feature, feature_dim_len

    # override
    def _calculate_reward(self, action: int): # -> np.float32:
        step_reward = 0.
        current_price = (self.raw_prices[self._current_tick])
        last_trade_price = (self.raw_prices[self._last_trade_tick])
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent))

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        step_reward = float(step_reward)

        return step_reward

    # override
    def max_possible_profit(self): #-> float:
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:

            if self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (2 - (current_price / last_trade_price)) * (1 - self.trade_fee_ask_percent
                                                                                  ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] >= self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (current_price / last_trade_price) * (1 - self.trade_fee_ask_percent
                                                                            ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            last_trade_tick = current_tick - 1

        return profit

    def __repr__(self): # -> str:
        return "Stock Trading Env"
