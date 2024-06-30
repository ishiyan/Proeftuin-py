from numbers import Real
from typing import Optional

from ..accounts.account import Account
from ..providers.provider import Provider
from ..aggregators.trade_aggregator import TradeAggregator
from ..frame import Frame 
from .renderer import Renderer

class CsvRenderer(Renderer):

    def __init__(self):
        self.account: Account = None

        self.columns = 'episode,provider,aggregator,step,reward,total reward,' \
            'open,high,low,close,volume,' \
            'position delta,position,balance,initial balance,' \
            'time weighted return,rate of return,' \
            'sharpe ratio,sortino ratio,calmar ratio'

        self.episode_number = None
        self.provider_name = None
        self.aggregator_name = None
        self.step_number = None
        self.reward = None
        self.total_reward = None
        self.open = None
        self.high = None
        self.low = None
        self.close_ = None
        self.volume = None
        self.position_delta = None
        self.position = None
        self.balance = None
        self.initial_balance = None
        self.time_weighted_return = None
        self.rate_of_return = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.calmar_ratio = None

    def reset(self, episode_number: int, episode_max_steps: Optional[int],
            account: Account, provider: Provider, aggregator: TradeAggregator,
            frame: Frame):
        self.account = account

        self.provider_name = provider.name
        self.aggregator_name = aggregator.name
        self.episode_number = episode_number
        self.step_number = 0
        self.total_reward = 0.0
        self.position = account.position.quantity_signed
        self.initial_balance = account.initial_balance
        self.time_weighted_return = 0.0
        self._append_step(frame, 0.0)

    def step(self, frame: Frame, reward: Real):
        self.step_number += 1
        self._append_step(frame, reward)

    def render(self):
        if self.episode_number is None:
            return self.columns
        row = f'{self.episode_number},{self.provider_name},{self.aggregator_name},' \
            f'{self.step_number},{self.reward},{self.total_reward},' \
            f'{self.open},{self.high},{self.low},{self.close_},{self.volume},' \
            f'{self.position_delta},{self.position},{self.balance},{self.initial_balance},' \
            f'{self.time_weighted_return},{self.rate_of_return},' \
            f'{self.sharpe_ratio},{self.sortino_ratio},{self.calmar_ratio}'
        return row
        
    def close(self):
        pass

    def _append_step(self, frame: Frame, reward: Real):
        self.reward = reward
        self.total_reward += reward
        self.open = frame.open
        self.high = frame.high
        self.low = frame.low
        self.close_ = frame.close
        self.volume = frame.volume

        position = self.account.position.quantity_signed
        self.position_delta = position - self.position
        self.position = position

        balance = self.account.balance
        balance_return = (balance / self.balance - 1.0) \
            if (self.balance != 0 and self.balance is not None) else 0.0
        self.balance = balance

        twr = (self.time_weighted_return + 1.0) * (balance_return + 1.0)
        self.time_weighted_return = twr - 1.0

        self.rate_of_return = self.account.report.ror
        val = self.account.report.sharpe_ratio
        self.sharpe_ratio = val if val is not None else 0.0
        val = self.account.report.sortino_ratio
        self.sortino_ratio = val if val is not None else 0.0
        val = self.account.report.calmar_ratio
        self.calmar_ratio = val if val is not None else 0.0
