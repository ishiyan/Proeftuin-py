from collections import OrderedDict
from numbers import Real
from typing import List, Optional

from ..accounts.account import Account
from ..providers.provider import Provider
from ..aggregators.trade_aggregator import TradeAggregator
from ..frame import Frame 
from .renderer import Renderer

class CsvRenderer(Renderer):

    def __init__(self, vec_env_index: Optional[int] = None):
        self.vec_env_index = vec_env_index
        self.vec_env_index_str = ''
        self.account: Account = None

        # Return on investment (ROI) of a position:
        # roi = (quantity_signed * (price - average_price) - commission) /
        #       (quantity_signed * average_price)
        
        # Cumulative return of an account:
        # cr = (previous_twr + 1) * (balance / previous_balance) - 1

        # Rate of return (ROR) of an account:
        # net_profit = SUM(quantity_signed * (price - average_price) - commission)
        # ror = net_profit / initial_balance

        self.columns = 'episode,provider,aggregator,step,reward,total reward,' \
            'open,high,low,close,volume,' \
            'position delta,position,balance,initial balance,account halted,' \
            'net profit,total commission,max drawdown pct,' \
            'cumulative return,rate of return,' \
            'return on investment,return on investment mean,return on investment std,' \
            'sharpe ratio,sortino ratio,calmar ratio,' \
            'roundtrip winning net pnl,roundtrip loosing net pnl,' \
            'roundtrip net profit pnl pct,roundtrip net winning pct,' \
            'roundtrip net loosing pct,roundtrip mean net winning loosing pct,' \
            'roundtrip max consecutive net winners,roundtrip max consecutive net loosers,' \
            'roundtrip mean maximum adverse excursion,roundtrip mean maximum favorable excursion,' \
            'roundtrip mean entry efficiency,roundtrip mean exit efficiency,' \
            'roundtrip mean total efficiency'

        if self.vec_env_index is not None:
            self.columns = f'vec_env_index,' + self.columns
            self.vec_env_index_str = f'{vec_env_index},'

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
        self.halted = None
        self.net_profit = None
        self.total_commission = None
        self.max_drawdown_percent = None
        self.cumulative_return = None
        self.rate_of_return = None
        self.return_on_investment = None
        self.return_on_investment_mean = None
        self.return_on_investment_std = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.calmar_ratio = None
        self.roundtrip_winning_net_pnl = None
        self.roundtrip_loosing_net_pnl = None
        self.roundtrip_net_profit_pnl_ratio = None
        self.roundtrip_net_winning_ratio = None
        self.roundtrip_net_loosing_ratio = None
        self.roundtrip_average_net_winning_loosing_ratio = None
        self.roundtrip_max_consecutive_net_winners = None
        self.roundtrip_max_consecutive_net_loosers = None
        self.roundtrip_average_maximum_adverse_excursion = None
        self.roundtrip_average_maximum_favorable_excursion = None
        self.roundtrip_average_entry_efficiency = None
        self.roundtrip_average_exit_efficiency = None
        self.roundtrip_average_total_efficiency = None

    def reset(self, episode_number: int, episode_max_steps: Optional[int],
            account: Account, provider: Provider, aggregator: TradeAggregator,
            frames: List[Frame], observation: OrderedDict):
        self.account = account
        self.performance = account.performance.daily
        self.performance_roundtrips = account.performance.roundtrips

        self.provider_name = provider.name
        self.aggregator_name = aggregator.name
        self.episode_number = episode_number
        self.step_number = 0
        self.total_reward = 0.0
        self.position = account.position.quantity_signed
        self.initial_balance = account.initial_balance

        self._append_step(frames[-1], 0.0)

    def step(self, frames: List[Frame], reward: Real, observation: OrderedDict):
        self.step_number += 1
        self._append_step(frames[-1], reward)

    def render(self):
        if self.episode_number is None:
            return self.columns
        row = f'{self.episode_number},' \
            f'{self.provider_name},' \
            f'{self.aggregator_name},' \
            f'{self.step_number},' \
            f'{self.reward},' \
            f'{self.total_reward},' \
            f'{self.open},{self.high},{self.low},{self.close_},{self.volume},' \
            f'{self.position_delta},' \
            f'{self.position},' \
            f'{self.balance},' \
            f'{self.initial_balance},' \
            f'{self.halted},' \
            f'{self.net_profit},' \
            f'{self.total_commission},' \
            f'{self.max_drawdown_percent},' \
            f'{self.cumulative_return},' \
            f'{self.rate_of_return},' \
            f'{self.return_on_investment},' \
            f'{self.return_on_investment_mean},' \
            f'{self.return_on_investment_std},' \
            f'{self.sharpe_ratio},' \
            f'{self.sortino_ratio},' \
            f'{self.calmar_ratio},' \
            f'{self.roundtrip_winning_net_pnl},' \
            f'{self.roundtrip_loosing_net_pnl},' \
            f'{self.roundtrip_net_profit_pnl_ratio},' \
            f'{self.roundtrip_net_winning_ratio},' \
            f'{self.roundtrip_net_loosing_ratio},' \
            f'{self.roundtrip_average_net_winning_loosing_ratio},' \
            f'{self.roundtrip_max_consecutive_net_winners},' \
            f'{self.roundtrip_max_consecutive_net_loosers},' \
            f'{self.roundtrip_average_maximum_adverse_excursion},' \
            f'{self.roundtrip_average_maximum_favorable_excursion},' \
            f'{self.roundtrip_average_entry_efficiency},' \
            f'{self.roundtrip_average_exit_efficiency},' \
            f'{self.roundtrip_average_total_efficiency}'

        if self.vec_env_index is not None:
            row = self.vec_env_index_str + row
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
        self.balance = self.account.balance
        self.halted = self.account.is_halted

        p = self.performance
        pr = self.performance_roundtrips

        self.net_profit = pr.net_profit_ratio
        self.total_commission = pr.total_commission
        v = p._drawdowns_cumulative_min
        self.max_drawdown_percent = v if v is not None else 0
        self.cumulative_return = p.cumulative_return

        self.rate_of_return = self.account.performance.rate_of_return
        v = pr.returns_on_investments
        self.return_on_investment = v[-1] if len(v) > 0 else 0
        v = pr.roi_mean
        self.return_on_investment_mean = v if v is not None else 0
        v = pr.roi_std
        self.return_on_investment_std = v if v is not None else 0

        v = p.sharpe_ratio()
        self.sharpe_ratio = v if v is not None else 0
        v = p.sortino_ratio()
        self.sortino_ratio = v if v is not None else 0
        v = p.calmar_ratio()
        self.calmar_ratio = v if v is not None else 0

        self.roundtrip_winning_net_pnl = pr.winning_net_pnl
        self.roundtrip_loosing_net_pnl = pr.loosing_net_pnl
        self.roundtrip_net_profit_pnl_ratio = pr.net_profit_pnl_ratio
        self.roundtrip_net_winning_ratio = pr.net_winning_ratio
        self.roundtrip_net_loosing_ratio = pr.net_loosing_ratio
        self.roundtrip_average_net_winning_loosing_ratio = pr.average_net_winning_loosing_ratio
        self.roundtrip_max_consecutive_net_winners = pr.max_consecutive_net_winners
        self.roundtrip_max_consecutive_net_loosers = pr.max_consecutive_net_loosers
        self.roundtrip_average_maximum_adverse_excursion = pr.average_maximum_adverse_excursion / 100
        self.roundtrip_average_maximum_favorable_excursion = pr.average_maximum_favorable_excursion / 100
        self.roundtrip_average_entry_efficiency = pr.average_entry_efficiency / 100
        self.roundtrip_average_exit_efficiency = pr.average_exit_efficiency / 100
        self.roundtrip_average_total_efficiency = pr.average_total_efficiency / 100
