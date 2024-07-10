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
        
        # Time weighted return (TWR) of an account:
        # twr = (previous_twr + 1) * (balance / previous_balance) - 1

        # Rate of return (ROR) of an account:
        # net_profit = SUM(quantity_signed * (price - average_price) - commission)
        # ror = net_profit / initial_balance

        self.columns = 'episode,provider,aggregator,step,reward,total reward,' \
            'open,high,low,close,volume,' \
            'position delta,position,balance,initial balance,account halted,' \
            'net profit,total commission,max drawdown pct,' \
            'time weighted return,rate of return,' \
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
        self.time_weighted_return = None
        self.rate_of_return = None
        self.return_on_investment = None
        self.return_on_investment_mean = None
        self.return_on_investment_std = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.calmar_ratio = None
        self.roundtrip_winning_net_pnl = None
        self.roundtrip_loosing_net_pnl = None
        self.roundtrip_net_profit_pnl_percentage = None
        self.roundtrip_net_winning_percentage = None
        self.roundtrip_net_loosing_percentage = None
        self.roundtrip_average_net_winning_loosing_percentage = None
        self.roundtrip_max_consecutive_net_winners = None
        self.roundtrip_max_consecutive_net_loosers = None
        self.roundtrip_average_maximum_adverse_excursion = None
        self.roundtrip_average_maximum_favorable_excursion = None
        self.roundtrip_average_entry_efficiency = None
        self.roundtrip_average_exit_efficiency = None
        self.roundtrip_average_total_efficiency = None

    def reset(self, episode_number: int, episode_max_steps: Optional[int],
            account: Account, provider: Provider, aggregator: TradeAggregator,
            frames: List[Frame]):
        self.account = account

        self.provider_name = provider.name
        self.aggregator_name = aggregator.name
        self.episode_number = episode_number
        self.step_number = 0
        self.total_reward = 0.0
        self.position = account.position.quantity_signed
        self.initial_balance = account.initial_balance
        self.time_weighted_return = 0.0

        self._append_step(frames[-1], 0.0)

    def step(self, frames: List[Frame], reward: Real):
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
            f'{self.time_weighted_return},' \
            f'{self.rate_of_return},' \
            f'{self.return_on_investment},' \
            f'{self.return_on_investment_mean},' \
            f'{self.return_on_investment_std},' \
            f'{self.sharpe_ratio},' \
            f'{self.sortino_ratio},' \
            f'{self.calmar_ratio},' \
            f'{self.roundtrip_winning_net_pnl},' \
            f'{self.roundtrip_loosing_net_pnl},' \
            f'{self.roundtrip_net_profit_pnl_percentage},' \
            f'{self.roundtrip_net_winning_percentage},' \
            f'{self.roundtrip_net_loosing_percentage},' \
            f'{self.roundtrip_average_net_winning_loosing_percentage},' \
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

        balance = self.account.balance
        balance_return = (balance / self.balance - 1.0) \
            if (self.balance != 0 and self.balance is not None) else 0.0
        self.balance = balance
        self.halted = self.account.is_halted
        self.net_profit = self.account.report.net_profit
        self.total_commission = self.account.report.total_commission
        self.max_drawdown_percent = self.account.report.max_drawdown_percent if \
            self.account.report.max_drawdown_percent is not None else 0.0

        twr = (self.time_weighted_return + 1.0) * (balance_return + 1.0)
        self.time_weighted_return = twr - 1.0

        self.rate_of_return = self.account.report.ror
        self.return_on_investment = self.account.report.returns_on_investments[-1] \
            if len(self.account.report.returns_on_investments) > 0 else 0.0
        self.return_on_investment_mean = self.account.report.roi_mean \
            if self.account.report.roi_mean is not None else 0.0
        self.return_on_investment_std = self.account.report.roi_std \
            if self.account.report.roi_std is not None else 0.0

        val = self.account.report.sharpe_ratio
        self.sharpe_ratio = val if val is not None else 0.0
        val = self.account.report.sortino_ratio
        self.sortino_ratio = val if val is not None else 0.0
        val = self.account.report.calmar_ratio
        self.calmar_ratio = val if val is not None else 0.0

        perf = self.account.position.roundtrip_performance
        self.roundtrip_winning_net_pnl = perf.winning_net_pnl
        self.roundtrip_loosing_net_pnl = perf.loosing_net_pnl
        self.roundtrip_net_profit_pnl_percentage = perf.net_profit_pnl_percentage
        self.roundtrip_net_winning_percentage = perf.net_winning_percentage / 100.0
        self.roundtrip_net_loosing_percentage = perf.net_loosing_percentage / 100.0
        self.roundtrip_average_net_winning_loosing_percentage = perf.average_net_winning_loosing_percentage / 100.0
        self.roundtrip_max_consecutive_net_winners = perf.max_consecutive_net_winners
        self.roundtrip_max_consecutive_net_loosers = perf.max_consecutive_net_loosers
        self.roundtrip_average_maximum_adverse_excursion = perf.average_maximum_adverse_excursion / 100.0
        self.roundtrip_average_maximum_favorable_excursion = perf.average_maximum_favorable_excursion / 100.0
        self.roundtrip_average_entry_efficiency = perf.average_entry_efficiency / 100.0
        self.roundtrip_average_exit_efficiency = perf.average_exit_efficiency / 100.0
        self.roundtrip_average_total_efficiency = perf.average_total_efficiency / 100.0
