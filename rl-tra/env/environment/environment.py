from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union
from collections import OrderedDict
from numbers import Real
from time import sleep
from datetime import datetime, timedelta, timezone
import logging

import numpy as np
import gymnasium as gym

from .frame import Frame 
from .providers import Provider, Trade
from .aggregators import TradeAggregator
from .actions import ActionScheme
from .rewards import RewardScheme
from .features import Feature, TradesFeature
from .broker import Broker
from .accounts import Account,DayCountConvention

from .renderers import CsvRenderer, MatplotlibPriceRenderer, MatplotlibObservationRenderer

def binance_commission(operation: str, amount: Real, price: Real) -> Real:
    return 0.0004 * abs(amount) * price

class Environment(Broker, gym.Env):
    """
    Gymnasium-compatible environment to simulate intraday trading episodes.
    """
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 3}

    def __init__(self,
        provider: Union[Provider, Sequence[Provider]],
        aggregator: TradeAggregator,
        action_scheme: ActionScheme,
        reward_scheme: RewardScheme,
        features_pipeline: Optional[Sequence[Feature]] = None,
        initial_balance: Real = 100000,
        halt_account_if_negative_balance: bool=True,
        annual_risk_free_rate: Real = 0,
        annual_target_return: Real = 0,
        day_count_convention: Literal['raw','30/360 us','30/360 us eom',
            '30/360 us nasd','30/360 eu','30/360 eu2','30/360 eu3',
            '30/360 eu+','30/365','act/360','act/365 fixed','act/365 nonleap',
            'act/act excel','act/act isda','act/act afb'] = 'raw',
        agent_order_delay: Union[Real, timedelta] = 3,
        broker_order_delay: Union[Real, timedelta] = 0.5,
        order_luck: float = 1.0,
        commission: Union[Real, Callable[[str, Real, Real], Real]] = binance_commission,
        warm_up_duration: Optional[Union[Real, timedelta]] = None,
        episode_max_duration: Optional[Union[Real, timedelta]] = None,
        episode_max_steps: Optional[int] = 512,
        delay_per_second: Optional[float] = None,
        instant_balance_update = False,
        max_frames_period = 4,
        max_trades_period = 4,
        render_mode: Optional[str] = None,
        render_observations: bool = False,
        log: Optional[logging.Logger] = None,
        vec_env_index: Optional[int] = None,
        **kwargs):
        """
        Args:
            provider Union[Provider, Sequence[Provider]]:
                An instance of provider data source, or a list of such providers.
            aggregator TradeAggregator:
                An instance of aggregator object, which is responsible for dividing
                a sequence of trades into frames.
            action_scheme ActionScheme:
                Action scheme to be used by all agents.
                It specifies how chosen actions are converted into buy/sell orders.
            reward_scheme RewardScheme:
                Reward scheme defines how reward is computed for an agent.
            features_pipeline Optional[Sequence[Feature]]:
                Specify a list of features to compute some additional information
                for agents. Features are called on each frame (trade) in the order
                as they are provided.
                Default: None
            initial_balance Real:
                Specify initial balance to start episode with.
                Default: 100000
            halt_account_if_negative_balance bool:
                Specify whether to halt the account if balance becomes negative.
            annual_risk_free_rate Real:
                Specify the annual risk free rate (1% is 0.01) to use
                in the account performance calculations.
                Default: 0.0
            annual_target_return Real:
                Specify the annual rarget return (1% is 0.01) to use
                in the account performance calculations.
                
                in context of Sortino ratio, it is the Minimum Acceptable
                Return (MAR, or Desired Target Return (DTR).
                Default: 0.0
            day_count_convention Literal['raw','30/360 us','30/360 us eom','30/360 us nasd','30/360 eu','30/360 eu2','30/360 eu3','30/360 eu+','30/365','act/360','act/365 fixed','act/365 nonleap','act/act excel','act/act isda','act/act afb']:
                Specify the day counting convention to use in the account
                performance calculations.
                Default: 'raw'
            agent_order_delay Union[Real, timedelta]:
                Specify a delay (in seconds) between the moment agent issues its
                action and the moment order reaches an exchange.
                Default: 3 (seconds)
            broker_order_delay Union[Real, timedelta]:
                Specify a delay (in seconds) between the moment when TakeProfit
                or StopLoss orders are signalled to be executed and the moment
                MarketOrder reaches an exchange.
                Default: 0.5 (seconds)
            order_luck float:
                Specify the probability for limit order to be executed in a
                scenario, when limit orders fail to be executed even if price
                reaches them. Simply because they were last in the order book
                and there were not enough corresponding market buy(sell) orders
                to fulfill them.

                For instance, if order_luck = 0.10, then 10% of limit orders
                will be executed even if price does not reach them.
                Default: 1.0 (Probability = 100%)
            commission Union[Real, Callable[[str, Real, Real], Real]]:
                Specify a commission as a fixed number or a callable function:
                ```python
                def binance_commission(operation: str, amount: Real, price: Real) -> Real:
                    return 0.0004 * abs(amount) * price
                ```
                Note: operation is either 'B' or 'S'
                Default: binance_commission (0.0004 * amount * price)
            warm_up_duration Optional[Union[Real, timedelta]]:
                Many features require some time (some frames) to start output a reasonable
                values. You may specify the number of seconds or a timedelta object.
                For instance, 10*60 or timedelta(minutes = 10) to warm up the environment.
                Default: None
            episode_max_duration Optional[Union[Real, timedelta]]:
                The maximum duration of an episode.
                You may specify the number of seconds or a timedelta object.
                For instance, 2*60*60 or timedelta(hours = 2).
                Default: None
            episode_max_steps Optional[int]:
                The maximum number of steps in an episode.
                This is the primary stopping condition for the episode.
                Default: 512
            delay_per_second Optional[float]:
                You may optionally specify the desired delay to limit CPU usage.
                It is a part of a second [0.0 ... 1.0) when process will go to sleep.
                May be actual in case you run into CPU overheating issues.
                Default: None
            instant_balance_update bool:
                In reality, in most cases, exchange will block your account if your
                balance becomes negative. This parameters specifies how often an agent
                balance will be updated. If True - balances will be updated after each
                trade (which is slow). If False - balances will be updated only at the
                frame end or when agent issues an order.
                Default: False
            max_frames_period int:
                Specify how many last frames to keep.
                Note: this limit will be increased if any of features in `features_pipeline`
                demands more frames to compute its values.
                Default: 4
            max_trades_period int:
                Specify how many last trades to keep.
                Note: this limit will be increased if any of features in `features_pipeline`
                demands more trades to compute its values.
                Default: 4
            render_mode Optional[str]:
                Specify the rendering mode. It may be 'ansi', 'rgb_array' or None.
                Default: None
            render_observations bool:
                This option is used only if render_mode is 'rgb_array'.
                Otherwise, it is ignored.

                It is useful when you want to see how observations are correlated
                and how observation series look like.

                Set it to True to render observations with the correlation heatmap.

                Set it to False to render rewards, price history, orders and account
                performance info.

                Default: False
            log Optional[logging.Logger]:
                Optionally specify a logger object to receive some info and debug messages.
                Default: None
            vec_env_index Optional[int]:
                Optionally specify the index of the environment in a vectorized environment.
        """
        if not isinstance(vec_env_index, int) and vec_env_index is not None:
            raise ValueError(f'Vector environment index {vec_env_index}' \
                'must be an integer number or None')
        self.vec_env_index = vec_env_index

        # Initialize the renderer.
        if render_mode is not None and \
            render_mode not in self.metadata['render_modes']:
            raise ValueError(f'Invalid render_mode. It must be one '
                             f'of the following: {self.metadata["render_modes"]}')
        self.render_mode: str = render_mode
        if self.render_mode == 'rgb_array':
            if render_observations:
                self.renderer = \
                    MatplotlibObservationRenderer(vec_env_index=vec_env_index)
            else:
                self.renderer = \
                    MatplotlibPriceRenderer(vec_env_index=vec_env_index)
        elif self.render_mode == 'ansi':
            self.renderer = CsvRenderer(vec_env_index=vec_env_index)
        else:
            self.renderer = None
        
        if warm_up_duration is None:
            warm_up_duration = timedelta(seconds=0.0)
        elif isinstance(warm_up_duration, Real):
            warm_up_duration = timedelta(seconds=float(warm_up_duration))
        elif isinstance(warm_up_duration, timedelta):
            pass
        else:
            raise ValueError(f'Invalid {warm_up_duration} value')
        assert warm_up_duration.total_seconds() >= 0
        self.warm_up_duration: timedelta = warm_up_duration

        # Maximum duration of an episode.
        if episode_max_duration is None:
            pass
        else:
            if isinstance(episode_max_duration, Real):
                episode_max_duration = timedelta(seconds=float(episode_max_duration))
            elif isinstance(episode_max_duration, timedelta):
                pass
            else:
                raise ValueError(f'Invalid episode_max_duration value: '
                    f'{episode_max_duration} of type {type(episode_max_duration)}')
            if episode_max_duration.total_seconds() <= 0:
                raise ValueError(f'episode_max_duration seconds '
                    f'{episode_max_duration} must be positive')
        self.episode_max_duration: Optional[timedelta] = episode_max_duration

        # Maximum number of steps in an episode.
        if episode_max_steps is None:
            pass
        elif isinstance(episode_max_steps, int):
            if episode_max_steps <= 0:
                raise ValueError(f'episode_max_steps {episode_max_steps} must be positive')
        else:
            raise ValueError(f'Invalid episode_max_steps value: {episode_max_steps} '
                             f'of type {type(episode_max_steps)}')
        self.episode_max_steps: Optional[int] = episode_max_steps

        # Setup delay to reduce CPU usage during reset().
        assert ((delay_per_second is None) or
            (isinstance(delay_per_second, float) and (0.0 <= delay_per_second < 1.0)))
        self.delay_per_second = delay_per_second

        # Initialize account.
        if not isinstance(initial_balance, Real) or (initial_balance <= 0):
            raise ValueError(f'Initial balance {initial_balance} must be a positive number')

        if not isinstance(annual_risk_free_rate, Real) or (annual_risk_free_rate < 0):
            raise ValueError(f'Annual risk free rate {annual_risk_free_rate} must not be a negative number')

        if not isinstance(annual_target_return, Real):
            raise ValueError(f'Annual target return {annual_target_return} must be a number')

        account = Account(initial_balance = initial_balance,
            annual_risk_free_rate = annual_risk_free_rate,
            annual_target_return = annual_target_return,
            day_count_convention = DayCountConvention.from_string(day_count_convention))

        # Setup logging.
        self._log = log if isinstance(log, logging.Logger) \
            else logging.getLogger(self.__class__.__name__)
        
        # Initialize parent Broker.
        super().__init__(account = account,
                         agent_order_delay = agent_order_delay,
                         broker_order_delay = broker_order_delay, # TODO: price_step
                         order_luck = order_luck,
                         commission = commission,
                         instant_balance_update = instant_balance_update,
                         halt_account_if_negative_balance = halt_account_if_negative_balance,
                         **kwargs)
        
        # Setup main objects.
        if isinstance(provider, Provider):
            self.providers = [provider]
        elif isinstance(provider, Sequence):
            if len(provider) < 1:
                raise ValueError(f'At least 1 data provider must be specified')
            if not all([isinstance(p, Provider) for p in provider]):
                raise ValueError(f'Some of objects in provider sequence {provider} ' \
                    'are not providers')
            self.providers = provider
        else:
            raise ValueError(f'Invalid provider {provider}')
        self.provider: Optional[Provider] = None
        self.aggregator = aggregator
        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.last_action = None
        self.features_pipeline = features_pipeline
        
        # Keeps list of features which require update on each processed trade.
        self.trades_features = []
        
        # Collect all state names.
        spaces = OrderedDict()

        # Calculate maximum periods for trades and frames.
        # Process features pipeline.
        max_period_frames = 0
        max_period_trades = 0
        if isinstance(features_pipeline, Sequence):
            # Iterate over features in pipeline.
            for feature in features_pipeline:
                # Set the account, some features extract account statistics.
                feature.account = self.account
                # Check if it is a TradesFeature.
                if isinstance(feature, TradesFeature):
                    # Add feature to a special list.
                    self.trades_features.append(feature)
                    # Check maximum requested trades_period for the feature.
                    trades_period = feature.trades_period
                    if isinstance(trades_period, int):
                        pass
                    elif isinstance(trades_period, Sequence):
                        trades_period = max(trades_period)
                    else:
                        trades_period = 0
                    if max_period_trades < trades_period:
                        max_period_trades = trades_period
                # Collect names from feature if they will be written to a state.
                for name, space in feature.spaces.items():
                    spaces[name] = space
                # Check maximum requested period for the feature.
                frames_period = feature.period
                if isinstance(frames_period, int):
                    pass
                elif isinstance(frames_period, Sequence):
                    frames_period = max(frames_period)
                else:
                    frames_period = 0
                if max_period_frames < frames_period:
                    max_period_frames = frames_period
        
        # Setup maximum requested periods.
        self.max_trades_period = max(max_period_trades, max_trades_period or 0)
        self.max_frames_period = max(max_period_frames, max_frames_period or 0)
        
        # Setup observation and action spaces.
        self.observation_space: gym.spaces.Space = gym.spaces.Dict(spaces)
        self.observation_names = tuple(spaces.keys())
        self.action_space: gym.spaces.Space = action_scheme.space

        # Episode variables.
        self.step_number: int = 0
        self.episode_number: int = 0
        self.episode_start_datetime: Optional[datetime] = None
        self.span_start_time: Optional[datetime] = None
        self.trades: List[Trade] = []
        self.frames: List[Frame] = []
        self.state: Optional[OrderedDict] = None
        self.rng: Optional[np.random.RandomState] = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.array, dict[str, Any]]:
        # This should also reset Broker class, which is also a parent class.
        # The Broker class should reset account and position.
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self.episode_number += 1
        self.step_number = 0
        self.aggregator.reset()
        self.action_scheme.reset()
        self.reward_scheme.reset()
        for feature in self.features_pipeline:
            feature.reset()

        # Reset episode variables.
        self.episode_start_datetime = None
        self.span_start_time = None
        self.trades.clear()
        self.frames.clear()

        # Setup random provider.
        if len(self.providers) == 1:
            self.provider = self.providers[0]
        else:
            if self.provider is not None:
                self.provider.close()
            idx = int(self.np_random.uniform(0, len(self.providers)))
            self.provider = self.providers[idx]

        # Initialize provider for a random or specified date (any other behavior is encoded in **kwargs).
        min_duration = self.episode_max_duration.total_seconds() \
            if self.episode_max_duration is not None else 0
        self.episode_start_datetime = self.provider.reset(
            episode_start_datetime = None,
            episode_min_duration = \
                min_duration + self.warm_up_duration.total_seconds(),
            rng = self.rng) + self.warm_up_duration

        # Reset span time.
        self.span_start_time = datetime.now(timezone.utc)

        # Read first frame.
        # If max_frames_period is positive, we need to read
        # max_frames_period frames, otherwise feature pipeline
        # may not function correctly, producing rubbish in state
        # output.
        for i in range(self.max_frames_period + 1):
            frame, truncated = self._get_next_frame()
            if (frame is None) or truncated:
                raise RuntimeError('Failed to get initial state for frame '
                    f'{i+1} of {self.max_frames_period} frames, episode '
                    f'starttime is {self.episode_start_datetime}')
        # Update the episode start datetime.
        if self.max_frames_period > 0:
            self.episode_start_datetime = frame.time_start

        # Initialize account performance after account reset.
        self.account.update_performance(
            price_high = frame.high, price_low = frame.low,
            #price_last = self.last_trade.price, datetime_last=self.last_trade.datetime,
            price_last = frame.close, datetime_last=frame.time_end,
            price_first=frame.open, datetime_first=frame.time_start)

        state = self._make_state()
        self._set_frame_info(frame)
        self.last_action = None

        if self.renderer is not None:
            self.renderer.reset(episode_number = self.episode_number,
                                episode_max_steps = self.episode_max_steps,
                                account = self.account, provider = self.provider,
                                aggregator = self.aggregator,
                                frames = self.frames, observation = state)
        return state, frame

    def step(self, action: Any) -> Tuple[Union[OrderedDict, None], float, bool, bool, Union[Frame, None]]:
        # Based on previous frame, actor made actions.
        # Convert these actions to orders.
        action_time = self.last_trade.datetime + self.agent_order_delay
        self.action_scheme.process_action(broker = self,
            account = self.account, action = action, time = action_time)
        self.last_action = action

        # Read next frame and execute orders.
        # Simulates a trading session for this step.
        frame, truncated = self._get_next_frame()
        if frame is not None:
            # Construct state for the agent.
            state = self._make_state()
            truncated = truncated and (state is not None)
        else:
            truncated = True
            state = None

        # Simulated trading session ended, update account performance.
        self.account.update_performance(
            price_high = frame.high, price_low = frame.low,
            #price_last = self.last_trade.price, datetime_last=self.last_trade.datetime,
            price_last = frame.close, datetime_last=frame.time_end,
            price_first=frame.open, datetime_first=frame.time_start)

        # Check for episode completion.
        self.step_number += 1
        if (not truncated) and (self.episode_max_steps is not None):
            truncated = self.step_number >= self.episode_max_steps
        if (not truncated) and (self.episode_max_duration is not None) \
            and (frame.time_end is not None):
            episode_duration = (frame.time_end - self.episode_start_datetime)
            truncated = episode_duration >= self.episode_max_duration

        if truncated:
            # Close agent position if done.
            # TODO: Since this is the last step, we can place closing orders
            # but we cannot process them. Kind of useless.
            if self.account.has_position:
                quantity_signed = self.account.position.quantity_signed
                price = self.last_trade.price
                commission = self._get_commission( \
                    operation = 'S' if (quantity_signed > 0) else 'B', \
                    amount = abs(quantity_signed),
                    price = price)
                self.account.close_position(datetime = self.last_trade.datetime, \
                    price = price, commission = commission, \
                    notes = 'closing on episode end')
            
        reward = self.reward_scheme.get_reward(env = self, account = self.account)
        terminated = (self.account.is_halted or self.account.balance <= 0) \
            if self.halt_account_if_negative_balance else False

        self._set_frame_info(frame)
        if self.renderer is not None:
            self.renderer.step(frames = self.frames, reward = reward,
                               observation = state)

        return state, reward, terminated, truncated, frame

    def _get_next_frame(self) -> Tuple[Union[Frame, None], bool]:
        # Iterate until the next frame.
        frame = None
        truncated = False
        while True:
            try:
                # Read next trade from provider.
                trade = next(self.provider)
                # Process trade to update statistics and execute orders.
                self.process_trade(trade)
                # Check if this trade belongs to the time period of our interest.
                if trade.datetime >= self.episode_start_datetime - self.warm_up_duration:
                    # Add trade to the list of trades.
                    self.trades.append(trade)
                    # Remove old trade.
                    if len(self.trades) > self.max_trades_period + 1:
                        del self.trades[0]
                    # Process trade to construct a frame.
                    frame = self.aggregator.aggregate(self.trades)
                    # Update features which require each trade.
                    for feature in self.trades_features:
                        feature.update(self.trades)
                # Perform delay if needed.
                if self.delay_per_second is not None:
                    # Calculate how much time have passed.
                    span_duration = float((datetime.now(timezone.utc) - self.span_start_time).microseconds)
                    if span_duration >= 1000000.0 * (1.0 - self.delay_per_second):
                        sleep(self.delay_per_second)
                        # Reset span timer.
                        self.span_start_time = datetime.now(timezone.utc)
                if frame is not None:
                    self._process_frame(frame)
                    # Return next frame, if its time >= episode_start_datetime.
                    if (frame.time_end is not None) and (frame.time_end >= self.episode_start_datetime):
                        break
            except StopIteration:
                # We get here when provider signals there are no more trades to process.
                # Force aggregator to return any last unfinished frame.
                frame = self.aggregator.finish()
                if frame is not None:
                    self._process_frame(frame)
                truncated = True
                break
        return frame, truncated
    
    def _process_frame(self, frame: Frame):
        # Save new frame.
        self.frames.append(frame)
        # Remove old frame.
        if len(self.frames) > self.max_frames_period + 1:
            del self.frames[0]
        # Collect state from feature pipeline.
        self.state = OrderedDict()
        for feature in self.features_pipeline:
            feature.process(self.frames, self.state)
            
    def _make_state(self) -> OrderedDict:
        assert isinstance(self.state, OrderedDict)
        return self.state

    def _set_frame_info(self, frame: Frame):
        if frame is not None:
            frame['vec_env_index'] = self.vec_env_index if self.vec_env_index is not None else 0
            frame['episode_number'] = self.episode_number
            frame['step_number'] = self.step_number
            frame['provider_name'] = self.provider.name
            frame['aggregator_name'] = self.aggregator.name
            frame['account_halted'] = self.account.is_halted

    def close(self):
        # Reset broker (it resets accounts).
        super().reset()
        self.provider.close()
        self.provider = None
        self.aggregator.reset()
        self.action_scheme.reset()
        self.reward_scheme.reset()
        # Reset episode variables.
        self.episode_start_datetime = None
        self.span_start_time = None
        self.trades.clear()
        self.frames.clear()
        if self.renderer is not None:
            self.renderer.close()

    def render(self, mode: Optional[str]=None) -> Optional[Union[str, np.array]]:
        return self.renderer.render() if self.renderer is not None else None
