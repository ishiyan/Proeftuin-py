<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
<br />
<span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">
Intraday Exchange Gym Environment</span> by
<a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/diovisgood/intraday" property="cc:attributionName" rel="cc:attributionURL">Pavel B. Chernov</a>
is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

# Intraday

https://github.com/diovisgood/intraday

This package provides gym compatible environment to **simulate intraday trading**
based on stream of trades, either historical or real-time.

![gif animation of trained agent](doc/render_ethusdt_trained.gif)

This project was inspired by [TensorTrade](https://github.com/tensortrade-org/tensortrade),
but it was written from scratch, and it is a completely original source code.

The main idea was to go deeper from candles to the actual stream of trades.
Because candles lose a lot of information, for example:

- How many trades during a period were initiated by buyers or by sellers?
- At what price did most of the trades happen during a period?
- At what price there were almost no activity?
- Were there many trades with small amounts or less trades with big amounts?
- etc.

## Installation

1. Download package
2. Cd into directory
3. Install package

```bash
git clone https://github.com/diovisgood/intraday.git
cd intraday
python setup.py install
```

### Quick start

Here is a simple script to run:

```python
from datetime import date, timedelta
from intraday.providers import BinanceArchiveProvider
from intraday.processor import IntervalProcessor
from intraday.features import EMA, Copy, PriceEncoder
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from intraday.env import SingleAgentEnv

provider = BinanceArchiveProvider(data_dir='.', symbol='ETHUSDT',
                                  date_from=date(2018, 5, 1), date_to=date(2018, 5, 31))
processor = IntervalProcessor(method='time', interval=5*60)
period = 1000
atr_name = f'ema_{period}_true_range'
features_pipeline = [
    PriceEncoder(source='close', write_to='both'),
    EMA(period=period, source='true_range', write_to='frame'),
    Copy(source=['volume'])
]
action_scheme = BuySellCloseAction()
reward_scheme = BalanceReward(norm_factor=atr_name)
env = SingleAgentEnv(
    provider=provider,
    processor=processor,
    features_pipeline=features_pipeline,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    initial_balance=1000000,
    warm_up_time=timedelta(hours=1)
)

state = env.reset()
while True:
    env.render('human')
    print(state)
    action = action_scheme.get_random_action()
    state, reward, done, frame = env.step(action)
    if done:
        break
        
env.close()
```

You should see something like this:

![gif animation of chart window](doc/render_ethusdt_untrained.gif)

### Screen explained

Here is some explaination of different values on this screen:

![screenshot with notes](doc/render_01.png)

Note that in the code above agent performed random actions,
that is why balance chart is constantly decreasing.

By the way, notice different colors on the balance line?
Each color describes a position agent was at that moment:

- Green: agent was in long position.
- Red: agent was in short position.
- Black: agent was not in position.

## Advantages

This project has several advantages over other trading environments:

1. Simulates order *delays*, like in the real life.
2. Takes into account bid-ask *spread*, when executing market orders.
3. Some limit orders are not executed even if price *touches* them, like in the real life. (Read below for explanation).
4. Supports *multiple agents* running the same episode.
5. Supports *idle penalty* to stimulate agents to perform actions and not just hold the initial balance.
6. Liquidates agent account when it goes *bankrupt* (like in the real life).
7. Supports adjustable *commissions* either a constant value or a callback function.

It provides you with:

- Gym compatible environment.
- Real exchange simulation under the hood.
- Different methods to split stream of trades into frames:
  - by time interval (as usual)
  - by traded volume
  - by number of trades (a.k.a. ticks)
- Different actions schemes, including the most popular one: `{Buy, Sell, Close}`.
- Support for many popular order types (it means you can easily write any desired action scheme):
  - MarketOrder
  - LimitOrder
  - StopOrder
  - TrailingStopOrder
  - TakeProfitOrder
- Support for different reward schemes.
  The most obvious one is: `BalanceReward`. Also there is: `ConstantReward`.
- Ability to train agent on different assets simultaneously.
  In this case the reward agent receives when trading different assets highly depends on their price.
  For example: `BTCUSD` price can move $4000 per day, while for `ETHUSD` it has range of about $100 per day.
  Hence the rewards based on balance will differ more than 10 times,
  making rewards from `BTCUSD` more valuable than those of `ETHUSD`, which is wrong!
  To solve this problem you can automatically divide balance reward with computed
  [**ATR**](https://en.wikipedia.org/wiki/Average_true_range) value.
  Thus making rewards from different assets to be in the same range.
  So it would be possible to train agent on different assets with different prices simultaneously.
- Support for many popular features and indicators:
  - ADL
  - CMF
  - Cumulative Sum
  - Efficiency Ratio
  - EMA
  - EOM
  - Fractals
  - Fractal Dimensions
  - Heiken Ashi
  - KAMA
  - OBV
  - Parabolic SAR
  - etc.
- Two Binance providers: `BinanceArchiveProvider` and `BinanceKlines`.
- Trades simulation, if for some reason you only have candles.
- Enhanced evaluation of agent trading performance:
  - roi
  - sharpe ratio
  - profit factor
  - sortino ratio
  - etc.

## Details

This project tries to simulate trading environment as realistically as possible.
A lot of trained models seem to perform good during training, but often fail to show
any positive result in real trading. Mainly because they ignore following issues.

### Order delays from agent to exchange

There is always a delay between the moment agent makes its decision and the
moment order reaches an exchange, due to network or exchange delays.
Typically, about 1..3 seconds.
This package allows you to specify it explicitly in `agent_order_delay` argument.

### Order delays from broker to exchange or inside exchange

For some kind of orders like: `TakeProfitOrder` or `StopOrder` there is a small internal delay
between the moment broker/exchange server decides to execute this order by placing the `MarketOrder`
and the moment when `MarketOrder` reaches an exchange. Typically, about 0.5...1 seconds.
This package allows you to specify it explicitly in `broker_order_delay` argument.

### Realistic bid-ask spread

When agent wants to buy or sell immediately it issues a `MarketOrder`.
This order should be executed by the best price which is available there in the order book.
But there is always a gap between best bid and best ask price, which is often ignored.

Imagine a situation: the last trade was a `sell at a price 103.5`,
and agent decides it should buy immediately.
But it can't buy at this price, since it was a sell trade, which was executed at the **best bid price**.
While buy order will be executed at **best ask price**.
The best ask price is typically higher than best bid price, say, for example: 104.0 (thus the spread is: 0.5)

We don't have order book since this is a simulated Exchange environment.
But this package analyzes the stream of buy and sell trades and estimates the mean and std
values for the bid-ask spread. It then uses the upper estimation of spread to choose
the realistic price for `MarketOrder` to be executed.

### Limit orders realistic execution

Another issue is with limit orders. Sometimes limit orders are not executed even if price *touches* them.
Simply because they were last in the order book and there were not enough corresponding market buy(sell)
orders to fulfill them.
This package lets you specify explicitly the probability for limit order to be executed in such scenario,
via `order_luck` argument.

### Support for multiple agents

Unlike most gym environments this package is able to run multiple agents on the same episode.
It does this by allocating different virtual broker's accounts for each agent.
So each agent has its own `initial_balance` and can perform its own trades.
This can be useful for some optimization algorithms like: Evolutions Strategies, CMA-ES, etc.

### Support for idle penalty

When trained in complicated stochastic environments agents often tend to **do nothing**.
This is a simple way to save its life or money balance.
A simple solution to the problem is to introduce some penalty for agent for being idle.

Thus, its balance will slightly decrease on each step even if it did not open long or short position.
This decrease is equal to the price range of the current frame multiplied by `idle_penalty` parameter.

### Liquidation of agent account when it goes bankrupt

In reality, in most cases, exchange will block your account if your balance becomes negative.
This means you can no longer buy or sell assets.

Maybe this is not very useful for the aim of agent training.
I suggest a better approach would be to let it make mistakes at the beginning of learning.
And after some time, as it matures, bring some real constraints.

This could easily be achieved by specifying very large `initial_balance` at the start of training.
And then reduce it to some realistic value over time.

### Adjustable commissions

Many researchers successfully train their models to trade different assets without commission.
Then, of course, these models fail in the real world.
Because if there are no commissions in training environments agents may learn to perform a lot of trades.
Which will lead to a great losses in the real world.

This package allows you to specify either a fixed commission, or a callback function.
For example:

```python
def binance_commission(operation: str, amount: float, price: float) -> float:
    return 0.0004 * abs(amount) * price
```

Note: operation in a string, one of: `{'B', 'S'}`

### Support for any kind of data providers

`BinanceArchiveProvider` - automatically downloads monthly trades archives from [binance.com](binance.com)
and converts them into `.feather` file format for faster loading.
All you need to do is to specify symbol name, for example: 'BTCUSDT', and dates range.

`BinanceKLines` - automatically downloads monthly candles archives from [binance.com](binance.com)
and also converts them into `.feather` file format for faster loading.
KLine is a binance candle with some additional fields.
If you want to investigate large time intervals: say 30 minutes or 1 hour - processing trades archives
becomes **too slow**. In this case you may want to use 1 minute klines as your data source.
Note, that in this case you should not rely on features which analyze trades,
because trades are *simulated* in case of klines or candles.

`MoexArchiveProvider` - provider which works with raw stream of trades from Moscow Exchange
saved in a special binary format: `*.trades.gz`.
In this format each file represents one trading session (one day).
Just ignore it if you don't access to these archives.

`SineProvider` - Sine wave generator with adjustable noise.
If you want to train a trading bot, this sine generator
would be a good test for it.
If your algorithm fails to learn to make profit even on such simple data,
it will never find any profit in a real market data.


## TODO

1. Not all `features/*.py` have corresponding unit-tests. Ideally, each feature should have good unit-testing.
2. It would be nice to have support for other major exchanges like Coinbase.
