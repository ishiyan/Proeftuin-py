from datetime import date
import math
import pandas as pd

from matplotlib import pyplot as plt
FIG_EXT = 'svg' # 'png' or 'svg'
if FIG_EXT == 'svg':
    # The following allows to save plots in SVG format.
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

from environment import BinanceMonthlyTradesProvider, BinanceMonthlyKlines1mToTradesProvider
from environment import IntervalTradeAggregator
from environment import BalanceReturnReward
from environment import BuySellHoldCloseAction
from environment import Environment
from environment import OhlcRatios, TimeEncoder, Scale, CopyPeriod
from environment import AccountCalmar, AccountSharpe, AccountSortino, AccountROI, AccountROR
from visualizations import plot_charts, plot_correlation_heatmap, fig_to_rgb_array, write_animated_gif

WHAT = 'binance-klines'
#WHAT = 'binance-trades'

SYMBOL = 'ETHUSDT'
if WHAT == 'binance-trades':
    dir = 'D:/data/binance_monthly_trades/'
    provider = BinanceMonthlyTradesProvider(data_dir = dir,
                symbol = SYMBOL,
                date_from = date(2024, 5, 1), date_to = date(2024, 5, 31))
elif WHAT == 'binance-klines':
    dir = 'data/binance_monthly_klines/'
    provider = BinanceMonthlyKlines1mToTradesProvider(data_dir = dir,
                symbol = SYMBOL,
                date_from = date(2023, 1, 1), date_to = date(2024, 5, 31))
else:
    raise ValueError(f'Unknown WHAT {WHAT}')

TIME_FRAME='1m'
#TIME_FRAME='1h'
TIME_FRAME_SECS = 1*60 if TIME_FRAME == '1m' else 60*60
aggregator = IntervalTradeAggregator(method='time',
                interval=TIME_FRAME_SECS, duration=(1, 8*60*60))

SCALE_METHOD = 'zscore'
SCALE_PERIOD = 196
COPY_PERIOD = 196

features_pipeline = [
    Scale(source=['open', 'high', 'low', 'close', 'volume'], method=SCALE_METHOD,
          period=SCALE_PERIOD, write_to='state'),
    OhlcRatios(write_to='state'),
    TimeEncoder(source=['time_start'], yday=True, wday=True, tday=True, write_to='state'),
]
features_pipeline_period = [
    Scale(source=['open', 'high', 'low', 'close', 'volume'], method=SCALE_METHOD,
          period=SCALE_PERIOD, write_to='frame'),
    OhlcRatios(write_to='frame'),
    TimeEncoder(source=['time_start'], yday=True, wday=True, tday=True, write_to='frame'),
    #AccountCalmar(write_to='frame'),
    #AccountSharpe(write_to='frame'),
    #AccountSortino(write_to='frame'),
    #AccountROI(write_to='frame'),
    #AccountROR(write_to='frame'),
    CopyPeriod(source=[
        ('open', -math.inf, math.inf),
        ('high', -math.inf, math.inf),
        ('low', -math.inf, math.inf),
        ('close', -math.inf, math.inf),
        (f'{SCALE_METHOD}{SCALE_PERIOD}_open', -math.inf, math.inf),
        (f'{SCALE_METHOD}{SCALE_PERIOD}_high', -math.inf, math.inf),
        (f'{SCALE_METHOD}{SCALE_PERIOD}_low', -math.inf, math.inf),
        (f'{SCALE_METHOD}{SCALE_PERIOD}_close', -math.inf, math.inf),
        (f'{SCALE_METHOD}{SCALE_PERIOD}_volume', -math.inf, math.inf),
        ('ol_hl', 0.0, 1.0),
        ('cl_hl', 0.0, 1.0),
        ('yday_time_start', 0.0, 1.0),
        ('wday_time_start', 0.0, 1.0),
        ('tday_time_start', 0.0, 1.0),
        #('roi', -math.inf, math.inf),
        #('ror', -math.inf, math.inf),
        #('sharpe', -math.inf, math.inf),
        #('calmar', -math.inf, math.inf),
        #('sortino', -math.inf, math.inf),
        ], copy_period=COPY_PERIOD)
]

env = Environment(
    provider=provider,
    aggregator=aggregator,
    features_pipeline=features_pipeline_period,
    action_scheme=BuySellHoldCloseAction(),
    reward_scheme=BalanceReturnReward(),
    warm_up_duration=None,
    episode_max_duration=None,#2*60*60,
    render_mode=None,
    initial_balance=10000,
    episode_max_steps=128#196
)

def make_charts(state, episode=None, step=None, show=False):
    if episode is not None:
        title = f', episode {episode}'
        if step is not None:
            title += f' step {step}'
    elif step is not None:
        title = f'step {step}'
    else:
        title = ''
    df = pd.DataFrame(state)
    plt.close()
    fig = plot_charts(df,
        title=f'{SCALE_METHOD}{SCALE_PERIOD} candlesticks{title}',
        candlestick_fields=[
            f'{SCALE_METHOD}{SCALE_PERIOD}_open_{COPY_PERIOD}',
            f'{SCALE_METHOD}{SCALE_PERIOD}_high_{COPY_PERIOD}',
            f'{SCALE_METHOD}{SCALE_PERIOD}_low_{COPY_PERIOD}',
            f'{SCALE_METHOD}{SCALE_PERIOD}_close_{COPY_PERIOD}'],
        #title_second=f'original price candlesticks',
        #candlestick_fields_second=[
        #    f'open_{COPY_PERIOD}',
        #    f'high_{COPY_PERIOD}',
        #    f'low_{COPY_PERIOD}',
        #    f'close_{COPY_PERIOD}'],
        pane_line_fields=[
            #[f'{SCALE_METHOD}{SCALE_PERIOD}_open_{COPY_PERIOD}'],
            #[f'{SCALE_METHOD}{SCALE_PERIOD}_high_{COPY_PERIOD}'],
            #[f'{SCALE_METHOD}{SCALE_PERIOD}_low_{COPY_PERIOD}'],
            #[f'{SCALE_METHOD}{SCALE_PERIOD}_close_{COPY_PERIOD}'],
            [f'{SCALE_METHOD}{SCALE_PERIOD}_volume_{COPY_PERIOD}'],
            [f'ol_hl_{COPY_PERIOD}'],
            [f'cl_hl_{COPY_PERIOD}'],
            [f'yday_time_start_{COPY_PERIOD}'],
            [f'wday_time_start_{COPY_PERIOD}'],
            [f'tday_time_start_{COPY_PERIOD}'],
            #[f'roi_{COPY_PERIOD}'],
            ],
        dark=True,
        show_legend=False,
        figsize=(12, 8)
        )
    rgb_array = fig_to_rgb_array(fig)
    if show:
        plt.show()
    plt.close()#fig)
    return rgb_array

def make_correlations(state, episode=None, step=None, show=False):
    if episode is not None:
        title = f'episode {episode}'
        if step is not None:
            title += f', step {step}'
    elif step is not None:
        title = f'step {step}'
    else:
        title = None
    df = pd.DataFrame(state)
    columns_to_drop = ['open_196', 'high_196', 'low_196', 'close_196']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    df = df.rename(columns={
        f'{SCALE_METHOD}{SCALE_PERIOD}_open_{COPY_PERIOD}': f'{SCALE_METHOD}_open',
        f'{SCALE_METHOD}{SCALE_PERIOD}_high_{COPY_PERIOD}': f'{SCALE_METHOD}_high',
        f'{SCALE_METHOD}{SCALE_PERIOD}_low_{COPY_PERIOD}': f'{SCALE_METHOD}_low',
        f'{SCALE_METHOD}{SCALE_PERIOD}_close_{COPY_PERIOD}': f'{SCALE_METHOD}_close',
        f'{SCALE_METHOD}{SCALE_PERIOD}_volume_{COPY_PERIOD}': f'{SCALE_METHOD}_volume',
        f'ol_hl_{COPY_PERIOD}': 'ol/hl',
        f'cl_hl_{COPY_PERIOD}': 'cl/hl',
        f'yday_time_start_{COPY_PERIOD}': 'yday',
        f'wday_time_start_{COPY_PERIOD}': 'wday',
        f'tday_time_start_{COPY_PERIOD}': 'tday',
        })
    print(df.columns)
    #df.dropna(inplace=True)
    plt.close()
    fig = plot_correlation_heatmap(df,
        title=title,
        cmap=None,
        coeff=True,
        coeff_color=None,
        dark=False,
        decimals=2,
        dpi=120,
        figsize=(8, 8)
        )
    rgb_array = fig_to_rgb_array(fig)
    if show:
        plt.show()
    plt.close()#fig)
    return rgb_array

#state, frame = env.reset()
#fig = make_correlations(state, show=True)

#"""
full_chart_rgb_arrays=[]
full_corr_rgb_arrays=[]
episode_chart_rgb_arrays=[]
episode_corr_rgb_arrays=[]
durations = []
for episode in range(100):
    step = 0
    state, frame = env.reset()
    while True:
        step += 1
        print(f'Episode {episode+1}, step {step}')
        #durations.append(0.3)
        rgb_array = make_charts(state, episode=episode, step=step, show=False)
        full_chart_rgb_arrays.append(rgb_array)
        #rgb_array = make_correlations(state, episode=episode, step=step, show=False)
        #full_corr_rgb_arrays.append(rgb_array)

        action = env.action_space.sample()
        state, reward, terminated, truncated, frame = env.step(action)
        if terminated or truncated:
            #durations.append(2.0)
            rgb_array = make_charts(state, episode=episode, step=step, show=False)
            full_chart_rgb_arrays.append(rgb_array)
            episode_chart_rgb_arrays.append(rgb_array)
            #rgb_array = make_correlations(state, episode=episode, step=step, show=False)
            #full_corr_rgb_arrays.append(rgb_array)
            #episode_corr_rgb_arrays.append(rgb_array)
            break
env.close()
dir = 'visualizations/state_features/'

write_animated_gif(rgb_arrays=full_chart_rgb_arrays, durations=durations, fps=5,
    filename=f'full_episodes_state_charts_{SCALE_METHOD}{SCALE_PERIOD}_{TIME_FRAME}_{SYMBOL}.gif',
    dir=dir)
full_chart_rgb_arrays=None
write_animated_gif(rgb_arrays=full_corr_rgb_arrays, durations=durations,
    filename=f'full_episodes_state_corr_{SCALE_METHOD}{SCALE_PERIOD}_{TIME_FRAME}_{SYMBOL}.gif',
    dir=dir)
full_corr_rgb_arrays=None
write_animated_gif(rgb_arrays=episode_chart_rgb_arrays, durations=durations, fps=2,
    filename=f'end_episodes_state_charts_{SCALE_METHOD}{SCALE_PERIOD}_{TIME_FRAME}_{SYMBOL}.gif',
    dir=dir)
episode_chart_rgb_arrays=None
write_animated_gif(rgb_arrays=episode_corr_rgb_arrays, durations=durations,
    filename=f'end_episodes_state_corr_{SCALE_METHOD}{SCALE_PERIOD}_{TIME_FRAME}_{SYMBOL}.gif',
    dir=dir)
episode_corr_rgb_arrays=None
durations=None
#"""
