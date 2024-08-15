from datetime import date, datetime, timezone
from typing import List
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

FIG_EXT = 'svg' # 'png' or 'svg'

if FIG_EXT == 'svg':
    # The following allows to save plots in SVG format.
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

from environment import Provider, BinanceMonthlyKlines1mToTradesProvider
from environment import TradeAggregator, IntervalTradeAggregator
from environment import Frame

from environment.accounts.performances import Ratios, Periodicity
from environment.accounts.daycounting import DayCountConvention

# https://matplotlib.org/stable/gallery/color/named_colors.html
D_UP = 'limegreen'
D_DN = 'tab:blue'
D_LINE = 'tab:blue'
D_FIG = '#202020'
D_AX = '#303030'
D_TXT = '#666666'
D_TIT = '#d0d0d0'
KDE_LINE = 'limegreen'
HIST_BAR = 'tab:blue'

DO_PERF_MSECS = False
DO_ALL = False
SHOW_CHARTS = True

DO_RETURNS = True or DO_ALL
DO_CUMULATIVE_RETURNS = False or DO_ALL
DO_COMPOUND_GROWTH_RATE = False or DO_ALL
DO_DRAWDOWN = False or DO_ALL
DO_CALMAR_RATIO = False or DO_ALL
DO_STERLING_RATIO = False or DO_ALL
DO_BURKE_RATIO = False or DO_ALL
DO_PAIN_INDEX = False or DO_ALL
DO_ULCER_INDEX = False or DO_ALL
DO_PAIN_RATIO = False or DO_ALL
DO_ULCER_RATIO = False or DO_ALL
DO_DRAWDOWN_BASED_RATIOS = True or DO_ALL

def plot_ratios(df, title, 
        panes: List[List[str]]=[],
        dark=True, show_legend=False, figsize=(8, 4)):
    fig = plt.figure(dpi=120, layout='constrained', figsize=figsize)
    if dark:
        fig.set_facecolor(D_FIG)
    ax_cnt = len(panes)
    ax_panes = []
    N = 1 #3
    gs = fig.add_gridspec(N+ax_cnt, 1)
    ax = fig.add_subplot(gs[:N, 0])
    for i in range(ax_cnt):
        ax_panes.append(fig.add_subplot(gs[N+i, 0], sharex = ax))
    for a in [ax] + ax_panes:
        if dark:
            a.set_facecolor(D_AX)
            a.grid(color=D_TXT)
            a.tick_params(labelbottom=False, labelsize='small', colors=D_TXT)
        else:
            a.tick_params(labelbottom=False, labelsize='small')
        a.grid(False)
        a.tick_params(labelbottom=False)
    if ax_cnt > 0:
        ax_panes[-1].tick_params(labelbottom=True)
    else:
        ax.tick_params(labelbottom=True)
    if dark:
        ax.set_title(title, fontsize='small', color=D_TIT)
    else:
        ax.set_title(title, fontsize='small')

    wick_width=.2
    body_width=.8
    up_color=D_UP
    down_color=D_DN

    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    # Plot up candlesticks
    ax.bar(up.index, up.close - up.open, body_width, bottom=up.open, color=up_color)
    ax.bar(up.index, up.high - up.close, wick_width, bottom=up.close, color=up_color)
    ax.bar(up.index, up.low - up.open, wick_width, bottom=up.open, color=up_color)
    # Plot down candlesticks
    ax.bar(down.index, down.open - down.close, body_width, bottom=down.close, color=down_color)
    ax.bar(down.index, down.high - down.open, wick_width, bottom=down.open, color=down_color)
    ax.bar(down.index, down.low - down.close, wick_width, bottom=down.close, color=down_color)
    # Plot panes
    for i, pane in enumerate(panes):
        for column in pane:
            ax_panes[i].plot(df.index, df[column], label=column, linewidth=1)#, color='tab:blue')
        if show_legend:
            legend = ax_panes[i].legend(loc='best', fontsize='small', )
            legend.get_frame().set_alpha(0.1)
            if dark:
                legend.get_frame().set_facecolor(D_FIG)
                for text in legend.get_texts():
                    text.set_color(color=D_TIT)
        else:
            ax_panes[i].legend().set_visible(False)
            tit = ' / '.join(pane)
            if dark:
                ax_panes[i].set_title(tit, fontsize='small', color=D_TIT)
            else:
                ax_panes[i].set_title(tit, fontsize='small')
    return fig

def plot_distribution_histogram(df, columns, bins='auto', dark=True,
                                show_legend=True, figsize=(4.8, 3.6)):
    """
        Don't plot multiple columns on the same histogram,
        because colors are ugly an plot is unreadable.
    """
    # bins: integer or 'auto', 'scott', 'rice', 'sturges', 'sqrt'    
    fig, ax = plt.subplots(dpi=120, layout='constrained', figsize=figsize)
    if dark:
        fig.set_facecolor(D_FIG)
        ax.set_facecolor(D_AX)
        ax.tick_params(labelbottom=True, labelsize='small', colors=D_TXT)
        ax.grid(color=D_TXT)
        ax.set_ylabel('probability density', fontsize='small', color=D_TXT)
    else:
        ax.tick_params(labelbottom=True, labelsize='small')
        ax.set_ylabel('probability density', fontsize='small')

    for column in columns:
        # Remove NaN values for kernel density estimate (KDE) calculation
        data = df[column].dropna()
        n, bins_, patches_ = ax.hist(data, bins=bins, density=True, color=HIST_BAR,
                edgecolor=D_AX if dark else 'white', label=column)
        # Calculate and plot KDE
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        kde_values = kde(x_range)
        # Scale KDE to match histogram
        max_hist_y_value = max(n)
        max_kde_y_value = max(kde_values)
        scale_factor = max_hist_y_value / max_kde_y_value
        ax.plot(x_range, kde_values * scale_factor, color=KDE_LINE,
                label=column + ' KDE')
    ax.grid(False)
    if show_legend:
        legend = ax.legend(loc='best', fontsize='small')
        legend.get_frame().set_alpha(0.7)
        if dark:
            legend.get_frame().set_facecolor(D_AX)
            legend.get_frame().set_edgecolor(D_FIG)
            for text in legend.get_texts():
                text.set_color(color=D_TIT)
    else:
        ax.legend().set_visible(False)
        tit = 'probability density of ' + ' / '.join(columns) + ' (bins: ' + str(bins) + ')'
        if dark:
            ax.set_title(tit, fontsize='small', color=D_TIT)
        else:
            ax.set_title(tit, fontsize='small')
    return fig

def get_frames(
        provider: Provider,
        aggregator:TradeAggregator,
        frame_count: int,
        datetime_cutoff: datetime
        ) -> List[Frame]:
    provider.reset(seek='first')
    aggregator.reset()
    frames: List[Frame] = []
    while True:
        if frame_count < 0:
            break
        try:
            trade = next(provider)
            if trade.datetime > datetime_cutoff:
                raise StopIteration
            frame = aggregator.aggregate([trade])
            if frame is not None:
                frames.append(frame)
                frame_count -= 1
        except StopIteration:
            frame = aggregator.finish()
            frames.append(frame)
            break
    return frames

def returns_from_frames(frames: List[Frame]) -> List[float]:
    returns = [0]
    for i in range(1, len(frames)):
        returns.append(frames[i].close / frames[i-1].close - 1)
    return returns

def plot_correllation_heatmap(df, title=None, cmap=None, coeff=False, coeff_color=None,
                                                        dark=True, feature_decimals=2):
    fig, ax = plt.subplots(dpi=120, layout='constrained')
    if dark:
        fig.set_facecolor(D_FIG)
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    if cmap is None:
        cmap = 'rainbow_r' # rainbow_r Spectral coolwarm_r bwr_r RdYlBu RdYlGn
    cax = ax.imshow(df, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
    cb = fig.colorbar(cax)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    if dark:
        cb.set_label('Spearman correlation', fontsize='small', color=D_TIT)
        cb.ax.tick_params(labelsize='small', colors=D_TXT)
    else:
        cb.set_label('Spearman correlation', fontsize='small')
        cb.ax.tick_params(labelsize='small')
    if title is not None:
        if dark:
            ax.set_title(title, color=D_TIT)
        else:
            ax.set_title(title, color=D_TIT)
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.columns)))
    if dark:
        ax.tick_params(axis='x', colors=D_TXT)
        ax.set_xticklabels(df.columns, rotation=45, fontsize='small', color=D_TIT)
        ax.tick_params(axis='y', colors=D_TXT)
        ax.set_yticklabels(df.columns, fontsize='small', color=D_TIT)
    else:
        ax.set_xticklabels(df.columns, rotation=45, fontsize='small')
        ax.set_yticklabels(df.columns, fontsize='small')
    if coeff:
        if coeff_color is None:
            coeff_color = 'black'
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                ax.text(j, i, f'{df.iloc[i, j]:.{feature_decimals}f}', ha='center', va='center',
                        fontsize='small', color=coeff_color)
    return fig

symbol = 'ETHUSDT'
dir = './data/binance_monthly_klines/'
provider = BinanceMonthlyKlines1mToTradesProvider(data_dir = dir, symbol = symbol,
            date_from = date(2024, 3, 1), date_to = date(2024, 6, 30), spread=0.5)
datetime_cutoff = datetime(2024, 6, 30, tzinfo=timezone.utc)

aggregator_1m = IntervalTradeAggregator(method='time',
                interval=1*60, duration=(1, 8*60*60))
aggregator_6h = IntervalTradeAggregator(method='time',
                interval=6*60*60, duration=(1, 800*60*60))

name_1m = f'{provider.name} {aggregator_1m.name}'
name_6h = f'{provider.name} {aggregator_6h.name}'

frame_count = 484 # how many previous frames to consider

RDM = 'readme/performance/'
LT = 'light/'
DK = 'dark/'
if not os.path.exists(RDM+DK):
    os.makedirs(RDM+DK, exist_ok=True)
if not os.path.exists(RDM+LT):
    os.makedirs(RDM+LT, exist_ok=True)

frames_6h = get_frames(provider, aggregator_6h,
    frame_count, datetime_cutoff)
returns_6h = returns_from_frames(frames_6h)

if DO_PERF_MSECS:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    msecs: List[float] = [0.0] * len(returns_6h)

    for j in range(100):
        ratios.reset()    
        for i in range(len(returns_6h)):
            start_time = time.time()
            ratios.add_return(
                return_=returns_6h[i],
                return_benchmark=0.,
                value=1.,
                time_start=frames_6h[i].time_start,
                time_end=frames_6h[i].time_end)
            msecs[i] += (time.time() - start_time) * 1000

    df['msecs in add_return'] = [element / 100 for element in msecs]
    for dark in [True]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['msecs in add_return']])
        plt.show()
        plt.close(fig)

if DO_RETURNS:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])
    df['6h returns'] = returns_6h

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['daily returns'] = ratios.returns
    ret_d = ratios.returns
    frp_d = ratios.fractional_periods

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['weekly returns'] = ratios.returns
    ret_w = ratios.returns
    frp_w = ratios.fractional_periods

    ratios = Ratios(
        periodicity=Periodicity.DAILY.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['monthly returns'] = ratios.returns
    ret_m = ratios.returns
    frp_m = ratios.fractional_periods

    ratios = Ratios(
        periodicity=Periodicity.QUARTERLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['quarterly returns'] = ratios.returns
    ret_q = ratios.returns
    frp_q = ratios.fractional_periods

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['annual returns'] = ratios.returns
    ret_a = ratios.returns
    frp_a = ratios.fractional_periods

    lines_csv = ['6h return,daily fractional period,daily return,weekly fractional period,weekly return,monthly fractional period,monthly return,annual fractional period,annual return']
    lines_table = [
        '| 6h return | daily fractional period | daily return | weekly fractional period | weekly return | monthly fractional period | monthly return | annual fractional period | annual return |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    table_decimals = 6
    for i in range(len(returns_6h)):
        line = f'{returns_6h[i]},{frp_d[i]},{ret_d[i]},{frp_w[i]},{ret_w[i]},{frp_m[i]},{ret_m[i]},{frp_a[i]},{ret_a[i]}'
        lines_csv.append(line)
        line = f'| {returns_6h[i]:.{table_decimals}f} | {frp_d[i]:.{table_decimals}f} | {ret_d[i]:.{table_decimals}f} | {frp_w[i]:.{table_decimals}f} | {ret_w[i]:.{table_decimals}f} | {frp_m[i]:.{table_decimals}f} | {ret_m[i]:.{table_decimals}f} | {frp_a[i]:.{table_decimals}f} | {ret_a[i]:.{table_decimals}f} |'
        lines_table.append(line)
    filename = RDM+name_6h+' returns'
    with open(filename+'.csv', 'w') as f:
        for line in lines_csv:
            f.write(line + '\n')
    with open(filename+'.txt', 'w') as f:
        for line in lines_table:
            f.write(line + '\n')

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['6h returns'], ['daily returns'], ['weekly returns'], ['monthly returns'], ['annual returns']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' returns.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    # Distribution returns.
    for column in ['6h returns', 'daily returns', 'weekly returns', 'monthly returns', 'annual returns']:
        for dark in [True, False]:
            fig = plot_distribution_histogram(df, [column], dark=dark)
            fig.savefig(RDM+(DK if dark else LT)+name_6h+' distr '+column+'.'+FIG_EXT)
            if SHOW_CHARTS:
                plt.show()
            plt.close(fig)

if DO_CUMULATIVE_RETURNS:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    cumulative_returns = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        cumulative_returns.append(ratios.cumulative_return)
    df['daily cumulative returns'] = cumulative_returns

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    cumulative_returns = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        cumulative_returns.append(ratios.cumulative_return)
    df['weekly cumulative returns'] = cumulative_returns

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    cumulative_returns = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        cumulative_returns.append(ratios.cumulative_return)
    df['monthly cumulative returns'] = cumulative_returns

    ratios = Ratios(
        periodicity=Periodicity.QUARTERLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    cumulative_returns = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        cumulative_returns.append(ratios.cumulative_return)
    df['quarterly cumulative returns'] = cumulative_returns

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    cumulative_returns = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        cumulative_returns.append(ratios.cumulative_return)
    df['annual cumulative returns'] = cumulative_returns

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily cumulative returns'], ['weekly cumulative returns'], ['monthly cumulative returns'], ['annual cumulative returns']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' cumulative returns.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily cumulative returns']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily cumulative returns.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly cumulative returns']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly cumulative returns.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly cumulative returns']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly cumulative returns.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual cumulative returns']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual cumulative returns.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_COMPOUND_GROWTH_RATE:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()

    compaund_grows_rate = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        compaund_grows_rate.append(ratios.compound_growth_rate())
    df['daily compaund grows rate'] = compaund_grows_rate

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    compaund_grows_rate = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        compaund_grows_rate.append(ratios.compound_growth_rate())
    df['weekly compaund grows rate'] = compaund_grows_rate

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    compaund_grows_rate = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        compaund_grows_rate.append(ratios.compound_growth_rate())
    df['monthly compaund grows rate'] = compaund_grows_rate

    ratios = Ratios(
        periodicity=Periodicity.QUARTERLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    compaund_grows_rate = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        compaund_grows_rate.append(ratios.compound_growth_rate())
    df['quarterly compaund grows rate'] = compaund_grows_rate

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    compaund_grows_rate = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        compaund_grows_rate.append(ratios.compound_growth_rate())
    df['annual compaund grows rate'] = compaund_grows_rate

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily compaund grows rate'], ['weekly compaund grows rate'], ['monthly compaund grows rate'], ['annual compaund grows rate']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' compaund grows rate.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily compaund grows rate']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily compaund grows rate.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly compaund grows rate']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly compaund grows rate.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly compaund grows rate']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly compaund grows rate.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual compaund grows rate']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual compaund grows rate.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_DRAWDOWN:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['daily drawdowns cumulative'] = ratios.drawdowns_cumulative
    df['daily drawdown peaks'] = ratios.drawdowns_peaks()
    df['daily drawdown continuous'] = ratios.drawdowns_continuous()

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['weekly drawdowns cumulative'] = ratios.drawdowns_cumulative
    df['weekly drawdown peaks'] = ratios.drawdowns_peaks()
    df['weekly drawdown continuous'] = ratios.drawdowns_continuous()

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['monthly drawdowns cumulative'] = ratios.drawdowns_cumulative
    df['monthly drawdown peaks'] = ratios.drawdowns_peaks()
    df['monthly drawdown continuous'] = ratios.drawdowns_continuous()

    ratios = Ratios(
        periodicity=Periodicity.QUARTERLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['quarterly drawdowns cumulative'] = ratios.drawdowns_cumulative
    df['quarterly drawdown peaks'] = ratios.drawdowns_peaks()
    df['quarterly drawdown continuous'] = ratios.drawdowns_continuous()

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
    df['annual drawdowns cumulative'] = ratios.drawdowns_cumulative
    df['annual drawdown peaks'] = ratios.drawdowns_peaks()
    df['annual drawdown continuous'] = ratios.drawdowns_continuous()

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily drawdowns cumulative'], ['daily drawdown peaks'], ['daily drawdown continuous']],
            figsize=(8, 8))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily drawdown.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly drawdowns cumulative'], ['weekly drawdown peaks'], ['weekly drawdown continuous']],
            figsize=(8, 8))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly drawdown.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly drawdowns cumulative'], ['monthly drawdown peaks'], ['monthly drawdown continuous']],
            figsize=(8, 8))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly drawdown.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual drawdowns cumulative'], ['annual drawdown peaks'], ['annual drawdown continuous']],
            figsize=(8, 8))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual drawdown.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_CALMAR_RATIO:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
    df['daily calmar ratio'] = calmar_ratio

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
    df['weekly calmar ratio'] = calmar_ratio

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
    df['monthly calmar ratio'] = calmar_ratio

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
    df['annual calmar ratio'] = calmar_ratio

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily calmar ratio'], ['weekly calmar ratio'], ['monthly calmar ratio'], ['annual calmar ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' calmar ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily calmar ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily calmar ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly calmar ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly calmar ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly calmar ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly calmar ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual calmar ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual calmar ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_STERLING_RATIO:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    sterling_ratio_0 = []
    sterling_ratio_5 = []
    sterling_ratio_10 = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        sterling_ratio_5.append(ratios.sterling_ratio(annual_excess_rate=5))
        sterling_ratio_10.append(ratios.sterling_ratio(annual_excess_rate=10))
    df['daily sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['daily sterling ratio, annual excess rate 5%'] = sterling_ratio_5
    df['daily sterling ratio, annual excess rate 10%'] = sterling_ratio_10

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    sterling_ratio_0 = []
    sterling_ratio_5 = []
    sterling_ratio_10 = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        sterling_ratio_5.append(ratios.sterling_ratio(annual_excess_rate=5))
        sterling_ratio_10.append(ratios.sterling_ratio(annual_excess_rate=10))
    df['weekly sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['weekly sterling ratio, annual excess rate 5%'] = sterling_ratio_5
    df['weekly sterling ratio, annual excess rate 10%'] = sterling_ratio_10

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    sterling_ratio_0 = []
    sterling_ratio_5 = []
    sterling_ratio_10 = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        sterling_ratio_5.append(ratios.sterling_ratio(annual_excess_rate=5))
        sterling_ratio_10.append(ratios.sterling_ratio(annual_excess_rate=10))
    df['monthly sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['monthly sterling ratio, annual excess rate 5%'] = sterling_ratio_5
    df['monthly sterling ratio, annual excess rate 10%'] = sterling_ratio_10

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    sterling_ratio_0 = []
    sterling_ratio_5 = []
    sterling_ratio_10 = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        sterling_ratio_5.append(ratios.sterling_ratio(annual_excess_rate=5))
        sterling_ratio_10.append(ratios.sterling_ratio(annual_excess_rate=10))
    df['annual sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['annual sterling ratio, annual excess rate 5%'] = sterling_ratio_5
    df['annual sterling ratio, annual excess rate 10%'] = sterling_ratio_10

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily sterling ratio, annual excess rate 5%'], ['weekly sterling ratio, annual excess rate 5%'], ['monthly sterling ratio, annual excess rate 5%'], ['annual sterling ratio, annual excess rate 5%']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' sterling ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily sterling ratio, annual excess rate 0%'], ['daily sterling ratio, annual excess rate 5%'], ['daily sterling ratio, annual excess rate 10%']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily sterling ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly sterling ratio, annual excess rate 0%'], ['weekly sterling ratio, annual excess rate 5%'], ['weekly sterling ratio, annual excess rate 10%']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly sterling ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly sterling ratio, annual excess rate 0%'], ['monthly sterling ratio, annual excess rate 5%'], ['monthly sterling ratio, annual excess rate 10%']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly sterling ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual sterling ratio, annual excess rate 0%'], ['annual sterling ratio, annual excess rate 5%'], ['annual sterling ratio, annual excess rate 10%']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual sterling ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_BURKE_RATIO:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    burke_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        burke_ratio.append(ratios.burke_ratio(modified=False))
    df['daily burke ratio'] = burke_ratio

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    burke_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        burke_ratio.append(ratios.burke_ratio(modified=False))
    df['weekly burke ratio'] = burke_ratio

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    burke_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        burke_ratio.append(ratios.burke_ratio(modified=False))
    df['monthly burke ratio'] = burke_ratio

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    burke_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        burke_ratio.append(ratios.burke_ratio(modified=False))
    df['annual burke ratio'] = burke_ratio

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily burke ratio'], ['weekly burke ratio'], ['monthly burke ratio'], ['annual burke ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' burke ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily burke ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily burke ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly burke ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly burke ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly burke ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly burke ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual burke ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual burke ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_PAIN_INDEX:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_index.append(ratios.pain_index())
    df['daily pain index'] = pain_index

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_index.append(ratios.pain_index())
    df['weekly pain index'] = pain_index

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_index.append(ratios.pain_index())
    df['monthly pain index'] = pain_index

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_index.append(ratios.pain_index())
    df['annual pain index'] = pain_index

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily pain index'], ['weekly pain index'], ['monthly pain index'], ['annual pain index']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' pain index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily pain index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily pain index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly pain index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly pain index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly pain index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly pain index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual pain index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual pain index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_ULCER_INDEX:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_index.append(ratios.ulcer_index())
    df['daily ulcer (martin) index'] = ulcer_index

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_index.append(ratios.ulcer_index())
    df['weekly ulcer (martin) index'] = ulcer_index

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_index.append(ratios.ulcer_index())
    df['monthly ulcer (martin) index'] = ulcer_index

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_index = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_index.append(ratios.ulcer_index())
    df['annual ulcer (martin) index'] = ulcer_index

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily ulcer (martin) index'], ['weekly ulcer (martin) index'], ['monthly ulcer (martin) index'], ['annual ulcer (martin) index']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' ulcer index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily ulcer (martin) index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily ulcer index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly ulcer (martin) index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly ulcer index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly ulcer (martin) index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly ulcer index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual ulcer (martin) index']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual ulcer index.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_PAIN_RATIO:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_ratio.append(ratios.pain_ratio())
    df['daily pain ratio'] = pain_ratio

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_ratio.append(ratios.pain_ratio())
    df['weekly pain ratio'] = pain_ratio

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_ratio.append(ratios.pain_ratio())
    df['monthly pain ratio'] = pain_ratio

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    pain_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        pain_ratio.append(ratios.pain_ratio())
    df['annual pain ratio'] = pain_ratio

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily pain ratio'], ['weekly pain ratio'], ['monthly pain ratio'], ['annual pain ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' pain ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily pain ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily pain ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly pain ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly pain ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly pain ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly pain ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual pain ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual pain ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_ULCER_RATIO:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_ratio.append(ratios.martin_ratio())
    df['daily ulcer (martin) ratio'] = ulcer_ratio

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_ratio.append(ratios.martin_ratio())
    df['weekly ulcer (martin) ratio'] = ulcer_ratio

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_ratio.append(ratios.martin_ratio())
    df['monthly ulcer (martin) ratio'] = ulcer_ratio

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        ulcer_ratio.append(ratios.martin_ratio())
    df['annual ulcer (martin) ratio'] = ulcer_ratio

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily ulcer (martin) ratio'], ['weekly ulcer (martin) ratio'], ['monthly ulcer (martin) ratio'], ['annual ulcer (martin) ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' ulcer ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily ulcer (martin) ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily ulcer ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly ulcer (martin) ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly ulcer ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly ulcer (martin) ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly ulcer ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual ulcer (martin) ratio']])
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual ulcer ratio.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)

if DO_DRAWDOWN_BASED_RATIOS:
    df = pd.DataFrame([f.__dict__ for f in frames_6h])

    ratios = Ratios(
        periodicity=Periodicity.DAILY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    sterling_ratio_0 = []
    burke_ratio = []
    pain_ratio = []
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        burke_ratio.append(ratios.burke_ratio(modified=False))
        pain_ratio.append(ratios.pain_ratio())
        ulcer_ratio.append(ratios.martin_ratio())
    df['daily calmar ratio'] = calmar_ratio
    df['daily sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['daily burke ratio'] = burke_ratio
    df['daily pain ratio'] = pain_ratio
    df['daily ulcer (martin) ratio'] = ulcer_ratio

    ratios = Ratios(
        periodicity=Periodicity.WEEKLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    sterling_ratio_0 = []
    burke_ratio = []
    pain_ratio = []
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        burke_ratio.append(ratios.burke_ratio(modified=False))
        pain_ratio.append(ratios.pain_ratio())
        ulcer_ratio.append(ratios.martin_ratio())
    df['weekly calmar ratio'] = calmar_ratio
    df['weekly sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['weekly burke ratio'] = burke_ratio
    df['weekly pain ratio'] = pain_ratio
    df['weekly ulcer (martin) ratio'] = ulcer_ratio

    ratios = Ratios(
        periodicity=Periodicity.MONTHLY,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    sterling_ratio_0 = []
    burke_ratio = []
    pain_ratio = []
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        burke_ratio.append(ratios.burke_ratio(modified=False))
        pain_ratio.append(ratios.pain_ratio())
        ulcer_ratio.append(ratios.martin_ratio())
    df['monthly calmar ratio'] = calmar_ratio
    df['monthly sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['monthly burke ratio'] = burke_ratio
    df['monthly pain ratio'] = pain_ratio
    df['monthly ulcer (martin) ratio'] = ulcer_ratio

    ratios = Ratios(
        periodicity=Periodicity.ANNUAL,
        annual_risk_free_rate = 0.,
        annual_target_return = 0.,
        day_count_convention = DayCountConvention.RAW)
    ratios.reset()
    calmar_ratio = []
    sterling_ratio_0 = []
    burke_ratio = []
    pain_ratio = []
    ulcer_ratio = []
    for i in range(len(returns_6h)):
        ratios.add_return(
            return_=returns_6h[i],
            return_benchmark=0.,
            value=1.,
            time_start=frames_6h[i].time_start,
            time_end=frames_6h[i].time_end)
        calmar_ratio.append(ratios.calmar_ratio())
        sterling_ratio_0.append(ratios.sterling_ratio(annual_excess_rate=0))
        burke_ratio.append(ratios.burke_ratio(modified=False))
        pain_ratio.append(ratios.pain_ratio())
        ulcer_ratio.append(ratios.martin_ratio())
    df['annual calmar ratio'] = calmar_ratio
    df['annual sterling ratio, annual excess rate 0%'] = sterling_ratio_0
    df['annual burke ratio'] = burke_ratio
    df['annual pain ratio'] = pain_ratio
    df['annual ulcer (martin) ratio'] = ulcer_ratio

    # Price chart.
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['daily calmar ratio'], ['daily sterling ratio, annual excess rate 0%'], ['daily burke ratio'], ['daily pain ratio'], ['daily ulcer (martin) ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' daily drawdown ratios.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['weekly calmar ratio'], ['weekly sterling ratio, annual excess rate 0%'], ['weekly burke ratio'], ['weekly pain ratio'], ['weekly ulcer (martin) ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' weekly drawdown ratios.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['monthly calmar ratio'], ['monthly sterling ratio, annual excess rate 0%'], ['monthly burke ratio'], ['monthly pain ratio'], ['monthly ulcer (martin) ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' monthly drawdown ratios.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
    for dark in [True, False]:
        fig = plot_ratios(df, name_6h, show_legend=False, dark=dark,
            panes=[['annual calmar ratio'], ['annual sterling ratio, annual excess rate 0%'], ['annual burke ratio'], ['annual pain ratio'], ['annual ulcer (martin) ratio']],
            figsize=(8, 10))
        fig.savefig(RDM+(DK if dark else LT)+name_6h+' annual drawdown ratios.'+FIG_EXT)
        if SHOW_CHARTS:
            plt.show()
        plt.close(fig)
