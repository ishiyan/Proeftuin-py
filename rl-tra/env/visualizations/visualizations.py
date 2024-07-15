from typing import List
import os
import io
import imageio

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

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

def plot_charts(df,
        title: str = None,
        candlestick_fields: List[str] = [],
        title_second: str = None,
        candlestick_fields_second: List[str] = [],
        pane_line_fields: List[List[str]]=[],
        dark=True,
        show_legend=False,
        dpi=120,
        figsize=(8, 4)
        ) -> Figure:
    fig = plt.figure(dpi=dpi, layout='constrained', figsize=figsize)
    if dark:
        fig.set_facecolor(D_FIG)
    # Each candlestick chart is 3x of each pane.
    cnt_cs1 = 3 if len(candlestick_fields) > 3 else 0
    cnt_cs2 = 3 if len(candlestick_fields_second) > 3 else 0
    cnt_pan = len(pane_line_fields)
    gs = fig.add_gridspec(cnt_cs1 + cnt_cs2 + cnt_pan, 1)
    ax_share = None
    ax_cs = []
    off = 0
    for i in [cnt_cs1, cnt_cs2]:
        if i > 0:
            if ax_share is None:
                ax = fig.add_subplot(gs[off:off+3, 0])
                ax_share = ax
            else:
                ax = fig.add_subplot(gs[off:off+3, 0], sharex = ax_share)
            ax_cs.append(ax)
            off += 3

    ax_pn = []
    for i in range(cnt_pan):
        if ax_share is None:
            ax = fig.add_subplot(gs[off+i, 0])
            ax_share = ax
        else:
            ax = fig.add_subplot(gs[off+i, 0], sharex = ax_share)
        ax_pn.append(ax)

    for a in ax_cs + ax_pn:
        if dark:
            a.set_facecolor(D_AX)
            a.grid(color=D_TXT)
            a.tick_params(labelbottom=False, labelsize='small', colors=D_TXT)
        else:
            a.tick_params(labelbottom=False, labelsize='small')
        a.grid(False)
        a.tick_params(labelbottom=False)
        
    if len(ax_pn) > 0:
        ax_pn[-1].tick_params(labelbottom=True)
    else:
        ax_cs[-1].tick_params(labelbottom=True)

    if cnt_cs1 > 0 and (title is not None) and len(title) > 0:
        if dark:
            ax_cs[0].set_title(title, color=D_TIT, fontsize='small')
        else:
            ax_cs[0].set_title(title, fontsize='small')
    if cnt_cs2 > 0 and (title_second is not None) and len(title_second) > 0:
        i = 1 if cnt_cs1 > 0 else 0
        if dark:
            ax_cs[i].set_title(title_second, color=D_TIT, fontsize='small')
        else:
            ax_cs[i].set_title(title_second, fontsize='small')

    wick_width=.2
    body_width=.8
    up_color=D_UP
    dn_color=D_DN

    off = 0
    for a in ax_cs:
        op = candlestick_fields[0] if off == 0 else candlestick_fields_second[0]
        hi = candlestick_fields[1] if off == 0 else candlestick_fields_second[1]
        lo = candlestick_fields[2] if off == 0 else candlestick_fields_second[2]
        cl = candlestick_fields[3] if off == 0 else candlestick_fields_second[3]
        off += 1
        up = df[df[cl] >= df[op]]
        dn = df[df[cl] < df[op]]
        # Plot up candlesticks
        a.bar(up.index, up[cl] - up[op], body_width, bottom=up[op], color=up_color)
        a.bar(up.index, up[hi] - up[cl], wick_width, bottom=up[cl], color=up_color)
        a.bar(up.index, up[lo] - up[op], wick_width, bottom=up[op], color=up_color)
        # Plot down candlesticks
        a.bar(dn.index, dn[op] - dn[cl], body_width, bottom=dn[cl], color=dn_color)
        a.bar(dn.index, dn[hi] - dn[op], wick_width, bottom=dn[op], color=dn_color)
        a.bar(dn.index, dn[lo] - dn[cl], wick_width, bottom=dn[cl], color=dn_color)

    # Plot panes
    for i, pane in enumerate(pane_line_fields):
        for col in pane:
            ax_pn[i].plot(df.index, df[col], label=col)#, color='tab:blue')
        if show_legend:
            legend = ax_pn[i].legend(loc='best', fontsize='small')
            legend.get_frame().set_alpha(0.1)
            if dark:
                legend.get_frame().set_facecolor(D_FIG)
                for text in legend.get_texts():
                    text.set_color(color=D_TIT)
        else:
            ax_pn[i].legend().set_visible(False)
            tit = ' / '.join(pane)
            if dark:
                ax_pn[i].set_title(tit, fontsize='small', color=D_TIT)
            else:
                ax_pn[i].set_title(tit, fontsize='small')
    return fig

def plot_correlation_heatmap(df,
        title=None,
        cmap=None,
        coeff=True,
        coeff_color=None,
        dark=True,
        decimals=2,
        dpi=120,
        figsize=(8, 8)
        ) -> Figure:
    fig, ax = plt.subplots(dpi=dpi, layout='constrained', figsize=figsize)
    if dark:
        fig.set_facecolor(D_FIG)
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    if cmap is None:
        cmap = 'rainbow_r' # rainbow_r Spectral coolwarm_r bwr_r RdYlBu RdYlGn
    df = df.corr()
    cax = ax.imshow(df, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1, aspect='auto')
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
            ax.set_title(title)
    rlen = range(len(df.columns))
    ax.set_xticks(rlen)
    ax.set_yticks(rlen)
    if dark:
        ax.tick_params(axis='x', colors=D_TXT)
        ax.set_xticklabels(df.columns, rotation=45, fontsize='small', color=D_TIT)
        ax.tick_params(axis='y', colors=D_TXT)
        ax.set_yticklabels(df.columns, fontsize='small', va='center', color=D_TIT)
    else:
        ax.set_xticklabels(df.columns, rotation=45, fontsize='small')
        ax.set_yticklabels(df.columns, va='center', fontsize='small')
    if coeff:
        if coeff_color is None:
            coeff_color = 'black'
        for i in rlen:
            for j in rlen:
                ax.text(j, i, f'{df.iloc[i, j]:.{decimals}f}', ha='center', va='center',
                        fontsize='small', color=coeff_color)
    return fig

def plot_distribution_histogram(df,
        columns,
        bins='auto',
        dark=True,
        show_legend=True,
        figsize=(4.8, 3.6)
        ) -> Figure:
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
        n, bins_, patches_ = ax.hist(data, bins=bins, density=True,
            color=HIST_BAR, edgecolor=D_AX if dark else 'white', label=column)
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

def df_from_frames_and_observations(frames, observations):
    df = pd.DataFrame([f.__dict__ for f in frames])
    for key in observations[0].keys():
        df[key] = [o[key] for o in observations]
    #df.set_index('time_start', inplace=True)
    return df

def fig_to_rgb_array(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    io_buf.seek(0)
    rgb_array = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return rgb_array

def write_animated_gif(dir, filename, rgb_arrays, durations_in_ms=[], constant_duration_in_ms=1000/3):
    if len(rgb_arrays) > 0:
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(dir):
            raise ValueError(f'Cannot create directory {dir}')
        if len(durations_in_ms) == 0:
            imageio.mimwrite(uri=dir+filename, ims=rgb_arrays, duration=constant_duration_in_ms)
        else:
            with imageio.get_writer(dir+filename, mode='I') as writer:
                for img, duration in zip(rgb_arrays, durations_in_ms):
                    writer.append_data(img, {'duration': duration})
