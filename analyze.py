#!/usr/bin/env python
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import numpy as nm

import soops.scoop_outputs as so
import soops.ioutils as io
import soops.plot_selected as sps

import sys
if 'sfepy' not in sys.path:
    sys.path.append(op.expanduser('~/projects/sfepy-git'))

import time_tensors as tt

filename = 'output/matrix-dw_laplace-8192-layouts-reports/results.h5'
#filename = 'output/tmp-plot-scatter/results.h5'
output_dir = 'output/tmp'

def load_results(filename):

    df = io.get_from_store(filename, 'df')
    par_keys = set(io.get_from_store(filename, 'par_keys').to_list())

    data = so.init_plugin_data(df, par_keys, output_dir, filename)
    data = tt.collect_stats(df, data)
    tt.check_rnorms(df, data)
    data = tt.setup_uniques(df, data)

    return df, data

def plot_per_lib1(ax, ldf, data, style_key='layout', mark='cqgvd0',
                  xkey='rtwwmean', order=None, show_legend=False):
    """
    Notes
    -----
    Includes libs missing due to timeout/memory requirements as empty rows.
    """
    if order is None:
        order = data.orders[0]

    else:
        assert(order in data.orders)

    sldf = ldf.sort_values(['lib', xkey])
    style_vals = (sldf[[style_key]]
                  .drop_duplicates()
                  .sort_values(style_key)[style_key])

    select = sps.select_by_keys(ldf, [style_key])
    styles = {style_key : {'color' : 'viridis'}}
    styles = sps.setup_plot_styles(select, styles)

    if ax is None:
        _, ax = plt.subplots()

    xvals = (ldf[[xkey]]
             .drop_duplicates()
             .sort_values([xkey], ignore_index=True)
             [xkey])
    if xvals.dtype == 'object':
        ax.set_xticks(nm.arange(len(xvals)))
        ax.set_xticklabels(xvals)

    libs = (ldf[['lib']]
            .drop_duplicates()
            .sort_values(['lib'], ignore_index=True)
            ['lib'])
    ax.set_yticks(nm.arange(len(libs)))
    ax.set_yticklabels(libs)

    ax.grid(True)
    used = None
    for style_val in style_vals:
        sdf = sldf[(sldf[style_key] == style_val) &
                   (sldf['order'] == order)]
        if not len(sdf): continue

        style_kwargs, indices, used = sps.get_row_style_used(
            sdf.iloc[0], select, {}, styles, used
        )
        if style_val == mark:
            style_kwargs.update({
                'color' : 'r',
                'zorder' : 100,
                'marker' : 'x',
            })
        else:
            style_kwargs.update({
                'alpha' : 0.6,
                'marker' : 'o',
            })
        if xvals.dtype == 'object':
            xs = nm.searchsorted(xvals, sdf[xkey])

        else:
            xs = sdf[xkey]

        ax.plot(xs, nm.searchsorted(libs, sdf['lib']), ls='None',
                **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used)

    return ax

def plot_layouts2(ax, ldf, data, xkey='rtwwmean', minor_ykey='spaths',
                  order=None, show_legend=False):
    """
    Notes
    -----
    Includes libs missing due to timeout/memory requirements as empty rows.
    """
    if order is None:
        order = data.orders[0]

    else:
        assert(order in data.orders)

    sldf = ldf.sort_values(['lib', xkey])

    select = sps.select_by_keys(ldf, ['layout'])
    styles = {'layout' : {'color' : 'viridis'}}
    styles = sps.setup_plot_styles(select, styles)

    if ax is None:
        _, ax = plt.subplots()

    lib_minors = (ldf[['lib', minor_ykey]]
                  .drop_duplicates()
                  .sort_values(['lib', minor_ykey], ignore_index=True))
    yticklabels = lib_minors.apply(lambda x: ': '.join(x), axis=1)
    groups = lib_minors.groupby('lib').groups
    yticks = nm.concatenate([ii + nm.linspace(0, 1, len(group) + 1)[:-1]
                             for ii, group in enumerate(groups.values())])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True)
    used = None
    for layout in data.uniques['layout']:
        sdf = sldf[(sldf['layout'] == layout) &
                   (sldf['order'] == order)]
        if not len(sdf): continue

        style_kwargs, indices, used = sps.get_row_style_used(
            sdf.iloc[0], select, {}, styles, used
        )
        if layout == 'cqgvd0':
            style_kwargs.update({
                'color' : 'r',
                'zorder' : 100,
                'marker' : 'x',
            })
        else:
            style_kwargs.update({
                'alpha' : 0.6,
                'marker' : 'o',
            })
        labels = sdf[['lib', minor_ykey]].apply(lambda x: ': '.join(x), axis=1)
        ax.plot(sdf[xkey], yticks[nm.searchsorted(yticklabels, labels)],
                ls='None', **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used)

    return ax

def main():
    df, data = load_results(filename)
    data = tt.select_data(df, data, omit_functions=['.*dat.*'])

    # data = tt.setup_styles(df, data)

    ldf = data.ldf
    fdf = data.fdf

    ax = plot_layouts1(None, ldf, data, xkey='rtwwmean', order=3)
    ax = plot_layouts2(None, ldf, data, xkey='rtwwmean', minor_ykey='spaths',
                       order=3)
    plt.show()
    from soops import shell; shell()

    ####### ... that spaths and opt are not 1:1...

    mii = pd.MultiIndex.from_arrays([ldf.lib, ldf.opt, ldf.spaths]).unique().sort_values()
    ldf.plot.scatter(x='opt', y='spaths')

    sdf = (ldf[['lib', 'opt', 'spaths']].drop_duplicates()
           .sort_values(['lib', 'opt', 'spaths'], ignore_index=True))

    cat = sdf['lib'].astype('category').cat
    sdf.plot.scatter(x='opt', y='spaths', color=cat.codes)


    sldf = ldf.sort_values('layout')

    cat = sldf['lib'].astype('category').cat

    sldf.plot.scatter(x='layout', y='rtwwmean', c=cat.codes, ls='None', marker='o', colormap='viridis')

    #######

    sldf = ldf.sort_values(['lib', 'rtwwmean'])

    select = sps.select_by_keys(ldf, ['lib'])
    styles = {'lib' : {'color' : 'viridis'}}
    styles = sps.setup_plot_styles(select, styles)
    #colors = sps.get_cat_style(select, 'lib', styles, 'color')

    fig, ax = plt.subplots()
    used = None
    for ii, lib in enumerate(data.uniques['lib']):
        sdf = sldf[sldf['lib'] == lib]
        style_kwargs, indices = sps.get_row_style(
            sdf.iloc[0], select, {}, styles
        )
        used = sps.update_used(used, indices)
        ax.plot(sdf['layout'], sdf['rtwwmean'], ls='None', marker='o', alpha=0.6,
                **style_kwargs)

    sps.add_legend(ax, select, styles, used)

    order = data.orders[0]
    fig, ax = plt.subplots()
    used = None
    for ii, lib in enumerate(data.uniques['lib']):
        sdf = sldf[(sldf['lib'] == lib) &
                   (sldf['order'] == order)]
        if not len(sdf): continue

        style_kwargs, indices = sps.get_row_style(
            sdf.iloc[0], select, {}, styles
        )
        used = sps.update_used(used, indices)
        ax.plot(sdf['layout'], sdf['rtwwmean'], ls='None', marker='o', alpha=0.6,
                **style_kwargs)

    sps.add_legend(ax, select, styles, used)

    #######

    plt.figure()
    colors = ldf['lib'].astype('category').cat.codes
    #markers = tuple(ldf['lib'].astype('category').values)
    ldf.plot.scatter(x='rtwwmean', y='rmmean', c=colors, colorbar=True, colormap='viridis', logx=True)

    plt.figure()
    colors = ldf['spaths'].astype('category').cat.codes
    ldf.plot.scatter(x='rtwwmean', y='rmmean', c=colors, colorbar=True, colormap='viridis', logx=True)

    gb1 = ldf.groupby(['lib'])
    gb2 = ldf.groupby(['spaths'])
    gb3 = ldf.groupby(['variant'])
    gb1.groups

    plt.figure()
    out1 = gb1['rtwwmean'].plot(ls='None', marker='o')
    plt.legend()

    plt.figure()
    out2 = gb2['rtwwmean'].plot(ls='None', marker='o')
    plt.legend()

    plt.figure()
    out3 = gb3['rtwwmean'].plot(ls='None', marker='o')
    plt.legend()

if __name__ == '__main__':
    main()
