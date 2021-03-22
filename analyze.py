#!/usr/bin/env python
"""
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os.path as op
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as nm

import soops as so
from soops.base import output, product, Struct
import soops.scoop_outputs as sc
import soops.ioutils as io
import soops.plot_selected as sps

import sys
if 'sfepy' not in sys.path:
    sys.path.append(op.expanduser('~/projects/sfepy-git'))

import time_tensors as tt

def load_results(filename, output_dir):

    df = io.get_from_store(filename, 'df')
    par_keys = set(io.get_from_store(filename, 'par_keys').to_list())

    data = sc.init_plugin_data(df, par_keys, output_dir, filename)
    data = tt.collect_stats(df, data)
    tt.check_rnorms(df, data)
    data = tt.setup_uniques(df, data)

    return df, data

def plot_per_lib1(ax, ldf, data, style_key='layout', mark='cqgvd0',
                  xkey='rtwwmean', all_ldf=None, show_legend=False):
    """
    Notes
    -----
    Includes libs missing due to timeout/memory requirements as empty rows.
    """
    if all_ldf is None:
        all_ldf = ldf

    sldf = ldf.sort_values(['lib', xkey])
    style_vals = (sldf[[style_key]]
                  .drop_duplicates()
                  .sort_values(style_key)[style_key])

    select = sps.select_by_keys(ldf, [style_key])
    styles = {style_key : {'color' : 'viridis'}}
    styles = sps.setup_plot_styles(select, styles)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel(xkey)
    xvals = (all_ldf[[xkey]]
             .drop_duplicates()
             .sort_values([xkey], ignore_index=True)
             [xkey])
    if xvals.dtype == 'object':
        ax.set_xticks(nm.arange(len(xvals)))
        ax.set_xticklabels(xvals)

    libs = (all_ldf[['lib']]
            .drop_duplicates()
            .sort_values(['lib'], ignore_index=True)
            ['lib'])
    ax.set_yticks(nm.arange(len(libs)))
    ax.set_yticklabels(libs)

    ax.grid(True)
    used = None
    for style_val in style_vals:
        sdf = sldf[(sldf[style_key] == style_val)]
        if not len(sdf): continue

        style_kwargs, indices, used = sps.get_row_style_used(
            sdf.iloc[0], select, {}, styles, used
        )
        if style_val == mark:
            style_kwargs.update({
                'color' : 'k',
                'zorder' : 100,
                'marker' : '+',
                'markersize' : 10,
                'mew' : 1,
            })
        else:
            style_kwargs.update({
                'alpha' : 0.6,
                'marker' : 'o',
                'mfc' : 'None',
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

def plot_per_lib2(ax, ldf, data, style_key='layout', mark='cqgvd0',
                  xkey='rtwwmean', minor_ykey='spaths', all_ldf=None,
                  style=None, format_labels=None, show_legend=False):
    """
    Notes
    -----
    Includes libs missing due to timeout/memory requirements as empty rows.
    """
    if all_ldf is None:
        all_ldf = ldf

    sldf = ldf.sort_values(['lib', xkey])
    style_vals = (sldf[[style_key]]
                  .drop_duplicates()
                  .sort_values(style_key)[style_key])

    select = sps.select_by_keys(ldf, [style_key])
    if style is None:
        style = {
            'color' : 'viridis',
            'alpha' : 0.6,
            'marker' : 'o',
            'mfc' : 'None',
        }
    styles = {style_key : style}
    styles = sps.setup_plot_styles(select, styles)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel(xkey)
    xvals = (all_ldf[[xkey]]
             .drop_duplicates()
             .sort_values([xkey], ignore_index=True)
             [xkey])
    if xvals.dtype == 'object':
        ax.set_xticks(nm.arange(len(xvals)))
        ax.set_xticklabels(xvals)

    if isinstance(minor_ykey, str):
        minor_ykey = [minor_ykey]
    lib_minors = (all_ldf[['lib'] + minor_ykey]
                  .drop_duplicates()
                  .sort_values(['lib'] + minor_ykey, ignore_index=True))
    yticklabels = lib_minors.apply(lambda x: ': '.join(x), axis=1)
    groups = lib_minors.groupby('lib').groups
    yticks = nm.concatenate([ii + nm.linspace(0, 1, len(group) + 1)[:-1]
                             for ii, group in enumerate(groups.values())])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.grid(True)
    used = None
    for style_val in style_vals:
        sdf = sldf[(sldf[style_key] == style_val)]
        if not len(sdf): continue

        style_kwargs, indices, used = sps.get_row_style_used(
            sdf.iloc[0], select, {}, styles, used
        )
        if style_val == mark:
            style_kwargs.update({
                'color' : 'k',
                'zorder' : 100,
                'marker' : '+',
                'markersize' : 10,
                'mew' : 2,
            })

        if xvals.dtype == 'object':
            xs = nm.searchsorted(xvals, sdf[xkey])

        else:
            xs = sdf[xkey]

        labels = sdf[['lib'] + minor_ykey].apply(lambda x: ': '.join(x), axis=1)
        ax.plot(xs, yticks[nm.searchsorted(yticklabels, labels)], ls='None',
                **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used, format_labels=format_labels,
                       loc='lower right', frame_alpha=0.8)

    return ax

def get_layout_group(layout):
    if layout == 'cqgvd0':
        return '0cqgvd0'

    elif layout.startswith('cq'):
        return '1cq*'

    elif layout.startswith('qc'):
        return '4qc*'

    elif layout.endswith('cq'):
        return '3*cq'

    elif layout.endswith('qc'):
        return '6*qc'

    elif ('c' in layout) and (layout.index('c') < layout.index('q')):
        return '2*c*q*'

    elif ('c' in layout) and (layout.index('c') > layout.index('q')):
        return '5*q*c*'

    else:
        return '7default'

def format_labels(key, iv, val):
    return val[1:]

helps = {
    'output_dir'
    : 'output directory',
    'results'
    : 'soops-scoop results',
    'analysis' : '',
    'no_show'
    : 'do not call matplotlib show()',
    'silent'
    : 'do not print messages to screen',
    'shell'
    : 'run ipython shell after all computations',
}

def main():
    opts = Struct(
        omit_functions = "'.*dat.*', '.*npq.*', '.*oeq.*', '.*_[01234]_.*'",
        limits = 'rtwwmean=4',
        xscale = 'log',
        xlim = 'auto=True',
        suffix = '.png',
    )

    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help=helps['output_dir'])
    parser.add_argument('results', help=helps['results'])
    parser.add_argument('--analysis', action='store', dest='analysis',
                        choices=['layouts'],
                        default='layouts',
                        help=helps['analysis'])
    for key, val in opts.items():
        helps[key] = '[default: %(default)s]'
        action = 'store'
        if val is True:
            action = 'store_false'

        elif val is False:
            action = 'store_true'

        if action == 'store':
            parser.add_argument('--' + key.replace('_', '-'),
                                type=type(val),
                                action=action, dest=key,
                                default=val, help=helps[key])
        else:
            parser.add_argument('--' + key.replace('_', '-'),
                                action=action, dest=key,
                                default=val, help=helps[key])
    parser.add_argument('-n', '--no-show',
                        action='store_false', dest='show',
                        default=True, help=helps['no_show'])
    parser.add_argument('--silent',
                        action='store_true', dest='silent',
                        default=False, help=helps['silent'])
    parser.add_argument('--shell',
                        action='store_true', dest='shell',
                        default=False, help=helps['shell'])
    options = parser.parse_args()

    options.omit_functions = so.parse_as_list(options.omit_functions)
    options.limits = so.parse_as_dict(options.limits)
    options.xlim = so.parse_as_dict(options.xlim)

    output_dir = options.output_dir
    indir = partial(op.join, output_dir)

    output.prefix = 'analyze:'
    filename = indir('output_log.txt')
    sc.ensure_path(filename)
    output.set_output(filename=filename, combined=options.silent == False)

    df, data = load_results(options.results, output_dir)
    data._ldf['lgroup'] = data._ldf['layout'].apply(get_layout_group)

    data = tt.select_data(df, data, omit_functions=options.omit_functions)

    # data = tt.setup_styles(df, data)

    ldf = data.ldf
    fdf = data.fdf

    output('ldf shape:', ldf.shape)
    output('fdf shape:', fdf.shape)

    if options.analysis == 'layouts':
        style = {
            'color' : 'viridis',
            # 'color' : ['k', 'b', 'g', 'c', 'r', 'm', 'y', 'tab:orange'],
            'mew' : 2,
            'marker' : ['+', 'o', 'v', '^', '<', '>', 's', 'd'],
            'alpha' : 0.8,
            'mfc' : 'None',
            'markersize' : 8,
        }
        xkeys = ['rtwwmean', 'rmmean']
        limit = options.limits.get('rtwwmean', ldf['rtwwmean'].max())
        for n_cell, order, xkey, upxkey in product(
                data.n_cell, data.orders, xkeys, upxkeys,
                contracts=[(2, 3)],
        ):
            sdf = ldf[(ldf['n_cell'] == n_cell) &
                      (ldf['order'] == order) &
                      (ldf['rtwwmean'] <= limit)]
            ax = plot_per_lib2(
                None, sdf, data, xkey=xkey,
                style_key='lgroup', mark='0cqgvd0',
                minor_ykey='spaths', all_ldf=ldf, style=style,
                format_labels=format_labels, show_legend=True,
            )
            xlim = options.xlim.get(xkey, {'auto' : True})
            ax.set_xlim(**xlim)
            ax.set_xscale(options.xscale)
            ax.axvline(1, color='r')
            fig = ax.figure
            plt.tight_layout()
            figname = ('{}-layout-n{}-o{}-{}{}'
                       .format(sdf['term_name'].iloc[0],
                               n_cell, order, xkey,
                               options.suffix))
            fig.savefig(indir(figname), bbox_inches='tight')

    else:
        # ldf.lgroup.hist()

        ax = plot_per_lib1(None, ldf[ldf['order'] == 3], data, xkey='layout',
                           style_key='rtwwmean', mark=None)
        ax = plot_per_lib1(None, ldf[ldf['order'] == 3], data, style_key='layout',
                           xkey='rtwwmean')
        ax = plot_per_lib2(None, ldf[ldf['order'] == 3], data, xkey='layout',
                           style_key='rtwwmean', mark=None,  minor_ykey='spaths')
        ax = plot_per_lib2(None, ldf[ldf['order'] == 3], data, xkey='rtwwmean',
                           minor_ykey='spaths')
        plt.show()
        ax = plot_per_lib2(None, ldf[(ldf['order'] == 3) & (ldf['lib'] == 'oe')] ,
                           data, xkey='rtwwmean', style_key='layout',
                           minor_ykey=['variant', 'opt', 'spaths'])
        ax = plot_per_lib2(None,
                           ldf[(ldf['order'] == 3)
                               & (ldf['lib'].isin(['oel', 'sfepy', 'oe']))],
                           data, xkey='rtwwmean', style_key='layout',
                           minor_ykey=['variant', 'opt', 'spaths'])
        ax = analyze.plot_per_lib2(None,
                                   ldf[(ldf['order'] == 2) &
                                       (ldf['lib'].isin(['np', 'oe'])) &
                                       (ldf['variant'].isin(['default', 'o']))],
                                   data,
                                   xkey='rtwwmean', style_key='layout',
                                   minor_ykey=['opt', 'spaths', 'variant'])

        ####### ... that spaths and opt are not 1:1...

        mii = pd.MultiIndex.from_arrays([ldf.lib, ldf.opt, ldf.spaths]).unique().sort_values()
        ldf.plot.scatter(x='opt', y='spaths')

        sdf = (ldf[['lib', 'opt', 'spaths']].drop_duplicates()
               .sort_values(['lib', 'opt', 'spaths'], ignore_index=True))

        cat = sdf['lib'].astype('category').cat
        sdf.plot.scatter(x='opt', y='spaths', color=cat.codes)

        # but spaths and (lib, opt) are 1:1
        sdf['lib-opt'] = sdf['lib'] + '-' + sdf['opt']
        sdf.plot.scatter(x='lib-opt', y='spaths', color=cat.codes)
        ax = sdf.plot.scatter(x='lib-opt', y='spaths', color=cat.codes)
        ax.grid()

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

        nm.mean([len(ii) for ii in ldf.groupby(['opt', 'order','lib'])['spaths'].unique()])

    if options.show:
        plt.show()

    if options.shell:
        from soops.base import shell; shell()

if __name__ == '__main__':
    main()
