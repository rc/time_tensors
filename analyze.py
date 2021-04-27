#!/usr/bin/env python
"""
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os.path as op
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as nm

import soops as so
from soops.base import output, product, Struct
import soops.scoop_outputs as sc
import soops.ioutils as io
import soops.formatting as sof
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

    Assumes a single term.
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
    yticks = nm.arange(len(lib_minors))
    groups = lib_minors.groupby('lib').groups
    ysplits, ylibs = zip(*[(group[-1] + 0.5, group[len(group)//2])
                           for group in groups.values()])
    def _get_yticklabels(x):
        label = ': '.join(x[minor_ykey])
        return (x['lib'] + ': ' + label) if x.name in ylibs else label
    yticklabels = lib_minors.apply(_get_yticklabels, axis=1)
    ysearch_labels = lib_minors.apply(lambda x: ': '.join(x), axis=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    for vy in ysplits:
        ax.axhline(vy, color='k', ls=':')

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
        ax.plot(xs, yticks[nm.searchsorted(ysearch_labels, labels)], ls='None',
                **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used, format_labels=format_labels,
                       loc='lower right', frame_alpha=0.8)

    return ax

def plot_per_n_cell(ax, ldf, ykeys=('n_cell', 'order'),
                    marker_key='lib', color_key='spaths',
                    xkey='rtwwmean', all_ldf=None, marker_style=None,
                    format_labels=None, show_legend=False):
    if all_ldf is None:
        all_ldf = ldf

    ykeys = list(ykeys)
    sldf = ldf.sort_values(ykeys)

    style_keys = [marker_key, color_key]
    style_vals = (sldf[style_keys]
                  .drop_duplicates()
                  .sort_values(style_keys).values)

    if marker_style is None:
        marker_style = {
            'lw' : 0.2,
            'mew' : 1.0,
            'marker' : ['o', '^', 'v', '<', 'x', '>', 's', '+', '.'],
            'alpha' : 1.0,
            'mfc' : 'None',
            'markersize' : 8,
        }

    select = sps.select_by_keys(ldf, style_keys)
    styles = {marker_key : marker_style,
              color_key : {'color' : 'nipy_spectral:max=0.95',}}
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

    ydf = (all_ldf[ykeys]
           .drop_duplicates()
           .sort_values(ykeys, ignore_index=True))
    yticks = nm.arange(len(ydf))
    groups = ydf.groupby(ykeys[0]).groups
    ysplits, ymajors = zip(*[(group[-1] + 0.5, group[len(group)//2])
                             for group in groups.values()])
    def _get_yticklabels(x):
        label = str(x[ykeys[1]])
        return ((str(x[ykeys[0]]) + ': ' + label) if x.name in ymajors
                else label)
    yticklabels = ydf.apply(_get_yticklabels, axis=1)
    ylabel_fun = lambda x: tuple(x)
    ysearch_labels = ydf.apply(ylabel_fun, axis=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    for vy in ysplits:
        ax.axhline(vy, color='k', ls=':')

    ax.grid(True)
    used = None
    for style_val in style_vals:
        sdf = sldf[(sldf[style_keys].values == style_val).all(axis=1)]
        if not len(sdf): continue

        style_kwargs, indices, used = sps.get_row_style_used(
            sdf.iloc[0], select, {}, styles, used
        )

        if xvals.dtype == 'object':
            xs = nm.searchsorted(xvals, sdf[xkey])

        else:
            xs = sdf[xkey]

        labels = sdf[ykeys].apply(ylabel_fun, axis=1)
        ax.plot(xs, yticks[nm.searchsorted(ysearch_labels, labels)],
                **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used, per_parameter=True,
                       format_labels=format_labels,
                       loc=['lower left', 'lower right'],
                       frame_alpha=0.8, ncol=1,
                       handlelength=1, handletextpad=0.4, columnspacing=0.2,
                       labelspacing=0.4)

    return ax

def plot_per_n_cell_t(ax, ldf, ykeys=('order', 'n_cell'),
                      marker_key='lib', color_key='spaths',
                      xkey='twwmean_rate', all_ldf=None, marker_style=None,
                      format_labels=None, show_legend=False):
    """
    Transposed plot_per_n_cell() for plotting cell rates, but same
    argument/variable names!
    """
    if all_ldf is None:
        all_ldf = ldf

    ykeys = list(ykeys)
    sldf = ldf.sort_values(ykeys)

    style_keys = [marker_key, color_key]
    style_vals = (sldf[style_keys]
                  .drop_duplicates()
                  .sort_values(style_keys).values)

    if marker_style is None:
        marker_style = {
            'lw' : 0.2,
            'mew' : 1.0,
            'marker' : ['o', '^', 'v', '<', 'x', '>', 's', '+', '.'],
            'alpha' : 1.0,
            'mfc' : 'None',
            'markersize' : 8,
        }

    select = sps.select_by_keys(ldf, style_keys)
    styles = {marker_key : marker_style,
              color_key : {'color' : 'nipy_spectral:max=0.95',}}
    styles = sps.setup_plot_styles(select, styles)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_ylabel(sof.escape_latex(xkey))

    ydf = (all_ldf[ykeys]
           .drop_duplicates()
           .sort_values(ykeys, ignore_index=True))
    yticks = nm.arange(len(ydf))
    groups = ydf.groupby(ykeys[0]).groups
    ysplits, ymajors = zip(*[(group[-1] + 0.5, group[len(group)//2])
                             for group in groups.values()])
    def _get_yticklabels(x):
        label = str(x[ykeys[1]])
        return ((str(x[ykeys[0]]) + ': ' + label) if x.name in ymajors
                else label)
    yticklabels = ydf.apply(_get_yticklabels, axis=1)
    ylabel_fun = lambda x: tuple(x)
    ysearch_labels = ydf.apply(ylabel_fun, axis=1)
    ax.set_xticks(yticks)
    ax.set_xticklabels(yticklabels, rotation=90)
    for vy in ysplits:
        ax.axvline(vy, color='k', ls=':')

    ax.grid(True)
    used = None
    for style_val in style_vals:
        sdf = sldf[(sldf[style_keys].values == style_val).all(axis=1)]
        if not len(sdf): continue

        style_kwargs, indices, used = sps.get_row_style_used(
            sdf.iloc[0], select, {}, styles, used
        )

        xs = sdf[xkey]
        labels = sdf[ykeys].apply(ylabel_fun, axis=1)
        ax.plot(yticks[nm.searchsorted(ysearch_labels, labels)], xs,
                **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used, per_parameter=True,
                       format_labels=format_labels,
                       loc=['lower left', 'upper right'],
                       frame_alpha=0.8, ncol=1,
                       handlelength=1, handletextpad=0.4, columnspacing=0.2,
                       labelspacing=0.4)

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

def format_labels2(key, iv, val):
    return val

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def get_spaths_per_opt(ldf, data):
    groups = ldf.groupby(['term_name', 'n_cell', 'order'])

    opts = ldf['opt'].unique()
    paths = {ii : {} for ii in opts}
    for ir, selection in enumerate(
            product(data.term_names, data.n_cell, data.orders)
    ):
        if not selection in groups.indices: continue

        sdf = ldf.iloc[groups.indices[selection]]
        spaths = sdf['spaths']
        sopts = sdf['opt'].unique()
        if (not len(sdf)) or (not sdf.tmean.notna().any()):
            output('-> no data, skipped!')
            continue

        for opt in sopts:
            path = spaths[sdf['opt'] == opt].iloc[0]
            paths[opt][selection] = path

    pdf = pd.DataFrame(paths)
    return pdf

def sort_spaths(spaths):
    ml = max(map(len, spaths))
    aux = [('00' * (ml - len(ii))) + ii for ii in spaths]
    ii = nm.argsort(aux)
    sspaths = spaths[ii]

    return sspaths

def sparsify_n_cell(nc):
    nnc = []
    last = None
    for ic in nc:
        if ic != last:
            nnc.append('{:,}'.format(ic))
            last = ic

        else:
            nnc.append('')

    return nnc

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
        fun_names="'sfepy'",
        limits = 'rtwwmean=4',
        plot_rc_params = "'text.usetex'=False",
        shorten_spaths = False,
        rate_mode = 'cell-counts',
        xscale = 'log',
        xlim = 'auto=True',
        suffix = '.png',
    )

    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help=helps['output_dir'])
    parser.add_argument('results', help=helps['results'])
    parser.add_argument('--analysis', action='store', dest='analysis',
                        choices=['layouts', 'all-terms', 'all-terms-rate',
                                 'n-dofs', 'fastest-times',
                                 'mem-usages-in-limit', 'times-mems-single-fun'],
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
    options.fun_names = so.parse_as_list(options.fun_names)
    options.limits = so.parse_as_dict(options.limits)
    options.plot_rc_params = so.parse_as_dict(options.plot_rc_params)
    options.xlim = so.parse_as_dict(options.xlim)

    output_dir = options.output_dir
    indir = partial(op.join, output_dir)

    output.prefix = 'analyze:'
    filename = indir('output_log.txt')
    sc.ensure_path(filename)
    output.set_output(filename=filename, combined=options.silent == False)

    df, data = load_results(options.results, output_dir)
    data._ldf['lgroup'] = data._ldf['layout'].apply(get_layout_group)
    if options.shorten_spaths:
        term_names = data._ldf['term_name'].unique()
        for term_name in term_names:
            isel = data._ldf['term_name'] == term_name
            subs = {val : ('{:02d}'.format(ii) if val != '-' else val)
                    for ii, val
                    in enumerate(sort_spaths(data._ldf.loc[isel, 'spaths']
                                             .unique()))}
            aux = data._ldf.loc[isel, 'spaths'].replace(subs)
            data._ldf.loc[isel, 'short_spaths'] = aux

            name = '{}-short-spaths-table.inc'.format(term_name)
            with open(indir(name), 'w') as fd:
                fd.write(pd.Series(subs)
                         .reset_index()
                         .to_latex(header=['contraction paths', 'abbreviations'],
                                   index=False))

    data = tt.select_data(df, data, omit_functions=options.omit_functions)

    # data = tt.setup_styles(df, data)

    ldf = data.ldf
    fdf = data.fdf

    output('ldf shape:', ldf.shape)
    output('fdf shape:', fdf.shape)

    plt.rcParams.update(options.plot_rc_params)

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
        upxkeys = ['twwmean [s]', 'mmean [MB]']
        limit = options.limits.get('rtwwmean', ldf['rtwwmean'].max())
        minor_ykey = 'spaths' if not options.shorten_spaths else 'short_spaths'
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
                minor_ykey=minor_ykey, all_ldf=ldf, style=style,
                format_labels=format_labels, show_legend=True,
            )
            xlim = options.xlim.get(xkey, {'auto' : True})
            ax.set_xlim(**xlim)
            ax.set_xscale(options.xscale)
            ax.xaxis.set_major_locator(mt.LogLocator(subs=(0.5, 1),
                                                     numticks=2))
            ax.xaxis.set_major_formatter(mt.StrMethodFormatter('{x:.2f}'))
            ax.xaxis.set_minor_locator(mt.LogLocator(subs=(0.5, 1),
                                                     numticks=2))
            ax.xaxis.set_minor_formatter(mt.StrMethodFormatter('{x:.2f}'))
            xlim = nm.array(ax.get_xlim())
            ax.axvline(1, color='r')

            pax = ax.twiny()
            pax.xaxis.set_ticks_position('bottom')
            pax.xaxis.set_label_position('bottom')
            pax.spines['bottom'].set_position(('axes', -0.15))
            make_patch_spines_invisible(pax)
            pax.spines['bottom'].set_visible(True)

            pxkey = upxkey.split()[0]
            pax.set_xlabel(upxkey)
            coef = sdf[pxkey].iloc[0] / sdf[xkey].iloc[0]
            pax.set_xlim(*(coef * xlim))
            pax.set_xscale(options.xscale)
            pax.xaxis.set_major_locator(mt.LogLocator(subs=(0.5, 1),
                                                      numticks=2))
            pax.xaxis.set_major_formatter(mt.StrMethodFormatter('{x:.2f}'))
            pax.xaxis.set_minor_locator(mt.LogLocator(subs=(0.5, 1),
                                                      numticks=2))
            pax.xaxis.set_minor_formatter(mt.StrMethodFormatter('{x:.2f}'))

            plt.tight_layout()
            figname = ('{}-layout-n{}-o{}-{}{}'
                       .format(sdf['term_name'].iloc[0],
                               n_cell, order, xkey,
                               options.suffix))
            fig = ax.figure
            fig.savefig(indir(figname), bbox_inches='tight')

    elif options.analysis == 'all-terms':
        term_names = ['dw_laplace::', 'dw_volume_dot:v:', 'dw_volume_dot:vm:',
                      'dw_convect::', 'dw_lin_elastic::', 'dw_laplace::u',
                      'dw_volume_dot:v:u', 'dw_volume_dot:vm:u', 'dw_convect::u',
                      'dw_lin_elastic::u']
        ldf = ldf[ldf['term_name'].isin(term_names)]

        gbt = ldf.groupby('term_name')
        thist = {key : len(val) for key, val in gbt.groups.items()}
        output(Struct(thist))

        pdf = get_spaths_per_opt(ldf, data)
        if options.shorten_spaths:
            pdf = pdf.replace(subs)
        output(pdf)

        xkeys = ['rtwwmean', 'rmmean']
        limit = options.limits.get('rtwwmean', ldf['rtwwmean'].max())
        for term_name, xkey in product(term_names, xkeys):
            sdf = ldf[(ldf['term_name'] == term_name) &
                      (ldf['rtwwmean'] <= limit)]

            if term_name == 'dw_convect::u':
                color_key = ('spaths' if not options.shorten_spaths else
                             'short_spaths')

            else:
                color_key = 'spaths'

            ax = plot_per_n_cell(
                None, sdf, ykeys=('order', 'n_cell'),
                marker_key='lib', color_key=color_key,
                xkey=xkey, all_ldf=ldf,
                format_labels=format_labels2, show_legend=True
            )
            xlim = options.xlim.get(xkey, {'auto' : True})
            ax.set_xlim(**xlim)
            ax.set_xscale(options.xscale)
            if xkey == 'rtwwmean':
                ax.xaxis.set_major_locator(mt.FixedLocator(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1, 1.5, 2, 3, 4, 5]
                ))

            else:
                ax.xaxis.set_major_locator(mt.LogLocator(subs=(0.5, 1),
                                                         numticks=5))
            ax.xaxis.set_major_formatter(mt.StrMethodFormatter('{x:.2f}'))
            ax.xaxis.set_minor_locator(mt.LogLocator(subs=(0.5, 1),
                                                     numticks=3))
            ax.xaxis.set_minor_formatter(mt.StrMethodFormatter('{x:.2f}'))
            ax.axvline(1, color='r')

            plt.tight_layout()
            figname = ('{}-n_cell-order-{}{}'
                       .format(term_name,
                               xkey,
                               options.suffix))
            fig = ax.figure
            fig.savefig(indir(figname), bbox_inches='tight')

    elif options.analysis == 'all-terms-rate':
        term_names = ['dw_laplace::', 'dw_volume_dot:v:', 'dw_volume_dot:vm:',
                      'dw_convect::', 'dw_lin_elastic::', 'dw_laplace::u',
                      'dw_volume_dot:v:u', 'dw_volume_dot:vm:u', 'dw_convect::u',
                      'dw_lin_elastic::u']
        # times: n_cell / s, memory: n_cell / MB
        # keys = ['tmean', 'twwmean', 'mmean', 'mwwmean']
        keys = ['twwmean', 'mmean']

        if options.rate_mode == 'cell-counts':
            for key in keys:
                ldf[key + '_rate'] = ldf['n_cell'] / ldf[key]

            def get_extreme(x):
                k0 = x.keys()[0]
                ii = nm.argmax(x[k0])
                return x.iloc[ii]

        elif options.rate_mode == 'cell-times':
            for key in keys:
                ldf[key + '_rate'] = ldf[key] / ldf['n_cell']

            def get_extreme(x):
                k0 = x.keys()[0]
                ii = nm.argmin(x[k0])
                return x.iloc[ii]

        elif options.rate_mode == 'result-sizes':
            size = nm.where(ldf['diff'].isna(),
                            ldf['c_vec_size_mb'],
                            ldf['c_mtx_size_mb'])

            for key in keys:
                ldf[key + '_rate'] = size / ldf[key]

            def get_extreme(x):
                k0 = x.keys()[0]
                ii = nm.argmax(x[k0])
                return x.iloc[ii]

        ldf = ldf[ldf['term_name'].isin(term_names)]


        gbt = ldf.groupby(['term_name', 'n_cell', 'order', 'lib'])

        for key in [key + '_rate' for key in keys]:
            aux = [key, 'spaths']
            if options.shorten_spaths: aux += ['short_spaths']
            rdf = gbt[aux].apply(get_extreme).reset_index()

            for term_name in term_names:
                sdf = rdf[(rdf['term_name'] == term_name)]

                if term_name == 'dw_convect::u':
                    color_key = ('spaths' if not options.shorten_spaths else
                                 'short_spaths')

                else:
                    color_key = 'spaths'

                ax = plot_per_n_cell_t(
                    None, sdf, ykeys=('order', 'n_cell'),
                    marker_key='lib', color_key=color_key,
                    xkey=key, all_ldf=rdf,
                    format_labels=format_labels2, show_legend=True
                )
                ax.set_yscale(options.xscale)

                plt.tight_layout()
                figname = ('{}-{}-rate-n_cell-order-{}{}'
                           .format(term_name,
                                   options.rate_mode,
                                   key,
                                   options.suffix))
                fig = ax.figure
                fig.savefig(indir(figname), bbox_inches='tight')

    elif options.analysis == 'n-dofs':
        sdf = (ldf[['term_name', 'n_cell', 'order', 'n_qp', 'n_dof', 'repeat',
                    'c_vec_size_mb', 'c_mtx_size_mb']]
               .drop_duplicates(ignore_index=True)
               .sort_values(['term_name', 'n_cell', 'order']))
        cdf = (sdf[sdf['term_name'] == 'dw_laplace::']
               [['n_cell', 'order', 'n_qp', 'n_dof', 'repeat',
                 'c_vec_size_mb', 'c_mtx_size_mb']]
               .reset_index(drop=True))
        aux = (sdf[sdf['term_name'] == 'dw_lin_elastic::']
               [['c_vec_size_mb', 'c_mtx_size_mb']]
               .reset_index(drop=True))
        for key in ['c_vec_size_mb', 'c_mtx_size_mb']:
            cdf[key + '_v'] = aux[key]

        # cdf['n_dof_v'] = aux
        cdf = cdf.astype({'n_cell' : int, 'order' : int, 'n_qp' : int,
                          'n_dof' : int, 'repeat' : int,
                          'c_vec_size_mb' : float, 'c_mtx_size_mb' : float,
                          'c_vec_size_mb_v' : float, 'c_mtx_size_mb_v' : float})
        cdf['n_cell'] = sparsify_n_cell(cdf['n_cell'].to_list())
        filename = indir('table-cdc.inc')
        # header = ['#cells', 'order', '#QP', 'scalar #DOFs', 'vector #DOFs']
        header = ['\#cells', 'order', '\#QP', '\#DOFs/comp.', 'repeat',
                  '$r_s$ [MB]', '$M_s$ [MB]', '$r_v$ [MB]', '$M_v$ [MB]']
        cdf.to_latex(filename, index=False, escape=False,
                     formatters=(['{}'.format] + (['{:,}'.format] * 4)
                                 + ['{:,.1f}'.format] * 4),
                     header=header, column_format='rrrrrrrrr')

    elif options.analysis == 'fastest-times':
        def get_extreme(x):
            k0 = x.keys()[0]
            ii = nm.argmin(x[k0])
            return x.iloc[ii]

        tn2key = {
            'dw_laplace::' : 'Laplacian' ,
            'dw_volume_dot:v:' : 'dot' ,
            'dw_volume_dot:vm:' : 'weighted dot',
            'dw_convect::' : 'NS convective',
            'dw_lin_elastic::' : 'elasticity',
            'dw_laplace::u' : 'Laplacian' ,
            'dw_volume_dot:v:u' : 'dot' ,
            'dw_volume_dot:vm:u' : 'weighted dot',
            'dw_convect::u' : 'NS convective',
            'dw_lin_elastic::u' : 'elasticity',
        }
        term_names = list(tn2key.keys())
        for ii, fname in enumerate(('table-fts-r.inc', 'table-fts-m.inc')):
            fts = {}
            for tn in term_names[5*ii:5*ii+5]:
                tdf = ldf[ldf['term_name'] == tn]
                rdf = tdf[tdf['lib'] == 'sfepy']
                rgb = rdf.groupby(['n_cell', 'order'])
                rmin = rgb['twwmean'].min()

                sdf = tdf[tdf['lib'] != 'sfepy']
                sgb = sdf.groupby(['n_cell', 'order'])
                smin = sgb[['twwmean', 'lib']].apply(get_extreme)

                aux = pd.concat((rmin, smin), axis=1)
                key = tn2key[tn]
                fts[key] = aux.apply(
                    lambda x: '{} ({} {:.1f})'.format(
                        sof.format_float_latex(x[1], '5.2f'),
                        x[2],
                        x[1] / x[0],
                    ),
                    axis=1,
                )

            ftdf = pd.concat(fts, axis=1).reset_index()
            ftdf['n_cell'] = sparsify_n_cell(ftdf['n_cell'].to_list())
            filename = indir(fname)
            header = ['\#cells', 'order'] + list(ftdf.keys())[2:]
            ftdf.to_latex(filename, index=False, escape=False, header=header,
                          column_format='rrlllll')

    elif options.analysis == 'mem-usages-in-limit':
        def get_min(x):
            k0 = x.keys()[0]
            ii = nm.argmin(x[k0])
            return x.iloc[ii]

        def get_max(x):
            k0 = x.keys()[0]
            ii = nm.argmax(x[k0])
            return x.iloc[ii]

        tn2key = {
            'dw_laplace::' : 'Laplacian' ,
            'dw_volume_dot:v:' : 'dot' ,
            'dw_volume_dot:vm:' : 'weighted dot',
            'dw_convect::' : 'NS convective',
            'dw_lin_elastic::' : 'elasticity',
            'dw_laplace::u' : 'Laplacian' ,
            'dw_volume_dot:v:u' : 'dot' ,
            'dw_volume_dot:vm:u' : 'weighted dot',
            'dw_convect::u' : 'NS convective',
            'dw_lin_elastic::u' : 'elasticity',
        }
        term_names = list(tn2key.keys())
        limit = options.limits.get('rtwwmean', ldf['rtwwmean'].max())
        for ii, fname in enumerate((
                'table-mus-min-r.inc', 'table-mus-min-m.inc',
                'table-mus-max-r.inc', 'table-mus-max-m.inc',
        )):
            fts = {}
            i2 = ii % 2
            for tn in term_names[5*i2:5*i2+5]:
                tdf = ldf[(ldf['term_name'] == tn) &
                          (ldf['rtwwmean'] <= limit)]

                sdf = tdf[tdf['lib'] != 'sfepy']
                sgb = sdf.groupby(['n_cell', 'order'])
                if 'min' in fname:
                    mdf = sgb[['mmean', 'rmmean', 'lib']].apply(get_min)

                else:
                    mdf = sgb[['mmean', 'rmmean', 'lib']].apply(get_max)

                key = tn2key[tn]
                fts[key] = mdf.apply(
                    lambda x: '{} ({} {:.1f})'.format(
                        sof.format_float_latex(x[0], '7.2f'),
                        x[2],
                        x[1],
                    ),
                    axis=1,
                )

            ftdf = pd.concat(fts, axis=1).reset_index()
            ftdf['n_cell'] = sparsify_n_cell(ftdf['n_cell'].to_list())
            filename = indir(fname)
            header = ['\#cells', 'order'] + list(ftdf.keys())[2:]
            ftdf.to_latex(filename, index=False, escape=False, header=header,
                          column_format='rrlllll')

    elif options.analysis == 'times-mems-single-fun':

        marker_style = {
            'lw' : 0.5,
            'mew' : 1.0,
            'marker' : ['o', '^', 'v', 'D', 's'],
            'alpha' : 1.0,
            'mfc' : 'None',
            'markersize' : 8,
        }

        xscale = 'log'
        yscale = 'log'

        tn2key = {
            'dw_laplace::' : 'Laplacian' ,
            'dw_volume_dot:v:' : 'vector dot' ,
            'dw_volume_dot:vm:' : 'weighted vec. dot',
            'dw_convect::' : 'NS convective',
            'dw_lin_elastic::' : 'elasticity',
            'dw_laplace::u' : 'Laplacian' ,
            'dw_volume_dot:v:u' : 'vector dot' ,
            'dw_volume_dot:vm:u' : 'weighted vec. dot',
            'dw_convect::u' : 'NS convective',
            'dw_lin_elastic::u' : 'elasticity',
        }
        term_names = list(tn2key.keys())
        def format_labels(key, iv, val):
            if key == 'order':
                return val

            else:
                return tn2key[val]

        for key, diff in product(['twwmean', 'mmean'], [False, True]):
            fig, ax = plt.subplots()

            tns = term_names[5*diff:5*diff+5]

            select = sps.select_by_keys(ldf, ['order'])
            select.update({'term_name' : tns})
            styles = {'term_name' : marker_style,
                      'order' : {'color' : 'tab10:kind=qualitative',}}
            styles = sps.setup_plot_styles(select, styles)

            ax.grid(True)
            used = None
            maxs = {ii : (0, 0) for ii in select['order']}
            for term_name, order, fun_name in product(
                    tns, data.orders, options.fun_names,
            ):
                is_diff = ldf['diff'].notna if diff else ldf['diff'].isna
                sdf = ldf[(ldf['term_name'] == term_name) &
                          (ldf['order'] == order) &
                          (ldf['fun_name'] == fun_name) &
                          (is_diff())]
                if not len(sdf): continue

                style_kwargs, indices, used = sps.get_row_style_used(
                    sdf.iloc[0], select, {}, styles, used
                )
                vx = sdf.n_cell.values
                means = sdf[key].values
                ax.plot(vx, means, **style_kwargs)

                imax = sdf[key].idxmax()
                if sdf.loc[imax, key] > maxs[order][1]:
                    maxs[order] = (sdf.loc[imax, 'n_cell'], sdf.loc[imax, key])

            sps.add_legend(ax, select, styles, used, per_parameter=True,
                           format_labels=format_labels,
                           loc=['center right', 'lower right'],
                           frame_alpha=0.8, ncol=1,
                           handlelength=1, handletextpad=0.4, columnspacing=0.2,
                           labelspacing=0.4)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_title('matrix mode' if diff else 'residual mode')
            ax.set_xlabel('n_cell')
            ax.set_ylabel(key)

            for order, (mx, my) in maxs.items():
                fmt = '{:.2f}' if my < 1 else '{:.1f}'
                ax.annotate(fmt.format(my), xy=(mx, my), xytext=(0, 15),
                            textcoords='offset points',
                            arrowprops=dict(facecolor='black',
                                            arrowstyle='->',
                                            shrinkA=0,
                                            shrinkB=0))

            plt.tight_layout()
            figname = ('{}-{}-{}{}'
                       .format(fun_name,
                               sdf['diff'].iloc[0] if diff else '-',
                               key,
                               options.suffix))
            fig = ax.figure
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

        #ldf['lib-opt'] = ldf['lib'] + '-' + ldf['opt']
        sldf = ldf[ldf['order'] == 1]
        sdf = (sldf[['lib', 'opt', 'spaths']].drop_duplicates()
               .sort_values(['lib', 'opt', 'spaths'], ignore_index=True))
        cat = sdf['lib'].astype('category').cat

        # sdf.plot.scatter(x='opt', y='spaths', color=cat.codes)

        # but spaths and (lib, opt) are 1:1
        sdf['lib-opt'] = sdf['lib'] + '-' + sdf['opt']
        #sdf.plot.scatter(x='lib-opt', y='spaths', color=cat.codes)
        ax = sdf.plot.scatter(x='lib-opt', y='spaths', color=cat.codes)
        ax.grid()

        sdf = (ldf[['n_cell', 'order', 'lib', 'opt', 'spaths']]
               .drop_duplicates()
               .sort_values(['n_cell', 'order', 'lib', 'opt', 'spaths'],
                            ignore_index=True))
        gb = sdf.groupby(['n_cell', 'order', 'lib', 'opt'])
        gb['spaths'].apply(lambda x: len(x))

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
