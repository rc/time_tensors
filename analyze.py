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
            'mew' : 2,
            'marker' : ['+', 'o', 'v', '^', '<', '>', 's', 'x', 'd'],
            'alpha' : 0.8,
            'mfc' : 'None',
            'markersize' : 8,
        }

    select = sps.select_by_keys(ldf, style_keys)
    styles = {marker_key : marker_style, color_key : {'color' : 'viridis',}}
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
        ax.plot(xs, yticks[nm.searchsorted(ysearch_labels, labels)], ls='None',
                **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used, format_labels=format_labels,
                       loc='lower right', frame_alpha=0.8)

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
    libs = sorted(ldf['lib'].unique())

    if marker_style is None:
        marker_style = {
            'mew' : 2,
            'marker' : ['+', 'o', 'v', '^', '<', '>', 's', 'x', 'd'],
            'alpha' : 0.8,
            'mfc' : 'None',
            'markersize' : 8,
        }

    select = sps.select_by_keys(ldf, style_keys)
    styles = {marker_key : marker_style, color_key : {'color' : 'viridis',}}
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
    for lib in libs:
        sdf = sldf[(sldf['lib'] == lib)]
        if not len(sdf): continue

        style_kwargs, indices, used = sps.get_row_style_used(
            sdf.iloc[0], select, {}, styles, used
        )

        xs = sdf[xkey]
        labels = sdf[ykeys].apply(ylabel_fun, axis=1)
        ax.plot(yticks[nm.searchsorted(ysearch_labels, labels)], xs, ls='None',
                **style_kwargs)

    if show_legend:
        sps.add_legend(ax, select, styles, used, format_labels=format_labels,
                       loc='upper right', frame_alpha=0.8)

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
        plot_rc_params = "'text.usetex'=False",
        shorten_spaths = False,
        xscale = 'log',
        xlim = 'auto=True',
        suffix = '.png',
    )

    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help=helps['output_dir'])
    parser.add_argument('results', help=helps['results'])
    parser.add_argument('--analysis', action='store', dest='analysis',
                        choices=['layouts', 'all-terms', 'all-terms-rate'],
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
                ax.xaxis.set_major_locator(mt.FixedLocator([0.5, 1, 2, 3, 4, 5]))

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
        for key in keys:
            ldf[key + '_rate'] = ldf['n_cell'] / ldf[key]

        ldf = ldf[ldf['term_name'].isin(term_names)]

        def get_max(x):
            k0 = x.keys()[0]
            ii = nm.argmax(x[k0])
            return x.iloc[ii]

        gbt = ldf.groupby(['term_name', 'n_cell', 'order', 'lib'])

        for key in [key + '_rate' for key in keys]:
            aux = [key, 'spaths']
            if options.shorten_spaths: aux += ['short_spaths']
            rdf = gbt[aux].apply(get_max).reset_index()

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
                figname = ('{}-rate-n_cell-order-{}{}'
                           .format(term_name,
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
