#!/usr/bin/env python
"""
Time tensor contractions using various einsum() implementations.
"""
import sys
sys.path.append('.')
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['XLA_FLAGS'] = ('--xla_cpu_multi_thread_eigen=false '
                           'intra_op_parallelism_threads=1')
import psutil

from functools import partial
from itertools import product
import gc
import re
import hashlib

import numpy as nm
try:
    import dask
    import dask.array as da

except ImportError:
    da = None

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

except ImportError:
    jnp = jax = None

try:
    import opt_einsum as oe

except ImportError:
    oe = None

try:
    import numba as nb
    from numba import jit as njit

except ImportError:
    nb = None
    def njit(*args, **kwargs):
        def _njit(fun):
            return fun
        return _njit

from mprof import read_mprofile_file

import pandas as pd

import soops as so
from sfepy.base.base import output
from sfepy.base.ioutils import ensure_path, save_options
from sfepy.base.timing import Timer
from sfepy.discrete.variables import expand_basis
from sfepy.discrete.fem import FEDomain, Field
import sfepy.discrete.fem.refine_hanging as rh
from sfepy.discrete.fem.geometry_element import GeometryElement
from sfepy.discrete import (FieldVariable, Material, Integral, PolySpace)
from sfepy.terms import Term
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.mechanics.matcoefs import stiffness_from_lame

import terms_multilinear; terms_multilinear

def get_run_info():
    # script_dir is added by soops-run, it is the normalized path to
    # this script.
    run_cmd = """
    rm {output_dir}/mprofile.dat; mprof run -T {sampling} -C -o {output_dir}/mprofile.dat time_tensors.py --mprof {output_dir}
    """
    run_cmd = ' '.join(run_cmd.split())

    # Arguments allowed to be missing in soops-run calls.
    opt_args = {
        '--n-cell' : '--n-cell={--n-cell}',
        '--refine' : '--refine',
        '--order' : '--order={--order}',
        '--quad-order' : '--quad-order={--quad-order}',
        '--term-name' : '--term-name={--term-name}',
        '--eval-mode' : '--eval-mode={--eval-mode}',
        '--variant' : '--variant={--variant}',
        '--layout' : '--layout={--layout}',
        '--diff' : '--diff={--diff}',
        '--select' : '--select={--select}',
        '--repeat' : '--repeat={--repeat}',
        '--affinity' : '--affinity={--affinity}',
        '--max-mem' : '--max-mem={--max-mem}',
        '--verbosity-eterm' : '--verbosity-eterm={--verbosity-eterm}',
        '--silent' : '--silent',
    }

    output_dir_key = 'output_dir'
    is_finished_basename = 'stats.csv'

    return run_cmd, opt_args, output_dir_key, is_finished_basename

def get_scoop_info():
    import soops.scoop_outputs as sc

    info = [
        ('options.txt', partial(
            sc.load_split_options,
            split_keys=None,
        ), True),
        ('stats.csv', sc.load_csv),
        ('mprofile.dat', load_mprofile),
        ('output_log.txt', scrape_output),
    ]

    return info

def load_mprofile(filename, rdata=None):
    mdata = read_mprofile_file(filename)
    mdata.pop('children')
    mdata.pop('cmd_line')
    return mdata

def scrape_output(filename, rdata=None):
    import soops.ioutils as io
    from ast import literal_eval

    out = {}
    with open(filename, 'r') as fd:
        line = io.skip_lines_to(fd, 'total system memory')
        out['mem_total_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'available system memory')
        out['mem_available_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'qsbg assumed size')
        out['qsbg_size_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'memory estimate [qsbg size]')
        out['mem_est_qsbg'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'memory estimate [MB]')
        out['mem_est_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'u shape')
        aux = literal_eval(line.split(':')[2].strip())
        out['n_dof'] = aux[0]

        line = io.skip_lines_to(fd, 'adc shape')
        aux = literal_eval(line.split(':')[2].strip())
        out['n_cdof'] = aux[1]

        line = io.skip_lines_to(fd, 'qsbg shape')
        aux = literal_eval(line.split(':')[2].strip())
        out['n_qp'] = aux[1]
        out['dim'] = aux[2]
        out['n_en'] = aux[3]

        line = io.skip_lines_to(fd, 'qvbg assumed size')
        out['qvbg_size_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'c vec size')
        out['c_vec_size_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'c mtx size')
        out['c_mtx_size_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'memory use [MB]')
        out['pre_mem_use_mb'] = literal_eval(line.split(':')[2].strip())

        line = io.skip_lines_to(fd, 'memory use [qsbg size]')
        out['pre_mem_use_qsbg'] = literal_eval(line.split(':')[2].strip())

    return out

def get_plugin_info():
    from soops.plugins import show_figures

    info = [
        collect_times,
        collect_mem_usages,
        select_data,
        setup_styles,
        plot_times,
        plot_mem_usages,
        plot_all_as_bars,
        plot_all_as_bars2,
        show_figures,
        plot_comparisons,
    ]

    return info

def collect_times(df, data=None):
    tkeys = [key for key in df.keys() if key.startswith('t_')]

    uniques = {key : val for key, val in data.par_uniques.items()
               if key not in ['output_dir', 'max_mem']}
    output('parameterization:')
    for key, val in uniques.items():
        output(key, val)

    df['index'] = df.index
    tdf = pd.melt(df, list(uniques.keys()) + ['index'], tkeys,
                  var_name='fun_name', value_name='t')
    tdf['fun_name'] = tdf['fun_name'].str[2:] # Strip 't_'.

    def fun(x):
        return x['t'] if nm.isfinite(x['t']).all() else [nm.nan] * x['repeat']
    tdf['t'] = tdf.apply(fun, axis=1)

    data._fun_names = [tkey[2:] for tkey in tkeys]
    data.uniques = uniques
    data._tdf = tdf
    return data

def collect_mem_usages(df, data=None):
    if 'func_timestamp' not in df:
        output('no memory profiling data!')
        data._mdf = None
        return data

    aux = pd.json_normalize(df['func_timestamp']).rename(
        lambda x: 'm_' + x.split('.')[-1].replace('eval_', ''), axis=1
    )
    mkeys = ['m_' + fun_name for fun_name in data._fun_names]
    if set(mkeys) != set(aux.keys()):
        output('wrong memory profiling data, ignoring!')
        data._mdf = None
        return data

    del df['func_timestamp']
    df = pd.concat([df, aux], axis=1)

    df['index'] = df.index
    mdf =  pd.melt(df, list(data.uniques.keys()) + ['index'], mkeys,
                   var_name='fun_name', value_name='ts')
    mdf['fun_name'] = mdf['fun_name'].str[2:] # Strip 'm_'.

    for term_name in data.par_uniques['term_name']:
        for order in data.par_uniques['order']:
            dfto = df[(df['term_name'] == term_name) &
                     (df['order'] == order)]
            mem_usage = dfto['mem_usage']
            mem_tss = dfto['timestamp']
            for fun_name in data._fun_names:
                indexer = ((mdf['term_name'] == term_name) &
                           (mdf['order'] == order) &
                           (mdf['fun_name'] == fun_name))
                sdf = mdf.loc[indexer]
                repeat = sdf.iloc[0]['repeat']
                mems = []
                for ii in dfto.index:
                    mu = nm.array(mem_usage.loc[ii])
                    tss = nm.array(mem_tss.loc[ii])
                    _ts = sdf[sdf['index'] == ii].iloc[0]['ts']
                    if (_ts is not nm.nan) and (len(_ts) == repeat):
                        for ts in _ts:
                            i0, i1 = nm.searchsorted(tss, ts[:2])

                            mmax = max(mu[i0:i1].max() if i1 > i0 else ts[3],
                                       ts[3])
                            mmin = min(mu[i0:i1].min() if i1 > i0 else ts[2],
                                       ts[2])
                            mem = mmax - mmin

                            mems.append(mem)

                    else:
                        if (_ts is not nm.nan) and len(_ts):
                            output('wrong memory profiling data for'
                                   ' {}/{} order: {} n_cell: {}!'
                                   .format(fun_name, term_name, order,
                                           sdf[sdf['index'] == ii]
                                           .iloc[0]['n_cell']))
                        mems.extend([nm.nan] * repeat)

                mems = nm.array(mems).reshape((-1, repeat))

                # This is to force a column with several values.
                mm = [pd.Series({'mems' : row.tolist()}) for row in mems]
                mdf.loc[indexer, 'mems'] = pd.DataFrame(mm, index=sdf.index)

    data._mdf = mdf
    return data

def select_data(df, data=None, term_names=None, n_cell=None, orders=None,
                functions=None):
    data.term_names = (data.par_uniques['term_name']
                       if term_names is None else term_names)
    data.n_cell = data.par_uniques['n_cell'] if n_cell is None else n_cell
    data.orders = data.par_uniques['order'] if orders is None else orders
    if functions is None:
        data.fun_names = data._fun_names
        data.tdf = data._tdf
        data.mdf = data._mdf

    else:
        fun_match = re.compile('|'.join(functions)).match
        data.fun_names = [fun for fun in data._fun_names if fun_match(fun)]

        indexer = data._tdf['fun_name'].isin(data.fun_names)
        data.tdf = data._tdf[indexer]
        data.mdf = data._mdf[indexer] if data._mdf is not None else None

    data.fun_hash = hashlib.sha256(''.join(data.fun_names)
                                   .encode('utf-8')).hexdigest()

    return data

def setup_styles(df, data=None, colormap_name='viridis', markers=None):
    import soops.plot_selected as sps

    if markers is None:
        from matplotlib.lines import Line2D
        markers = list(Line2D.filled_markers)

    select = sps.normalize_selected(data.uniques)
    select['fun_name'] = data.fun_names

    styles = {key : {} for key in select.keys()}
    styles['term_name'] = {'ls' : ['-', '--', '-.'], 'lw' : 2, 'alpha' : 0.8}
    styles['order'] = {'color' : colormap_name}
    styles['fun_name'] = {'marker' : markers,
                          'mfc' : 'None', 'ms' : 8}
    styles = sps.setup_plot_styles(select, styles)

    data.select = select
    data.styles = styles
    return data

def format_labels(key, iv, val):
    if key == 'term_name':
        return 'term: {}'.format(val)

    elif key == 'fun_name':
        return '{}'.format(val)

    else:
        return '{}: {}'.format(key, val)

def _onpick_line(event, lines):
    line = event.artist
    output(lines[line])

def plot_times(df, data=None, xscale='log', yscale='log',
               prefix='', suffix='.pdf'):
    import soops.plot_selected as sps
    import matplotlib.pyplot as plt

    select = data.select.copy()
    select['fun_name'] = data.fun_names
    styles = data.styles

    tdf = data.tdf

    fig, ax = plt.subplots()
    used = None
    lines = {}
    for term_name in data.par_uniques['term_name']:
        if term_name not in data.term_names: continue
        for order in data.par_uniques['order']:
            if order not in data.orders: continue
            for fun_name in data.fun_names:
                if fun_name not in data._fun_names: continue
                print(term_name, order, fun_name)
                sdf = tdf[(tdf['term_name'] == term_name) &
                          (tdf['order'] == order) &
                          (tdf['fun_name'] == fun_name)]
                vx = sdf.n_cell.values
                times = sdf['t'].to_list()

                means = nm.nanmean(times, axis=1)
                emins = means - nm.nanmin(times, axis=1)
                emaxs = nm.nanmax(times, axis=1) - means
                style_kwargs, indices = sps.get_row_style(
                    sdf.iloc[0], select, {}, styles
                )
                used = sps.update_used(used, indices)
                line = plt.errorbar(vx, means, yerr=[emins, emaxs],
                                    ecolor=style_kwargs['color'],
                                    elinewidth=1, capsize=2,
                                    **style_kwargs)[0]
                line.set_picker(True)
                lines[line] = (fun_name, order)

    sps.add_legend(ax, select, styles, used, format_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel('n_cell')
    ax.set_ylabel('time [s]')
    plt.tight_layout()

    fig.savefig(os.path.join(data.output_dir, prefix + 'times' + suffix),
                bbox_inches='tight')

    fig.canvas.mpl_connect('pick_event', partial(_onpick_line, lines=lines))

def plot_mem_usages(df, data=None, xscale='log', yscale='symlog',
                    prefix='', suffix='.pdf'):
    import soops.plot_selected as sps
    import matplotlib.pyplot as plt

    select = data.select.copy()
    select['fun_name'] = data.fun_names
    styles = data.styles

    mdf = data.mdf

    fig, ax = plt.subplots()
    used = None
    lines = {}
    for term_name in data.par_uniques['term_name']:
        if term_name not in data.term_names: continue
        for order in data.par_uniques['order']:
            if order not in data.orders: continue
            for fun_name in data.fun_names:
                if fun_name not in data._fun_names: continue
                print(term_name, order, fun_name)
                sdf = mdf[(mdf['term_name'] == term_name) &
                          (mdf['order'] == order) &
                          (mdf['fun_name'] == fun_name)]
                vx = sdf.n_cell.values
                mems = sdf['mems'].to_list()

                means = nm.nanmean(mems, axis=1)
                emins = means - nm.nanmin(mems, axis=1)
                emaxs = nm.nanmax(mems, axis=1) - means
                style_kwargs, indices = sps.get_row_style(
                    sdf.iloc[0], select, {}, styles
                )
                used = sps.update_used(used, indices)

                line = plt.errorbar(vx, means, yerr=[emins, emaxs],
                                    ecolor=style_kwargs['color'],
                                    elinewidth=1, capsize=2,
                                    **style_kwargs)[0]
                line.set_picker(True)
                lines[line] = (fun_name, order)

    sps.add_legend(ax, select, styles, used, format_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel('n_cell')
    ax.set_ylabel('memory [MB]')
    plt.tight_layout()

    fig.savefig(os.path.join(data.output_dir, prefix + 'mem_usages' + suffix),
                bbox_inches='tight')

    fig.canvas.mpl_connect('pick_event', partial(_onpick_line, lines=lines))

def set_ylim(ax, mi, ma, yscale):
    if yscale == 'linear':
        ax.set_ylim(0, 1.2 * ma)

    else:
        mi = max(mi, 1e-3)
        ax.set_ylim(0.8 * mi, 1.2 * ma)

def get_yticks(mi, ma, yscale):
    if yscale == 'linear':
        yticks = nm.linspace(mi, ma, 5)

    else:
        mi = max(mi, 1e-3)
        yticks = nm.logspace(nm.log10(mi), nm.log10(ma), 3)

    return yticks

def get_stats(sdf, key):
    vals = sdf[key].to_list()
    means = nm.nanmean(vals, axis=1)
    emins = means - nm.nanmin(vals, axis=1)
    emaxs = nm.nanmax(vals, axis=1) - means

    return means, emins, emaxs

def plot_all_as_bars(df, data=None, tcolormap_name='viridis',
                     mcolormap_name='plasma', yscale='log',
                     prefix='', suffix='.pdf'):
    import soops.plot_selected as sps
    from soops.formatting import format_float_latex
    import matplotlib.pyplot as plt

    tdf = data.tdf
    mdf = data.mdf

    select = {}
    select['tn_cell'] = tdf['n_cell'].unique()

    mit = nm.nanmin(tdf['t'].to_list())
    mat = nm.nanmax(tdf['t'].to_list())
    tyticks = get_yticks(mit, mat, yscale)
    tyticks_labels = [format_float_latex(ii, 1) for ii in tyticks]

    styles = {}
    styles['tn_cell'] = {'color' : tcolormap_name}

    if mdf is not None:
        select['mn_cell'] = mdf['n_cell'].unique()
        styles['mn_cell'] = {'color' : mcolormap_name}

        mim = nm.nanmin(mdf['mems'].to_list())
        mam = nm.nanmax(mdf['mems'].to_list())
        myticks = get_yticks(mim, mam, yscale)
        myticks_labels = [format_float_latex(ii, 1) for ii in myticks]

    styles = sps.setup_plot_styles(select, styles)
    tcolors = styles['tn_cell']['color']
    if mdf is not None:
        mcolors = styles['mn_cell']['color']

    fig, axs = plt.subplots(len(data.orders) * len(data.term_names),
                            figsize=(12, 8), squeeze=False)
    axs = axs.T[0]
    axs2 = []
    for ax in axs:
        ax.grid(which='both', axis='y')
        set_ylim(ax, mit, mat, yscale)
        ax.set_yscale(yscale)
        ax.set_yticks(tyticks)
        ax.set_yticklabels(tyticks_labels)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if mdf is not None:
            ax2 = ax.twinx()
            ax2.grid(which='both', axis='y')
            set_ylim(ax2, mim, mam, yscale)
            ax2.set_yscale(yscale)
            ax2.set_yticks(myticks)
            ax2.set_yticklabels(myticks_labels)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.get_xaxis().set_visible(False)
            axs2.append(ax2)

    nax = len(axs)

    sx = 3
    ia = 0
    for term_name in data.par_uniques['term_name']:
        if term_name not in data.term_names: continue
        for io, order in enumerate(data.par_uniques['order']):
            if order not in data.orders: continue
            ax = axs[ia]
            if mdf is not None:
                ax2 = axs2[ia]
            bx = 0

            xts = []
            functions = []
            for it, fun_name in enumerate(data.fun_names):
                if fun_name not in data._fun_names: continue
                tsdf = tdf[(tdf['term_name'] == term_name) &
                           (tdf['order'] == order) &
                           (tdf['fun_name'] == fun_name)]
                if not len(tsdf): continue

                functions.append(fun_name)

                vx = tsdf.n_cell.values
                tmeans, temins, temaxs = get_stats(tsdf, 't')

                xs = bx + nm.arange(len(vx))
                ax.bar(xs, tmeans, width=0.8, align='edge',
                       yerr=[temins, temaxs], bottom=ax.get_ylim()[0],
                       color=tcolors, capsize=2)

                xts.append(xs[-1])

                if mdf is not None:
                    msdf = mdf[(mdf['term_name'] == term_name) &
                               (mdf['order'] == order) &
                               (mdf['fun_name'] == fun_name)]
                    mmeans, memins, memaxs = get_stats(msdf, 'mems')

                    xs = xs[-1] + sx + nm.arange(len(vx))
                    ax2.bar(xs, mmeans, width=0.8, align='edge',
                            yerr=[memins, memaxs], bottom=ax2.get_ylim()[0],
                            color=mcolors, capsize=2)

                bx = xs[-1] + 2 * sx

                if it < len(data.fun_names):
                    ax.axvline(bx - sx, color='k', lw=0.5)

            ax.set_title('{}/order {}'.format(term_name, order))
            ax.set_xlim(0, bx + 1 - 2 * sx)
            if ia + 1 < nax:
                ax.get_xaxis().set_visible(False)

            else:
                ax.set_xticks(xts)
                ax.set_xticklabels(functions)

            ia += 1

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)

    lines, labels = sps.get_legend_items(select, styles)
    leg = fig.legend(lines, labels)
    if leg is not None:
        leg.get_frame().set_alpha(0.5)

    fig.savefig(os.path.join(data.output_dir, prefix + 'all_bars' + suffix),
                bbox_inches='tight')

def plot_all_as_bars2(df, data=None, tcolormap_name='viridis',
                      mcolormap_name='plasma', yscale='linear',
                      prefix='', suffix='.pdf'):
    import soops.plot_selected as sps
    from soops.formatting import format_float_latex
    import matplotlib.pyplot as plt

    tdf = data.tdf
    mdf = data.mdf

    select = {}
    select['tfunction'] = tdf['fun_name'].unique()

    mit = nm.nanmin(tdf['t'].to_list())
    mat = nm.nanmax(tdf['t'].to_list())
    tyticks = get_yticks(mit, mat, yscale)
    tyticks_labels = [format_float_latex(ii, 1) for ii in tyticks]

    styles = {}
    styles['tfunction'] = {'color' : tcolormap_name}

    if mdf is not None:
        select['mfunction'] = mdf['fun_name'].unique()
        styles['mfunction'] = {'color' : mcolormap_name}

        mim = nm.nanmin(mdf['mems'].to_list())
        mam = nm.nanmax(mdf['mems'].to_list())
        myticks = get_yticks(mim, mam, yscale)
        myticks_labels = [format_float_latex(ii, 1) for ii in myticks]

    styles = sps.setup_plot_styles(select, styles)
    tcolors = styles['tfunction']['color']
    if mdf is not None:
        mcolors = styles['mfunction']['color']

    fig, axs = plt.subplots(len(data.term_names) * len(data.n_cell),
                            figsize=(12, 8), squeeze=False)
    axs = axs.T[0]
    axs2 = []
    for ax in axs:
        ax.grid(which='both', axis='y')
        set_ylim(ax, mit, mat, yscale)
        ax.set_yscale(yscale)
        ax.set_yticks(tyticks)
        ax.set_yticklabels(tyticks_labels)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if mdf is not None:
            ax2 = ax.twinx()
            ax2.grid(which='both', axis='y')
            set_ylim(ax2, mim, mam, yscale)
            ax2.set_yscale(yscale)
            ax2.set_yticks(myticks)
            ax2.set_yticklabels(myticks_labels)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.get_xaxis().set_visible(False)
            axs2.append(ax2)

    nax = len(axs)

    sx = 3
    ia = 0
    for term_name in data.par_uniques['term_name']:
        if term_name not in data.term_names: continue
        for ic, n_cell in enumerate(data.par_uniques['n_cell']):
            if n_cell not in data.n_cell: continue
            ax = axs[ia]
            if mdf is not None:
                ax2 = axs2[ia]
            bx = 0

            xts = []
            orders = []
            for io, order in enumerate(data.par_uniques['order']):
                if order not in data.orders: continue
                tsdf = tdf[(tdf['term_name'] == term_name) &
                           (tdf['n_cell'] == n_cell) &
                           (tdf['order'] == order)]
                if not len(tsdf): continue

                orders.append(order)

                vx = tsdf['fun_name'].values
                tmeans, temins, temaxs = get_stats(tsdf, 't')

                xs = bx + nm.arange(len(vx))
                ax.bar(xs, tmeans, width=0.8, align='edge',
                       yerr=[temins, temaxs], bottom=ax.get_ylim()[0],
                       color=tcolors, capsize=2)

                xts.append(xs[-1])

                if mdf is not None:
                    msdf = mdf[(mdf['term_name'] == term_name) &
                               (mdf['n_cell'] == n_cell) &
                               (mdf['order'] == order)]
                    mmeans, memins, memaxs = get_stats(msdf, 'mems')

                    xs = xs[-1] + sx + nm.arange(len(vx))
                    ax2.bar(xs, mmeans, width=0.8, align='edge',
                            yerr=[memins, memaxs], bottom=ax2.get_ylim()[0],
                            color=mcolors, capsize=2)

                bx = xs[-1] + 2 * sx

                if io < len(data.orders):
                        ax.axvline(bx - sx, color='k', lw=0.5)

            ax.set_title('{}/{} cells'.format(term_name, n_cell))
            ax.set_xlim(0, bx + 1 - 2 * sx)
            if ia + 1 < nax:
                ax.get_xaxis().set_visible(False)

            else:
                ax.set_xticks(xts)
                ax.set_xticklabels(orders)

            ia += 1

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)

    lines, labels = sps.get_legend_items(select, styles)
    leg = fig.legend(lines, labels)
    if leg is not None:
        leg.get_frame().set_alpha(0.5)

    fig.savefig(os.path.join(data.output_dir, prefix + 'all_bars2' + suffix),
                bbox_inches='tight')

def plot_comparisons(df, data=None, colormap_name='tab10:qualitative',
                     yscale='linear', figsize=(8, 6), prefix='', suffix='.png',
                     sort='time', number=None):
    import soops.plot_selected as sps
    import matplotlib.pyplot as plt

    tdf = data.tdf
    mdf = data.mdf

    select = {}
    select['fun_name'] = tdf['fun_name'].unique()
    styles = {}
    styles['fun_name'] = {'color' : colormap_name}

    styles = sps.setup_plot_styles(select, styles)
    colors = styles['fun_name']['color']

    fig, axs = plt.subplots(1 + (mdf is not None), figsize=figsize,
                            sharex=True, squeeze=False)
    for ifig, (term_name, n_cell, order) in enumerate(
            product(data.term_names, data.n_cell, data.orders)
    ):
        output(term_name, n_cell, order)

        tsdf = tdf[(tdf['term_name'] == term_name) &
                   (tdf['n_cell'] == n_cell) &
                   (tdf['order'] == order)]
        if not len(tsdf): continue

        n_dof = df.loc[tsdf['index'].values[0], 'n_dof']
        if nm.isfinite(n_dof):
            n_dof = int(n_dof)

        vx = tsdf['fun_name']
        tmeans, temins, temaxs = get_stats(tsdf, 't')

        if mdf is not None:
            msdf = mdf[(mdf['term_name'] == term_name) &
                       (mdf['n_cell'] == n_cell) &
                       (mdf['order'] == order)]
            mmeans, memins, memaxs = get_stats(msdf, 'mems')

        if sort == 'time':
            ii = nm.argsort(tmeans)

        elif (sort == 'memory') and mdf is not None:
            ii = nm.argsort(mmeans)

        if sort != 'none':
            if number is not None:
                ii = ii[:number]

            vx = vx.iloc[ii]
            tmeans = tmeans[ii]
            temins = temins[ii]
            temaxs = temaxs[ii]
            if mdf is not None:
                mmeans = mmeans[ii]
                memins = memins[ii]
                memaxs = memaxs[ii]

        xs = nm.arange(len(vx))

        diff = tsdf['diff'].values[0]
        if diff is None: diff = '-'

        ax = axs[0, 0]
        ax.cla()
        ax.set_title('{}, diff: {}, #cells: {}, order: {}, #DOFs: {}'
                     .format(term_name, diff, n_cell, order, n_dof))
        ax.grid(which='both', axis='y')
        ax.bar(xs, tmeans, width=0.8, align='center',
               yerr=[temins, temaxs], bottom=ax.get_ylim()[0],
               color=colors, capsize=2)
        ax.set_yscale(yscale)
        ax.set_ylabel('time [s]')

        if mdf is not None:
            ax.xaxis.set_visible(False)

            ax = axs[1, 0]
            ax.cla()
            ax.grid(which='both', axis='y')
            ax.bar(xs, mmeans, width=0.8, align='center',
                   yerr=[memins, memaxs], bottom=ax.get_ylim()[0],
                   color=colors, capsize=2)
            ax.set_yscale(yscale)
            ax.set_ylabel('memory [MB]')

        ax = axs[-1, 0]
        ax.set_xticks(xs)
        ax.set_xticklabels(vx, rotation='vertical')

        plt.tight_layout()
        filename = (prefix
                    + '{}-{:03d}-{:03d}-{}-{}-{}-{}-{}'
                    .format(data.fun_hash[:8], len(vx), ifig,
                            term_name, diff, n_cell, order, yscale)
                    + suffix)
        fig.savefig(os.path.join(data.output_dir, filename),
                    bbox_inches='tight')

def create_domain(n_cell, refine, timer):
    timer.start()
    mesh = gen_block_mesh((n_cell, 1, 1), (n_cell + 1, 2, 2), (0, 0, 0),
                          name='')
    output('generate mesh: {} s'.format(timer.stop()))
    timer.start()
    domain = FEDomain('el', mesh)
    output('create domain: {} s'.format(timer.stop()))

    subs = None
    if (n_cell > 1) and refine:
        refine_cells = nm.zeros(domain.mesh.n_el, dtype=nm.uint8)
        refine_cells[n_cell // 2] = 1
        domain, subs = rh.refine(domain, refine_cells, subs=subs)

    timer.start()
    omega = domain.create_region('omega', 'all')
    output('create omega: {} s'.format(timer.stop()))

    return mesh, domain, subs, omega

def get_v_sol(coors):
    x0 = coors.min(axis=0)
    x1 = coors.max(axis=0)
    dims = x1 - x0

    cc = (coors - x0) / dims[None, :]
    return cc

def get_s_sol(coors):
    x0 = coors.min(axis=0)
    x1 = coors.max(axis=0)
    dims = x1 - x0

    cc = (coors[:, 0] - x0[0]) / dims[0]
    return cc

def set_sol(uvar, mesh, timer):
    timer.start()
    if uvar.n_components == mesh.dim:
        uvar.set_from_function(get_v_sol)

    else:
        uvar.set_from_function(get_s_sol)

    output('set state {}: {} s'.format(uvar.name, timer.stop()))
    uvec = uvar()
    return uvec

def create_terms(_create_term, timer):
    timer.start()
    term = _create_term()
    term.setup()
    term.standalone_setup()
    output('create setup term: {} s'.format(timer.stop()))

    timer.start()
    eterm = _create_term('e')
    eterm.setup()
    eterm.standalone_setup()
    output('create setup eterm: {} s'.format(timer.stop()))

    return term, eterm

def setup_data(order, quad_order, n_cell, term_name='dw_convect',
               eval_mode='weak', variant=None, refine=False):

    integral = Integral('i', order=quad_order)

    timer = Timer('')

    mesh, domain, subs, omega = create_domain(n_cell, refine, timer)

    if (term_name in ('dw_convect', 'dw_div', 'dw_lin_elastic')
        or ('vector' in variant)):
        n_c = mesh.dim

    else:
        n_c = 1

    timer.start()
    field = Field.from_args('fu', nm.float64, n_c, omega,
                            approx_order=order)
    if subs is not None:
        field.substitute_dofs(subs)
    output('create field: {} s'.format(timer.stop()))

    timer.start()
    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    output('create variables: {} s'.format(timer.stop()))

    if term_name in ('dw_lin_elastic',) or ('material' in variant):
        timer.start()
        if term_name == 'dw_volume_dot':
            mat = Material('m', val=nm.ones((n_c, n_c), dtype=nm.float64))

        elif term_name == 'dw_lin_elastic':
            mat = Material('m', D=stiffness_from_lame(dim=3, lam=2.0, mu=1.0))

        else:
            raise ValueError(term_name)

        output('create material: {} s'.format(timer.stop()))

    uvec = set_sol(u, mesh, timer)

    def _create_term(prefix=''):
        if term_name == 'dw_convect':
            term = Term.new('dw_{}convect(v, u)'.format(prefix),
                            integral=integral,
                            region=omega, v=v, u=u)

        elif term_name == 'dw_laplace':
            if eval_mode == 'weak':
                term = Term.new('dw_{}laplace(v, u)'.format(prefix),
                                integral=integral,
                                region=omega, v=v, u=u)

            else:
                term = Term.new('dw_{}laplace(u, u)'.format(prefix),
                                integral=integral,
                                region=omega, u=u)

        elif term_name == 'dw_volume_dot':
            if eval_mode == 'weak':
                if 'material' in variant:
                    tstr = 'dw_{}volume_dot(m.val, v, u)'
                    targs = {'m' : mat, 'v' : v, 'u' : u}

                else:
                    tstr = 'dw_{}volume_dot(v, u)'
                    targs = {'v' : v, 'u' : u}

            else:
                if 'material' in variant:
                    tstr = 'dw_{}volume_dot(m.val, u, u)'
                    targs = {'m' : mat, 'u' : u}

                else:
                    tstr = 'dw_{}volume_dot(u, u)'
                    targs = {'u' : u}

            term = Term.new(tstr.format(prefix), integral=integral,
                            region=omega, **targs)

        elif term_name == 'dw_div':
            term = Term.new('dw_{}div(v)'.format(prefix),
                            integral=integral,
                            region=omega, v=v)

        elif term_name == 'dw_lin_elastic':
            if eval_mode == 'weak':
                term = Term.new('dw_{}lin_elastic(m.D, v, u)'.format(prefix),
                                integral=integral,
                                region=omega, m=mat, v=v, u=u)

            else:
                term = Term.new('dw_{}lin_elastic(m.D, u, u)'.format(prefix),
                                integral=integral,
                                region=omega, m=mat, u=u)

        else:
            raise ValueError(term_name)

        return term

    term, eterm = create_terms(_create_term, timer)

    return uvec, term, eterm

def setup_data_mixed(order1, order2, quad_order, n_cell, term_name='dw_stokes',
                     eval_mode='weak', variant='div', refine=False):

    integral = Integral('i', order=quad_order)

    timer = Timer('')

    mesh, domain, subs, omega = create_domain(n_cell, refine, timer)

    if term_name in ('dw_stokes',):
        n_c1 = mesh.dim
        n_c2 = 1

    else:
        raise ValueError(term_name)

    timer.start()
    field1 = Field.from_args('f1', nm.float64, n_c1, omega,
                             approx_order=order1)
    field2 = Field.from_args('f2', nm.float64, n_c2, omega,
                             approx_order=order2)
    if subs is not None:
        field1.substitute_dofs(subs)
        field2.substitute_dofs(subs)
    output('create fields: {} s'.format(timer.stop()))

    timer.start()
    u1 = FieldVariable('u1', 'unknown', field1)
    v1 = FieldVariable('v1', 'test', field1, primary_var_name='u1')
    u2 = FieldVariable('u2', 'unknown', field2)
    v2 = FieldVariable('v2', 'test', field2, primary_var_name='u2')
    output('create variables: {} s'.format(timer.stop()))

    if eval_mode == 'eval':
        uvec = set_sol(u1, mesh, timer)
        set_sol(u2, mesh, timer)

    elif variant == 'div':
        uvec = set_sol(u1, mesh, timer)

    else:
        uvec = set_sol(u2, mesh, timer)

    def _create_term(prefix=''):
        if term_name == 'dw_stokes':
            if variant == 'div':
                if eval_mode == 'weak':
                    term = Term.new('dw_{}stokes(u1, v2)'.format(prefix),
                                    integral=integral,
                                    region=omega, v2=v2, u1=u1)

                else:
                    term = Term.new('dw_{}stokes(u1, u2)'.format(prefix),
                                    integral=integral,
                                    region=omega, u2=u2, u1=u1)

            else:
                if eval_mode == 'weak':
                    term = Term.new('dw_{}stokes(v1, u2)'.format(prefix),
                                    integral=integral,
                                    region=omega, v1=v1, u2=u2)

                else:
                    term = Term.new('dw_{}stokes(u1, u2)'.format(prefix),
                                    integral=integral,
                                    region=omega, u1=u1, u2=u2)

        else:
            raise ValueError(term_name)

        return term

    term, eterm = create_terms(_create_term, timer)

    return uvec, term, eterm

def _get_shape(expr, *arrays):
    lhs, output = expr.split('->')
    inputs = lhs.split(',')

    sizes = {}
    for term, array in zip(inputs, arrays):
        for k, d in zip(term, array.shape):
            sizes[k] = d

    out_shape = tuple(sizes[k] for k in output)
    return sizes, out_shape

def _expand_sbg(basis, dpn):
    dim, n_ep = basis.shape[-2:]
    vg = nm.zeros(basis.shape[:2] + (dpn, dim, dim * n_ep))
    for ir in range(dpn):
        vg[..., ir, :, n_ep*ir:n_ep*(ir+1)] = basis
    return vg

def get_evals_dw_convect(options, term, eterm,
                         dets, qsb, qsbg, qvb, qvbg, state, adc):
    if not options.mprof:
        def profile(fun):
            return fun

    else:
        profile = globals()['profile']

    @profile
    def eval_numpy_einsum2():
        uc = state()[adc]

        if options.diff == 'u':
            v1 = nm.einsum('cqab,qji,cqjkl,qkn,cn->cil',
                           dets, qvb[0], qvbg, qvb[0], uc,
                           optimize='greedy')
            v2 = nm.einsum('cqab,qji,cqjkl,cl,qkn->cin',
                           dets, qvb[0], qvbg, uc, qvb[0],
                           optimize='greedy')
            return v1 + v2, 0

        else:
            return nm.einsum('cqab,qji,cqjkl,cl,qkn,cn->ci',
                             dets, qvb[0], qvbg, uc, qvb[0], uc,
                             optimize='greedy'), 0
    @profile
    def eval_numpy_einsum_qsb():
        uc = state()[adc]

        n_cell, n_ed = uc.shape
        ucc = uc.reshape((dets.shape[0], -1, qsb.shape[-1]))
        ee = nm.eye(ucc.shape[-2])
        if options.diff == 'u':
            val1 = nm.einsum('cqab,qzy,jx,cqkY,jX,qzn,ckn->cxyXY',
                             dets, qsb[0], ee, qsbg, ee, qsb[0], ucc,
                             optimize='greedy')
            v1 = val1.reshape((n_cell, n_ed, n_ed))
            val2 = nm.einsum('cqab,qzy,jx,cqkl,cjl,qzY,kX->cxyXY',
                             dets, qsb[0], ee, qsbg, ucc, qsb[0], ee,
                             optimize='greedy')
            v2 = val2.reshape((n_cell, n_ed, n_ed))
            return v1 + v2, 0

        else:
            val2 = nm.einsum('cqab,qzy,jx,cqkl,cjl,qzn,ckn->cxy',
                             dets, qsb[0], ee, qsbg, ucc, qsb[0], ucc,
                             optimize='greedy')
            v2 = val2.reshape((n_cell, n_ed))
            # no time difference with the above
            # out = nm.empty((n_cell, n_ed), dtype=nm.float64)
            # vout = out.reshape(ucc.shape)
            # v2 = nm.einsum('cqab,qzy,jx,cqkl,cjl,qzn,ckn->cxy',
            #                dets, qsb[0], ee, qsbg, ucc, qsb[0], ucc,
            #                out=vout,
            #                optimize='greedy')

            return v2, 0

    @profile
    def eval_numpy_einsum3():
        # Slower than eval_numpy_einsum2().
        uc = state()[adc]

        if options.diff == 'u':
            aux = nm.einsum('cqab,qji,cqjkl->cqikl',
                           dets, qvb[0], qvbg,
                           optimize='greedy')
            v1 = nm.einsum('cqikl,qkn,cn->cil',
                           aux, qvb[0], uc,
                           optimize='greedy')
            v2 = nm.einsum('cqikl,cl,qkn->cin',
                           aux, uc, qvb[0],
                           optimize='greedy')
            return v1 + v2, 0

        else:
            return nm.einsum('cqab,qji,cqjkl,cl,qkn,cn->ci',
                             dets, qvb[0], qvbg, uc, qvb[0], uc,
                             optimize='greedy'), 0

    @profile
    def eval_opt_einsum1a():
        uc = state()[adc]

        if options.diff == 'u':
            v1 = oe.contract('cqab,qji,cqjkl,qkn,cn->cil',
                             dets, qvb[0], qvbg, qvb[0], uc,
                             optimize='auto')
            v2 = oe.contract('cqab,qji,cqjkl,cl,qkn->cin',
                             dets, qvb[0], qvbg, uc, qvb[0],
                             optimize='auto')
            return v1 + v2, 0

        else:
            return oe.contract('cqab,qji,cqjkl,cl,qkn,cn->ci',
                               dets, qvb[0], qvbg, uc, qvb[0], uc,
                               optimize='auto'), 0

    @profile
    def eval_opt_einsum1g():
        uc = state()[adc]

        if options.diff == 'u':
            v1 = oe.contract('cqab,qji,cqjkl,qkn,cn->cil',
                             dets, qvb[0], qvbg, qvb[0], uc,
                             optimize='greedy')
            v2 = oe.contract('cqab,qji,cqjkl,cl,qkn->cin',
                             dets, qvb[0], qvbg, uc, qvb[0],
                             optimize='greedy')
            return v1 + v2, 0

        else:
            return oe.contract('cqab,qji,cqjkl,cl,qkn,cn->ci',
                               dets, qvb[0], qvbg, uc, qvb[0], uc,
                               optimize='greedy'), 0

    @profile
    def eval_opt_einsum1dp():
        uc = state()[adc]

        if options.diff == 'u':
            v1 = oe.contract('cqab,qji,cqjkl,qkn,cn->cil',
                             dets, qvb[0], qvbg, qvb[0], uc,
                             optimize='dynamic-programming')
            v2 = oe.contract('cqab,qji,cqjkl,cl,qkn->cin',
                             dets, qvb[0], qvbg, uc, qvb[0],
                             optimize='dynamic-programming')
            return v1 + v2, 0

        else:
            return oe.contract('cqab,qji,cqjkl,cl,qkn,cn->ci',
                               dets, qvb[0], qvbg, uc, qvb[0], uc,
                               optimize='dynamic-programming'), 0

    @profile
    def eval_opt_einsum_qsb():
        uc = state()[adc]
        n_cell, n_ed = uc.shape
        ucc = uc.reshape((dets.shape[0], -1, qsb.shape[-1]))
        ee = nm.eye(ucc.shape[-2])

        if options.diff == 'u':
            val1 = oe.contract('cqab,qzy,jx,cqkY,jX,qzn,ckn->cxyXY',
                               dets, qsb[0], ee, qsbg, ee, qsb[0], ucc,
                               optimize='greedy')
            v1 = val1.reshape((n_cell, n_ed, n_ed))
            val2 = oe.contract('cqab,qzy,jx,cqkl,cjl,qzY,kX->cxyXY',
                               dets, qsb[0], ee, qsbg, ucc, qsb[0], ee,
                               optimize='greedy')
            v2 = val2.reshape((n_cell, n_ed, n_ed))
            # print(nm.abs(_v2 - v2).max())
            # from sfepy.base.base import debug; debug()
            return v1 + v2, 0

        else:
            # val2 = oe.contract('cqab,qrd,is,cqie,cje,qvf,cjf->csd',
            #                    #'cqab,qzy,jx,cqkl,cjl,qzn,ckn->cxy',
            #                    dets, qsb[0], ee, qsbg, ucc, qsb[0], ucc,
            #                    optimize='greedy')
            # no time difference with the above
            v2 = nm.empty((n_cell, n_ed), dtype=nm.float64)
            vout = v2.reshape(ucc.shape)
            oe.contract('cqab,qzy,jx,cqkl,cjl,qzn,ckn->cxy',
                        dets, qsb[0], ee, qsbg, ucc, qsb[0], ucc,
                        out=vout,
                        optimize='greedy')

            return v2, 0

    n_cell, n_qp, dim, n_en = qsbg.shape
    n_c = dim

    qbs = [qsb[0, :, 0, ir].copy(order='F') for ir in range(n_en)]
    qbgs = [qsbg[..., ir].copy(order='F') for ir in range(n_en)]
    det = dets[..., 0, 0].copy(order='F')

    @profile
    def eval_opt_einsum_nl1f():
        uc = state()[adc]
        n_cell, n_ed = uc.shape
        ucc = uc.reshape((dets.shape[0], -1, qsb.shape[-1]))
        ee = nm.eye(ucc.shape[-2])

        opt = 'dynamic-programming'
        tt = Timer(start=True)
        qgu = oe.contract('cqkl,cjl->cqkj', qsbg, ucc, optimize=opt)
        qu = oe.contract('qzn,ckn->cqk', qsb[0], ucc, optimize=opt)
        print(tt.stop())
        if options.diff == 'u':
            out = nm.empty((n_cell, n_c * n_en, n_c * n_en), dtype=nm.float64)
            path1, info1 = oe.contract_path('cq,q,jx,cqk,jX,cqk->cxX',
                                            det, qbs[0], ee,
                                            qbgs[0], ee, qu,
                                            optimize=opt)
            print(path1)
            print(info1)
            path2, info2 = oe.contract_path('cq,q,jx,cqkj,q,kX->cxX',
                                             det, qbs[0], ee,
                                             qgu, qbs[0], ee,
                                             optimize=opt)
            print(path2)
            print(info2)
            for ir in range(n_en): # y
                rqb = qbs[ir]
                for ic in range(n_en): # Y
                    cqbg = qbgs[ic]
                    cqb = qbs[ic]
                    v1 = oe.contract('cq,q,jx,cqk,jX,cqk->cxX',
                                     det, rqb, ee,
                                     cqbg, ee, qu,
                                     optimize=path1)
                    v2 = oe.contract('cq,q,jx,cqkj,q,kX->cxX',
                                     det, rqb, ee,
                                     qgu, cqb, ee,
                                     optimize=path2)
                    out[:, ir::n_en, ic::n_en] = v1 + v2

            return out, 0

        else:
            raise NotImplementedError

    qbs2 = [qsb[0, :, 0, ir].copy(order='C') for ir in range(n_en)]
    qbgs2 = [qsbg[..., ir].transpose(2, 0, 1).copy(order='C')
             for ir in range(n_en)]
    det2 = dets[..., 0, 0].copy(order='C')

    @profile
    def eval_opt_einsum_nl2c():
        uc = state()[adc]
        n_cell, n_ed = uc.shape
        ucc = uc.reshape((dets.shape[0], -1, qsb.shape[-1]))
        ee = nm.eye(ucc.shape[-2])

        opt = 'dynamic-programming'
        tt = Timer(start=True)
        qgu = oe.contract('cqkl,cjl->kjcq', qsbg, ucc, optimize=opt)
        qu = oe.contract('qzn,ckn->kcq', qsb[0], ucc, optimize=opt)
        print(tt.stop())
        if options.diff == 'u':
            out = nm.empty((n_cell, n_c * n_en, n_c * n_en), dtype=nm.float64)
            path1, info1 = oe.contract_path('cq,q,jx,kcq,jX,kcq->cxX',
                                            det2, qbs2[0], ee,
                                            qbgs2[0], ee, qu,
                                            optimize=opt)
            print(path1)
            print(info1)
            path2, info2 = oe.contract_path('cq,q,jx,kjcq,q,kX->cxX',
                                             det2, qbs2[0], ee,
                                             qgu, qbs2[0], ee,
                                             optimize=opt)
            print(path2)
            print(info2)
            for ir in range(n_en): # y
                rqb = qbs2[ir]
                for ic in range(n_en): # Y
                    cqbg = qbgs2[ic]
                    cqb = qbs2[ic]
                    v1 = oe.contract('cq,q,jx,kcq,jX,kcq->cxX',
                                     det2, rqb, ee,
                                     cqbg, ee, qu,
                                     optimize=path1)
                    v2 = oe.contract('cq,q,jx,kjcq,q,kX->cxX',
                                     det2, rqb, ee,
                                     qgu, cqb, ee,
                                     optimize=path2)
                    out[:, ir::n_en, ic::n_en] = v1 + v2

            return out, 0

        else:
            raise NotImplementedError

    @profile
    def eval_opt_einsum2a():
        uc = state()[adc]

        if options.diff == 'u':
            with oe.shared_intermediates():
                v1 = oe.contract('cqab,qji,cqjkl,qkn,cn->cil',
                                 dets, qvb[0], qvbg, qvb[0], uc,
                                 optimize='auto')
                v2 = oe.contract('cqab,qji,cqjkl,cl,qkn->cin',
                                 dets, qvb[0], qvbg, uc, qvb[0],
                                 optimize='auto')
            return v1 + v2, 0

        else:
            return oe.contract('cqab,qji,cqjkl,cl,qkn,cn->ci',
                               dets, qvb[0], qvbg, uc, qvb[0], uc,
                               optimize='auto'), 0

    @profile
    def eval_opt_einsum2dp():
        uc = state()[adc]

        if options.diff == 'u':
            with oe.shared_intermediates():
                v1 = oe.contract('cqab,qji,cqjkl,qkn,cn->cil',
                                 dets, qvb[0], qvbg, qvb[0], uc,
                                 optimize='dynamic-programming')
                v2 = oe.contract('cqab,qji,cqjkl,cl,qkn->cin',
                                 dets, qvb[0], qvbg, uc, qvb[0],
                                 optimize='dynamic-programming')
            return v1 + v2, 0

        else:
            return oe.contract('cqab,qji,cqjkl,cl,qkn,cn->ci',
                               dets, qvb[0], qvbg, uc, qvb[0], uc,
                               optimize='dynamic-programming'), 0

    def eval_jax2(dets, Fs, Gs, u):
        out = jnp.einsum('qab,qji,qjkl,l,qkn,n->i', dets, Fs, Gs, u, Fs, u)
        return out

    if jax is not None:
        eval_jax2_grad = jax.jacobian(eval_jax2, 3)

    @profile
    def eval_jax_einsum1():
        if jax is None: return nm.zeros(1), 1
        uc = state()[adc]
        f = 0
        vm = (0, None, 0, 0)
        if options.diff is None:
            f = jax.jit(jax.vmap(eval_jax2, vm, 0))(dets, qvb[0], qvbg, uc)

        elif options.diff == 'u':
            f = jax.jit(jax.vmap(eval_jax2_grad, vm, 0))(dets, qvb[0], qvbg, uc)

        return f, 0

    @jax.jit
    def _eval_jax_einsum2_qsb(dets, qsb, qsbg, dofs, adc):
        uc = dofs[adc]
        n_cell, n_ed = uc.shape
        ucc = uc.reshape((dets.shape[0], -1, qsb.shape[-1]))
        ee = nm.eye(ucc.shape[-2])
        if options.diff == 'u':
            val1 = jnp.einsum('cqab,qzy,jx,cqkY,jX,qzn,ckn->cxyXY',
                              dets, qsb[0], ee, qsbg, ee, qsb[0], ucc,
                              optimize='greedy')
            v1 = val1.reshape((n_cell, n_ed, n_ed))
            val2 = jnp.einsum('cqab,qzy,jx,cqkl,cjl,qzY,kX->cxyXY',
                              dets, qsb[0], ee, qsbg, ucc, qsb[0], ee,
                              optimize='greedy')
            v2 = val2.reshape((n_cell, n_ed, n_ed))
            return v1 + v2

        else:
            val = jnp.einsum('cqab,qzy,jx,cqkl,cjl,qzn,ckn->cxy',
                             dets, qsb[0], ee, qsbg, ucc, qsb[0], ucc,
                             optimize='greedy')
            v = val.reshape((n_cell, n_ed))

            return v

    @profile
    def eval_jax_einsum2_qsb():
        val = _eval_jax_einsum2_qsb(dets, qsb, qsbg, state(), adc)
        return nm.asarray(val), 0

    _dets = dets[..., 0, 0]
    bf = qsb[0, :, 0]
    dofs = state()
    uc = dofs[adc]
    n_cell, n_ed = uc.shape
    ucc = uc.reshape((dets.shape[0], -1, qsb.shape[-1]))
    ee = nm.eye(ucc.shape[-2])

    @jax.jit
    def _eval_jax_einsum2_qsb2(dets, qsb, qsbg, ucc, ee):
        if options.diff == 'u':
            val1 = jnp.einsum('cq,qy,jx,cqkY,jX,qn,ckn->cxyXY',
                              dets, qsb, ee, qsbg, ee, qsb, ucc,
                              optimize='greedy')
            val2 = jnp.einsum('cq,qy,jx,cqkl,cjl,qY,kX->cxyXY',
                              dets, qsb, ee, qsbg, ucc, qsb, ee,
                              optimize='greedy')
            return (val1 + val2).reshape((n_cell, n_ed, n_ed))

        else:
            val = jnp.einsum('cq,qy,jx,cqkl,cjl,qn,ckn->cxy',
                             dets, qsb, ee, qsbg, ucc, qsb, ucc,
                             optimize='greedy')
            return val.reshape((n_cell, n_ed))

    @profile
    def eval_jax_einsum2_qsb2():
        val = _eval_jax_einsum2_qsb2(_dets, bf, qsbg, ucc, ee)
        return nm.asarray(val), 0

    @profile
    def eval_dask_einsum1():
        uc = state()[adc]

        if options.diff == 'u':
            v1 = da.einsum('cqab,qji,cqjkl,qkn,cn->cil',
                           dets, qvb[0], qvbg, qvb[0], uc,
                           optimize='greedy')
            v2 = da.einsum('cqab,qji,cqjkl,cl,qkn->cin',
                           dets, qvb[0], qvbg, uc, qvb[0],
                           optimize='greedy')
            return (v1 + v2).compute(scheduler='single-threaded'), 0

        else:
            return da.einsum('cqab,qji,cqjkl,cl,qkn,cn->ci',
                             dets, qvb[0], qvbg, uc, qvb[0], uc,
                             optimize='greedy').compute(
                                 scheduler='single-threaded'
                             ), 0

    evaluators = {
        # 'numpy_einsum1' : (eval_numpy_einsum1, 0, True), # unusably slow
        'numpy_einsum2' : (eval_numpy_einsum2, 0, nm),
        'numpy_einsum_qsb' : (eval_numpy_einsum_qsb, 0, nm),
        # 'numpy_einsum3' : (eval_numpy_einsum3, 0, nm), # slow, memory hog
        #'opt_einsum1a' : (eval_opt_einsum1a, 0, oe),
        'opt_einsum1g' : (eval_opt_einsum1g, 0, oe),
        'opt_einsum1dp' : (eval_opt_einsum1dp, 0, oe),
        'opt_einsum_qsb' : (eval_opt_einsum_qsb, 0, oe),
        'opt_einsum_nl1f' : (eval_opt_einsum_nl1f, 0, oe),
        'opt_einsum_nl2c' : (eval_opt_einsum_nl2c, 0, oe),
        #'opt_einsum2a' : (eval_opt_einsum2a, 0, oe), # more memory than opt_einsum1*
        'opt_einsum2dp' : (eval_opt_einsum2dp, 0, oe), # more memory than opt_einsum1*
        'dask_einsum1' : (eval_dask_einsum1, 0, da),
        # 'jax_einsum1' : (eval_jax_einsum1, 0, jnp), # meddles with memory profiler
         'jax_einsum2_qsb' : (eval_jax_einsum2_qsb, 0, jnp),
         'jax_einsum2_qsb2' : (eval_jax_einsum2_qsb2, 0, jnp),
    }

    return evaluators

def get_evals_dw_laplace(options, term, eterm,
                         dets, qsb, qsbg, qvb, qvbg, state, adc):
    n_cell, n_qp, dim, n_en = qsbg.shape

    detsf = dets.copy(order='F')
    qsbgf = qsbg.copy(order='F')

    dets2 = dets[..., 0, 0]
    qsbg2 = qsbg.transpose((2, 3, 0, 1)).copy(order='C')
    qsbg4 = qsbg.transpose((3, 2, 0, 1)).copy(order='C')
    adc2 = adc.T.copy(order='C')
    dets3 = dets[..., 0, 0].transpose((1, 0)).copy(order='C')
    qsbg3 = qsbg.transpose((2, 3, 1, 0)).copy(order='C')
    qsbg3a = qsbg.transpose((3, 2, 1, 0)).copy(order='C')

    qbgs1f = [qsbg[..., ir].copy(order='F') for ir in range(n_en)]
    dets1f = dets[..., 0, 0].copy(order='F')
    qbgs1c = [qsbg[..., ir].copy(order='C') for ir in range(n_en)]
    dets1c = dets[..., 0, 0].copy(order='C')

    qbgs2f = [qsbg[..., ir].transpose(1, 2, 0).copy(order='F')
              for ir in range(n_en)]
    dets2f = dets[..., 0, 0].T.copy(order='F')
    qbgs2c = [qsbg[..., ir].transpose(1, 2, 0).copy(order='C')
              for ir in range(n_en)]
    dets2c = dets[..., 0, 0].T.copy(order='C')

    qbgs3f = [qsbg[..., ir].transpose(1, 0, 2).copy(order='F')
              for ir in range(n_en)]
    qbgs3c = [qsbg[..., ir].transpose(1, 0, 2).copy(order='C')
              for ir in range(n_en)]

    if not options.mprof:
        def profile(fun):
            return fun

    else:
        profile = globals()['profile']

    @profile
    def eval_numpy_einsum2():
        if options.diff == 'u':
            return nm.einsum('cqab,cqjk,cqjn->ckn',
                             dets, qsbg, qsbg,
                             optimize='greedy'), 0

        else:
            uc = state()[adc]
            return nm.einsum('cqab,cqjk,cqjn,cn->ck',
                             dets, qsbg, qsbg, uc,
                             optimize='greedy'), 0

    @profile
    def eval_opt_einsum1a():
        if options.diff == 'u':
            return oe.contract('cqab,cqjk,cqjn->ckn',
                               dets, qsbg, qsbg,
                               optimize='auto'), 0

        else:
            uc = state()[adc]
            return oe.contract('cqab,cqjk,cqjn,cn->ck',
                               dets, qsbg, qsbg, uc,
                               optimize='auto'), 0

    @profile
    def eval_opt_einsum1g():
        if options.diff == 'u':
            return oe.contract('cqab,cqjk,cqjn->ckn',
                               dets, qsbg, qsbg,
                               optimize='greedy'), 0

        else:
            uc = state()[adc]
            return oe.contract('cqab,cqjk,cqjn,cn->ck',
                               dets, qsbg, qsbg, uc,
                               optimize='greedy'), 0


    @profile
    def eval_opt_einsum1dp():
        if options.diff == 'u':
            return oe.contract('cqab,cqjk,cqjn->ckn',
                               dets, qsbg, qsbg,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc]
            return oe.contract('cqab,cqjk,cqjn,cn->ck',
                               dets, qsbg, qsbg, uc,
                               optimize='dynamic-programming'), 0

    @profile
    def eval_opt_einsum1dp2():
        if options.diff == 'u':
            return oe.contract('cq,cqjk,cqjn->ckn',
                               dets[..., 0, 0], qsbg, qsbg,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc]
            return oe.contract('cq,cqjk,cqjn,cn->ck',
                               dets[..., 0, 0], qsbg, qsbg, uc,
                               optimize='dynamic-programming'), 0

    @profile
    def eval_opt_einsum1dp2_nl1f():
        det = dets1f
        qbgs = qbgs1f
        if options.diff == 'u':
            out = nm.empty((n_cell, n_en, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('cq,cqj,cqj->c',
                                               det, qbgs[0], qbgs[0],
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                for ic in range(n_en):
                    cqbg = qbgs[ic]
                    aux = oe.contract('cq,cqj,cqj->c',
                                      det, rqbg, cqbg,
                                      optimize=path)
                    out[:, ir, ic] = aux

            return out, 0

        else:
            uc = state()[adc]
            uq = oe.contract('cqjn,cqn->cqj', qsbgf, uc[:, None, :],
                             optimize='dynamic-programming')

            out = nm.empty((n_cell, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('cq,cqj,cqj->c',
                                               det, qbgs[0], uq,
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                aux = oe.contract('cq,cqj,cqj->c',
                                  det, rqbg, uq,
                                  optimize=path)
                out[:, ir] = aux

            return out, 0

    @profile
    def eval_opt_einsum1dp2_nl1c():
        det = dets1c
        qbgs = qbgs1c
        if options.diff == 'u':
            out = nm.empty((n_cell, n_en, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('cq,cqj,cqj->c',
                                               det, qbgs[0], qbgs[0],
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                for ic in range(n_en):
                    cqbg = qbgs[ic]
                    aux = oe.contract('cq,cqj,cqj->c',
                                      det, rqbg, cqbg,
                                      optimize=path)
                    out[:, ir, ic] = aux

            return out, 0

        else:
            uc = state()[adc]
            uq = oe.contract('cqjn,cqn->cqj', qsbg, uc[:, None, :],
                             optimize='dynamic-programming')

            out = nm.empty((n_cell, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('cq,cqj,cqj->c',
                                               det, qbgs[0], uq,
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                aux = oe.contract('cq,cqj,cqj->c',
                                  det, rqbg, uq,
                                  optimize=path)
                out[:, ir] = aux

            return out, 0

    @profile
    def eval_opt_einsum1dp2_nl2f():
        det = dets2f
        qbgs = qbgs2f
        if options.diff == 'u':
            out = nm.empty((n_cell, n_en, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qjc,qjc->c',
                                               det, qbgs[0], qbgs[0],
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                for ic in range(n_en):
                    cqbg = qbgs[ic]
                    aux = oe.contract('qc,qjc,qjc->c',
                                      det, rqbg, cqbg,
                                      optimize=path)
                    out[:, ir, ic] = aux

            return out, 0

        else:
            uc = state()[adc]
            uq = oe.contract('cqjn,cqn->qjc', qsbgf, uc[:, None, :],
                             optimize='dynamic-programming')

            out = nm.empty((n_cell, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qjc,qjc->c',
                                               det, qbgs[0], uq,
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                aux = oe.contract('qc,qjc,qjc->c',
                                  det, rqbg, uq,
                                  optimize=path)
                out[:, ir] = aux

            return out, 0

    @profile
    def eval_opt_einsum1dp2_nl2c():
        det = dets2c
        qbgs = qbgs2c
        if options.diff == 'u':
            out = nm.empty((n_cell, n_en, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qjc,qjc->c',
                                               det, qbgs[0], qbgs[0],
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                for ic in range(n_en):
                    cqbg = qbgs[ic]
                    aux = oe.contract('qc,qjc,qjc->c',
                                      det, rqbg, cqbg,
                                      optimize=path)
                    out[:, ir, ic] = aux

            return out, 0

        else:
            uc = state()[adc]
            uq = oe.contract('cqjn,cqn->qjc', qsbg, uc[:, None, :],
                             optimize='dynamic-programming')

            out = nm.empty((n_cell, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qjc,qjc->c',
                                               det, qbgs[0], uq,
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                aux = oe.contract('qc,qjc,qjc->c',
                                  det, rqbg, uq,
                                  optimize=path)
                out[:, ir] = aux

            return out, 0

    @profile
    def eval_opt_einsum1dp2_nl3f():
        det = dets2f
        qbgs = qbgs3f
        if options.diff == 'u':
            out = nm.empty((n_cell, n_en, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qcj,qcj->c',
                                               det, qbgs[0], qbgs[0],
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                for ic in range(n_en):
                    cqbg = qbgs[ic]
                    aux = oe.contract('qc,qcj,qcj->c',
                                      det, rqbg, cqbg,
                                      optimize=path)
                    out[:, ir, ic] = aux

            return out, 0

        else:
            uc = state()[adc]
            uq = oe.contract('cqjn,cqn->qcj', qsbgf, uc[:, None, :],
                             optimize='dynamic-programming')

            out = nm.empty((n_cell, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qcj,qcj->c',
                                               det, qbgs[0], uq,
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                aux = oe.contract('qc,qcj,qcj->c',
                                  det, rqbg, uq,
                                  optimize=path)
                out[:, ir] = aux

            return out, 0

    @profile
    def eval_opt_einsum1dp2_nl3c():
        det = dets2c
        qbgs = qbgs3c
        if options.diff == 'u':
            out = nm.empty((n_cell, n_en, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qcj,qcj->c',
                                               det, qbgs[0], qbgs[0],
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                for ic in range(n_en):
                    cqbg = qbgs[ic]
                    aux = oe.contract('qc,qcj,qcj->c',
                                      det, rqbg, cqbg,
                                      optimize=path)
                    out[:, ir, ic] = aux

            return out, 0

        else:
            uc = state()[adc]
            uq = oe.contract('cqjn,cqn->qcj', qsbg, uc[:, None, :],
                             optimize='dynamic-programming')

            out = nm.empty((n_cell, n_en), dtype=nm.float64)
            path, path_info = oe.contract_path('qc,qcj,qcj->c',
                                               det, qbgs[0], uq,
                                               optimize='dynamic-programming')
            #print(path_info)
            for ir in range(n_en):
                rqbg = qbgs[ir]
                aux = oe.contract('qc,qcj,qcj->c',
                                  det, rqbg, uq,
                                  optimize=path)
                out[:, ir] = aux

            return out, 0

    @profile
    def eval_opt_einsum1dp3():
        if options.diff == 'u':
            return oe.contract('cq,jkcq,jncq->knc',
                               dets2, qsbg2, qsbg2,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc2]
            return oe.contract('cq,jkcq,jncq,nc->kc',
                               dets2, qsbg2, qsbg2, uc,
                               optimize='dynamic-programming'), 0

    @profile
    def eval_opt_einsum1dp4():
        if options.diff == 'u':
            return oe.contract('cq,jkcq,jncq->ckn',
                               dets2, qsbg2, qsbg2,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc2]
            return oe.contract('cq,jkcq,jncq,nc->ck',
                               dets2, qsbg2, qsbg2, uc,
                               optimize='dynamic-programming'), 0

    @profile
    def eval_opt_einsum1dp4a():
        if options.diff == 'u':
            return oe.contract('cq,kjcq,njcq->ckn',
                               dets2, qsbg4, qsbg4,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc2]
            return oe.contract('cq,kjcq,njcq,nc->ck',
                               dets2, qsbg4, qsbg4, uc,
                               optimize='dynamic-programming'), 0

    @profile
    def eval_opt_einsum1dp4b():
        if options.diff == 'u':
            return oe.contract('cq,jkcq,jncq->ckn',
                               dets[..., 0, 0], qsbg2, qsbg2,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc2]
            return oe.contract('cq,jkcq,jncq,nc->ck',
                               dets[..., 0, 0], qsbg2, qsbg2, uc,
                               optimize='dynamic-programming'), 0
    @profile
    def eval_opt_einsum1dp5():
        if options.diff == 'u':
            return oe.contract('qc,jkqc,jnqc->ckn',
                               dets3, qsbg3, qsbg3,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc2]
            return oe.contract('qc,jkqc,jnqc,nc->ck',
                               dets3, qsbg3, qsbg3, uc,
                               optimize='dynamic-programming'), 0

    @profile
    def eval_opt_einsum1dp5a():
        if options.diff == 'u':
            return oe.contract('qc,kjqc,njqc->ckn',
                               dets3, qsbg3a, qsbg3a,
                               optimize='dynamic-programming'), 0

        else:
            uc = state()[adc2]
            return oe.contract('qc,jkqc,jnqc,nc->ck',
                               dets3, qsbg3, qsbg3, uc,
                               optimize='dynamic-programming'), 0

    def eval_jax2(dets, Gs, u):
        out = jnp.einsum('qab,qjk,qjn,n->k',
                         dets, Gs, Gs, u)
        return out

    if jax is not None:
        eval_jax2_grad = jax.jacobian(eval_jax2, 2)

    @profile
    def eval_jax_einsum1():
        if jax is None: return nm.zeros(1), 1
        uc = state()[adc]
        f = 0
        vm = (0, 0, 0)
        if options.diff is None:
            f = jax.jit(jax.vmap(eval_jax2, vm, 0))(dets, qsbg, uc)

        elif options.diff == 'u':
            f = jax.jit(jax.vmap(eval_jax2_grad, vm, 0))(dets, qsbg, uc)

        return f, 0

    @profile
    def eval_dask_einsum1():
        if options.diff == 'u':
            return da.einsum('cqab,cqjk,cqjn->ckn',
                             dets, qsbg, qsbg,
                             optimize='greedy').compute(
                                 scheduler='single-threaded'
                             ), 0

        else:
            uc = state()[adc]
            return da.einsum('cqab,cqjk,cqjn,cn->ck',
                             dets, qsbg, qsbg, uc,
                             optimize='greedy').compute(
                                 scheduler='single-threaded'
                             ), 0

    @profile
    def eval_dask_einsum2():
        if options.diff == 'u':
            return da.einsum('cqab,cqjk,cqjn->ckn',
                             dets, qsbg, qsbg,
                             optimize='greedy').compute(
                                 scheduler='threads'
                             ), 0

        else:
            uc = state()[adc]
            return da.einsum('cqab,cqjk,cqjn,cn->ck',
                             dets, qsbg, qsbg, uc,
                             optimize='greedy').compute(
                                 scheduler='threads'
                             ), 0

    @profile
    def eval_opt_einsum_loop():
        _dets = dets[..., 0, 0]
        if options.diff == 'u':
            sizes, out_shape = _get_shape('cq,cqjk,cqjn->ckn',
                                          _dets, qsbg, qsbg)
            out = nm.empty(out_shape, dtype=nm.float64)
            path, path_info =  oe.contract_path('q,qjk,qjn->kn',
                                                _dets[0], qsbg[0], qsbg[0],
                                                optimize='auto')
            for c in range(sizes['c']):
                sbg = qsbg[c]
                out[c] = oe.contract('q,qjk,qjn->kn', _dets[c], sbg, sbg,
                                     optimize=path)

            return out, 0

        else:
            uc = state()[adc]
            sizes, out_shape = _get_shape('cq,cqjk,cqjn,cn->ck',
                                          _dets, qsbg, qsbg, uc)
            out = nm.empty(out_shape, dtype=nm.float64)
            path, path_info =  oe.contract_path('q,qjk,qjn,n->k',
                                                _dets[0], qsbg[0], qsbg[0],
                                                uc[0], optimize='auto')
            for c in range(sizes['c']):
                sbg = qsbg[c]
                out[c] = oe.contract('q,qjk,qjn,n->k', _dets[c], sbg, sbg,
                                     uc[c], optimize=path)

            return out, 0

    @njit('(float64[:,:], float64[:,:,:,:])',
          nopython=True)
    def _eval_numba_loops_m(det, bg):
        out = nm.zeros((n_cell, n_en, n_en), dtype=nm.float64)
        for icell in range(n_cell):
            for iqp in range(n_qp):
                for ir in range(n_en):
                    for ic in range(n_en):
                        aux = 0.0
                        for ii in range(dim):
                            aux += bg[icell,iqp,ii,ir] * bg[icell,iqp,ii,ic]
                        out[icell, ir, ic] += aux * det[icell, iqp]
        return out, 0

    @njit('(float64[:,:], float64[:,:,:,:], float64[:], int32[:,:])',
          nopython=True)
    def _eval_numba_loops_r(det, bg, dofs, adc):
        out = nm.zeros((n_cell, n_en), dtype=nm.float64)
        for icell in range(n_cell):
            for iqp in range(n_qp):
                ug = nm.zeros(dim, dtype=nm.float64)
                for ir in range(n_en):
                    dd = dofs[adc[icell, ir]]
                    for ii in range(dim):
                        ug[ii] += bg[icell,iqp,ii,ir] * dd

                for ir in range(n_en):
                    aux = 0.0
                    for ii in range(dim):
                        aux += bg[icell,iqp,ii,ir] * ug[ii]
                    out[icell, ir] += aux * det[icell, iqp]
        return out, 0

    @profile
    def eval_numba_loops():
        if options.diff == 'u':
            return _eval_numba_loops_m(dets[..., 0, 0], qsbg)

        else:
            return _eval_numba_loops_r(dets[..., 0, 0], qsbg, state(), adc)

    @profile
    def eval_opt_einsum_dask(coef=1000):
        n_cell, n_qp, dim, n_en = qsbg.shape
        # print(coef * qsbg.nbytes / n_cell)
        _dets = da.from_array(dets, chunks=(coef, n_qp, 1, 1), name='dets')
        _qsbg = da.from_array(qsbg, chunks=(coef, n_qp, dim, n_en), name='qsbg')
        if options.diff == 'u':
            return oe.contract(
                'cqab,cqjk,cqjn->ckn',
                _dets, _qsbg, _qsbg,
                optimize='dynamic-programming', backend='dask').compute(
                    scheduler='single-threaded'
            ), 0

        else:
            uc = state()[adc]
            return oe.contract(
                'cqab,cqjk,cqjn,cn->ck',
                _dets, _qsbg, _qsbg, uc,
                optimize='dynamic-programming', backend='dask').compute(
                    scheduler='single-threaded'
            ), 0

    evaluators = {
        'numpy_einsum2' : (eval_numpy_einsum2, 0, nm),
        'opt_einsum1a' : (eval_opt_einsum1a, 0, oe),
        # 'opt_einsum1g' : (eval_opt_einsum1g, 0, oe), # Uses too much memory in this case
        'opt_einsum1dp' : (eval_opt_einsum1dp, 0, oe),
        'opt_einsum1dp2' : (eval_opt_einsum1dp2, 0, oe),
        'opt_einsum1dp2_nl1f' : (eval_opt_einsum1dp2_nl1f, 0, oe),
        'opt_einsum1dp2_nl1c' : (eval_opt_einsum1dp2_nl1c, 0, oe),
        'opt_einsum1dp2_nl2f' : (eval_opt_einsum1dp2_nl2f, 0, oe),
        'opt_einsum1dp2_nl2c' : (eval_opt_einsum1dp2_nl2c, 0, oe),
        'opt_einsum1dp2_nl3f' : (eval_opt_einsum1dp2_nl3f, 0, oe),
        'opt_einsum1dp2_nl3c' : (eval_opt_einsum1dp2_nl3c, 0, oe),
        # 'opt_einsum1dp3' : (eval_opt_einsum1dp3, 0, oe),
        'opt_einsum1dp4' : (eval_opt_einsum1dp4, 0, oe),
        'opt_einsum1dp4a' : (eval_opt_einsum1dp4a, 0, oe),
        'opt_einsum1dp4b' : (eval_opt_einsum1dp4b, 0, oe),
        'opt_einsum1dp5' : (eval_opt_einsum1dp5, 0, oe),
        'opt_einsum1dp5a' : (eval_opt_einsum1dp5a, 0, oe),
        'dask_einsum1' : (eval_dask_einsum1, 0, da),
        'dask_einsum2' : (eval_dask_einsum2, 0, da),
        'opt_einsum_loop' : (eval_opt_einsum_loop, 0, oe),
        'numba_loops' : (eval_numba_loops, 0, nb),
        'opt_einsum_dask' : (eval_opt_einsum_dask, 0, oe and da),
        # 'jax_einsum1' : (eval_jax_einsum1, 0, jnp), # meddles with memory profiler
    }

    return evaluators

def get_evals_micro(options, term, eterm,
                    dets, qsb, qsbg, qvb, qvbg, state, adc):
    if not options.mprof:
        def profile(fun):
            return fun

    else:
        profile = globals()['profile']

    qsb = qsb[0, :, 0]

    conn = state.field.get_econn(eterm.get_dof_conn_type(), eterm.region)
    dofs_vec = state().reshape((-1, state.n_components))
    # axis 0: cells, axis 1: node, axis 2: component
    dofs1 = dofs_vec[conn]
    # axis 0: cells, axis 1: component, axis 2: node
    dofs2 = dofs_vec[conn].transpose((0, 2, 1))

    """
    eval_u_*() -> GEMM
    eval_gu_*() -> no BLAS
    """

    @profile
    def eval_u_1():
        out = oe.contract('qn,cnk->cqk', qsb, dofs1,
                          optimize='dynamic-programming')
        return out, 0

    @profile
    def eval_u_2():
        out = oe.contract('qn,ckn->cqk', qsb, dofs2,
                          optimize='dynamic-programming')
        return out, 0

    @profile
    def eval_u_1q():
        out = oe.contract('qn,cnk->ckq', qsb, dofs1,
                          optimize='dynamic-programming')
        return out, 0

    @profile
    def eval_u_2q():
        out = oe.contract('qn,ckn->ckq', qsb, dofs2,
                          optimize='dynamic-programming')
        return out, 0

    @profile
    def eval_gu_1():
        out = oe.contract('cqjn,cnk->cqjk', qsbg, dofs1,
                          optimize='dynamic-programming')
        return out, 0

    @profile
    def eval_gu_2():
        out = oe.contract('cqjn,ckn->cqjk', qsbg, dofs2,
                          optimize='dynamic-programming')
        return out, 0

    @profile
    def eval_gu_1q():
        out = oe.contract('cqjn,cnk->cjkq', qsbg, dofs1,
                          optimize='dynamic-programming')
        return out, 0

    @profile
    def eval_gu_2q():
        out = oe.contract('cqjn,ckn->cjkq', qsbg, dofs2,
                          optimize='dynamic-programming')
        return out, 0

    evaluators = {
        'u_1' : (eval_u_1, 0, oe),
        'u_2' : (eval_u_2, 0, oe),
        'u_1q' : (eval_u_1q, 0, oe),
        'u_2q' : (eval_u_2q, 0, oe),
        'gu_1' : (eval_gu_1, 0, oe),
        'gu_2' : (eval_gu_2, 0, oe),
        'gu_1q' : (eval_gu_1q, 0, oe),
        'gu_2q' : (eval_gu_2q, 0, oe),
    }

    return evaluators

def get_evals_sfepy(options, term, eterm,
                    dets, qsb, qsbg, qvb, qvbg, state, adc):
    if not options.mprof:
        def profile(fun):
            return fun

    else:
        profile = globals()['profile']


    backends = {
        'numpy' : ['greedy', 'optimal'],
        'numpy_loop' : ['greedy', 'optimal'],
        'opt_einsum'
        : ['dp:flops', 'dp:size', 'greedy', 'branch-2', 'auto', 'optimal'],
        'opt_einsum_loop'
        : ['dp:flops', 'dp:size', 'greedy', 'branch-2', 'auto', 'optimal'],
        'jax' : ['greedy', 'optimal'],
        'jax_vmap' : ['greedy', 'optimal'],
        'dask_single' : ['greedy', 'optimal'],
        'dask_threads' : ['greedy', 'optimal'],
    }
    layouts = ['cqijd0', 'cqdij0', 'ijd0cq', 'dji0cq', 'ijd0qc', 'dji0qc']

    abbrevs = {
        'numpy' : 'np',
        'numpy_loop' : 'npl',
        'opt_einsum' : 'oe',
        'opt_einsum_loop' : 'oel',
        'jax' : 'jx',
        'jax_vmap' : 'jxv',
        'dask_single' : 'das',
        'dask_threads' : 'dat',
        'greedy' : 'gre',
        'optimal' : 'opt',
        'dp:flops' : 'dpf',
        'dp:size' : 'dps',
        'branch-2' : 'br2',
        'auto' : 'aut',
    }

    evaluators = {
    }

    @profile
    def eval_sfepy_term():
        return term.evaluate(mode=options.eval_mode,
                             diff_var=options.diff,
                             standalone=False, ret_status=True)

    evaluators['sfepy_term'] =  (eval_sfepy_term, 0, True)

    def _make_evaluator(backend, optimize, layout, name):
        def _eval_eterm():
            if 'threads' in backend:
                this = psutil.Process()
                affinity = this.cpu_affinity()
                this.cpu_affinity([])
            eterm.set_backend(backend=backend, optimize=optimize, layout=layout)
            out = eterm.evaluate(mode=options.eval_mode,
                                 diff_var=options.diff,
                                 standalone=False, ret_status=True)
            return out
            if 'threads' in backend:
                this.cpu_affinity(affinity)
        _eval_eterm.__name__ = name
        _eval_eterm = profile(_eval_eterm)

        return _eval_eterm

    can = terms_multilinear.ETermBase.can_backend
    for backend, optimizes in backends.items():
        for optimize, layout in product(optimizes, layouts):
            name = 'eval_eterm_{}_{}_{}'.format(abbrevs[backend],
                                                abbrevs[optimize],
                                                layout)
            if ':' in optimize:
                _, minimize = optimize.split(':')
                optimize = oe.DynamicProgramming(minimize=minimize)

            fun = _make_evaluator(backend, optimize, layout, name)
            evaluators[name[5:]] = (fun, 0, can[backend])

    @profile
    def eval_eterm_oe_dpf_das(c_chunk_size=10):
        eterm.set_backend(
            backend='opt_einsum_dask_single',
            optimize='dynamic-programming',
            c_chunk_size=c_chunk_size,
        )
        return eterm.evaluate(mode=options.eval_mode,
                              diff_var=options.diff,
                              standalone=False, ret_status=True)

    @profile
    def eval_eterm_oe_dpf_dat(c_chunk_size=10):
        this = psutil.Process()
        affinity = this.cpu_affinity()
        this.cpu_affinity([])
        eterm.set_backend(
            backend='opt_einsum_dask_threads',
            optimize='dynamic-programming',
            c_chunk_size=c_chunk_size,
        )
        out = eterm.evaluate(mode=options.eval_mode,
                             diff_var=options.diff,
                             standalone=False, ret_status=True)
        this.cpu_affinity(affinity)
        return out

    evaluators.update({
        'eterm_oe_dpf_das' : (eval_eterm_oe_dpf_das, 0, oe and da),
        'eterm_oe_dpf_dat' : (eval_eterm_oe_dpf_dat, 0, oe and da),
    })

    return evaluators

def run_evaluator(key, fun, arg_no, can_use, options, timer,
                  ref_res=None):
    output(key)
    stats = {}
    times = stats.setdefault('t_' + key, [])
    norms = stats.setdefault('norm_' + key, [])
    rnorms = stats.setdefault('rnorm_' + key, [])
    for ir in range(options.repeat):
        timer.start()
        res = fun()[arg_no]
        times.append(timer.stop())
        output('result shape:', res.shape)
        res = res.reshape(-1)
        if ref_res is None:
            ref_res = res
        norms.append(nm.linalg.norm(res))
        rnorms.append(nm.linalg.norm(res - ref_res))
        output('|result|: {} ({:.1e}) in {} s'
               .format(norms[-1], rnorms[-1], times[-1]))
        del res
        gc.collect()

    return stats, ref_res

helps = {
    'output_dir'
    : 'output directory',
    'n_cell'
    : 'the number of cells [default: %(default)s]',
    'refine'
    : 'refine a single cell to test hanging nodes',
    'order'
    : 'displacement field approximation order [default: %(default)s]',
    'quad_order'
    : 'quadrature order [default: 2 * approximation order]',
    'term_name'
    : 'the sfepy term to time [default: %(default)s]',
    'eval_mode'
    : 'the term evaluation mode [default: %(default)s]',
    'variant'
    : 'the term variant [default: %(default)s]',
    'layout'
    : 'the term argument arrays layout [default: %(default)s]',
    'diff'
    : 'if given, differentiate w.r.t. this variable [default: %(default)s]',
    'select'
    : ' evaluation functions selection [default: %(default)s]',
    'repeat'
    : 'the number of term implementation evaluations [default: %(default)s]',
    'micro'
    : 'evaluate micro-functions instead of terms',
    'mprof'
    : 'indicates a run under memory_profiler',
    'affinity'
    : ' CPU affinity [default: %(default)s]',
    'max_mem'
    : ' max. memory estimate in qsbg size units [default: %(default)s]',
    'verbosity_eterm'
    : ' ETermBase verbosity level [default: %(default)s]',
    'silent'
    : 'do not print messages to screen',
    'debug'
    : 'automatically start debugger when an exception is raised',
}

def main():
    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help=helps['output_dir'])
    parser.add_argument('--n-cell', metavar='int', type=int,
                        action='store', dest='n_cell',
                        default=100, help=helps['n_cell'])
    parser.add_argument('--refine',
                        action='store_true', dest='refine',
                        default=False, help=helps['refine'])
    parser.add_argument('--order', metavar='int', type=int,
                        action='store', dest='order',
                        default=1, help=helps['order'])
    parser.add_argument('--quad-order', metavar='int', type=int,
                        action='store', dest='quad_order',
                        default=None, help=helps['quad_order'])
    parser.add_argument('-t', '--term-name',
                        action='store', dest='term_name',
                        choices=['dw_convect', 'dw_laplace', 'dw_volume_dot',
                                 'dw_div', 'dw_stokes', 'dw_lin_elastic'],
                        default='dw_convect', help=helps['term_name'])
    parser.add_argument('--eval-mode',
                        action='store', dest='eval_mode',
                        choices=['weak', 'eval'],
                        default='weak', help=helps['eval_mode'])
    parser.add_argument('--variant',
                        action='store', dest='variant',
                        choices=[None, '', 'scalar', 'vector',
                                 'scalar-material', 'vector-material',
                                 'div', 'grad'],
                        default=None, help=helps['variant'])
    parser.add_argument('--layout',
                        action='store', dest='layout',
                        choices=['C', 'F'],
                        default='C', help=helps['layout'])
    parser.add_argument('--diff',
                        metavar='variable name',
                        action='store', dest='diff',
                        default=None, help=helps['diff'])
    parser.add_argument('--select',
                        metavar='functions list',
                        action='store', dest='select',
                        default='all', help=helps['select'])
    parser.add_argument('--repeat', metavar='int', type=int,
                        action='store', dest='repeat',
                        default=1, help=helps['repeat'])
    parser.add_argument('--micro',
                        action='store_true', dest='micro',
                        default=False, help=helps['micro'])
    parser.add_argument('--mprof',
                        action='store_true', dest='mprof',
                        default=False, help=helps['mprof'])
    parser.add_argument('--affinity',
                        action='store', dest='affinity',
                        default='', help=helps['affinity'])
    parser.add_argument('--max-mem',
                        action='store', dest='max_mem',
                        default='total=20', help=helps['max_mem'])
    parser.add_argument('--verbosity-eterm', type=int,
                        action='store', dest='verbosity_eterm',
                        choices=[0, 1, 2, 3],
                        default=0, help=helps['verbosity_eterm'])
    parser.add_argument('--silent',
                        action='store_true', dest='silent',
                        default=False, help=helps['silent'])
    parser.add_argument('--debug',
                        action='store_true', dest='debug',
                        default=False, help=helps['debug'])
    options = parser.parse_args()

    if options.debug:
        from sfepy.base.base import debug_on_error; debug_on_error()

    if options.quad_order is None:
        options.quad_order = 2 * options.order

    if options.variant is None:
        options.variant = ''

    options.select = so.parse_as_list(options.select, free_word=True)
    options.affinity = so.parse_as_list(options.affinity)
    options.max_mem = so.parse_as_dict(options.max_mem, free_word=True)

    output_dir = options.output_dir
    output.prefix = 'time_tensors:'
    filename = os.path.join(output_dir, 'output_log.txt')
    ensure_path(filename)
    output.set_output(filename=filename, combined=options.silent == False)

    filename = os.path.join(output_dir, 'options.txt')
    save_options(filename, [('options', vars(options))],
                 quote_command_line=True)

    output('numpy:', nm.__version__)
    output('opt_einsum:', oe.__version__ if oe is not None else 'not available')
    output('dask:', dask.__version__ if da is not None else 'not available')
    output('jax:', jax.__version__ if jnp is not None else 'not available')

    to_mb = lambda x: x / 1000**2

    mem = psutil.virtual_memory()
    output('total system memory [MB]: {:.2f}'.format(to_mb(mem.total)))
    output('available system memory [MB]: {:.2f}'
           .format(mem.available / 1000**2))

    # Estimate qsbg size.
    ps = PolySpace.any_from_args('ps', GeometryElement('3_8'), options.order)
    integral = Integral('i', order=options.quad_order)
    _, weights = integral.get_qp('3_8')

    n_cell = options.n_cell
    n_qp = len(weights)
    n_en = ps.n_nod
    dim = 3

    qsbg_shape = (n_cell, n_qp, dim, n_en)
    qsbg_size = nm.prod(qsbg_shape) * 8
    output('qsbg assumed size [MB]: {:.2f}'.format(to_mb(qsbg_size)))

    max_mem = options.max_mem['total']
    output('memory estimate [qsbg size]:', max_mem)
    mem_est = max_mem * qsbg_size
    output('memory estimate [MB]: {:.2f}'.format(mem_est / 1000**2))

    if mem_est > mem.available:
        raise MemoryError('insufficient memory for timing!'
                          ' ({:.2f} [MB] > {:.2f} [MB])'
                          .format(to_mb(mem_est), to_mb(mem.available)))

    if options.term_name not in ['dw_stokes']:
        uvec, term, eterm = setup_data(
            order=options.order,
            quad_order=options.quad_order,
            n_cell=options.n_cell,
            term_name=options.term_name,
            eval_mode=options.eval_mode,
            variant=options.variant,
            refine=options.refine,
        )

    else:
        uvec, term, eterm = setup_data_mixed(
            order1=options.order+1, order2=options.order,
            quad_order=options.quad_order,
            n_cell=options.n_cell,
            term_name=options.term_name,
            eval_mode=options.eval_mode,
            variant=options.variant,
            refine=options.refine,
        )

    eterm.verbosity = options.verbosity_eterm

    timer = Timer('')
    timer.start()
    vg, geo = term.get_mapping(term.args[-1])
    output('reference element mapping: {} s'.format(timer.stop()))

    dets = vg.det
    dim = vg.dim

    state = term.args[-1]
    dc_type = term.get_dof_conn_type()
    # Assumes no E(P)BCs are present!
    adc = state.get_dof_conn(dc_type)

    n_cell, n_qp, dim, n_en, n_c = term.get_data_shape(state)
    n_cdof = n_c * n_en

    output('u shape:', uvec.shape)
    output('adc shape:', adc.shape)
    output('u size [MB]:', to_mb(uvec.nbytes))
    output('adc size [MB]:', to_mb(adc.nbytes))

    qsb = vg.bf
    qsbg = vg.bfg

    output('qsbg shape:', qsbg.shape)
    output('qvbg shape:', (n_cell, n_qp, n_c, dim, dim * n_en))

    size = (n_cell * n_qp * n_c * dim * dim * n_en) * 8
    output('qvbg assumed size [MB]:', to_mb(size))

    if options.term_name == 'dw_convect':
        qvb = expand_basis(qsb, dim)
        qvbg = _expand_sbg(qsbg, dim)

    else:
        qvb = nm.zeros(0)
        qvbg = nm.zeros(0)

    output('qsbg size [MB]:', to_mb(qsbg.nbytes))
    output('qvbg size [MB]:', to_mb(qvbg.nbytes))

    vec_shape = (n_cell, n_cdof)
    mtx_shape = (n_cell, n_cdof, n_cdof)
    output('c vec shape:', vec_shape)
    output('c mtx shape:', mtx_shape)
    output('c vec size [MB]:', to_mb(n_cell * n_cdof * 8))
    output('c mtx size [MB]:', to_mb(n_cell * n_cdof**2 * 8))

    pid = os.getpid()
    this = psutil.Process(pid)
    this.cpu_affinity(options.affinity)
    mem_this = this.memory_info()
    memory_use = mem_this.rss
    output('memory use [MB]: {:.2f}'.format(to_mb(memory_use)))
    output('memory use [qsbg size]: {:.2f}'.format(memory_use / qsbg_size))

    if options.micro:
        evaluators = get_evals_micro(
            options, term, eterm, dets, qsb, qsbg, qvb, qvbg, state, adc
        )

    else:
        evaluators = get_evals_sfepy(
            options, term, eterm, dets, qsb, qsbg, qvb, qvbg, state, adc
        )

        if options.term_name == 'dw_convect':
            evaluators.update(get_evals_dw_convect(
                options, term, eterm, dets, qsb, qsbg, qvb, qvbg, state, adc
            ))

        elif options.term_name == 'dw_laplace':
            evaluators.update(get_evals_dw_laplace(
                options, term, eterm, dets, qsb, qsbg, qvb, qvbg, state, adc
            ))

    if options.select[0] == 'all':
        options.select = list(evaluators.keys())

    all_stats = {}

    if not options.micro:
        key = 'sfepy_term'
        fun, arg_no, can_use = evaluators.pop(key)
        stats, ref_res = run_evaluator(key, fun, arg_no, can_use, options,
                                       timer)
        all_stats.update(stats)

        if options.layout == 'F':
            for iv, arg in enumerate(eterm.args):
                if isinstance(arg, FieldVariable):
                    ag, _ = eterm.get_mapping(arg)
                    ag.det = nm.require(ag.det, requirements='F')
                    ag.bfg = nm.require(ag.bfg, requirements='F')

                else:
                    if arg[0] is not None:
                        key = (eterm.region.name, eterm.integral.order)
                        mat = arg[0].get_data(key, arg[1])
                        arg[0].datas[key][arg[1]] = nm.require(mat,
                                                               requirements='F')
    else:
        ref_res = 0

    select_match = re.compile('|'.join(options.select)).match
    mem_matches = [re.compile(key).match for key in options.max_mem.keys()]
    max_mems = list(options.max_mem.values())
    for key, (fun, arg_no, can_use) in evaluators.items():
        if not can_use: continue
        if select_match(key) is None: continue

        skip = False
        for im, match in enumerate(mem_matches):
            if match(key):
                max_mem = max_mems[im]

                fun_mem_est = max_mem * qsbg_size
                output('{} memory estimate [MB]: {:.2f}'
                       .format(key, fun_mem_est / 1000**2))
                if fun_mem_est > mem.total:
                    output('-> skipping!')
                    skip = True
                    break

        if skip: continue

        try:
            stats, _ = run_evaluator(key, fun, arg_no, can_use, options, timer,
                                     ref_res=ref_res)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            if options.debug:
                raise

            else:
                output('{} failed with:'.format(key))
                output(exc)

        else:
            all_stats.update(stats)

    df = pd.DataFrame(all_stats)
    df.index.rename('evaluation', inplace=True)

    filename = os.path.join(options.output_dir, 'stats.csv')
    df.to_csv(filename)

if __name__ == '__main__':
    main()
