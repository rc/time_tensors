#!/usr/bin/env python
"""
fenics also assembles -> compare with full problem.evaluate()!
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os.path as op
import psutil
import numpy as nm
from functools import partial
import gc

try:
    import fenics as fe
    from ffc.fiatinterface import create_quadrature as cquad

except:
    pass

from sfepy.base.base import output
from sfepy.discrete.fem import FEDomain, Field
from sfepy.discrete import (FieldVariable, Integral, Equation, Equations,
                            Problem)
from sfepy.terms import Term
from sfepy.mesh.mesh_generators import gen_block_mesh

import soops as so
from soops.base import product
from soops.timing import Timer
import soops.plot_selected as sps

try:
    _ = profile

except NameError:
    profile = lambda x: x

def get_run_info():
    # script_dir is added by soops-run, it is the normalized path to
    # this script.
    run_cmd = """
    rm {output_dir}/mprofile.dat; mprof run -T {sampling} -C -o {output_dir}/mprofile.dat try_packages.py {output_dir}
    """
    run_cmd = ' '.join(run_cmd.split())

    # Arguments allowed to be missing in soops-run calls.
    opt_args = {
        '--package' : '--package={--package}',
        '--form' : '--form={--form}',
        '--n-cell' : '--n-cell={--n-cell}',
        '--order' : '--order={--order}',
        '--repeat' : '--repeat={--repeat}',
        '--affinity' : '--affinity={--affinity}',
        '--silent' : '--silent',
    }

    output_dir_key = 'output_dir'

    def is_finished(pars, options):
        output_dir = pars[output_dir_key]
        filename = op.join(output_dir, 'output_log.txt')
        ok = op.exists(filename)
        if ok:
            with open(filename, 'r') as fd:
                for line in fd:
                    if 'times:' in line:
                        break

                else:
                    ok = False

        return ok

    return run_cmd, opt_args, output_dir_key, is_finished

def get_scoop_info():
    import soops.scoop_outputs as sc

    info = [
        ('options.txt', partial(
            sc.load_split_options,
            split_keys=None,
        ), True),
        ('mprofile.dat', load_mprofile),
        ('output_log.txt', scrape_output),
    ]

    return info

def load_mprofile(filename, rdata=None):
    from mprof import read_mprofile_file

    mdata = read_mprofile_file(filename)
    mdata.pop('children')
    mdata.pop('cmd_line')
    mdata['mem_usage'] = nm.array(mdata['mem_usage'])
    mdata['timestamp'] = nm.array(mdata['timestamp'])
    mdata['func_timestamp'] = {key.split('_')[-2] : val
                               for key, val in mdata['func_timestamp'].items()}

    return mdata

def scrape_output(filename, rdata=None):
    import soops.ioutils as io
    from ast import literal_eval

    out = {}
    with open(filename, 'r') as fd:
        out['t'] = []
        for ir in range(rdata['repeat']):
            line = io.skip_lines_to(fd, 'repeat:')
            if line:
                line = line.split(':')[2].strip().split()

            else:
                break

            out['t'].append(literal_eval(line[-1]))
            out['mtx_size'] = literal_eval(line[-2])

        if len(out['t']) < rdata['repeat']:
            out['t'].extend([nm.nan] * (rdata['repeat'] - len(out['t'])))

    return out

def get_plugin_info():
    from soops.plugins import show_figures

    info = [
        collect_stats,
        get_ratios,
        plot_results,
        show_figures,
    ]

    return info

def _get_mem(drow):
    mu = drow['mem_usage']
    tss = drow['timestamp']
    ts = drow['func_timestamp'].get(drow['package'])
    if ts is None:
        return nm.nan

    ts = ts[0]

    i0, i1 = nm.searchsorted(tss, ts[:2])
    if i1 > i0:
        mmax = max(mu[i0:i1].max(), ts[3])
        mmin = min(mu[i0:i1].min(), ts[2])

    else:
        mmax = ts[3]
        mmin = ts[2]

    mem = mmax - mmin
    return mem

def collect_stats(df, data=None):
    import time_tensors as tt

    stat_keys = ('mean', 'min', 'max', 'emin', 'emax', 'wwmean')
    for key, val in zip(['t' + ii for ii in stat_keys],
                        tt.get_stats(df, 't')):
        df[key] = val

    if 'func_timestamp' in df:
        df['mem'] = df.apply(_get_mem, axis=1)

    return data

def get_ratios(df, data=None, term_names=None):
    import pandas as pd
    import soops.formatting as sof

    _format_ratios = partial(sof.format_float_latex, prec='5.2f')

    tn2key = {
        'dw_laplace::u' : 'Laplacian',
        'dw_volume_dot::u' : 'scalar dot',
        'dw_volume_dot:v:u' : 'vector dot',
        'dw_convect::u' : 'NS convective',
    }
    st2key = {
        'twwmean'
        : r'med($\bar T^{\rm ww}_{\rm sfepy} / \bar T^{\rm ww}_{\rm fenics}$)',
        'mem'
        : r'med($M^{\rm max}_{\rm sfepy} / M^{\rm max}_{\rm fenics}$)',
    }

    if term_names is None:
        term_names = data.par_uniques['form']
    orders = data.par_uniques['order']
    packages = data.par_uniques['package']
    p0, p1  = packages
    ratios = {}
    for key in ['twwmean', 'mem']:
        if key not in df: continue
        ratio = ratios.setdefault(st2key[key], {})
        for term_name, order in product(
                term_names, orders,
        ):
            sdf = df[(df['form'] == term_name) &
                     (df['order'] == order)]
            if not len(sdf): continue

            _ratios = sdf.groupby('n_cell')[key].apply(
                lambda x: x.iloc[1] / x.iloc[0]
            )
            ratio[tn2key[term_name], order] = _ratios.median()

    rdf = data.rdf = pd.DataFrame.from_dict(ratios)

    indir = partial(op.join, data.output_dir)
    filename = indir('table-packages-ratios.inc')
    with pd.option_context("max_colwidth", 1000):
        rdf.T.to_latex(filename, index=True, escape=False, na_rep='-',
                       columns=data.rdf.T.columns.sort_values(),
                       float_format=_format_ratios)

    return data

def _format_labels(key, iv, val, tn2key=None):
    if key == 'form':
        return tn2key[val]

    else:
        return val

def plot_results(df, data=None, term_names=None, prefix='', suffix='.png'):
    import matplotlib.pyplot as plt
    indir = partial(op.join, data.output_dir)

    plt.rcParams.update({
        'text.usetex' : True,
        'font.size' : 14.0,
        'legend.fontsize' : 12.0
    })

    twwmean_label = r'$\bar T^{\rm ww}$ [s]'
    mem_label = r'$M^{\rm max}$ [MB]'

    marker_style = {
        'lw' : 1,
        'mew' : 1.0,
        'marker' : ['o', '^', 'v', 'D', 's'],
        'alpha' : 1.0,
        'mfc' : 'None',
        'markersize' : 8,
    }

    xscale = 'log'
    yscale = 'log'

    tn2key = {
        'dw_laplace::u' : 'Laplacian',
        'dw_volume_dot::u' : 'scalar dot',
        'dw_volume_dot:v:u' : 'vector dot',
        'dw_convect::u' : 'NS convective',
    }
    if term_names is None:
        term_names = list(tn2key.keys())

    is_form_legend = len(term_names) > 1

    # Omit runs with no timing stats.
    df = df.dropna(subset=['twwmean'])

    ylabels = {'twwmean' : twwmean_label, 'mem' : mem_label}
    orders = data.par_uniques['order']
    packages = data.par_uniques['package']
    for key in ['twwmean', 'mem']:
        if key not in df: continue
        fig, ax = plt.subplots()

        select = sps.select_by_keys(df, ['order', 'package'])
        if is_form_legend:
            select.update({'form' : term_names})
        styles = {'form' : marker_style,
                  'order' : {'color' : 'tab10:kind=qualitative',},
                  'package' : {'ls' : ['--', '-'],}}
        if not is_form_legend:
            styles['package'].update({
                'marker' : ['x', 'o'],
                'lw' : 1,
                'mew' : 1.0,
                'mfc' : 'None',
                'markersize' : 8,
            })
        styles = sps.setup_plot_styles(select, styles)

        ax.grid(True)
        used = None
        maxs = {ii : (0, 0) for ii in select['order']}
        for term_name, order, package in product(
                term_names, orders, packages,
        ):
            sdf = df[(df['form'] == term_name) &
                     (df['order'] == order) &
                     (df['package'] == package)]
            if not len(sdf): continue

            style_kwargs, indices, used = sps.get_row_style_used(
                sdf.iloc[0], select, {}, styles, used
            )
            vx = sdf.n_cell.values
            means = sdf[key].values
            ax.plot(vx, means, **style_kwargs)

            imax = sdf[key].idxmax()
            if nm.isfinite(imax) and (sdf.loc[imax, key] > maxs[order][1]):
                maxs[order] = (sdf.loc[imax, 'n_cell'], sdf.loc[imax, key])

        sps.add_legend(ax, select, styles, used, per_parameter=False,
                       format_labels=partial(_format_labels, tn2key=tn2key),
                       loc='best',
                       frame_alpha=0.8, ncol=1,
                       handlelength=2, handletextpad=0.4, columnspacing=0.2,
                       labelspacing=0.2)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(r'\#cells')
        ax.set_ylabel(ylabels[key])

        for order, (mx, my) in maxs.items():
            fmt = '{:.2f}' if my < 1 else '{:.1f}'
            ax.annotate(fmt.format(my), xy=(mx, my), xytext=(-5, 15),
                        textcoords='offset points',
                        arrowprops=dict(facecolor='black',
                                        arrowstyle='->',
                                        shrinkA=0,
                                        shrinkB=0))

        plt.tight_layout()
        figname = ('packages-{}-{}{}'.format(prefix, key, suffix))
        fig = ax.figure
        fig.savefig(indir(figname), bbox_inches='tight')

def print_fenics_n_qp():
    shape = 'hexahedron'
    scheme = 'default'

    for deg in range(1, 12):
        points, weights = cquad(shape, deg, scheme)
        print('degree:', deg, 'n_qp:', len(points))


def get_nc(form):
    nc = 3 if (':v' in form) or (form in {'dw_convect::u',}) else 1
    return nc

@profile
def assemble_sfepy_form(form, n_cell, order, repeat, eterm_options=None):
    mesh = gen_block_mesh((n_cell, 1, 1), (n_cell + 1, 2, 2), (0, 0, 0),
                          name='')
    domain = FEDomain('el', mesh)
    omega = domain.create_region('omega', 'all')

    nc = get_nc(form)
    field = Field.from_args('fu', nm.float64, nc, omega,
                            approx_order=order)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    if form == 'dw_convect::u':
        u.set_constant(1.0)

    form = form.split(':')[0]

    timer = Timer()
    times = []
    for ir in range(repeat):
        timer.start()
        integral = Integral('i', order=0)
        term = Term.new('dw_zero(v, u)',
                        integral=integral, region=omega, v=v, u=u)
        eq = Equation('eq', term)
        eqs = Equations([eq])

        pb = Problem('pb', equations=eqs)
        mtx = pb.evaluate('{}.{}.omega(v, u)'.format(form, 2 * order),
                          mode='weak', dw_mode='matrix',
                          eterm_options=eterm_options)
        times.append(timer.stop())
        output('repeat:', ir, mtx.shape[0], times[-1])
        del mtx
        gc.collect()

    return times

@profile
def assemble_fenics_form(form, n_cell, order, repeat):
    mesh = fe.BoxMesh.create([fe.Point(0,0,0), fe.Point(n_cell, 1, 1)],
                             [n_cell, 1, 1],
                             fe.CellType.Type.hexahedron)

    nc = get_nc(form)
    if nc > 1:
        V = fe.VectorFunctionSpace(mesh, 'Lagrange', order)

    else:
        V = fe.FunctionSpace(mesh, 'Lagrange', order)

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    if form == 'dw_convect::u':
        u0 = fe.Function(V)

    fcc_pars = {
        'quadrature_degree': 2 * order,
        'optimize' : True,
        'log_level' : 1
    }

    timer = Timer()
    times = []
    for ir in range(repeat):
        timer.start()
        if form == 'dw_laplace::u':
            term = fe.dot(fe.grad(u), fe.grad(v))*fe.dx

        elif form == 'dw_volume_dot::u':
            term = fe.dot(u, v)*fe.dx

        elif form == 'dw_volume_dot:v:u':
            term = fe.dot(u, v)*fe.dx

        elif form == 'dw_convect::u':
            term = (fe.inner(fe.grad(u0)*u, v)*fe.dx +
                    fe.inner(fe.grad(u)*u0, v)*fe.dx)

        mtx = fe.assemble(term, form_compiler_parameters=fcc_pars)
        times.append(timer.stop())
        output('repeat:', ir, mtx.size(0), times[-1])
        del mtx
        gc.collect()

    return times

helps = {
    'output_dir'
    : 'output directory',
    'silent'
    : 'do not print messages to screen',
    'shell'
    : 'run ipython shell after all computations',
}

def main():
    opts = so.Struct(
        package = ('sfepy', 'fenics'),
        form = ('dw_laplace::u', 'dw_dot::u', 'dw_dot:v:u', 'dw_convect::u',
                'de_laplace::u', 'de_dot::u', 'de_dot:v:u', 'de_convect::u'),
        eterm_options = ("verbosity=0, backend_args={backend='numpy', "
                         "optimize='optimal', layout=None}"),
        n_cell = 1024,
        order = 1,
        repeat = 2,
        affinity = '',
    )
    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help=helps['output_dir'])
    for key, val in opts.items():
        helps[key] = '[default: %(default)s]'
        action = 'store'
        choices = None
        if val is True:
            action = 'store_false'

        elif val is False:
            action = 'store_true'

        elif isinstance(val, tuple):
            choices = val
            val = val[0]

        if action == 'store':
            parser.add_argument('--' + key.replace('_', '-'),
                                type=type(val),
                                action=action, dest=key, choices=choices,
                                default=val, help=helps[key])
        else:
            parser.add_argument('--' + key.replace('_', '-'),
                                action=action, dest=key,
                                default=val, help=helps[key])
    parser.add_argument('--silent',
                        action='store_true', dest='silent',
                        default=False, help=helps['silent'])
    parser.add_argument('--shell',
                        action='store_true', dest='shell',
                        default=False, help=helps['shell'])
    options = parser.parse_args()

    options.affinity = so.parse_as_list(options.affinity)
    options.eterm_options = so.parse_as_dict(options.eterm_options)

    output_dir = options.output_dir
    output.prefix = 'try_packages:'
    filename = op.join(output_dir, 'output_log.txt')
    so.ensure_path(filename)
    output.set_output(filename=filename, combined=options.silent == False)

    filename = op.join(output_dir, 'options.txt')
    so.save_options(filename, [('options', vars(options))],
                    quote_command_line=True)

    this = psutil.Process()
    this.cpu_affinity(options.affinity)

    if options.package == 'sfepy':
        times = assemble_sfepy_form(options.form, options.n_cell,
                                    options.order, options.repeat,
                                    eterm_options=options.eterm_options)

    elif options.package == 'fenics':
        fe.set_log_active(False)

        print_fenics_n_qp()
        times = assemble_fenics_form(options.form, options.n_cell,
                                     options.order, options.repeat)

    output('times:', times)
    if options.shell:
        from soops.base import shell; shell()

if __name__ == '__main__':
    main()
