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
import psutil

from functools import partial
import gc

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

import pandas as pd

from sfepy.base.base import output, Struct
from sfepy.base.ioutils import ensure_path, save_options
from sfepy.base.timing import Timer
from sfepy.discrete.variables import expand_basis

def get_run_info():
    # script_dir is added by soops-run, it is the normalized path to
    # this script.
    run_cmd = """
    {python} time_tensors.py {output_dir}
    """
    run_cmd = ' '.join(run_cmd.split())

    # Arguments allowed to be missing in soops-run calls.
    opt_args = {
        '--n-cell' : '--n-cell={--n-cell}',
        '--order' : '--order={--order}',
        '--quad-order' : '--quad-order={--quad-order}',
        '--term-name' : '--term-name={--term-name}',
        '--diff' : '--diff={--diff}',
        '--repeat' : '--repeat={--repeat}',
        '--mprof' : '--mprof',
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
        )),
        ('stats.csv', sc.load_csv),
        ('output_log.txt', scrape_output),
    ]

    return info

def scrape_output(filename, rdata=None):
    out = {}

    return out

def get_plugin_info():
    from soops.plugins import show_figures

    info = [show_figures]

    return info

def get_v_sol(coors):
    x0 = coors.min(axis=0)
    x1 = coors.max(axis=0)
    dims = x1 - x0

    cc = (coors - x0) / dims[None, :]
    return cc

def _expand_sbg(basis, dpn):
    dim, n_ep = basis.shape[-2:]
    vg = nm.zeros(basis.shape[:2] + (dpn, dim, dim * n_ep))
    for ir in range(dpn):
        vg[..., ir, :, n_ep*ir:n_ep*(ir+1)] = basis
    return vg

def setup_data(order, quad_order, n_cell, term_name='dw_convect'):
    from sfepy.discrete.fem import FEDomain, Field
    from sfepy.discrete import (FieldVariable, Integral)
    from sfepy.terms import Term
    from sfepy.mesh.mesh_generators import gen_block_mesh

    integral = Integral('i', order=quad_order)

    mesh = gen_block_mesh((n_cell, 1, 1), (n_cell + 1, 2, 2), (0, 0, 0),
                          name='')
    domain = FEDomain('el', mesh)

    omega = domain.create_region('omega', 'all')

    field = Field.from_args('fu', nm.float64, mesh.dim, omega,
                            approx_order=order)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    u.set_from_function(get_v_sol)
    uvec = u()

    if term_name == 'dw_convect':
        term = Term.new('dw_convect(v, u)', integral=integral,
                        region=omega, v=v, u=u)

    term.setup()
    term.standalone_setup()

    return uvec, term

helps = {
    'output_dir'
    : 'output directory',
    'n_cell'
    : 'the number of cells [default: %(default)s]',
    'order'
    : 'displacement field approximation order [default: %(default)s]',
    'quad_order'
    : 'quadrature order [default: 2 * approximation order]',
    'term_name'
    : 'the sfepy term to time [default: %(default)s]',
    'diff'
    : 'if given, differentiate w.r.t. this variable [default: %(default)s]',
    'repeat'
    : 'the number of term implementation evaluations [default: %(default)s]',
    'mprof'
    : 'indicates a run under memory_profiler',
    'silent'
    : 'do not print messages to screen',
}

def main():
    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help=helps['output_dir'])
    parser.add_argument('--n-cell', metavar='int', type=int,
                        action='store', dest='n_cell',
                        default=100, help=helps['n_cell'])
    parser.add_argument('--order', metavar='int', type=int,
                        action='store', dest='order',
                        default=1, help=helps['order'])
    parser.add_argument('--quad-order', metavar='int', type=int,
                        action='store', dest='quad_order',
                        default=None, help=helps['quad_order'])
    parser.add_argument('-t', '--term-name',
                        action='store', dest='term_name',
                        choices=['dw_convect'],
                        default='dw_convect', help=helps['term_name'])
    parser.add_argument('--diff',
                        metavar='variable name',
                        action='store', dest='diff',
                        default=None, help=helps['diff'])
    parser.add_argument('--repeat', metavar='int', type=int,
                        action='store', dest='repeat',
                        default=1, help=helps['repeat'])
    parser.add_argument('--mprof',
                        action='store_true', dest='mprof',
                        default=False, help=helps['mprof'])
    parser.add_argument('--silent',
                        action='store_true', dest='silent',
                        default=False, help=helps['silent'])
    options = parser.parse_args()

    if options.quad_order is None:
        options.quad_order = 2 * options.order

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

    uvec, term = setup_data(
        order=options.order,
        quad_order=options.quad_order,
        n_cell=options.n_cell,
        term_name=options.term_name
    )

    vg, geo = term.get_mapping(term.args[1])
    dets = vg.det
    dim = vg.dim

    qsb = vg.bf
    qsbg = vg.bfg

    qvb = expand_basis(qsb, dim)
    qvbg = _expand_sbg(qsbg, dim)

    output('qsbg shape:', qsbg.shape)
    output('qvbg shape:', qvbg.shape)
    output('qsbg size [MB]:', qsbg.nbytes / 1000**2)
    output('qvbg size [MB]:', qvbg.nbytes / 1000**2)

    state = term.args[1]
    dc_type = term.get_dof_conn_type()
    # Assumes no E(P)BCs are present!
    adc = state.get_dof_conn(dc_type)

    n_elc, n_qpc, dim, n_enc, n_cc = term.get_data_shape(state)
    n_cdof = n_cc * n_enc

    output('u shape:', state().shape)
    output('adc shape:', adc.shape)
    output('u size [MB]:', uvec.nbytes / 1000**2)
    output('adc size [MB]:', adc.nbytes / 1000**2)

    vec_shape = (n_elc, n_cdof)
    mtx_shape = (n_elc, n_cdof, n_cdof)
    output('c vec shape:', vec_shape)
    output('c mtx shape:', mtx_shape)
    output('c vec size [MB]:', (n_elc * n_cdof * 8) / 1000**2)
    output('c mtx size [MB]:', (n_elc * n_cdof**2 * 8) / 1000**2)

    pid = os.getpid()
    this = psutil.Process(pid)
    mem_this = this.memory_info()
    memory_use = mem_this.rss
    output('memory use [MB]: {:.2f}'.format(memory_use / 1000**2))
    mem = psutil.virtual_memory()
    output('total system memory [MB]: {:.2f}'.format(mem.total / 1000**2))
    output('available system memory [MB]: {:.2f}'
           .format(mem.available / 1000**2))

    coef = 3 if options.diff is None else 4
    if (coef * memory_use) > mem.total:
        raise MemoryError('insufficient memory for timing!')

    if not options.mprof:
        def profile(fun):
            return fun

    @profile
    def eval_sfepy_term():
        return term.evaluate(mode='weak',
                             diff_var=options.diff,
                             standalone=False, ret_status=True)

    @profile
    def eval_numpy_einsum1():
        # Unusably slow - not using optimize arg of einsum().
        uc = state()[adc]
        return nm.einsum('cqab,qji,cqjkl,cl,qkn,cn->ci',
                         dets, qvb[0], qvbg, uc, qvb[0], uc), 0

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
            # aa = oe.contract_path('cqab,qji,cqjkl,qkn,cn->cil',
            #                       dets, qvb[0], qvbg, qvb[0], uc)
            # bb = oe.contract_path('cqab,qji,cqjkl,cl,qkn->cin',
            #                       dets, qvb[0], qvbg, uc, qvb[0])
            # from sfepy.base.base import debug; debug()
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

    # -> to functions to enable memory profiling
    evaluators = {
        'sfepy_term' : (eval_sfepy_term, 0),
        # 'numpy_einsum1' : (eval_numpy_einsum1, 0), # unusably slow
        'numpy_einsum2' : (eval_numpy_einsum2, 0),
        # 'numpy_einsum3' : (eval_numpy_einsum3, 0), # slow, memory hog
        'opt_einsum1a' : (eval_opt_einsum1a, 0),
        'opt_einsum1g' : (eval_opt_einsum1g, 0),
        'opt_einsum1dp' : (eval_opt_einsum1dp, 0),
        #'opt_einsum2a' : (eval_opt_einsum2a, 0), # more memory than opt_einsum1*
        'opt_einsum2dp' : (eval_opt_einsum2dp, 0), # more memory than opt_einsum1*
        'dask_einsum1' : (eval_dask_einsum1, 0), # how to limit to 1 thread?
        # 'jax_einsum1' : (eval_jax_einsum1, 0), # meddles with memory profiler
    }

    results = {}

    timer = Timer('')

    for key, (fun, arg_no) in evaluators.items():
        output(key)
        times = results.setdefault('t_' + key, [])
        norms = results.setdefault('norm_' + key, [])
        for ir in range(options.repeat):
            timer.start()
            res = fun()[arg_no]
            times.append(timer.stop())
            norms.append(nm.linalg.norm(res.reshape(-1)))
            output('|result|: {} in {} s'.format(norms[-1], times[-1]))
            del res
            gc.collect()
            #results['res_' + key] = res

        # res_term = results.get('res_sfepy_term')
        # output('difference w.r.t. term:',
        #        nm.linalg.norm(res.ravel() - res_term.ravel()))

    df = pd.DataFrame(results)
    df.index.rename('evaluation', inplace=True)

    filename = os.path.join(options.output_dir, 'stats.csv')
    df.to_csv(filename)

if __name__ == '__main__':
    main()
