#!/usr/bin/env python
"""
fenics also assembles -> compare with full problem.evaluate()!
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os.path as op
import numpy as nm
import gc

import fenics as fe
from ffc.fiatinterface import create_quadrature as cquad

from sfepy.base.base import output
from sfepy.discrete.fem import FEDomain, Field
from sfepy.discrete import (FieldVariable, Integral, Equation, Equations,
                            Problem)
from sfepy.terms import Term
from sfepy.mesh.mesh_generators import gen_block_mesh

import soops as so
from soops.timing import Timer

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
        '--silent' : '--silent',
    }

    output_dir_key = 'output_dir'

    return run_cmd, opt_args, output_dir_key, 'output_log.txt'

def print_fenics_n_qp():
    shape = 'hexahedron'
    scheme = 'default'

    for deg in range(1, 12):
        points, weights = cquad(shape, deg, scheme)
        print('degree:', deg, 'n_qp:', len(points))

@profile
def assemble_sfepy_form(form, n_cell, order, repeat):
    mesh = gen_block_mesh((n_cell, 1, 1), (n_cell + 1, 2, 2), (0, 0, 0),
                          name='')
    domain = FEDomain('el', mesh)
    omega = domain.create_region('omega', 'all')
    field = Field.from_args('fu', nm.float64, 1, omega,
                            approx_order=order)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

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
                          mode='weak', dw_mode='matrix')
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

    V = fe.FunctionSpace(mesh, 'Q', order)

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    fcc_pars = {
        'quadrature_degree': 2 * order,
        'optimize' : True,
        'log_level' : 1
    }

    timer = Timer()
    times = []
    for ir in range(repeat):
        timer.start()
        if form == 'dw_laplace':
            term = fe.dot(u, v)*fe.dx

        elif form == 'dw_volume_dot':
            term = fe.dot(fe.grad(u), fe.grad(v))*fe.dx

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
        form = ('dw_laplace', 'dw_volume_dot'),
        n_cell = 1024,
        order = 1,
        repeat = 2,
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

    output_dir = options.output_dir
    output.prefix = 'try_packages:'
    filename = op.join(output_dir, 'output_log.txt')
    so.ensure_path(filename)
    output.set_output(filename=filename, combined=options.silent == False)

    filename = op.join(output_dir, 'options.txt')
    so.save_options(filename, [('options', vars(options))],
                    quote_command_line=True)

    if options.package == 'sfepy':
        times = assemble_sfepy_form(options.form, options.n_cell,
                                    options.order, options.repeat)

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
