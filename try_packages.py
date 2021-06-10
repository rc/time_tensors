"""
fenics also assembles -> compare with full problem.evaluate()!
"""
import numpy as nm

import fenics as fe
from ffc.fiatinterface import create_quadrature as cquad

from sfepy.discrete.fem import FEDomain, Field
from sfepy.discrete import (FieldVariable, Integral, Equation, Equations,
                            Problem)
from sfepy.terms import Term
from sfepy.mesh.mesh_generators import gen_block_mesh

from soops import Struct
from soops.timing import Timer

fe.set_log_active(True)

shape = 'hexahedron'
scheme = 'default'

for deg in range(1, 12):
    points, weights = cquad(shape, deg, scheme)
    print(deg, len(points))

options = Struct(
    #n_cell = 1024,
    #n_cell = 32768,
    n_cell = 262144,
    #n_cell = 1048576,
    order = 1,
    repeat = 2,
)

def assemble_sfepy_form(n_cell, order, repeat):
    mesh = gen_block_mesh((n_cell, 1, 1), (n_cell + 1, 2, 2), (0, 0, 0),
                          name='')
    domain = FEDomain('el', mesh)
    omega = domain.create_region('omega', 'all')
    field = Field.from_args('fu', nm.float64, 1, omega,
                            approx_order=order)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    timer = Timer()
    for ir in range(repeat):
        timer.start()
        integral = Integral('i', order=0)
        term = Term.new('dw_zero(v, u)',
                        integral=integral, region=omega, v=v, u=u)
        eq = Equation('eq', term)
        eqs = Equations([eq])

        pb = Problem('pb', equations=eqs)
        mtx = pb.evaluate('dw_volume_dot.{}.omega(v, u)'.format(2 * order),
                          mode='weak', dw_mode='matrix')
        # mtx = pb.evaluate('dw_laplace.{}.omega(v, u)'.format(2 * order),
        #                   mode='weak', dw_mode='matrix')
        print(timer.stop())
        print(mtx.shape[0])

def assemble_fenics_form(n_cell, order, repeat):
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
    for ir in range(repeat):
        timer.start()
        # term = fe.dot(fe.grad(u), fe.grad(v))*fe.dx
        term = fe.dot(u, v)*fe.dx
        mtx = fe.assemble(term, form_compiler_parameters=fcc_pars)
        print(timer.stop())
        print(mtx.size(0))

assemble_fenics_form(options.n_cell, options.order, options.repeat)
assemble_sfepy_form(options.n_cell, options.order, options.repeat)
