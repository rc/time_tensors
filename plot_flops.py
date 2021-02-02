"""
Plot flops per cell of evaluations of various terms.
"""
import sys
sys.path.append('.')
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as nm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import soops as so
import soops.plot_selected as sps

from sfepy.base.base import Struct
from sfepy.discrete.fem.geometry_element import GeometryElement
from sfepy.discrete import (Integral, PolySpace)
from sfepy.mechanics.tensors import dim2sym
import sfepy.terms.terms_multilinear as tm

def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    """
    Computes the number of FLOPS in the contraction.

    Taken from NumPy.
    """

    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor

def _compute_size_by_dict(indices, idx_dict):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Taken from NumPy.
    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret

def get_naive_cost(ebuilder, operands):
    cost = 0
    for ia, (expr, ops) in enumerate(zip(ebuilder.get_expressions(),
                                         operands)):
        print(expr)
        sizes = ebuilder.get_sizes(ia, operands)
        print(sizes)

        input_sets = [set(val) for val in ebuilder.subscripts[ia]]
        indices = set(''.join(ebuilder.subscripts[ia]))
        is_inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
        _cost = _flop_count(indices, is_inner_product, len(input_sets), sizes)
        cost += _cost

    return cost

helps = {
    'output_dir'
    : 'output directory',
    'orders'
    : 'field approximation orders [default: %(default)s]',
    'quad_orders'
    : 'quadrature orders [default: 2 * approximation orders]',
    'term_names'
    : 'use given terms [default: %(default)s]',
    'diff'
    : 'if given, differentiate w.r.t. this variable [default: %(default)s]',
    'no_show'
    : 'do not call matplotlib show()',
    'shell'
    : 'run ipython shell after all computations',
}

def main():
    default_term_names = ', '.join((
        'dw_convect', 'dw_laplace',
        'dw_volume_dot:scalar', 'dw_volume_dot:scalar-material',
        'dw_volume_dot:vector','dw_volume_dot:vector-material',
        'dw_div', 'dw_lin_elastic'
    ))
    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help=helps['output_dir'])
    parser.add_argument('--orders', metavar='int[,int,...]',
                        action='store', dest='orders',
                        default='1,2,3,4,5,6,7,8,9,10', help=helps['orders'])
    parser.add_argument('--quad-orders', metavar='int[,int,...]',
                        action='store', dest='quad_orders',
                        default=None, help=helps['quad_orders'])
    parser.add_argument('--term-names', metavar='list',
                        action='store', dest='term_names',
                        default=default_term_names, help=helps['term_names'])
    parser.add_argument('--diff',
                        metavar='variable name',
                        action='store', dest='diff',
                        default=None, help=helps['diff'])
    parser.add_argument('-n', '--no-show',
                        action='store_false', dest='show',
                        default=True, help=helps['no_show'])
    parser.add_argument('--shell',
                        action='store_true', dest='shell',
                        default=False, help=helps['shell'])
    options = parser.parse_args()

    options.orders = so.parse_as_list(options.orders)
    if options.quad_orders is not None:
        options.quad_orders = so.parse_as_list(options.quad_orders)

    else:
        options.quad_orders = [2 * ii for ii in options.orders]

    options.term_names = so.parse_as_list(options.term_names, free_word=True)

    n_cell = 1
    dim = 3
    sym = dim2sym(dim)

    orders = options.orders

    all_costs = {}
    for _term_name in options.term_names:
        aux = _term_name.split(':')
        term_name = aux[0]
        variant = aux[1] if len(aux) == 2 else ''

        if (term_name in ('dw_convect', 'dw_div', 'dw_lin_elastic')
            or ('vector' in variant)):
            n_c = dim

        else:
            n_c = 1

        if (term_name == 'dw_convect') and (options.diff == 'u'):
            n_add = 2

        else:
            n_add = 1

        costs = all_costs.setdefault(_term_name, [])
        for order, quad_order in zip(orders, options.quad_orders):
            ps = PolySpace.any_from_args('ps', GeometryElement('3_8'), order)
            integral = Integral('i', order=quad_order)
            _, weights = integral.get_qp('3_8')

            n_qp = len(weights)
            n_en = ps.n_nod

            expr_cache = {}
            ebuilder = tm.ExpressionBuilder(n_add, expr_cache)

            virtual = tm.ExpressionArg(
                name='v',
                bf=nm.empty((1, n_qp, 1, n_en)),
                bfg=nm.empty((n_cell, n_qp, dim, n_en)),
                det=nm.empty((n_cell, n_qp)),
                n_components=n_c,
                dim=dim,
                kind='virtual',
            )

            arg = Struct(evaluate_cache={
                'dofs' : {0 : {'u' : nm.empty((n_cell, n_c, n_en))}}
            })
            state = tm.ExpressionArg(
                name='u',
                arg=arg,
                bf=nm.empty((1, n_qp, 1, n_en)),
                bfg=nm.empty((n_cell, n_qp, dim, n_en)),
                det=nm.empty((n_cell, n_qp)),
                n_components=n_c,
                dim=dim,
                kind='state',
            )
            if term_name == 'dw_convect':
                eargs = [virtual, state, state]
                ebuilder.build('i,i.j,j', *eargs, diff_var=options.diff)

            elif term_name == 'dw_laplace':
                eargs = [virtual, state]
                ebuilder.build('0.j,0.j', *eargs, diff_var=options.diff)

            elif term_name == 'dw_volume_dot':
                if 'material' not in variant:
                    eargs = [virtual, state]
                    ebuilder.build('i,i', *eargs, diff_var=options.diff)

                else:
                    mat = tm.ExpressionArg(
                        name='c',
                        arg=nm.empty((n_cell, n_qp, n_c, n_c)),
                        kind='ndarray',
                    )
                    eargs = [mat, virtual, state]
                    ebuilder.build('ij,i,j', *eargs, diff_var=options.diff)

            elif term_name == 'dw_div':
                eargs = [virtual]
                ebuilder.build('i.i', *eargs, diff_var=options.diff)

            elif term_name == 'dw_lin_elastic':
                mat = tm.ExpressionArg(
                    name='D',
                    arg=nm.empty((n_cell, n_qp, sym, sym)),
                    kind='ndarray',
                )
                eargs = [mat, virtual, state]
                ebuilder.build('IK,s(i:j)->I,s(k:l)->K', *eargs,
                               diff_var=options.diff)

            else:
                continue

            operands = tm.get_einsum_ops(eargs, ebuilder, expr_cache)
            cost = get_naive_cost(ebuilder, operands)
            costs.append(cost)

    markers = list(Line2D.filled_markers)
    select = sps.normalize_selected({'term' : list(all_costs.keys())})
    styles = {'term' : {'color' : 'tab10:qualitative', 'marker' : markers,
                        'mfc' : 'None', 'ms' : 8}}
    styles = sps.setup_plot_styles(select, styles)

    fig, ax = plt.subplots()
    used = None
    for key, costs in all_costs.items():
        style_kwargs, indices = sps.get_row_style(
            {'term' : key}, select, {}, styles
        )
        used = sps.update_used(used, indices)
        ax.semilogy(orders, costs, **style_kwargs)

    mode = 'vector' if options.diff is None else 'matrix'
    ax.set_title('{} mode'.format(mode))
    ax.set_xticks(orders)
    ax.set_xlabel('order')
    ax.set_ylabel('flops per cell')
    ax.grid(which='both', axis='y')
    sps.add_legend(ax, select, styles, used,
                   format_labels=lambda key, iv, val: val[3:])
    plt.tight_layout()

    filename = os.path.join(options.output_dir, 'flops-{}.pdf'.format(mode))
    so.ensure_path(filename)
    fig.savefig(filename, bbox_inches='tight')

    if options.show:
        plt.show()

    if options.shell:
        from soops.base import shell; shell()

if __name__ == '__main__':
    main()
