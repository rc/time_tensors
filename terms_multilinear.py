import numpy as nm

try:
    import dask.array as da

except ImportError:
    da = None

try:
    import opt_einsum as oe

except ImportError:
    oe = None

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

except ImportError:
    jnp = jax = None

from pyparsing import (Word, Suppress, oneOf, OneOrMore, delimitedList,
                       Combine, alphas, Literal)

from sfepy.base.base import output, Struct
from sfepy.base.timing import Timer
from sfepy.discrete import FieldVariable
from sfepy.mechanics.tensors import dim2sym
from sfepy.terms.terms import Term
from sfepy.terms import register_term

def _get_char_map(c1, c2):
    mm = {}
    for ic, char in enumerate(c1):
        if char in mm:
            print(char, '->eq?', mm[char], c2[ic])
            if mm[char] != c2[ic]:
                mm[char] += c2[ic]
        else:
            mm[char] = c2[ic]

    return mm

def append_all(seqs, item, ii=None):
    if ii is None:
        for seq in seqs:
            seq.append(item)

    else:
        seqs[ii].append(item)

def get_sizes(indices, operands):
    sizes = {}

    for iis, op in zip(indices, operands):
        for ii, size in zip(iis, op.shape):
            sizes[ii] = size

    return sizes

class ExpressionArg(Struct):

    @staticmethod
    def from_term_arg(arg, term, cache):
        if isinstance(arg, FieldVariable) and arg.is_virtual():
            ag, _ = term.get_mapping(arg)
            obj = ExpressionArg(name=arg.name, qsb=ag.bf, qsbg=ag.bfg,
                                det=ag.det[..., 0, 0],
                                n_components=arg.n_components,
                                dim=arg.dim,
                                kind='virtual')

        elif isinstance(arg, FieldVariable) and arg.is_state_or_parameter():
            dofs = cache.get(arg.name)
            if dofs is None:
                conn = arg.field.get_econn(term.get_dof_conn_type(),
                                           term.region)
                dofs_vec = arg().reshape((-1, arg.n_components))
                # # axis 0: cells, axis 1: node, axis 2: component
                # dofs = dofs_vec[conn]
                # axis 0: cells, axis 1: component, axis 2: node
                dofs = dofs_vec[conn].transpose((0, 2, 1))
                if arg.n_components == 1:
                    dofs.shape = (dofs.shape[0], -1)
                cache[arg.name] = dofs

            ag, _ = term.get_mapping(arg)
            obj = ExpressionArg(name=arg.name, qsb=ag.bf, qsbg=ag.bfg,
                                det=ag.det[..., 0, 0], dofs=dofs,
                                n_components=arg.n_components,
                                dim=arg.dim,
                                kind='state')

        elif isinstance(arg, nm.ndarray):
            aux = term.get_args()
            # Find arg in term arguments using a loop (numpy arrays cannot be
            # compared) to get its name.
            ii = [ii for ii in range(len(term.args)) if aux[ii] is arg][0]
            obj = ExpressionArg(name='.'.join(term.arg_names[ii]), val=arg,
                                kind='ndarray')

        else:
            raise ValueError('unknown argument type! ({})'
                             .format(type(arg)))

        return obj

class ExpressionBuilder(Struct):
    letters = 'defgh'
    _aux_letters = 'rstuvwxyz'

    def __init__(self, n_add, cache):
        self.n_add = n_add
        self.subscripts = [[] for ia in range(n_add)]
        self.operands = [[] for ia in range(n_add)]
        self.operand_names = [[] for ia in range(n_add)]
        self.out_subscripts = ['c' for ia in range(n_add)]
        self.ia = 0
        self.cache = cache
        self.aux_letters = iter(self._aux_letters)

    def make_eye(self, size):
        key = 'I{}'.format(size)
        ee = self.cache.get(key)
        if ee is None:
            ee = nm.eye(size)
            self.cache[key] = ee

        return ee

    def make_psg(self, dim):
        key = 'Psg{}'.format(dim)
        psg = self.cache.get(key)
        if psg is None:
            sym = dim2sym(dim)
            psg = nm.zeros((dim, dim, sym))
            if dim == 3:
                psg[0, [0,1,2], [0,3,4]] = 1
                psg[1, [0,1,2], [3,1,5]] = 1
                psg[2, [0,1,2], [4,5,2]] = 1

            elif dim == 2:
                psg[0, [0,1], [0,2]] = 1
                psg[1, [0,1], [2,1]] = 1

            self.cache[key] = psg

        return psg

    def add_constant(self, val, name):
        append_all(self.subscripts, 'cq')
        append_all(self.operands, val)
        append_all(self.operand_names, name)

    def add_bfg(self, iin, ein, qsbg, name):
        append_all(self.subscripts, 'cq{}{}'.format(ein[2], iin))
        append_all(self.operands, qsbg)
        append_all(self.operand_names, name)

    def add_bf(self, iin, ein, qsb, name):
        key = 'qsb{}'.format(id(qsb))
        _qsb  = self.cache.get(key)
        if qsb.shape[0] > 1: # cell-depending basis.
            append_all(self.subscripts, 'cq{}'.format(iin))
            if _qsb is None:
                _qsb = qsb[:, :, 0]
                self.cache[key] = _qsb

        else:
            append_all(self.subscripts, 'q{}'.format(iin))
            if _qsb is None:
                _qsb = qsb[0, :, 0]
                self.cache[key] = _qsb

        append_all(self.operands, _qsb)
        append_all(self.operand_names, name)

    def add_eye(self, iic, ein, eye, iia=None):
        append_all(self.subscripts, '{}{}'.format(ein[0], iic), ii=iia)
        append_all(self.operands, eye, ii=iia)
        append_all(self.operand_names, 'I', ii=iia)

    def add_psg(self, iic, ein, psg, iia=None):
        append_all(self.subscripts, '{}{}{}'.format(iic, ein[2], ein[0]),
                   ii=iia)
        append_all(self.operands, psg, ii=iia)
        append_all(self.operand_names, 'Psg', ii=iia)

    def add_arg_dofs(self, iin, ein, arg, iia=None):
        if arg.n_components > 1:
            #term = 'c{}{}'.format(iin, ein[0])
            term = 'c{}{}'.format(ein[0], iin)

        else:
            term = 'c{}'.format(iin)

        append_all(self.subscripts, term, ii=iia)
        append_all(self.operands, arg.dofs, ii=iia)
        append_all(self.operand_names, arg.name + '.dofs', ii=iia)

    def add_virtual_arg(self, arg, ii, ein, modifier):
        iin = self.letters[ii] # node (qs basis index)
        if ('.' in ein) or (':' in ein): # derivative, symmetric gradient
            self.add_bfg(iin, ein, arg.qsbg, arg.name + '.bfg')

        else:
            self.add_bf(iin, ein, arg.qsb, arg.name + '.bf')

        out_letters = iin

        if arg.n_components > 1:
            iic = next(self.aux_letters) # component
            if ':' not in ein:
                ee = self.make_eye(arg.n_components)
                self.add_eye(iic, ein, ee)

            else: # symmetric gradient
                if modifier[0][0] == 's': # vector storage
                    psg = self.make_psg(arg.dim)
                    self.add_psg(iic, ein, psg)

                else:
                    raise ValueError('unknown argument modifier! ({})'
                                     .format(modifier))

            out_letters = iic + out_letters

        for iia in range(self.n_add):
            self.out_subscripts[iia] += out_letters

    def add_state_arg(self, arg, ii, ein, modifier, diff_var):
        iin = self.letters[ii] # node (qs basis index)
        if ('.' in ein) or (':' in ein): # derivative, symmetric gradient
            self.add_bfg(iin, ein, arg.qsbg, arg.name + '.bfg')

        else:
            self.add_bf(iin, ein, arg.qsb, arg.name + '.bf')

        out_letters = iin

        if (diff_var != arg.name):
            if ':' not in ein:
                self.add_arg_dofs(iin, ein, arg)

            else: # symmetric gradient
                if modifier[0][0] == 's': # vector storage
                    iic = next(self.aux_letters) # component
                    psg = self.make_psg(arg.dim)
                    self.add_psg(iic, ein, psg)
                    self.add_arg_dofs(iin, [iic], arg)

                else:
                    raise ValueError('unknown argument modifier! ({})'
                                     .format(modifier))

        else:
            if arg.n_components > 1:
                iic = next(self.aux_letters) # component
                if ':' not in ein:
                    ee = self.make_eye(arg.n_components)

                else: # symmetric gradient
                    if modifier[0][0] == 's': # vector storage
                        psg = self.make_psg(arg.dim)

                    else:
                        raise ValueError('unknown argument modifier! ({})'
                                         .format(modifier))

                out_letters = iic + out_letters

            for iia in range(self.n_add):
                if iia != self.ia:
                    self.add_arg_dofs(iin, ein, arg, iia)

                elif arg.n_components > 1:
                    if ':' not in ein:
                        self.add_eye(iic, ein, ee, iia)

                    else:
                        self.add_psg(iic, ein, psg, iia)

            self.out_subscripts[self.ia] += out_letters
            self.ia += 1

    def add_material_arg(self, arg, ii, ein):
        append_all(self.subscripts, 'cq{}'.format(ein))
        append_all(self.operands, arg.val)
        append_all(self.operand_names, arg.name)

    def build(self, texpr, *args, diff_var=None):
        eins, modifiers = parse_term_expression(texpr)

        # Virtual variable must be the first variable.
        # Numpy arrays cannot be compared -> use a loop.
        for iv, arg in enumerate(args):
            if arg.kind == 'virtual':
                self.add_constant(arg.det, 'J')
                self.add_virtual_arg(arg, iv, eins[iv], modifiers[iv])
                break
        else:
            iv = -1
            for ip, arg in enumerate(args):
                if arg.is_state_or_parameter:
                    self.add_constant(arg.det, 'J')
                    break
            else:
                raise ValueError('no FieldVariable in arguments!')

        for ii, ein in enumerate(eins):
            if ii == iv: continue
            arg = args[ii]

            if arg.kind == 'ndarray':
                self.add_material_arg(arg, ii, ein)

            elif arg.kind == 'state':
                self.add_state_arg(arg, ii, ein, modifiers[ii], diff_var)

            else:
                raise ValueError('unknown argument type! ({})'
                                 .format(type(arg)))

    @staticmethod
    def join_subscripts(subscripts, out_subscripts):
        return ','.join(subscripts) + '->' + out_subscripts

    def get_expressions(self):
        expressions = [self.join_subscripts(self.subscripts[ia],
                                            self.out_subscripts[ia])
                       for ia in range(self.n_add)]
        return expressions

    def get_sizes(self, ia):
        return get_sizes(self.subscripts[ia], self.operands[ia])

    def get_output_shape(self, ia):
        return tuple(self.get_sizes(ia)[ii] for ii in self.out_subscripts[ia])

    def print_shapes(self):
        for ia in range(self.n_add):
            sizes = self.get_sizes(ia)
            output(sizes)
            out_shape = self.get_output_shape(ia)
            output(self.out_subscripts[ia], out_shape, '=')

            for name, ii, op in zip(self.operand_names[ia],
                                    self.subscripts[ia],
                                    self.operands[ia]):
                output('  {:10}{:8}{}'.format(name, ii, op.shape))

    def transform(self, transformation='loop'):
        if transformation == 'loop':
            expressions, poperands, liis = [], [], []

            for subscripts, out_subscripts, operands in zip(
                    self.subscripts, self.out_subscripts, self.operands
            ):
                lii = [ii for ii, subs in enumerate(subscripts)
                       if subs.startswith('c')]
                tsubs = [subs[1:] if ii in lii else subs
                         for ii, subs in enumerate(subscripts)]
                tout_subs = out_subscripts[1:]
                expr = self.join_subscripts(tsubs, tout_subs)
                pops = [op[0] if ii in lii else op
                        for ii, op in enumerate(operands)]

                expressions.append(expr)
                poperands.append(pops)
                liis.append(lii)

            return expressions, poperands, liis

        else:
            raise ValueError('unknown transformation! ({})'
                             .format(transformation))

def collect_modifiers(modifiers):
    def _collect_modifiers(toks):
        if len(toks) > 1:
            out = []
            modifiers.append([])
            for ii, mod in enumerate(toks[::3]):
                tok = toks[3*ii+1]
                tok = tok.replace(tok[0], toks[2])
                modifiers[-1].append(list(toks))
                out.append(tok)
            return out

        else:
            modifiers.append(None)
            return toks
    return _collect_modifiers

def parse_term_expression(texpr):
    mods = 's'
    lparen, rparen = map(Suppress, '()')
    simple_arg = Word(alphas + '.:0')
    arrow = Literal('->').suppress()
    letter = Word(alphas, exact=1)
    mod_arg = oneOf(mods) + lparen + simple_arg + rparen + arrow + letter
    arg = OneOrMore(simple_arg ^ mod_arg)
    modifiers = []
    arg.setParseAction(collect_modifiers(modifiers))

    parser = delimitedList(Combine(arg))
    eins = parser.parseString(texpr, parseAll=True)
    return eins, modifiers

class ETermBase(Struct):
    """
    Reserved letters:

    c .. cells
    q .. quadrature points
    d-h .. DOFs axes
    r-z .. auxiliary axes
    """
    verbosity = 0

    can_backend = {
        'numpy' : nm,
        'numpy_loop' : nm,
        'opt_einsum' : oe,
        'opt_einsum_loop' : oe,
        'jax' : jnp,
        'jax_vmap' : jnp,
        'dask_single' : da,
        'dask_threads' : da,
        'opt_einsum_dask_single' : oe and da,
        'opt_einsum_dask_threads' : oe and da,
    }

    def set_backend(self, backend='numpy', optimize=True, **kwargs):
        if backend not in self.can_backend.keys():
            raise ValueError('backend {} not in {}!'
                             .format(self.backend, self.can_backend.keys()))

        if not self.can_backend[backend]:
            raise ValueError('backend {} is not available!'.format(backend))

        if (hasattr(self, 'backend')
            and (backend == self.backend) and (optimize == self.optimize)):
            return

        self.backend = backend
        self.optimize = optimize
        self.backend_kwargs = kwargs
        self.paths, self.path_infos = None, None
        self.eval_einsum = None

    def build_expression(self, texpr, *args, diff_var=None):
        timer = Timer('')
        timer.start()

        if diff_var is not None:
            n_add = len([arg.name for arg in args
                         if (isinstance(arg, FieldVariable)
                             and (arg.name == diff_var))])

        else:
            n_add = 1

        expr_cache = {}
        self.ebuilder = ExpressionBuilder(n_add, expr_cache)
        eargs = [ExpressionArg.from_term_arg(arg, self, expr_cache)
                 for arg in args]
        self.ebuilder.build(texpr, *eargs, diff_var=diff_var)

        if self.verbosity:
            output('build expression: {} s'.format(timer.stop()))

    def get_paths(self, expressions, operands):
        memory_limit = self.backend_kwargs.get('memory_limit')

        if ('numpy' in self.backend) or self.backend.startswith('dask'):
            optimize = (self.optimize if memory_limit is None
                        else (self.optimize, memory_limit))
            paths, path_infos = zip(*[nm.einsum_path(
                expressions[ia], *operands[ia],
                optimize=optimize,
            ) for ia in range(len(operands))])

        elif 'opt_einsum' in self.backend:
            paths, path_infos = zip(*[oe.contract_path(
                expressions[ia], *operands[ia],
                optimize=self.optimize,
                memory_limit=memory_limit,
            ) for ia in range(len(operands))])

        elif 'jax' in self.backend:
            paths, path_infos = zip(*[jnp.einsum_path(
                expressions[ia], *operands[ia],
                optimize=self.optimize,
            ) for ia in range(len(operands))])

        else:
            raise ValueError('unsupported backend! ({})'.format(self.backend))

        return paths, path_infos

    def make_function(self, texpr, *args, diff_var=None):
        timer = Timer('')
        timer.start()
        if hasattr(self, 'eval_einsum') and (self.eval_einsum is not None):
            if self.verbosity:
                output('einsum setup: {} s'.format(timer.stop()))
            return self.eval_einsum

        if not hasattr(self, 'ebuilder'):
            self.build_expression(texpr, *args, diff_var=diff_var)

        if not hasattr(self, 'paths') or (self.paths is None):
            self.parsed_expressions = self.ebuilder.get_expressions()
            if self.verbosity:
                output(self.parsed_expressions)
            if self.verbosity > 1:
                self.ebuilder.print_shapes()

            self.paths, self.path_infos = self.get_paths(
                self.parsed_expressions,
                self.ebuilder.operands,
            )
            if self.verbosity > 2:
                for path, path_info in zip(self.paths, self.path_infos):
                    output(path)
                    output(path_info)

        operands = self.ebuilder.operands
        n_add = len(operands)

        if self.backend in ('numpy', 'opt_einsum'):
            contract = {'numpy' : nm.einsum,
                        'opt_einsum' : oe.contract}[self.backend]
            def eval_einsum(out, eshape):
                if operands[0][0].flags.c_contiguous:
                    # This is very slow if vout layout differs from operands
                    # layout.
                    vout = out.reshape(eshape)
                    contract(self.parsed_expressions[0], *operands[0],
                             out=vout,
                             optimize=self.paths[0])

                else:
                    aux = contract(self.parsed_expressions[0], *operands[0],
                                   optimize=self.paths[0])
                    out[:] = aux.reshape(out.shape)

                for ia in range(1, n_add):
                    aux = contract(self.parsed_expressions[ia],
                                   *operands[ia],
                                   optimize=self.paths[ia])
                    out[:] += aux.reshape(out.shape)

        elif self.backend in ('numpy_loop', 'opt_einsum_loop'):
            expressions, poperands, liis = self.ebuilder.transform('loop')
            paths, path_infos = self.get_paths(expressions, poperands)
            n_cell = self.ebuilder.get_sizes(0)['c']
            transforms = lambda ia: lambda ic: [op[ic] if ii in liis[ia]
                                                else op
                                                for ii, op
                                                in enumerate(operands[ia])]
            if self.verbosity > 2:
                for path, path_info in zip(paths, path_infos):
                    output(path)
                    output(path_info)

            contract = {'numpy_loop' : nm.einsum,
                        'opt_einsum_loop' : oe.contract}[self.backend]
            def eval_einsum(out, eshape):
                vout = out.reshape(eshape)
                get_ops = transforms(0)
                for ic in range(n_cell):
                    ops = get_ops(ic)
                    contract(expressions[0], *ops, out=vout[ic],
                             optimize=paths[0])
                for ia in range(1, n_add):
                    get_ops = transforms(ia)
                    for ic in range(n_cell):
                        ops = get_ops(ic)
                        vout[ic] += contract(expressions[ia], *ops,
                                             optimize=paths[ia])

        elif self.backend == 'jax':
            @jax.partial(jax.jit, static_argnums=(0, 1, 2))
            def _eval_einsum(expressions, paths, n_add, operands):
                val = jnp.einsum(expressions[0], *operands[0],
                                 optimize=paths[0])
                for ia in range(1, n_add):
                    val += jnp.einsum(expressions[ia], *operands[ia],
                                      optimize=paths[ia])
                return val

            def eval_einsum(out, eshape):
                aux = _eval_einsum(self.parsed_expressions, self.paths, n_add,
                                   operands)
                out[:] = nm.asarray(aux.reshape(out.shape))

        elif self.backend == 'jax_vmap':
            expressions, poperands, liis = self.ebuilder.transform('loop')
            paths, path_infos = self.get_paths(expressions, poperands)
            if self.verbosity > 2:
                for path, path_info in zip(paths, path_infos):
                    output(path)
                    output(path_info)

            def _eval_einsum_cell(expressions, paths, n_add, operands):
                val = jnp.einsum(expressions[0], *operands[0],
                                 optimize=paths[0])
                for ia in range(1, n_add):
                    val += jnp.einsum(expressions[ia], *operands[ia],
                                      optimize=paths[ia])
                return val

            vm = [[0 if ii in iis else None for ii in range(len(ops))]
                  for ops, iis in zip(operands, liis)]
            vms = (None, None, None, vm)
            _eval_einsum = jax.jit(jax.vmap(_eval_einsum_cell, vms, 0),
                                   static_argnums=(0, 1, 2))

            def eval_einsum(out, eshape):
                aux = _eval_einsum(expressions, paths, n_add,
                                   operands)
                out[:] = nm.asarray(aux.reshape(out.shape))

        elif self.backend.startswith('dask'):
            scheduler = {'dask_single' : 'single-threaded',
                         'dask_threads' : 'threads'}[self.backend]
            def eval_einsum(out, eshape):
                _out = da.einsum(self.parsed_expressions[0], *operands[0],
                                 optimize=self.paths[0])
                for ia in range(1, n_add):
                    aux = da.einsum(self.parsed_expressions[ia],
                                    *operands[ia],
                                    optimize=self.paths[ia])
                    _out += aux

                out[:] = _out.compute(scheduler=scheduler).reshape(out.shape)

        elif self.backend.startswith('opt_einsum_dask'):
            scheduler = {'opt_einsum_dask_single' : 'single-threaded',
                         'opt_einsum_dask_threads' : 'threads'}[self.backend]

            da_operands = []
            c_chunk_size = self.backend_kwargs.get('c_chunk_size')
            for ia in range(self.ebuilder.n_add):
                da_ops = []
                for name, ii, op in zip(self.ebuilder.operand_names[ia],
                                        self.ebuilder.subscripts[ia],
                                        operands[ia]):
                    if 'c' in ii:
                        if c_chunk_size is None:
                            chunks = 'auto'

                        else:
                            chunks = (c_chunk_size,) + op.shape[1:]
                            da_op = da.from_array(op, chunks=chunks, name=name)

                    else:
                        da_op = op

                    da_ops.append(da_op)
                da_operands.append(da_ops)

            def eval_einsum(out, eshape):
                _out = oe.contract(self.parsed_expressions[0], *da_operands[0],
                                   optimize=self.paths[0],
                                   backend='dask')
                for ia in range(1, n_add):
                    aux = oe.contract(self.parsed_expressions[ia],
                                      *da_operands[ia],
                                      optimize=self.paths[ia],
                                      backend='dask')
                    _out += aux

                out[:] = _out.compute(scheduler=scheduler).reshape(out.shape)

        else:
            raise ValueError('unsupported backend! ({})'.format(self.backend))

        self.eval_einsum = eval_einsum

        if self.verbosity:
            output('einsum setup: {} s'.format(timer.stop()))

        return eval_einsum

    @staticmethod
    def function(out, eval_einsum, eshape):
        tt = Timer('')
        tt.start()
        eval_einsum(out, eshape)
        output('eval_einsum: {} s'.format(tt.stop()))
        return 0

    def get_fargs(self, *args, **kwargs):
        mode, term_mode, diff_var = args[-3:]

        if mode == 'weak':
            vvar = self.get_virtual_variable()
            n_elr, n_qpr, dim, n_enr, n_cr = self.get_data_shape(vvar)

            if diff_var is not None:
                varc = self.get_variables(as_list=False)[diff_var]
                n_elc, n_qpc, dim, n_enc, n_cc = self.get_data_shape(varc)
                eshape = tuple([n_elr]
                               + ([n_cr] if n_cr > 1 else [])
                               + [n_enr]
                               + ([n_cc] if n_cc > 1 else [])
                               + [n_enc])

            else:
                eshape = (n_elr, n_cr, n_enr) if n_cr > 1 else (n_elr, n_enr)

        else:
            if diff_var is not None:
                raise ValueError('cannot differentiate in {} mode!'
                                 .format(mode))

            # self.ebuilder is created in self.get_eval_shape() call by Term.
            eshape = self.ebuilder.get_output_shape(0)

        eval_einsum = self.get_function(*args, **kwargs)

        return eval_einsum, eshape

    def get_eval_shape(self, *args, **kwargs):
        mode, term_mode, diff_var = args[-3:]
        if diff_var is not None:
            raise ValueError('cannot differentiate in {} mode!'
                             .format(mode))

        self.get_function(*args, **kwargs)

        out_shape = self.ebuilder.get_output_shape(0)

        operands = self.ebuilder.operands[0]
        dtype = nm.find_common_type([op.dtype for op in operands], [])

        return out_shape, dtype

class ELaplaceTerm(ETermBase, Term):
    name = 'dw_elaplace'
    arg_types = (('opt_material', 'virtual', 'state'),
                 ('opt_material', 'parameter_1', 'parameter_2'))
    arg_shapes = [{'opt_material' : '1, 1', 'virtual' : (1, 'state'),
                   'state' : 1, 'parameter_1' : 1, 'parameter_2' : 1},
                  {'opt_material' : None}]
    modes = ('weak', 'eval')

    def get_function(self, mat, virtual, state, mode=None, term_mode=None,
                     diff_var=None, **kwargs):
        """
        diff_var not needed here(?), but Term passes it in *args.
        """
        if mat is None:
            fun = self.make_function(
                '0.j,0.j', virtual, state, diff_var=diff_var,
            )

        else:
            fun = self.make_function(
                'jk,0.j,0.k', mat, virtual, state, diff_var=diff_var,
            )

        return fun

register_term(ELaplaceTerm)

class EVolumeDotTerm(ETermBase, Term):
    name = 'dw_evolume_dot'
    arg_types = (('opt_material', 'virtual', 'state'),
                 ('opt_material', 'parameter_1', 'parameter_2'))
    arg_shapes = [{'opt_material' : '1, 1', 'virtual' : (1, 'state'),
                   'state' : 1, 'parameter_1' : 1, 'parameter_2' : 1},
                  {'opt_material' : None},
                  {'opt_material' : '1, 1', 'virtual' : ('D', 'state'),
                   'state' : 'D', 'parameter_1' : 'D', 'parameter_2' : 'D'},
                  {'opt_material' : 'D, D'},
                  {'opt_material' : None}]
    modes = ('weak', 'eval')

    def get_function(self, mat, virtual, state, mode=None, term_mode=None,
                     diff_var=None, **kwargs):
        if mat is None:
            fun = self.make_function(
                'i,i', virtual, state, diff_var=diff_var,
            )

        else:
            fun = self.make_function(
                'ij,i,j', mat, virtual, state, diff_var=diff_var,
            )

        return fun

register_term(EVolumeDotTerm)

class EConvectTerm(ETermBase, Term):
    name = 'dw_econvect'
    arg_types = ('virtual', 'state')
    arg_shapes = {'virtual' : ('D', 'state'), 'state' : 'D'}

    def get_function(self, virtual, state, mode=None, term_mode=None,
                     diff_var=None, **kwargs):
        return self.make_function(
            'i,i.j,j', virtual, state, state, diff_var=diff_var,
        )

register_term(EConvectTerm)

class EDivTerm(ETermBase, Term):
    name = 'dw_ediv'
    arg_types = ('opt_material', 'virtual')
    arg_shapes = [{'opt_material' : '1, 1', 'virtual' : ('D', None)},
                  {'opt_material' : None}]

    def get_function(self, mat, virtual, mode=None, term_mode=None,
                     diff_var=None, **kwargs):
        if mat is None:
            fun = self.make_function(
                'i.i', virtual, diff_var=diff_var,
            )

        else:
            fun = self.make_function(
                '0,i.i', mat, virtual, diff_var=diff_var,
            )

        return fun

register_term(EDivTerm)

class EStokesTerm(ETermBase, Term):
    name = 'dw_estokes'
    arg_types = (('opt_material', 'virtual', 'state'),
                 ('opt_material', 'state', 'virtual'),
                 ('opt_material', 'parameter_v', 'parameter_s'))
    arg_shapes = [{'opt_material' : '1, 1',
                   'virtual/grad' : ('D', None), 'state/grad' : 1,
                   'virtual/div' : (1, None), 'state/div' : 'D',
                   'parameter_v' : 'D', 'parameter_s' : 1},
                  {'opt_material' : None}]
    modes = ('grad', 'div', 'eval')

    def get_function(self, coef, var_v, var_s, mode=None, term_mode=None,
                     diff_var=None, **kwargs):
        if coef is None:
            fun = self.make_function(
                'i.i,0', var_v, var_s, diff_var=diff_var,
            )

        else:
            fun = self.make_function(
                '0,i.i,0', coef, var_v, var_s, diff_var=diff_var,
            )

        return fun

register_term(EStokesTerm)

class ELinearElasticTerm(ETermBase, Term):
    name = 'dw_elin_elastic'
    arg_types = (('material', 'virtual', 'state'),
                 ('material', 'parameter_1', 'parameter_2'))
    arg_shapes = {'material' : 'S, S', 'virtual' : ('D', 'state'),
                  'state' : 'D', 'parameter_1' : 'D', 'parameter_2' : 'D'}
    modes = ('weak', 'eval')

    def get_function(self, mat, virtual, state, mode=None, term_mode=None,
                     diff_var=None, **kwargs):
        return self.make_function(
            'IK,s(i:j)->I,s(k:l)->K', mat, virtual, state, diff_var=diff_var,
        )

register_term(ELinearElasticTerm)
