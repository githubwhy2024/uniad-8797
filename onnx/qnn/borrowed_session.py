"""Synchronous FP32 CPU candidate that borrows suitable NumPy input buffers.

All native ABI, finite/integer/bool guards and output ownership remain the
source NativeSession.run implementation. Only its input packing call changes.
Caller-owned inputs must remain stable until the synchronous call returns.
"""
import ast
import copy
import hashlib
import inspect
import textwrap
import numpy as np
import session as source

SOURCE_SHA256 = '70ae8f1961ef455aa93f0bedbcf6688895a96a5ae3b9dba1e498ac5a661947df'
if source.sha(source.__file__) != SOURCE_SHA256:
    raise ValueError('borrowed input candidate requires its exact source session')

TRANSPORT = dict(borrowed_calls=0, borrowed_bytes=0, copied_calls=0, copied_bytes=0)


class BorrowedNativeSession(source.NativeSession):
    def _prepare_input(self, value):
        # Preserve the original aligned, writable, contiguous native buffer
        # requirement, including a copy for readonly/strided/unaligned arrays.
        if value.flags.c_contiguous and value.flags.aligned and value.flags.writeable:
            TRANSPORT['borrowed_calls'] += 1
            TRANSPORT['borrowed_bytes'] += value.nbytes
            return value
        result = np.array(value, copy=True, order='C')
        TRANSPORT['copied_calls'] += 1
        TRANSPORT['copied_bytes'] += result.nbytes
        return result


original = ast.parse(textwrap.dedent(inspect.getsource(source.NativeSession.run)))
candidate = copy.deepcopy(original)
expected = ast.dump(ast.parse("np.array(v, copy=True, order='C')", mode='eval').body)


class PackInputs(ast.NodeTransformer):
    changed = 0

    def visit_Call(self, node):
        if ast.dump(node) == expected:
            self.changed += 1
            return ast.copy_location(ast.parse('self._prepare_input(v)', mode='eval').body, node)
        return self.generic_visit(node)


transform = PackInputs()
candidate = ast.fix_missing_locations(transform.visit(candidate))
if transform.changed != 1:
    raise ValueError('expected exactly one source input packing call')
namespace = dict(vars(source))
exec(compile(candidate, __file__, 'exec'), namespace)
BorrowedNativeSession.run = namespace['run']
PROVENANCE = dict(source_session_sha256=SOURCE_SHA256,
                  original_run_ast_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
                  candidate_run_ast_sha256=hashlib.sha256(ast.dump(candidate).encode()).hexdigest(),
                  replacements=transform.changed,
                  scope='Only input packing changes. No precision, arithmetic, output allocation, '
                        'ABI/finite/integer/bool guard, or synchronous native execution changes.')
