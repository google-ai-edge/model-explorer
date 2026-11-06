# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checks that model_debugger.server's dependencies only go downward.

Every module of the package belongs to a layer: the directory it is in, or
`server` for the facade, server.py. A module imports only from its own layer
and from the layers below it in LAYERS. `testing` is outside the order: only
tests (`*_test.py`) and `testing` itself import it, and it may import any
layer. A test imports what its module may, and `testing`. Imports of other
packages, the protocol's model_debugger.protocol among them, are not checked.
Importing the package itself, as `from model_debugger import server` or a
bare `import model_debugger.server`, names no layer and is refused. Only
import statements are read: a module that importlib loads is not seen.

Usage: python3 -I ci/check_server_layers.py /path/to/model_debugger/server
"""

import ast
import pathlib
import sys

# The layers, lowest first.
LAYERS = (
    'util',  # errors, JSON files, the clock
    'architectures',  # what a model is: its skeleton, from its config.json
    'tap',  # what a model's tensors are, and a copy that observes them
    'graph',  # a model's graph, the converter's, and its anchors' binding
    'capture',  # a chat's capture on disk
    'analysis',  # the token, logits and KV comparisons of captures
    'runner',  # launching runners and talking to them
    'session',  # the Session itself: stores, jobs, Model Servers
    'service',  # the interface to the UI: the HTTP API and its views
    'server',  # the facade, server.py
    'cli',  # the command line
)
_TESTING = 'testing'
_PACKAGE = 'model_debugger.server'


def _layer(parts: tuple[str, ...]) -> str:
  """The layer of a module, by its path's parts under the package.

  A module in the package's own directory is a layer of its own, as are its
  tests; its __init__.py, which every module needs, is in the lowest.

  Args:
      parts: Relative path components under the package root.

  Returns:
      The layer name for the module.
  """
  if len(parts) > 1:
    return parts[0]
  stem = pathlib.PurePath(parts[0]).stem
  if stem == '__init__':
    return LAYERS[0]
  return stem.removesuffix('_test')


def _imported_layers(tree: ast.AST) -> list[tuple[int, str]]:
  """The package's layers a module imports, with the line of each import.

  Args:
      tree: The module's syntax tree.

  Returns:
      (line, layer) for each name an import statement binds in the package:
      '.' for a relative import, '' for the package itself or a package
      above it, else the directory or module under the package it is in.
  """
  found = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      modules = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        found.append((node.lineno, '.'))
        continue
      # The name imported may be a module of node.module or a name in
      # it; either way, it is in node.module's layer or below it.
      modules = [f'{node.module}.{alias.name}' for alias in node.names]
    else:
      continue
    for module in modules:
      if module == _PACKAGE or _PACKAGE.startswith(module + '.'):
        found.append((node.lineno, ''))
      elif module.startswith(_PACKAGE + '.'):
        found.append((node.lineno, module[len(_PACKAGE) + 1 :].split('.')[0]))
  return found


def check(root: pathlib.Path) -> list[str]:
  """Returns every import under `root` that does not go downward.

  Args:
      root: The package's directory, model_debugger/server.

  Returns:
      One line per violation, as `path:line: what`.
  """
  violations = []
  for path in sorted(root.rglob('*.py')):
    relative = path.relative_to(root)
    if '__pycache__' in relative.parts:
      continue
    layer = _layer(relative.parts)
    is_test = path.stem.endswith('_test')
    if layer != _TESTING and layer not in LAYERS:
      violations.append(
          f'{relative}: {layer} is in no layer; add it to LAYERS.'
      )
      continue
    tree = ast.parse(path.read_text(), str(path))
    for line, target in _imported_layers(tree):
      where = f'{relative}:{line}'
      if target == '.':
        violations.append(f'{where}: a relative import.')
      elif not target:
        violations.append(
            f'{where}: imports the package, which names no layer; '
            'import its modules.'
        )
      elif target == _TESTING:
        if not is_test and layer != _TESTING:
          violations.append(
              f'{where}: {layer} imports testing, which only tests may.'
          )
      elif target not in LAYERS:
        violations.append(f'{where}: {target} is in no layer.')
      elif layer != _TESTING and LAYERS.index(target) > LAYERS.index(layer):
        violations.append(
            f'{where}: {layer} imports {target}, a layer above it.'
        )
  return violations


def main(argv: list[str]) -> int:
  """Prints every violation and returns 1 if there is any, else 0."""
  if len(argv) != 2:
    print(__doc__.strip().splitlines()[-1], file=sys.stderr)
    return 2
  violations = check(pathlib.Path(argv[1]))
  for violation in violations:
    print(f'check_server_layers: {violation}', file=sys.stderr)
  if violations:
    print(
        'check_server_layers: dependencies go only downward: '
        + ' < '.join(LAYERS),
        file=sys.stderr,
    )
    return 1
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
