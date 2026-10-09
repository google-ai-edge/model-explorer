#!/usr/bin/env python3
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

"""Fail on unused imports (stdlib-only subset of pyflakes).

Mark intentional side-effect imports with `# noqa` on the import line.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCOPES = (
    'src/capture/pytorch/package/ai_edge_debugger_pytorch',
    'src/capture/pytorch/tests',
    'src/server/package/model_explorer_debugger',
    'src/runner/python/package/model_debugger_runner',
    'src/contracts/python/package/model_debugger_contracts',
    'src/runner/python/tools',
    'tools',
)


def unused_imports(path: pathlib.Path) -> list[tuple[str, int]]:
  """Returns a list of (name, line) for unused imports in path."""
  source = path.read_text(encoding='utf-8')
  tree = ast.parse(source, filename=str(path))
  lines = source.splitlines()
  imported = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        imported[(alias.asname or alias.name).split('.')[0]] = node.lineno
    elif isinstance(node, ast.ImportFrom):
      for alias in node.names:
        if alias.name != '*':
          imported[alias.asname or alias.name] = node.lineno
  used = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Name):
      used.add(node.id)
    elif isinstance(node, ast.Attribute):
      base = node
      while isinstance(base, ast.Attribute):
        base = base.value
      if isinstance(base, ast.Name):
        used.add(base.id)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets
    ):
      used.update(
          c.value
          for c in ast.walk(node.value)
          if isinstance(c, ast.Constant) and isinstance(c.value, str)
      )
  return [
      (name, line)
      for name, line in sorted(imported.items(), key=lambda item: item[1])
      if name not in used and 'noqa' not in lines[line - 1]
  ]


def main() -> None:
  """Scans repository Python sources and reports unused imports."""
  problems = []
  for scope in SCOPES:
    for path in sorted((ROOT / scope).rglob('*.py')):
      if '__pycache__' in path.parts or '.egg-info' in str(path):
        continue
      problems.extend(
          f'{path.relative_to(ROOT)}:{line}: unused import {name}'
          for name, line in unused_imports(path)
      )
  if problems:
    raise SystemExit('\n'.join(problems))
  print('Unused imports: none.')


if __name__ == '__main__':
  main()
