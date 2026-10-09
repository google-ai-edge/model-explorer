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
# ==============================================================================

"""Report import cycles inside the Server package.

Cycles hidden behind function-local imports still couple modules; this check
keeps the list from growing. KNOWN_CYCLES records the cycles present when
the check was introduced (2026-09-15); remove entries as they are broken.
"""

import ast
import pathlib
import sys

PACKAGE = (
    pathlib.Path(__file__).resolve().parents[1]
    / 'src/server/package/model_explorer_debugger'
)
# The known cycles were broken on 2026-09-15 (fsutil, device_ids,
# runner_launch, cross_runtime_common).
KNOWN_CYCLES: tuple[set[str], ...] = ()


def module_names() -> dict[str, pathlib.Path]:
  """Maps module dot-paths to their corresponding filesystem paths."""
  names = {}
  for path in PACKAGE.rglob('*.py'):
    relative = path.relative_to(PACKAGE).with_suffix('')
    parts = list(relative.parts)
    if parts[-1] == '__init__':
      parts = parts[:-1]
    names['.'.join(parts) or '__init__'] = path
  return names


def edges(names: dict[str, pathlib.Path]) -> dict[str, set[str]]:
  """Constructs an adjacency list representing imports between modules."""
  graph = {name: set() for name in names}
  for name, path in names.items():
    tree = ast.parse(path.read_text(), filename=str(path))
    package_parts = list(path.relative_to(PACKAGE).parts[:-1])
    for node in ast.walk(tree):
      if not isinstance(node, ast.ImportFrom):
        continue
      if node.level:
        base = (
            package_parts[: len(package_parts) - (node.level - 1)]
            if node.level > 1
            else package_parts
        )
        target = '.'.join(base + ([node.module] if node.module else []))
      elif node.module and node.module.startswith('model_explorer_debugger'):
        target = node.module[len('model_explorer_debugger') :].lstrip('.')
      else:
        continue
      candidates = [target] + [
          (target + '.' + alias.name).lstrip('.') for alias in node.names
      ]
      for candidate in candidates:
        if candidate in names and candidate != name:
          graph[name].add(candidate)
  return graph


def cycles(graph: dict[str, set[str]]) -> list[set[str]]:
  """Finds strongly connected components of size > 1 via Tarjan's algorithm."""
  index, low, stack, on_stack, found, counter = {}, {}, [], set(), [], [0]
  sys.setrecursionlimit(10000)

  def visit(node: str) -> None:
    index[node] = low[node] = counter[0]
    counter[0] += 1
    stack.append(node)
    on_stack.add(node)
    for target in graph[node]:
      if target not in index:
        visit(target)
        low[node] = min(low[node], low[target])
      elif target in on_stack:
        low[node] = min(low[node], index[target])
    if low[node] == index[node]:
      component = []
      while True:
        item = stack.pop()
        on_stack.discard(item)
        component.append(item)
        if item == node:
          break
      if len(component) > 1:
        found.append(set(component))

  for node in graph:
    if node not in index:
      visit(node)
  return found


def main() -> None:
  """Discovers and reports import cycles in the Server package."""
  found = cycles(edges(module_names()))
  unexpected = [c for c in found if c not in KNOWN_CYCLES]
  resolved = [c for c in KNOWN_CYCLES if c not in found]
  for cycle in found:
    status = 'known' if cycle in KNOWN_CYCLES else 'NEW'
    print(f'cycle ({status}): ' + ' <-> '.join(sorted(cycle)))
  if unexpected:
    raise SystemExit(
        f'check_import_cycles: {len(unexpected)} new import cycle(s); break'
        ' them or document them in KNOWN_CYCLES'
    )
  if resolved:
    print(
        'Resolved cycles still listed in KNOWN_CYCLES (remove them): '
        + '; '.join(' <-> '.join(sorted(c)) for c in resolved)
    )
  print(f'Import cycles: {len(found)} known, 0 new.')


if __name__ == '__main__':
  main()
