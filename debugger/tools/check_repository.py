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

"""Check source boundaries, required resources, and repository hygiene."""

import ast
import hashlib
import pathlib
import re
import subprocess

GRAPH_EXECUTION_UPSTREAM_PREFIX: str = 'src/ui/public/graph-execution/upstream/'
MODEL_EXPLORER_UPSTREAM_DIR: str = (
    f'{GRAPH_EXECUTION_UPSTREAM_PREFIX}model-explorer'
)
REQUIRED_FILES: tuple[str, ...] = (
    'src/ui/package.json',
    'src/ui/package-lock.json',
    'src/ui/angular.json',
    'src/ui/tsconfig.app.json',
    'src/ui/src/main.ts',
    'src/ui/src/index.html',
    'src/ui/src/proxy.conf.json',
    'src/ui/scripts/version_graph_execution.mjs',
    'src/ui/public/graph-execution/index.html',
    'src/ui/public/graph-execution/execution-data.js',
    'src/ui/public/graph-execution/execution-view.js',
    f'{MODEL_EXPLORER_UPSTREAM_DIR}/PROVENANCE.json',
    f'{MODEL_EXPLORER_UPSTREAM_DIR}/dist/main_browser.js',
    f'{MODEL_EXPLORER_UPSTREAM_DIR}/dist/worker.js',
    'examples/gemma4-e2b/semantic.json',
    'ci/build_ui.sh',
    'pyproject.toml',
    'src/server/pyproject.toml',
    'src/server/package/model_explorer_debugger/server.py',
    'src/server/package/model_explorer_debugger/asgi.py',
    'ci/test_server.sh',
    'ci/check_repository.sh',
    'ci/test_contracts.sh',
    'ci/test_python_runner.sh',
    'ci/check_apple.sh',
    'src/server/package/model_explorer_debugger/runtime/gemma4_e2b_taps.json',
    'src/capture/pytorch/README.md',
    'src/capture/pytorch/pyproject.toml',
    'src/capture/pytorch/package/ai_edge_debugger_pytorch/__init__.py',
    'src/capture/pytorch/package/ai_edge_debugger_pytorch/capture.py',
    'src/capture/pytorch/tests/test_capture.py',
    'src/contracts/python/package/model_debugger_contracts/__init__.py',
    'src/runner/python/package/model_debugger_runner/python_runner_host.py',
    'src/runner/apple/Runner/RunnerApp.swift',
    'src/runner/apple/Core/NativeRunner.swift',
    'src/runner/apple/ModelDebuggerRunner.xcodeproj/project.pbxproj',
    'src/runner/ui/index.html',
    'src/runner/ui/debugger-logo.svg',
    'src/runner/apple/Brand/debugger-logo-1024.png',
    'src/runner/apple/Brand/source.sha256',
    'src/runner/apple/Brand/LICENSE-model-explorer.txt',
    'src/runner/apple/CLiteRTLM/LICENSE',
    'src/runner/apple/CLiteRTLM/engine.h',
    'src/runner/apple/CLiteRTLM/conversation.h',
    'src/runner/apple/CLiteRTLM/experimental.h',
    'src/runner/apple/CLiteRTLM/module.modulemap',
)
GENERATED_DIRECTORIES: frozenset[str] = frozenset({
    '.venv',
    '__pycache__',
    'node_modules',
    '.angular',
    '.build',
    'Vendor',
    'xcuserdata',
    'build',
    'dist',
    'out-tsc',
    'test-results',
    'playwright-report',
    '.local-runtime',
    '.debugger-artifacts',
})
# This dist is a pinned runtime dependency, not output from this repository.
PINNED_RUNTIME_DIRECTORIES: frozenset[str] = frozenset({
    f'{MODEL_EXPLORER_UPSTREAM_DIR}/dist',
})
SOURCE_SUFFIXES: frozenset[str] = frozenset({
    '.py',
    '.sh',
    '.swift',
    '.toml',
    '.html',
    '.js',
    '.ts',
    '.pbxproj',
})
# Every first-party source of these kinds carries the shared Apache 2.0 header;
# copy.bara.sky enforces the same rule on export.
LICENSE_HEADER_SUFFIXES: frozenset[str] = frozenset({
    '.cjs',
    '.css',
    '.html',
    '.js',
    '.mjs',
    '.py',
    '.scss',
    '.sh',
    '.swift',
    '.ts',
})
# Pinned upstream files and verbatim upstream evidence keep their own
# license headers.
LICENSE_HEADER_EXEMPT_PREFIXES: tuple[str, ...] = (
    'examples/gemma4-e2b/evidence/',
    GRAPH_EXECUTION_UPSTREAM_PREFIX,
)
LICENSE_HEADER_PATTERN: re.Pattern[str] = re.compile(
    r'Copyright 20\d\d The AI Edge Model Explorer Authors\.'
    r'[\s\S]*?Licensed under the Apache License, Version 2\.0'
)
# The header must start within this many leading characters, which leaves room
# for a shebang or doctype line.
LICENSE_HEADER_WINDOW = 256


def generated_source_path(name: str) -> bool:
  """Returns True if the path points to a generated or build artifact."""
  parts = pathlib.Path(name).parts
  return any(
      (
          part in GENERATED_DIRECTORIES
          and '/'.join(parts[: index + 1]) not in PINNED_RUNTIME_DIRECTORIES
      )
      or part.endswith('.egg-info')
      or part.startswith('.debugger-')
      for index, part in enumerate(parts)
  )


def validate_repository_file(
    name: str,
    root: pathlib.Path,
    required: set[str],
    errors: list[str],
) -> pathlib.Path | None:
  """Validates that a relative path stays within the repository root.

  Args:
    name: Relative path of the file to check.
    root: Base repository root directory.
    required: Set of relative paths that are required to exist.
    errors: Accumulator list collecting repository validation error strings.

  Returns:
    Path to the regular file within the repository, or None if skipped/invalid.
  """
  relative = pathlib.Path(name)
  if relative.is_absolute() or '..' in relative.parts:
    errors.append(f'{name}: source must stay inside this repository')
    return None
  path = root
  for part in relative.parts:
    path /= part
    if path.is_symlink():
      errors.append(f'{name}: file or ancestor is a symlink')
      return None
  if not path.exists():
    # git ls-files includes unstaged deletions. Only required resources
    # must remain present; removing ordinary source is a valid change.
    if name in required:
      errors.append(f'{name}: required source/resource is missing')
    return None
  if not path.is_file() or not path.resolve().is_relative_to(root):
    errors.append(
        f'{name}: source must be a regular file inside this repository'
    )
    return None
  return path


def validate_python_import_boundaries(
    name: str, content: str, errors: list[str]
) -> None:
  """Validates Python syntax and contracts/runner package import boundaries."""
  try:
    tree = ast.parse(content, filename=name)
  except SyntaxError as error:
    errors.append(f'{name}:{error.lineno}: {error.msg}')
    return
  if name.startswith((
      'src/capture/pytorch/',
      'src/runner/python/',
      'src/contracts/python/',
  )):
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        modules = [item.name for item in node.names]
      elif isinstance(node, ast.ImportFrom):
        modules = [node.module or '']
      else:
        modules = []
      if any(
          module.split('.')[0] == 'model_explorer_debugger'
          for module in modules
      ):
        errors.append(f'{name}: execution/contracts package imports Server')
      if name.startswith('src/capture/pytorch/') and any(
          module.split('.')[0] == 'model_debugger_runner' for module in modules
      ):
        errors.append(f'{name}: capture package imports Runner')


def validate_source_content(
    name: str, path: pathlib.Path, errors: list[str]
) -> None:
  """Validates source text encoding, path markers, and AST boundaries."""
  try:
    content = path.read_text(encoding='utf-8')
  except UnicodeError:
    errors.append(f'{name}: source must be UTF-8 text')
    return
  if '/'.join(('', 'model-explorer-debugger', '')) in content:
    errors.append(f'{name}: executable source references the original checkout')
  if path.suffix == '.py':
    validate_python_import_boundaries(name, content, errors)


def validate_branding_assets(
    checked: dict[str, pathlib.Path], errors: list[str]
) -> None:
  """Verifies the Runner SVG logo hash matches the approved raster source."""
  svg = checked.get('src/runner/ui/debugger-logo.svg')
  logo_hash = checked.get('src/runner/apple/Brand/source.sha256')
  if (
      svg
      and logo_hash
      and hashlib.sha256(svg.read_bytes()).hexdigest()
      != logo_hash.read_text(encoding='utf-8').strip()
  ):
    errors.append(
        'Runner logo: SVG differs from the approved native raster source hash'
    )


def validate_license_header(
    name: str, path: pathlib.Path, errors: list[str]
) -> None:
  """Validates that a first-party source file starts with the license header."""
  if path.suffix not in LICENSE_HEADER_SUFFIXES or name.startswith(
      LICENSE_HEADER_EXEMPT_PREFIXES
  ):
    return
  try:
    content = path.read_text(encoding='utf-8')
  except UnicodeError:
    return  # validate_source_content reports the encoding error.
  if not content.strip():
    return  # An empty file, such as a package __init__.py, has no code.
  match = LICENSE_HEADER_PATTERN.search(content)
  if match is None or match.start() > LICENSE_HEADER_WINDOW:
    errors.append(f'{name}: missing the Apache 2.0 license header')


def main() -> None:
  """Verifies source boundaries, required resources, and repository hygiene."""
  root = pathlib.Path(__file__).resolve().parents[1]
  errors = []
  required = set(REQUIRED_FILES)
  try:
    visible = (
        subprocess.check_output(
            [
                'git',
                'ls-files',
                '--cached',
                '--others',
                '--exclude-standard',
                '-z',
            ],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .split('\0')
    )
    names = set(filter(None, visible)) | required
  except (subprocess.CalledProcessError, FileNotFoundError):
    # Fall back to filesystem rglob when git is unavailable.
    names = set()
    for path in root.rglob('*'):
      if path.is_file():
        rel = path.relative_to(root).as_posix()
        if not generated_source_path(rel) and not rel.startswith('.git/'):
          names.add(rel)
    names |= required
  checked = {}

  for name in sorted(names):
    path = validate_repository_file(name, root, required, errors)
    if path is None:
      continue
    checked[name] = path
    generated = generated_source_path(name)
    local_file = path.name == '.DS_Store' or (
        path.name != '.env.example'
        and (path.name == '.env' or path.name.startswith('.env.'))
    )
    if (
        generated
        or local_file
        or path.suffix in {'.pyc', '.pyo', '.pyd', '.profraw', '.xcuserstate'}
    ):
      errors.append(
          f'{name}: generated or machine-local artifact is included in source'
      )
    if (
        name.startswith(('src/', 'ci/', 'tools/'))
        and path.suffix in SOURCE_SUFFIXES
    ):
      validate_source_content(name, path, errors)
    if not generated:
      validate_license_header(name, path, errors)

  validate_branding_assets(checked, errors)
  if errors:
    raise SystemExit('\n'.join(errors))
  print(
      f'Repository checks passed: {len(checked)} current source/resource files.'
  )


if __name__ == '__main__':
  main()
