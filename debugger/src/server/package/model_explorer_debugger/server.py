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

"""CLI entry point serving a saved-session API or a local runtime workspace.

Both modes are served over FastAPI/Uvicorn. Started without flags from an
editable checkout, the Server opens `.local-runtime/workspace`, serves the built
UI when `src/ui/dist` exists, listens on 8080 and finds LiteRT preparation tools
in the sibling `debugger_runtime` checkout. `--runtime-root` and
`--pytorch-root` are remembered in the workspace once given. When 8080 is busy
and no `--port` was given, the next free port is used and shown; an explicit
`--port` must be free.
"""

import argparse
import json
from pathlib import Path
import socket
import sys
from urllib.request import urlopen

from model_explorer_debugger.analysis import AnalysisPool
from model_explorer_debugger.asgi import WorkspaceLease, create_app
from model_explorer_debugger.fsutil import atomic_json
from model_explorer_debugger.session_registry import SessionRegistry
from model_explorer_debugger.store import SessionStore
import uvicorn

DEFAULT_PORT = 8080
PORT_SPAN = 20
PACKAGE = (
    __package__  # also correct under `python -m`, where __name__ is __main__
)
SETTINGS_FILE = 'server-settings.json'
REMEMBERED = ('runtime_root', 'pytorch_root')
UI_BUILD = Path('src/ui/dist/model_explorer_debugger/browser')
WORKSPACE = Path('.local-runtime/workspace')


def checkout_root():
  """Returns the repository this package is installed from.

  Returns None for a detached install.
  """
  parents = Path(__file__).resolve().parents
  root = parents[4] if len(parents) > 4 else None
  return (
      root if root and (root / 'src' / 'server' / 'package').is_dir() else None
  )


def build_parser():
  parser = argparse.ArgumentParser(
      prog='model_explorer_debugger.server',
      description=(
          'Model Debugger Server. Without flags: the checkout workspace, its'
          ' built UI and port %d.'
      )
      % DEFAULT_PORT,
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument(
      '--data',
      type=Path,
      help='Read one saved session directory instead of a workspace',
  )
  mode.add_argument(
      '--workspace',
      type=Path,
      help='Writable workspace (default: <checkout>/%s)' % WORKSPACE,
  )
  parser.add_argument(
      '--runtime-root',
      type=Path,
      help=(
          'debugger_runtime checkout with its .venv for LiteRT capture'
          ' preparation (remembered; default: the remembered value, then the'
          ' sibling checkout)'
      ),
  )
  parser.add_argument(
      '--pytorch-root',
      type=Path,
      help='PyTorch runtime directory with its .venv (remembered)',
  )
  parser.add_argument(
      '--pytorch-model',
      type=Path,
      action='append',
      default=[],
      help=(
          'Register a local Hugging Face model directory (repeatable;'
          ' registrations persist)'
      ),
  )
  parser.add_argument(
      '--model',
      type=Path,
      help=(
          'Register a local .litertlm file without uploading its bytes'
          ' (registration persists)'
      ),
  )
  parser.add_argument('--tap-manifest', type=Path)
  parser.add_argument(
      '--semantic',
      type=Path,
      help='Reviewed semantic graph associated with the tap profile',
  )
  parser.add_argument(
      '--ui',
      type=Path,
      help='Built UI directory (default: <checkout>/%s when built)' % UI_BUILD,
  )
  parser.add_argument(
      '--port',
      type=int,
      help=(
          'Port to listen on; it must be free (default: %d, or the next free'
          ' port up to %d when %d is busy)'
      )
      % (DEFAULT_PORT, DEFAULT_PORT + PORT_SPAN - 1, DEFAULT_PORT),
  )
  parser.add_argument('--analysis-workers', type=int, default=2)
  parser.add_argument('--analysis-waiting', type=int, default=8)
  parser.add_argument('--analysis-timeout', type=float, default=60)
  parser.add_argument('--analysis-memory-mib', type=int, default=8192)
  return parser


def _read_settings(path):
  try:
    value = json.loads(path.read_text())
  except (OSError, ValueError):
    return {}
  return value if isinstance(value, dict) else {}


def port_in_use(port, host='127.0.0.1'):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
      probe.bind((host, port))
    except OSError:
      return True
  return False


def occupant(port, host='127.0.0.1'):
  """Names what holds a busy port when it is one of ours.

  This lets the summary or error say so.
  """
  try:
    with urlopen(
        'http://%s:%d/api/diagnostics' % (host, port), timeout=1
    ) as response:
      value = json.loads(response.read())
  except Exception:
    return 'in use'
  return (
      'used by another Model Debugger Server'
      if isinstance(value, dict) and value.get('package') == PACKAGE
      else 'in use'
  )


def pick_port(
    requested, host='127.0.0.1', default=DEFAULT_PORT, span=PORT_SPAN
):
  """Selects a free port to listen on.

  An explicit port must be free; the default moves to the next free port, then
  to any free port.

  Returns (port, note); the note explains a move for the startup summary.
  """
  if requested is not None:
    if port_in_use(requested, host):
      raise SystemExit(
          'port %d is %s; choose another --port'
          % (requested, occupant(requested, host))
      )
    return requested, None
  if not port_in_use(default, host):
    return default, None
  note = '%d %s' % (default, occupant(default, host))
  for port in range(default + 1, default + span):
    if not port_in_use(port, host):
      return port, note
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.bind((host, 0))
    return probe.getsockname()[1], note


def resolve(args, checkout=None):
  """Fills in defaults and remembered settings.

  Returns {option: source} for the startup summary.
  """
  checkout = checkout_root() if checkout is None else checkout
  sources = {}
  if args.data is None and args.workspace is None:
    args.workspace = (checkout or Path.cwd()) / WORKSPACE
    sources['workspace'] = 'default'
  if args.ui is None and checkout:
    candidate = checkout / UI_BUILD
    if (candidate / 'index.html').is_file():
      args.ui = candidate
      sources['ui'] = 'checkout build'
  if args.workspace is None:
    return sources
  settings_path = args.workspace / SETTINGS_FILE
  remembered = _read_settings(settings_path)
  updated = dict(remembered)
  for key in REMEMBERED:
    given = getattr(args, key)
    if given is not None:
      updated[key] = str(Path(given).resolve())
      sources[key] = 'flag'
    elif remembered.get(key):
      setattr(args, key, Path(remembered[key]))
      sources[key] = 'remembered in %s' % settings_path
  if args.runtime_root is None and checkout:
    sibling = checkout.parent / 'debugger_runtime'
    if (sibling / '.venv' / 'bin' / 'python').is_file():
      args.runtime_root = sibling
      sources['runtime_root'] = 'sibling checkout'
  if updated != remembered:
    atomic_json(settings_path, updated)
  return sources


def summary(args, sources):
  """Builds a human-readable effective configuration, one line per decision."""

  def line(label, value, key):
    origin = sources.get(key)
    return '  %-14s %s%s' % (
        label + ':',
        value,
        ' (%s)' % origin if origin else '',
    )

  lines = ['Model Debugger Server']
  if args.data is not None:
    lines.append(line('saved session', args.data, 'data'))
  else:
    lines.append(line('workspace', args.workspace, 'workspace'))
  lines.append(
      line('UI', args.ui or 'API only; build src/ui to serve the UI', 'ui')
  )
  if args.data is None:
    if args.runtime_root is None:
      runtime = (
          'not configured; LiteRT capture preparation is unavailable (pass'
          ' --runtime-root once)'
      )
    elif not (Path(args.runtime_root) / '.venv' / 'bin' / 'python').is_file():
      runtime = (
          '%s (no .venv/bin/python; LiteRT capture preparation will fail)'
          % args.runtime_root
      )
    else:
      runtime = str(args.runtime_root)
    lines.append(line('runtime root', runtime, 'runtime_root'))
    if args.pytorch_root is not None:
      lines.append(line('pytorch root', args.pytorch_root, 'pytorch_root'))
    if args.model is not None:
      lines.append(
          line(
              'model',
              '%s (registered; the flag is not needed again)' % args.model,
              'model',
          )
      )
  lines.append(
      line(
          'URL',
          'http://127.0.0.1:%d/'
          % (args.port if args.port is not None else DEFAULT_PORT),
          'port',
      )
  )
  return '\n'.join(lines)


def main(argv=None):
  parser = build_parser()
  args = parser.parse_args(argv)
  if (
      args.analysis_workers < 1
      or args.analysis_waiting < 0
      or args.analysis_timeout <= 0
      or args.analysis_memory_mib < 256
  ):
    parser.error('Invalid analysis limits')
  sources = resolve(args)
  # The lease comes first: a second start on the same workspace fails there, not
  # on a moved port.
  lease = WorkspaceLease(args.workspace) if args.workspace else None
  pool = None
  app = None
  try:
    args.port, moved = pick_port(args.port)
    if moved:
      sources['port'] = moved
    print(summary(args, sources), file=sys.stderr, flush=True)
    registry = None
    if args.workspace:
      registry = SessionRegistry(
          args.workspace, args.runtime_root, args.pytorch_root
      )
      for path in args.pytorch_model:
        registry.register_pytorch_model(path)
      if args.model:
        registry.register_model(args.model, args.tap_manifest, args.semantic)
    pool = AnalysisPool(
        args.analysis_workers,
        args.analysis_waiting,
        args.analysis_timeout,
        args.analysis_memory_mib * 1024**2,
    )
    app = create_app(
        SessionStore(args.data) if args.data else None,
        registry,
        args.ui,
        pool,
        lease,
    )
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=args.port,
        workers=1,
        limit_concurrency=128,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=45,
    )
  finally:
    if app:
      app.state.application.close()
    elif pool:
      pool.close()
    if lease:
      lease.close()


if __name__ == '__main__':
  main()
