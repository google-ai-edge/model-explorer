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

"""Startup defaults: the checkout workspace, its built UI and port 8080.

Also covers remembered tool roots.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from model_explorer_debugger import server


@contextmanager
def listening():
  """An OS-assigned busy port that nothing else answers on."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(('127.0.0.1', 0))
    sock.listen(1)
    yield sock.getsockname()[1]


@contextmanager
def another_server(payload):
  """A minimal HTTP server whose /api/diagnostics answers with `payload`."""

  class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
      body = json.dumps(payload).encode()
      self.send_response(200)
      self.send_header('Content-Type', 'application/json')
      self.send_header('Content-Length', str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def log_message(self, *args):
      pass

  httpd = HTTPServer(('127.0.0.1', 0), Handler)
  thread = threading.Thread(target=httpd.serve_forever, daemon=True)
  thread.start()
  try:
    yield httpd.server_address[1]
  finally:
    httpd.shutdown()
    httpd.server_close()
    thread.join(5)


class PortSelectionTests(unittest.TestCase):

  def test_the_default_port_is_used_when_free(self):
    with listening() as busy:
      pass  # closed again: the port is free now
    self.assertEqual(server.pick_port(None, default=busy), (busy, None))

  def test_a_busy_default_port_moves_to_the_next_free_port_and_says_why(self):
    with listening() as busy:
      port, note = server.pick_port(None, default=busy)
      self.assertNotEqual(port, busy)
      self.assertFalse(server.port_in_use(port))
      self.assertLess(port - busy, server.PORT_SPAN)
      self.assertEqual(note, '%d in use' % busy)

  def test_an_exhausted_span_falls_back_to_any_free_port(self):
    with listening() as busy:
      port, note = server.pick_port(None, default=busy, span=1)
      self.assertNotEqual(port, busy)
      self.assertFalse(server.port_in_use(port))
      self.assertEqual(note, '%d in use' % busy)

  def test_an_explicit_port_must_be_free(self):
    with listening() as busy:
      with self.assertRaises(SystemExit) as raised:
        server.pick_port(busy)
      self.assertEqual(
          str(raised.exception),
          'port %d is in use; choose another --port' % busy,
      )
    self.assertEqual(server.pick_port(busy), (busy, None))

  def test_a_port_held_by_another_model_debugger_server_is_named(self):
    with another_server({'status': 'ok', 'package': server.PACKAGE}) as busy:
      port, note = server.pick_port(None, default=busy)
      self.assertNotEqual(port, busy)
      self.assertEqual(note, '%d used by another Model Debugger Server' % busy)
      with self.assertRaises(SystemExit) as raised:
        server.pick_port(busy)
      self.assertIn(
          'used by another Model Debugger Server', str(raised.exception)
      )
    with another_server({'status': 'ok'}) as busy:
      self.assertEqual(
          server.pick_port(None, default=busy)[1], '%d in use' % busy
      )

  def test_the_command_line_entry_point_names_the_occupant_too(self):
    # The package marker must still match under `python -m`, where the module
    # is __main__.
    import subprocess
    import sys

    with (
        another_server(
            {'status': 'ok', 'package': 'model_explorer_debugger'}
        ) as busy,
        tempfile.TemporaryDirectory() as data,
    ):
      result = subprocess.run(
          [
              sys.executable,
              '-m',
              'model_explorer_debugger.server',
              '--data',
              data,
              '--port',
              str(busy),
          ],
          capture_output=True,
          text=True,
          timeout=60,
      )
    self.assertEqual(result.returncode, 1, result.stderr)
    self.assertIn(
        'port %d is used by another Model Debugger Server; choose another'
        ' --port' % busy,
        result.stderr,
    )


def fake_checkout(root, ui=True, sibling=True):
  (root / 'src' / 'server' / 'package').mkdir(parents=True)
  if ui:
    browser = root / server.UI_BUILD
    browser.mkdir(parents=True)
    (browser / 'index.html').write_text('<!doctype html>')
  if sibling:
    python = root.parent / 'debugger_runtime' / '.venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.write_text('')
  return root


class ServerCliTests(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name).resolve()
    self.checkout = fake_checkout(self.root / 'ai-edge-debugger')

  def parse(self, *argv, checkout=None):
    args = server.build_parser().parse_args(list(argv))
    return args, server.resolve(
        args, checkout=self.checkout if checkout is None else checkout
    )

  def test_no_flags_use_the_checkout_workspace_ui_port_and_sibling_runtime(
      self,
  ):
    args, sources = self.parse()
    self.assertEqual(args.workspace, self.checkout / server.WORKSPACE)
    self.assertEqual(args.ui, self.checkout / server.UI_BUILD)
    self.assertIsNone(
        args.port, 'the port is chosen at startup so a busy 8080 can move'
    )
    self.assertEqual(server.DEFAULT_PORT, 8080)
    self.assertEqual(args.runtime_root, self.root / 'debugger_runtime')
    self.assertIsNone(args.pytorch_root)
    self.assertEqual(
        sources,
        {
            'workspace': 'default',
            'ui': 'checkout build',
            'runtime_root': 'sibling checkout',
        },
    )
    self.assertFalse(
        (args.workspace / server.SETTINGS_FILE).exists(),
        'detected values are not remembered',
    )
    args.port, moved = (
        server.DEFAULT_PORT + 1,
        '%d in use' % server.DEFAULT_PORT,
    )
    sources['port'] = moved
    text = server.summary(args, sources)
    self.assertIn('URL:           http://127.0.0.1:8081/ (8080 in use)', text)
    self.assertIn('sibling checkout', text)

  def test_explicit_flags_win_and_are_not_attributed_to_defaults(self):
    args, sources = self.parse(
        '--workspace',
        str(self.root / 'ws'),
        '--ui',
        str(self.root / 'ui'),
        '--port',
        '9000',
    )
    self.assertEqual(
        (args.workspace, args.ui, args.port),
        (self.root / 'ws', self.root / 'ui', 9000),
    )
    self.assertNotIn('workspace', sources)
    self.assertNotIn('ui', sources)

  def test_without_a_ui_build_the_server_is_api_only(self):
    checkout = fake_checkout(
        self.root / 'alone' / 'bare', ui=False, sibling=False
    )
    args, sources = self.parse(checkout=checkout)
    self.assertIsNone(args.ui)
    self.assertIsNone(args.runtime_root)
    self.assertIn('API only', server.summary(args, sources))
    self.assertIn('not configured', server.summary(args, sources))

  def test_runtime_and_pytorch_roots_are_remembered_per_workspace(self):
    workspace, first, second = (
        self.root / 'ws',
        self.root / 'runtime-a',
        self.root / 'runtime-b',
    )
    args, sources = self.parse(
        '--workspace', str(workspace), '--runtime-root', str(first)
    )
    self.assertEqual(sources['runtime_root'], 'flag')
    self.assertEqual(
        json.loads((workspace / server.SETTINGS_FILE).read_text()),
        {'runtime_root': str(first)},
    )
    args, sources = self.parse('--workspace', str(workspace))
    self.assertEqual(
        args.runtime_root,
        first,
        'the remembered root beats the sibling checkout',
    )
    self.assertTrue(sources['runtime_root'].startswith('remembered in '))
    args, _ = self.parse(
        '--workspace',
        str(workspace),
        '--runtime-root',
        str(second),
        '--pytorch-root',
        str(second),
    )
    self.assertEqual(
        json.loads((workspace / server.SETTINGS_FILE).read_text()),
        {'runtime_root': str(second), 'pytorch_root': str(second)},
    )
    args, sources = self.parse('--workspace', str(workspace))
    self.assertEqual((args.runtime_root, args.pytorch_root), (second, second))
    (workspace / server.SETTINGS_FILE).write_text('not json')
    args, sources = self.parse('--workspace', str(workspace))
    self.assertEqual(
        args.runtime_root,
        self.root / 'debugger_runtime',
        'a corrupt settings file is ignored',
    )

  def test_saved_session_mode_has_no_workspace_and_remembers_nothing(self):
    data = self.root / 'saved'
    args, sources = self.parse('--data', str(data), '--runtime-root', str(data))
    self.assertIsNone(args.workspace)
    self.assertEqual(args.ui, self.checkout / server.UI_BUILD)
    self.assertEqual(sources, {'ui': 'checkout build'})
    self.assertEqual(list(self.root.rglob(server.SETTINGS_FILE)), [])
    self.assertIn('saved session', server.summary(args, sources))

  def test_data_and_workspace_stay_mutually_exclusive(self):
    with self.assertRaises(SystemExit):
      server.build_parser().parse_args(['--data', 'a', '--workspace', 'b'])

  def test_detached_install_falls_back_to_the_working_directory(self):
    with (
        patch.object(server, 'checkout_root', return_value=None),
        patch.object(Path, 'cwd', return_value=self.root / 'elsewhere'),
    ):
      args = server.build_parser().parse_args([])
      sources = server.resolve(args)
    self.assertEqual(args.workspace, self.root / 'elsewhere' / server.WORKSPACE)
    self.assertIsNone(args.ui)
    self.assertIsNone(args.runtime_root)
    self.assertEqual(sources, {'workspace': 'default'})

  def test_checkout_root_recognizes_this_checkout(self):
    root = server.checkout_root()
    self.assertIsNotNone(root)
    self.assertTrue(
        (
            root
            / 'src'
            / 'server'
            / 'package'
            / 'model_explorer_debugger'
            / 'server.py'
        ).is_file()
    )


if __name__ == '__main__':
  unittest.main()
