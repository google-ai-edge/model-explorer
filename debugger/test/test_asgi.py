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
"""Tests for ASGI application endpoints and workspace lease lifecycle."""

import asyncio
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi.testclient import TestClient
import inline_asgi
from model_explorer_debugger import asgi
from model_explorer_debugger import store as store_module
from model_explorer_debugger.analysis import AnalysisBusy
from model_explorer_debugger.session_metadata import SessionMetadata
from model_explorer_debugger.store import SessionStore
from starlette.requests import Request
from test_job_admission import JobAdmissionTests

ROOT = Path(__file__).resolve().parents[1] / 'examples/gemma4-e2b'


class DirectAnalysis:

  def query(self, *args, **kwargs):
    raise AnalysisBusy('Analysis queue is full')

  def diagnostics(self):
    return {'active': 0}

  def close(self):
    pass


class ASGITests(unittest.TestCase):

  def test_host_origin_input_static_and_health(self):
    app = asgi.create_app(SessionStore(ROOT), analysis=DirectAnalysis())
    with TestClient(app, base_url='http://127.0.0.1') as client:
      self.assertEqual(client.get('/api/health').status_code, 200)
      self.assertEqual(
          client.get(
              '/api/health', headers={'Host': 'evil.example'}
          ).status_code,
          403,
      )
      self.assertEqual(
          client.post(
              '/api/sessions/rename',
              json={},
              headers={'Origin': 'http://evil.example'},
          ).status_code,
          403,
      )
      self.assertEqual(
          client.post('/api/sessions/rename', json=[]).status_code, 400
      )
      self.assertEqual(client.get('/missing').status_code, 404)
      result = client.get('/api/overview')
      self.assertEqual(result.status_code, 503)
      self.assertEqual(result.headers['retry-after'], '1')
      self.assertTrue(result.headers['x-request-id'])
      self.assertEqual(client.get('/api/session').status_code, 200)
    self.assertTrue(app.state.application.closed)

  def test_workspace_has_one_coordinator(self):
    with tempfile.TemporaryDirectory() as root:
      lease = asgi.WorkspaceLease(root)
      try:
        with self.assertRaisesRegex(RuntimeError, 'active server'):
          asgi.WorkspaceLease(root)
      finally:
        lease.close()
      asgi.WorkspaceLease(root).close()

  def test_server_faults_are_logged_and_reported_without_details(self):
    faults = [
        KeyError('internal_key'),
        TypeError('unsupported operand'),
        RuntimeError('invariant broken'),
        OSError(28, 'No space left on device', '/private/workspace/x.bin'),
    ]
    for fault in faults:
      with self.subTest(fault=type(fault).__name__):
        app = asgi.create_app(SessionStore(ROOT), analysis=DirectAnalysis())
        with (
            mock.patch.object(
                app.state.application,
                'get',
                side_effect=fault,
                autospec=True,
                spec_set=True,
            ),
            TestClient(app, base_url='http://127.0.0.1') as client,
            self.assertLogs('model_explorer_debugger.http', 'ERROR'),
        ):
          response = client.get('/api/session')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {'error': 'internal_server_error'})

  def test_missing_user_file_is_404_without_its_path(self):
    store = SessionStore(ROOT)
    # The record content is irrelevant: the tensor read itself is faked.
    store._resources['resource'] = {}
    missing = FileNotFoundError(2, 'No such file', '/private/workspace/x.bin')
    with (
        mock.patch.object(
            store_module,
            'load_tensor',
            side_effect=missing,
            autospec=True,
            spec_set=True,
        ),
        inline_asgi.client(store) as http,
    ):
      response = http.get(
          '/api/resource?' + urlencode({'resource_id': 'resource'})
      )
    self.assertEqual(response.status_code, 404)
    self.assertNotIn('/private', response.text)

  def test_missing_server_file_is_500_without_its_path(self):
    app = asgi.create_app(SessionStore(ROOT), analysis=DirectAnalysis())
    missing = FileNotFoundError(2, 'No such file', '/usr/local/bin/iproxy')
    with (
        mock.patch.object(
            app.state.application,
            'get',
            side_effect=missing,
            autospec=True,
            spec_set=True,
        ),
        TestClient(app, base_url='http://127.0.0.1') as client,
        self.assertLogs('model_explorer_debugger.http', 'ERROR'),
    ):
      response = client.get('/api/session')
    self.assertEqual(response.status_code, 500)
    self.assertEqual(response.json(), {'error': 'internal_server_error'})

  def test_client_faults_are_400_and_unknown_resources_are_404(self):
    with inline_asgi.client(SessionStore(ROOT)) as http:
      response = http.get('/api/kv-analysis')
      self.assertEqual(response.status_code, 400)
      self.assertEqual(response.json(), {'error': 'missing_field: turn'})
      response = http.post('/api/resources/compare', json={'target': 'x'})
      self.assertEqual(response.status_code, 400)
      self.assertEqual(response.json(), {'error': 'missing_field: reference'})
      response = http.get('/api/comparisons?batch=999999')
      self.assertEqual(response.status_code, 404)
      self.assertEqual(response.json(), {'error': 'unknown_batch'})
      response = http.get('/api/resource?resource_id=missing')
      self.assertEqual(response.status_code, 404)
      self.assertEqual(response.json(), {'error': 'unknown_resource'})

  def test_streaming_upload(self):
    app = asgi.create_app(SessionStore(ROOT), analysis=DirectAnalysis())
    metadata = mock.create_autospec(
        SessionMetadata, instance=True, spec_set=True
    )

    def consume(stream, size, name):
      data = stream.read(size)
      self.assertEqual(data, b'x' * 200000)
      return {'name': name, 'size': len(data)}

    metadata.upload.side_effect = consume
    app.state.application.metadata = metadata
    with TestClient(app, base_url='http://127.0.0.1') as client:
      response = client.post(
          '/api/artifacts/upload',
          content=b'x' * 200000,
          headers={
              'Content-Type': 'application/octet-stream',
              'X-File-Name': 'test.bin',
          },
      )
      self.assertEqual(response.status_code, 200, response.text)
      self.assertEqual(response.json()['size'], 200000)

  def test_sse_resume_drains_every_page_of_terminal_job(self):
    case = JobAdmissionTests()
    case.setUp()
    self.addCleanup(case.doCleanups)
    job = case.initialize()
    with case.manager.lock:
      internal = case.manager.jobs[job['id']]
      for number in range(600):
        case.manager.record_event(internal, 'progress', message=str(number))
    with mock.patch(
        'model_explorer_debugger.jobs.JobManager',
        return_value=case.manager,
        autospec=True,
        spec_set=True,
    ):
      app = asgi.create_app(registry=case.registry, analysis=DirectAnalysis())
    with TestClient(app, base_url='http://127.0.0.1') as client:
      response = client.get(
          '/api/jobs/' + job['id'] + '/events', headers={'Last-Event-ID': '10'}
      )
      self.assertEqual(response.status_code, 200)
      ids = [
          int(line[4:])
          for line in response.text.splitlines()
          if line.startswith('id: ')
      ]
      self.assertEqual(ids, list(range(11, internal['sequence'] + 1)))

  def test_truncated_upload_removes_partial_file(self):
    with tempfile.TemporaryDirectory() as root:
      metadata = SessionMetadata(SimpleNamespace(root=Path(root), session={}))
      messages = iter(
          [{'type': 'http.request', 'body': b'a' * 100000, 'more_body': False}]
      )

      async def receive():
        return next(messages)

      request = Request(
          {
              'type': 'http',
              'headers': [
                  (b'content-length', b'200000'),
                  (b'content-type', b'application/octet-stream'),
                  (b'x-file-name', b'capture.bin'),
              ],
          },
          receive,
      )
      with self.assertRaisesRegex(ValueError, 'interrupted'):
        asyncio.run(asgi.upload(request, metadata))
      self.assertEqual(list(Path(root).rglob('capture.bin')), [])

  def test_stalled_upload_is_interrupted_and_cleaned_up(self):
    with tempfile.TemporaryDirectory() as root:
      metadata = SessionMetadata(SimpleNamespace(root=Path(root), session={}))

      async def receive():
        await asyncio.sleep(3600)

      request = Request(
          {
              'type': 'http',
              'headers': [
                  (b'content-length', b'10'),
                  (b'content-type', b'application/octet-stream'),
                  (b'x-file-name', b'stalled.bin'),
              ],
          },
          receive,
      )
      with (
          mock.patch.object(asgi, 'UPLOAD_IDLE_TIMEOUT_SECONDS', 0.05),
          self.assertRaisesRegex(ValueError, 'stalled'),
      ):
        asyncio.run(asgi.upload(request, metadata))
      self.assertEqual(list(Path(root).rglob('stalled.bin')), [])

  def test_slow_but_steady_upload_outlives_the_idle_timeout(self):
    with tempfile.TemporaryDirectory() as root:
      metadata = SessionMetadata(SimpleNamespace(root=Path(root), session={}))
      chunks = [b'a' * 4, b'b' * 3, b'c' * 3]

      async def receive():
        await asyncio.sleep(0.03)
        body = chunks.pop(0)
        return {'type': 'http.request', 'body': body, 'more_body': bool(chunks)}

      request = Request(
          {
              'type': 'http',
              'headers': [
                  (b'content-length', b'10'),
                  (b'content-type', b'application/octet-stream'),
                  (b'x-file-name', b'steady.bin'),
              ],
          },
          receive,
      )
      # Total transfer (~90 ms) exceeds the idle timeout; each gap does not.
      with mock.patch.object(asgi, 'UPLOAD_IDLE_TIMEOUT_SECONDS', 0.05):
        result = asyncio.run(asgi.upload(request, metadata))
      self.assertEqual(
          (Path(root) / result['artifact']).read_bytes(), b'aaaabbbccc'
      )

  def test_cli_socket_startup_and_shutdown(self):
    with tempfile.TemporaryDirectory() as root:
      with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
      process = subprocess.Popen(
          [
              sys.executable,
              '-m',
              'model_explorer_debugger.server',
              '--workspace',
              root,
              '--port',
              str(port),
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.PIPE,
      )
      try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
          try:
            with urlopen(
                f'http://127.0.0.1:{port}/api/diagnostics', timeout=1
            ) as response:
              self.assertIn(b'sqlite', response.read())
              break
          except OSError:
            if process.poll() is not None:
              self.fail(process.stderr.read().decode())
            time.sleep(0.05)
        else:
          self.fail('Server startup deadline exceeded')
        with self.assertRaises(RuntimeError):
          asgi.WorkspaceLease(root)
      finally:
        process.terminate()
        _, errors = process.communicate(timeout=10)
      self.assertIn(process.returncode, (0, -15), errors.decode())
      self.assertIn('Application shutdown complete', errors.decode())
      asgi.WorkspaceLease(root).close()
