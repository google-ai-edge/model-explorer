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
"""Tests for background analysis worker pools."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import threading
import time
import unittest
from model_explorer_debugger import analysis
from model_explorer_debugger import errors
from model_explorer_debugger.store import SessionStore
from model_explorer_debugger.store_cache import StoreCache

ROOT = Path(__file__).resolve().parents[1] / 'examples/gemma4-e2b'


def probe(store, path, query, payload):
  if path == 'crash':
    os._exit(19)
  if path == 'memory':
    allocated = bytearray(200 * 1024**2)
    time.sleep(2)
    return len(allocated)
  if path == 'slow':
    time.sleep(2)
  if path == 'rlimit':
    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    return {'soft': soft, 'infinity': resource.RLIM_INFINITY}
  if path == 'invalid':
    raise ValueError('turn must be positive')
  if path == 'unknown':
    raise errors.NotFound('unknown_batch')
  if path == 'busy':
    raise analysis.AnalysisBusy('KV analysis workers are busy; retry shortly.')
  if path == 'bug':
    return query['missing'] + 1
  return {'pid': os.getpid(), 'capture': str(store.root)}


class AnalysisPoolTests(unittest.TestCase):

  def pool(self, **kwargs):
    pool = analysis.AnalysisPool(evaluator=probe, **kwargs)
    self.addCleanup(pool.close)
    return pool

  def test_worker_enforces_rlimit_as(self):
    if os.uname().sysname == 'Darwin':
      self.skipTest('RLIMIT_AS is not supported on Darwin')
    pool = self.pool(timeout=10)
    rlimit_info = pool.query(ROOT, 'rlimit', {})
    self.assertNotEqual(rlimit_info['soft'], rlimit_info['infinity'])
    self.assertGreaterEqual(rlimit_info['soft'], 2 * 1024**3)

  def test_reuses_process_and_replaces_crashed_worker(self):
    pool = self.pool(timeout=10)
    first = pool.query(ROOT, 'pid', {})
    self.assertNotEqual(first['pid'], os.getpid())
    # Two slots are used in round-robin order; both persist.
    pool.query(ROOT, 'pid', {})
    self.assertEqual(first, pool.query(ROOT, 'pid', {}))
    with self.assertRaises(analysis.AnalysisBusy):
      pool.query(ROOT, 'crash', {})
    self.assertTrue(pool.query(ROOT, 'pid', {})['pid'])
    self.assertEqual(pool.diagnostics()['restarts'], 1)

  def test_timeout_cancel_and_overload_release_capacity(self):
    pool = self.pool(workers=1, waiting=0, timeout=10)
    pool.query(ROOT, 'pid', {})
    with ThreadPoolExecutor() as executor:
      cancelled = threading.Event()
      result = executor.submit(
          pool.query, ROOT, 'slow', {}, None, cancelled.is_set
      )
      deadline = time.monotonic() + 3
      while pool.diagnostics()['active'] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
      with self.assertRaisesRegex(analysis.AnalysisBusy, 'queue is full'):
        pool.query(ROOT, 'pid', {})
      cancelled.set()
      with self.assertRaisesRegex(analysis.AnalysisBusy, 'cancelled'):
        result.result(timeout=5)
    pool.timeout = 0.2
    with self.assertRaisesRegex(analysis.AnalysisBusy, 'deadline'):
      pool.query(ROOT, 'slow', {})
    pool.timeout = 10
    self.assertTrue(pool.query(ROOT, 'pid', {})['pid'])

  def test_same_queries_share_one_execution(self):
    pool = self.pool(workers=2, timeout=10)
    with ThreadPoolExecutor() as executor:
      first = executor.submit(pool.query, ROOT, 'slow', {})
      time.sleep(0.2)
      second = executor.submit(pool.query, ROOT, 'slow', {})
      self.assertEqual(first.result(), second.result())
    self.assertEqual(pool.diagnostics()['processes'], 1)

  def test_worker_errors_keep_their_client_or_server_category(self):
    pool = self.pool(workers=1, timeout=10)
    with self.assertRaisesRegex(ValueError, 'turn must be positive') as error:
      pool.query(ROOT, 'invalid', {})
    self.assertNotIsInstance(error.exception, errors.NotFound)
    with self.assertRaisesRegex(errors.NotFound, 'unknown_batch'):
      pool.query(ROOT, 'unknown', {})
    with self.assertRaisesRegex(analysis.AnalysisBusy, 'retry shortly'):
      pool.query(ROOT, 'busy', {})
    with self.assertRaisesRegex(analysis.AnalysisFailed, 'KeyError') as error:
      pool.query(ROOT, 'bug', {})
    self.assertNotIsInstance(
        error.exception, (ValueError, analysis.AnalysisBusy)
    )
    # Errors raised by the evaluator never cost the worker process.
    self.assertEqual(pool.diagnostics()['restarts'], 0)

  def test_real_analysis_matches_direct_result(self):
    pool = analysis.AnalysisPool(timeout=20)
    self.addCleanup(pool.close)
    self.assertEqual(
        pool.query(ROOT, '/api/overview', {}),
        analysis.evaluate(SessionStore(ROOT), '/api/overview', {}),
    )

  def test_cache_reloads_when_evidence_changes(self):
    with tempfile.TemporaryDirectory() as tmp:
      target = Path(tmp) / 'capture'
      shutil.copytree(ROOT, target, copy_function=os.link)
      target.chmod(0o755)
      cache = StoreCache(max_entries=1)
      first = cache.get(target)
      self.assertIs(first, cache.get(target))
      (target / 'new-evidence.txt').write_text('changed')
      self.assertIsNot(first, cache.get(target))

  def test_memory_limit_recovers_on_macos(self):
    if sys.platform != 'darwin':
      self.skipTest('macOS RSS supervisor; other platforms also use RLIMIT_AS')
    pool = self.pool(workers=1, memory_bytes=128 * 1024**2, timeout=10)
    with self.assertRaisesRegex(analysis.AnalysisBusy, 'memory limit'):
      pool.query(ROOT, 'memory', {})
    self.assertTrue(pool.query(ROOT, 'pid', {})['pid'])
