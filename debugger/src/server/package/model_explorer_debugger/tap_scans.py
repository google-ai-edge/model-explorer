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

"""One queued structural scanner, separate from session creation/native jobs."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from threading import RLock
from .fsutil import atomic_json


class TapScans:

  def __init__(self, registry):
    self.registry = registry
    self.lock = RLock()
    self.pool = ThreadPoolExecutor(max_workers=1)
    self.pending = set()

  def artifact(self, identity):
    with self.registry.lock:
      artifact = self.registry.state['artifacts'].get(identity)
      if not artifact:
        raise ValueError('Unknown registered model')
      return dict(artifact)

  def get(self, identity):
    artifact = self.artifact(identity)
    path = self.registry.root / 'tap-scans' / identity / 'result.json'
    with self.lock:
      if identity in self.pending:
        return {'status': 'scanning'}
      if path.is_file():
        result = json.loads(path.read_text())
        stat = Path(artifact['path']).stat()
        if result.get('source_stat') != [stat.st_size, stat.st_mtime_ns]:
          return {
              'status': 'failed',
              'error': 'Model changed. Upload or register it again.',
          }
        return result
      return {'status': 'idle'}

  def start(self, identity):
    artifact = self.artifact(identity)
    if not self.registry.capabilities().get('tap_profiles'):
      raise ValueError('Configure the local LiteRT-LM tap toolkit first')
    with self.lock:
      current = self.get(identity)
      if current['status'] in ('completed', 'scanning'):
        return current
      self.pending.add(identity)
      self.pool.submit(self._scan, identity, artifact)
      return {'status': 'scanning'}

  def _scan(self, identity, artifact):
    folder = self.registry.root / 'tap-scans' / identity
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    try:
      stat = Path(artifact['path']).stat()
      request = dict(
          model=artifact['path'],
          model_sha256=artifact['sha256'],
          runtime_root=str(self.registry.runtime_root),
          result=str(folder / 'worker-result.json'),
      )
      atomic_json(folder / 'request.json', request)
      env = {
          **os.environ,
          'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
          'PYTHONDONTWRITEBYTECODE': '1',
      }
      with (folder / 'scan.log').open('wb') as log:
        subprocess.run(
            [
                str(self.registry.runtime_root / '.venv/bin/python'),
                '-m',
                'model_explorer_debugger.runtime.scan_taps',
                str(folder / 'request.json'),
            ],
            env=env,
            stdout=log,
            stderr=log,
            check=True,
            timeout=120,
        )
      result = json.loads((folder / 'worker-result.json').read_text())
      after = Path(artifact['path']).stat()
      if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError('Model changed during scanning')
      result['source_stat'] = [stat.st_size, stat.st_mtime_ns]
    except Exception as error:
      result = {'status': 'failed', 'error': str(error)}
      if Path(artifact['path']).is_file():
        stat = Path(artifact['path']).stat()
        result['source_stat'] = [stat.st_size, stat.st_mtime_ns]
    finally:
      with self.lock:
        atomic_json(result_path, result)
        self.pending.discard(identity)

  def close(self):
    self.pool.shutdown(wait=True, cancel_futures=True)
