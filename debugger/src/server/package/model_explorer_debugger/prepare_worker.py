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

"""Runs the LiteRT-LM tap-preparation subprocess for each run of a Session."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .fsutil import atomic_json

PREPARE_TIMEOUT_SECONDS = 600
CANCEL_GRACE_SECONDS = 5
LOW_DISK_BYTES = 256 * 1024**2


class PrepareWorker:

  def __init__(self, manager):
    self.manager = manager

  def run(self, job, record, artifacts):
    manager, registry = self.manager, self.manager.registry
    directory = manager.directory(job)
    results, prepared = {}, {}
    for run in record['runs']:
      if run['runtime'] == 'PyTorch':
        continue
      artifact = artifacts[run['id']]
      if artifact['sha256'] in prepared:
        results[run['id']] = prepared[artifact['sha256']]
        manager.record_event(
            job,
            'progress',
            runId=run['id'],
            message='Reusing Reference debug model',
        )
        continue
      request = manager._request(record, run, artifact, directory, 'prepare')
      request_path = directory / (run['id'] + '-request.json')
      atomic_json(request_path, request)
      env = {
          **os.environ,
          'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
          'PYTHONUNBUFFERED': '1',
      }
      with (directory / (run['id'] + '.log')).open('wb') as log:
        process = subprocess.Popen(
            registry.worker_command(run, 'prepare', request_path),
            env=env,
            stdout=log,
            stderr=log,
        )
        with manager.lock:
          manager.processes[job['id']] = process
        try:
          self._follow(job, run['id'], directory, process)
        finally:
          if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
      result = json.loads((directory / run['id'] / 'result.json').read_text())
      results[run['id']] = prepared[artifact['sha256']] = result
    self._register(record, artifacts, results)
    return results

  def _follow(self, job, role, directory, process):
    """Relay worker events until it exits; cancel on request or low disk."""
    manager = self.manager
    cursor, began, cancel_at = 0, time.monotonic(), None
    while process.poll() is None:
      path = directory / role / 'events.jsonl'
      if path.exists():
        lines = [
            line
            for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n')
        ]
        for line in lines[cursor:]:
          event = json.loads(line)
          manager._emit(job, role, event.pop('type'), event)
        cursor = len(lines)
      if manager.stopping or manager._cancelled(job):
        cancel_at = cancel_at or time.monotonic()
        if time.monotonic() - cancel_at > CANCEL_GRACE_SECONDS:
          raise InterruptedError('Preparation stopped')
      if time.monotonic() - began > PREPARE_TIMEOUT_SECONDS:
        raise TimeoutError('Preparation exceeded 10 minutes')
      if shutil.disk_usage(directory).free < LOW_DISK_BYTES:
        raise ValueError('Capture stopped because disk space is low')
      time.sleep(0.08)
    if manager._cancelled(job):
      raise InterruptedError('Preparation stopped')
    if process.returncode:
      raise RuntimeError(f'Preparation failed; see {role}.log')

  def _register(self, record, artifacts, results):
    registry = self.manager.registry
    runs, point_selections = [], {}
    for run in record['runs']:
      if run['runtime'] == 'PyTorch':
        runs.append(run)
        continue
      result = results[run['id']]
      semantic = artifacts[run['id']].get('semantic')
      if not semantic and (
          record.get('tap_profile') != 'custom-outputs-v1'
          or result.get('reviewed_e2b')
      ):
        candidate = (
            Path(__file__).resolve().parents[4]
            / 'examples/gemma4-e2b/semantic.json'
        )
        semantic = str(candidate) if candidate.is_file() else None
      artifact_id = registry.register_model(
          result['model'], result['manifest'], semantic
      )
      runs.append({**run, 'artifact': artifact_id, 'source': 'registered'})
      if record.get('tap_points', {}).get(run['artifact']):
        point_selections[artifact_id] = record['tap_points'][run['artifact']]
    registry.update(
        record['id'],
        runs=runs,
        tap_points=point_selections,
        tap_prepared=True,
        tap_preparation=results,
        status='draft',
        initialized=False,
        notice='Tensor capture prepared. Initialize to start.',
    )
