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

"""Session configuration rules shared by both Session backends.

The backends are the saved-capture metadata and the workspace registry.
"""

import os
from pathlib import Path
import shutil
from uuid import uuid4

# Upload policy shared by the HTTP transport and the capability report.
UPLOAD_LIMIT_BYTES = 20 * 1024**3
UPLOAD_IDLE_TIMEOUT_SECONDS = 60
NAME_LIMIT = 160
COPY_NAME_LIMIT = 150
RUN_FIELDS = (
    'runtime',
    'backend',
    'precision',
    'runnerBuild',
    'artifact',
    'source',
    'device',
    'repository',
    'revision',
    'sourceUrl',
    'cpuThreads',
    'audioBackend',
    'audioCpuThreads',
    'visionBackend',
    'prefillBatchSizes',
    'contextLength',
)


def validate_name(value):
  if (
      not isinstance(value, str)
      or not value.strip()
      or len(value.strip()) > NAME_LIMIT
  ):
    raise ValueError(f'Session name must contain 1–{NAME_LIMIT} characters.')
  if any(ord(char) < 32 for char in value):
    raise ValueError('Session name cannot contain control characters.')
  return value.strip()


def copy_name(name):
  """Returns the default name of a duplicated Session or Chat.

  Both backends truncate identically.
  """
  return name[:COPY_NAME_LIMIT] + ' (copy)'


def chat_config_locked(record):
  """Returns whether a Chat's conversation settings are immutable.

  They are once anything was captured or the Chat is read-only.
  """
  return bool(
      record.get('has_capture')
      or record.get('successful_turns')
      or record.get('read_only')
  )


def validate_config(payload):
  if not isinstance(payload, dict):
    raise ValueError('Invalid session configuration.')
  name = validate_name(payload.get('name'))
  model = payload.get('model')
  if not isinstance(model, str) or not model.strip() or len(model) > 200:
    raise ValueError('Model is required (maximum 200 characters).')
  runs = payload.get('runs')
  if (
      not isinstance(runs, list)
      or len(runs) != 2
      or {r.get('id') for r in runs if isinstance(r, dict)} != {'ref', 'target'}
  ):
    raise ValueError('Reference and Target configurations are required.')
  clean = []
  for run in runs:
    record = {'id': run['id']}
    for key in RUN_FIELDS:
      value = run.get(key, '')
      if (
          not isinstance(value, str)
          or len(value) > 2048
          or any(ord(c) < 32 for c in value)
      ):
        raise ValueError('Invalid runtime field: ' + key)
      record[key] = value.strip()
    record['forceF32'] = run.get('forceF32', False)
    if not isinstance(record['forceF32'], bool):
      raise ValueError('Activation dtype must be a boolean.')
    for key in ('cpuThreads', 'audioCpuThreads', 'contextLength'):
      value = record[key]
      if value and (not value.isdigit() or not 1 <= int(value) <= 2147483647):
        raise ValueError(key + ' must be a positive integer.')
    for key in ('backend', 'audioBackend', 'visionBackend'):
      allowed = (
          ('', 'CPU', 'MPS', 'CUDA')
          if key == 'backend' and record['runtime'] == 'PyTorch'
          else ('', 'CPU', 'GPU')
      )
      if record[key] not in allowed:
        raise ValueError('Unsupported backend.')
    sizes = record['prefillBatchSizes']
    if sizes and any(
        not n.strip().isdigit() or not 1 <= int(n) <= 2147483647
        for n in sizes.split(',')
    ):
      raise ValueError(
          'Prefill batch sizes must be positive integers separated by commas.'
      )
    if not record['artifact'] or (
        record['source'] == 'huggingface' and not record['repository']
    ):
      raise ValueError('Both model artifacts are required.')
    if not record['runtime']:
      raise ValueError('Both runtimes are required.')
    clean.append(record)
  return {'name': name, 'model': model.strip(), 'runs': clean}


def upload_artifact(root, stream, size, name):
  """Store one uploaded model file, fsynced.

  The file goes under `root/.debugger-artifacts/<uuid>/<name>`.
  """
  root = Path(root)
  if (
      not isinstance(name, str)
      or not name
      or name in ('.', '..')
      or Path(name).name != name
      or any(ord(c) < 32 for c in name)
  ):
    raise ValueError('Invalid file name.')
  if not 0 < size <= UPLOAD_LIMIT_BYTES:
    raise ValueError(
        f'Choose a non-empty file smaller than {UPLOAD_LIMIT_BYTES // 1024**3}'
        ' GiB.'
    )
  if shutil.disk_usage(root).free < size + 64 * 1024**2:
    raise ValueError('Not enough disk space for this model artifact.')
  directory = root / '.debugger-artifacts' / str(uuid4())
  directory.mkdir(parents=True)
  path = directory / name
  try:
    with path.open('xb') as file:
      remaining = size
      while remaining:
        block = stream.read(min(1024**2, remaining))
        if not block:
          raise ValueError('Upload interrupted.')
        file.write(block)
        remaining -= len(block)
      file.flush()
      os.fsync(file.fileno())
  except BaseException:
    path.unlink(missing_ok=True)
    directory.rmdir()
    raise
  return {'artifact': str(path.relative_to(root)), 'name': name, 'size': size}
