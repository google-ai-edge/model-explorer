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

"""Index verified LiteRT-LM dumps without converting their tensor payloads."""

import json
from pathlib import Path
import shutil

from model_debugger_contracts.schema import validate
from safetensors import SafetensorError, safe_open

# TFLite tensor types and Safetensors header types describe the same stored
# dtype.
DTYPES = {
    0: ('F32', 'float32'),
    1: ('F16', 'float16'),
    2: ('I32', 'int32'),
    3: ('U8', 'uint8'),
    4: ('I64', 'int64'),
    6: ('BOOL', 'bool'),
    7: ('I16', 'int16'),
    9: ('I8', 'int8'),
    10: ('F64', 'float64'),
    12: ('U64', 'uint64'),
    15: ('U32', 'uint32'),
    16: ('U16', 'uint16'),
    18: ('BF16', 'bfloat16'),
}


def collect(
    capture_dir, manifest_path, destination, backend, *, reference=False
):
  """Validate exact tap identities, then export an index of unchanged shards.

  Workers use reference=True: paths are relative to the run
  (destination.parent), so the export contains metadata only. Publishing makes
  an independent saved copy of each shard. Standalone exports copy each selected
  shard once instead.
  """
  capture_dir = Path(capture_dir).resolve()
  destination = Path(destination).resolve()
  if destination.exists():
    raise ValueError('Capture export destination must be new')
  run_root = destination.parent
  if reference and not capture_dir.is_relative_to(run_root):
    raise ValueError('Capture directory must stay inside the run')
  manifest = validate(
      'tap-manifest', json.loads(Path(manifest_path).read_text())
  )
  taps = manifest['taps']
  lookup = {
      (tap['signature'], 'post_' + tap['output_name']): tap for tap in taps
  }
  if not taps or len(lookup) != len(taps):
    raise ValueError('Empty or duplicate tap manifest')
  records, seen, files = [], set(), {}
  for candidate in sorted(capture_dir.rglob('*.safetensors')):
    path = candidate.resolve()
    if not path.is_relative_to(capture_dir):
      raise ValueError('Invalid captured tensor path')
    try:
      with safe_open(path, framework='numpy') as file:
        metadata = file.metadata() or {}
        signature, raw_step = metadata.get('signature'), metadata.get('step')
        for key in file.keys():
          tap = lookup.get((signature, key))
          if tap is None:
            continue
          try:
            step = int(raw_step)
          except (TypeError, ValueError) as error:
            raise ValueError('Missing/invalid step metadata') from error
          if step < 0:
            raise ValueError('Missing/invalid step metadata')
          session = str(path.parent.relative_to(capture_dir))
          identity = session, signature, key, step
          if identity in seen:
            raise ValueError(f'Duplicate capture: {identity}')
          seen.add(identity)
          tensor = file.get_slice(key)
          expected = DTYPES.get(tap['tensor_type'])
          if (
              expected is None
              or tensor.get_dtype() != expected[0]
              or tensor.get_shape() != tap['shape']
          ):
            raise ValueError(
                f'Capture dtype/shape differs from tapped tensor: {path}'
            )
          if path not in files:
            files[path] = (
                str(path.relative_to(run_root))
                if reference
                else f'tensors/{len(files):06d}.safetensors'
            )
          records.append({
              **tap,
              'backend_requested': backend,
              'session': session,
              'step': step,
              'phase': 'prefill' if 'prefill' in signature else 'decode',
              'source': str(path),
              'format': 'safetensors',
              'key': key,
              'path': files[path],
              'dtype': expected[1],
          })
    except SafetensorError as error:
      raise ValueError(f'Invalid captured Safetensors file: {path}') from error
  if not records:
    raise ValueError(
        'No matching post-tap tensors captured; inference success alone is'
        ' insufficient'
    )
  groups = {}
  for record in records:
    group = record['session'], record['signature'], record['step']
    groups.setdefault(group, set()).add(record['key'])
  for (_, signature, step), captured in groups.items():
    expected = {key for sig, key in lookup if sig == signature}
    if missing := expected - captured:
      raise ValueError(
          f'Incomplete tap capture at {signature}/{step}: {sorted(missing)}'
      )
  result = {
      'format_version': 2,
      'tensor_root': 'run' if reference else 'export',
      'tapped_sha256': manifest['tapped_sha256'],
      'backend_requested': backend,
      'tensors': records,
      'uncaptured_signatures': sorted(
          {t['signature'] for t in taps} - {t['signature'] for t in records}
      ),
  }
  validate('capture-index-v2', result)
  destination.mkdir(parents=True)
  try:
    if not reference:
      (destination / 'tensors').mkdir()
      for path, relative in files.items():
        shutil.copy2(path, destination / relative)
    (destination / 'capture_index.json').write_text(
        json.dumps(result, indent=2) + '\n'
    )
  except BaseException:
    shutil.rmtree(destination)
    raise
  return result
