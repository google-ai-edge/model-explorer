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

"""Index existing CaptureRun shards without changing their tensor payloads."""

from collections.abc import Mapping
import json
import pathlib
from typing import Any
from model_debugger_contracts import schema


def export_capture(
    capture: Any,
    directory: str | pathlib.Path,
    input_identity: Mapping[str, Any],
) -> dict[str, Any]:
  """Index complete generation evidence without rewriting shards."""
  # Lazy import required by test_package_boundary.py so model_debugger_runner
  # can be imported without optional ML libraries installed.
  from safetensors import safe_open  # pylint: disable=g-import-not-at-top

  directory = pathlib.Path(directory).resolve()
  source = pathlib.Path(capture.capture_dir).resolve()
  if not source.is_relative_to(directory):
    raise ValueError('PyTorch capture is outside its run directory')
  rows = [
      json.loads(line)
      for line in (source / 'manifest.jsonl').read_text().splitlines()
      if line
  ]
  records = []
  for row in rows:
    if row.get('scope', 'module') != 'module':
      raise ValueError(
          'Only module tensors belong in the PyTorch capture export'
      )
    path = (source / row['shard']).resolve()
    if not path.is_relative_to(source) or path.suffix != '.safetensors':
      raise ValueError('Invalid PyTorch capture shard path')
    with safe_open(path, framework='pt', device='cpu') as shard:
      tensor = shard.get_tensor(row['slot'])
      if (
          list(tensor.shape) != list(row['shape'])
          or str(tensor.dtype) != row['dtype']
      ):
        raise ValueError('PyTorch capture metadata differs from its raw tensor')
    record = dict(
        format='safetensors',
        path=str(path.relative_to(directory)),
        key=row['slot'],
        shape=list(row['shape']),
        dtype=row['dtype'].removeprefix('torch.'),
        module_path=row['path'],
        module_type=row['cls'],
        edge=row['when'],
        invocation=row.get('invocation', row.get('repeat', 0)),
        call_seq=row['call_seq'],
        output=row.get('index', 0),
        output_tree=[row.get('index', 0)],
        layer=row['layer'],
        kind=row['kind'],
        phase=row['phase'],
        step=row['step'],
        capture_key=row.get('capture_key', row['key']),
        scope='module',
    )
    for field in (
        'forward_id',
        'module_call_id',
        'output_path',
        'turn',
        'pos_offset',
    ):
      if field in row:
        record[field] = row[field]
    records.append(record)
  if not records:
    raise ValueError('No PyTorch module tensors captured')
  topology = dict(
      blocks_path=capture.topology.blocks_path,
      n_layers=capture.topology.n_layers,
      width=capture.topology.width,
      sites=[
          dict(
              module_path=site.path,
              module_type=type(site.module).__name__,
              layer=site.layer,
              kind=site.kind,
          )
          for site in capture.sites
      ],
  )
  result = dict(
      format_version=2,
      tensor_root='run',
      runtime='PyTorch',
      export_scope='all',
      capture_scope='generation',
      input_identity=input_identity,
      topology=topology,
      skipped=dict(capture.sink.skipped),
      tensors=records,
  )
  resources = []
  for scope, folder in (('boundary', 'boundaries'), ('kv', 'kv')):
    manifest = source / folder / 'manifest.jsonl'
    for line in manifest.read_text().splitlines() if manifest.is_file() else []:
      if not line.strip():
        continue
      row = json.loads(line)
      path = (manifest.parent / row['shard']).resolve()
      if (
          row.get('scope') != scope
          or not path.is_relative_to(source)
          or path.suffix != '.safetensors'
      ):
        raise ValueError('Invalid PyTorch resource shard')
      with safe_open(path, framework='pt', device='cpu') as shard:
        tensor = shard.get_tensor(row['slot'])
        if (
            list(tensor.shape) != row['shape']
            or str(tensor.dtype) != row['dtype']
        ):
          raise ValueError(
              'PyTorch resource metadata differs from its raw tensor'
          )
      resources.append({
          **{
              key: value
              for key, value in row.items()
              if key not in ('shard', 'slot', 'key')
          },
          'format': 'safetensors',
          'path': str(path.relative_to(directory)),
          'key': row['slot'],
          'capture_key': row['key'],
          'dtype': row['dtype'].removeprefix('torch.'),
      })
  result.update(
      forwards=capture.forwards,
      kv_snapshots=capture.kv_snapshots,
      token_records=capture.token_records,
      generation=capture.generation,
      resources=resources,
      forward_input_proofs=capture.run.get('forward_input_proofs', []),
  )
  schema.validate('capture-index-v2', result)
  destination = directory / 'export'
  destination.mkdir()
  (destination / 'capture_index.json').write_text(
      json.dumps(result, indent=2) + '\n'
  )
  return result
