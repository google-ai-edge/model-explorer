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
# ==============================================================================

"""Create a clearly synthetic, immutable 128k saved session for API/browser QA.

Used by test/test_kv_range.py; also runnable as a CLI (--output DIRECTORY).
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src/server/package'), str(ROOT / 'test')]
import numpy as np
from safetensors.numpy import save_file
from model_explorer_debugger.store import SessionStore
from test_pytorch_telemetry import PyTorchTelemetryTests

POSITIONS = 131_072


def create_fixture(output, positions=POSITIONS):
  """Three layers, four KV heads, four channels.

  At most one 8 MiB array is written per write.
  """
  output = Path(output).resolve()
  if output.exists():
    raise ValueError(
        'Choose a new output directory; existing captures are never'
        ' overwritten.'
    )
  fixture = PyTorchTelemetryTests()
  fixture.setUp()
  try:
    job, results = fixture.fixture()
    original = fixture.publish(job, results)
    shutil.copytree(original.root, output)
  finally:
    fixture.doCleanups()
  store = SessionStore(output)
  data = deepcopy(store._telemetry)
  templates = {
      role: next(
          row
          for row in data['resources']
          if row.get('scope') == 'kv'
          and row.get('moment') == 'terminal'
          and row['run'] == role
      )
      for role in ('ref', 'target')
  }
  snapshots = {
      role: next(
          row
          for row in data['kv_snapshots']
          if row['moment'] == 'terminal' and row['run'] == role
      )
      for role in ('ref', 'target')
  }
  data['resources'] = [
      row for row in data['resources'] if row.get('scope') != 'kv'
  ]
  data['kv_snapshots'] = []
  spikes = [
      dict(layer=0, kind='key', position=65537, head=2, channel=1, delta=16.0),
      dict(
          layer=1, kind='value', position=131071, head=3, channel=0, delta=32.0
      ),
      dict(layer=2, kind='key', position=17, head=0, channel=2, delta=-8.0),
  ]
  spikes += [
      dict(
          layer=0,
          kind='key',
          position=1025 + index * 2048,
          head=0,
          channel=0,
          delta=4.0,
      )
      for index in range(60)
  ]
  spikes = [spike for spike in spikes if spike['position'] < positions]
  for role in ('ref', 'target'):
    snapshot = deepcopy(snapshots[role])
    snapshot.update(
        id=f'SYNTHETIC-128K:{role}:kv:2',
        processed_token_count=positions,
        layers=[],
    )
    for layer in range(3):
      geometry = dict(
          layer=layer,
          layer_type='SYNTHETIC',
          state='available',
          layout=['batch', 'kv_head', 'sequence', 'head_dim'],
          capacity=positions,
          valid_length=positions,
          logical_start=0,
          logical_end=positions,
          processed_token_count=positions,
          tensors=[],
      )
      for kind in ('key', 'value'):
        if role == 'target' and layer == 2 and kind == 'value':
          continue
        values = np.ones((1, 4, positions, 4), dtype=np.float32)
        if layer == 0 and kind == 'value' and positions > 2048:
          values[:, :, 2048, :] = 0
        if role == 'target':
          for spike in spikes:
            if spike['layer'] == layer and spike['kind'] == kind:
              values[
                  0, spike['head'], spike['position'], spike['channel']
              ] += spike['delta']
          if layer == 1 and kind == 'key' and positions > 4096:
            values[0, 1, 4096, 0] = np.nan
        relative = f'tensors/SYNTHETIC-128K-{role}-{layer}-{kind}.safetensors'
        path = output / relative
        save_file(
            {'kv': values},
            path,
            metadata={'synthetic': 'true', 'positions': str(positions)},
        )
        del values
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = f'SYNTHETIC-128K-{role}-{layer}-{kind}'
        resource = deepcopy(templates[role])
        resource.update(
            {key: value for key, value in geometry.items() if key != 'tensors'}
        )
        resource.update(
            id=identity,
            kind=kind,
            path=relative,
            key='kv',
            shape=[1, 4, positions, 4],
            dtype='float32',
            sha256=checksum,
            capture_key=identity,
        )
        data['resources'].append(resource)
        geometry['tensors'].append(
            dict(
                kind=kind,
                resource_id=identity,
                key=identity,
                shape=resource['shape'],
                dtype='float32',
                storage_status='stored',
            )
        )
      snapshot['layers'].append(geometry)
    data['kv_snapshots'].append(snapshot)
  (output / 'telemetry.json').write_text(json.dumps(data, indent=2) + '\n')
  session = store.session
  session['name'] = (
      f'SYNTHETIC KV capacity · {positions:,} positions · no inference'
  )
  (output / 'session.json').write_text(json.dumps(session, indent=2) + '\n')
  manifest = dict(
      synthetic=True,
      inference_run=False,
      capture_root=str(output),
      positions=positions,
      layers=3,
      heads=4,
      channels=4,
      spikes=spikes,
      missing_side=dict(layer=2, kind='value', role='target'),
      zero_norm=dict(layer=0, kind='value', position=2048),
      non_finite=dict(layer=1, kind='key', position=4096, head=1),
      expected_find=dict(formula='relative_l2 > 100%', count=len(spikes)),
  )
  (output / 'SYNTHETIC.json').write_text(json.dumps(manifest, indent=2) + '\n')
  return manifest


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  output = (
      args.output
      or Path(tempfile.mkdtemp(prefix='kv-128k-synthetic-')) / 'capture'
  )
  print(json.dumps(create_fixture(output), indent=2))
