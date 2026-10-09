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

"""Read one configured saved session and compare explicitly indexed anchors."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from model_explorer_debugger import errors

from .capture_telemetry import COLLECTIONS, empty_telemetry
from .fsutil import file_digest
from .metrics import KEYS, compare
from .node_details import NodeDetails
from .tensor_io import load_tensor


class SessionStore:

  def __init__(self, root):
    self.root = Path(root).resolve()
    self.session = self.read('session.json')
    self.source_session_fingerprint = hashlib.sha256(
        json.dumps(self.session, sort_keys=True).encode()
    ).hexdigest()
    self.semantic = self.read('semantic.json')
    self.execution = self.read('execution.json')
    self.tensors = self.read('tensor_index.json')['tensors']
    self._telemetry = (
        self.read('telemetry.json')
        if (self.root / 'telemetry.json').is_file()
        else empty_telemetry()
    )
    if self._telemetry.get('format_version') != 1 or any(
        not isinstance(self._telemetry.get(name), list) for name in COLLECTIONS
    ):
      raise ValueError('invalid_capture_telemetry')
    from .runtime.litert_storage import prepare_saved_resources

    prepare_saved_resources(
        self._telemetry['resources'],
        self.session.get('runs', []),
        root=self.root,
        conversation=self.session.get('conversation', []),
    )
    self._resources = {
        resource['id']: resource for resource in self._telemetry['resources']
    }
    if len(self._resources) != len(self._telemetry['resources']):
      raise ValueError('duplicate_execution_resource')
    # Older saved sessions already contain sampling events but no token links.
    # Derive their read-time view from evidence; keep immutable JSON unchanged.
    from .token_sources import link_token_batches

    runtimes = {
        run['id']: run['runtime'] for run in self.session.get('runs', [])
    }
    for turn in self.session.get('turns', []):
      link_token_batches(self.session, self._telemetry, turn['n'], runtimes)
    self._resource_hashes = set()
    self.node_details = NodeDetails(self)
    if len({b['batch'] for b in self.session['batches']}) != len(
        self.session['batches']
    ):
      raise ValueError('duplicate_batch')
    for layer in self.semantic['layers']:
      if not 0 <= layer['def'] < len(self.semantic['semantic_graph']):
        raise ValueError('invalid_layer_definition')

  def read(self, name):
    return json.loads((self.root / name).read_text())

  def telemetry(self, turn=None, run=None):
    """Return indexed evidence for all saved turns or one selection."""
    return {
        'format_version': 1,
        **{
            name: [
                deepcopy(row)
                for row in self._telemetry[name]
                if (turn is None or row.get('turn') == turn)
                and (run is None or row.get('run') == run)
            ]
            for name in COLLECTIONS
        },
    }

  def register_pair(self, manifest, roots):
    """Register an explicit pair and copy its validated evidence here."""
    from .cross_runtime_pairs import compare_pair, register_pair

    try:
      report = compare_pair(manifest, roots)
    except ValueError as error:
      if 'native replay proof required' in str(error):
        raise ValueError(
            'Explicit cross-runtime comparison needs a Runner that records'
            ' token_replay_proof; the current native Runner does not'
            ' produce it.'
        ) from error
      raise
    self._validate_pair_binding(report)
    return register_pair(self.root / 'explicit_pairs', manifest, roots)

  def compare_pair(self, pair_id):
    from .cross_runtime_pairs import read_pair

    report = read_pair(self.root / 'explicit_pairs', pair_id)
    self._validate_pair_binding(report)
    return report

  def _validate_pair_binding(self, report):
    observation = report['observation']
    if not any(
        turn['n'] == observation['turn'] for turn in self.session['turns']
    ):
      raise ValueError('explicit_pair_turn_mismatch')
    for role, evidence in report['runs'].items():
      run = next((row for row in self.session['runs'] if row['id'] == role), {})
      if (
          run.get('runtime') != evidence['runtime']
          or run.get('provenance', {}).get('model_sha256')
          != evidence['model']['artifact_sha256']
      ):
        raise ValueError('explicit_pair_session_model_mismatch')
      raw = (
          evidence['observation_tensor']
          if report.get('scope') == 'kv'
          else evidence['original_tensor']
      )
      role_observation = evidence.get('observation', observation)
      if role_observation.get('turn') != observation['turn']:
        raise ValueError('explicit_pair_role_turn_mismatch')
      if not any(
          row['run'] == role
          and row['turn'] == observation['turn']
          and row['phase'] == role_observation['phase']
          and row.get('sha256') == raw.get('sha256')
          and row.get('key') == raw.get('key')
          and row.get('shape') == raw.get('shape')
          and row.get('dtype') == raw.get('dtype')
          for row in self.tensors
      ):
        raise ValueError('explicit_pair_observation_tensor_mismatch')

  def explicit_pairs(self):
    """List validated explicit reports without changing automatic samples."""
    directory = self.root / 'explicit_pairs'
    if not directory.is_dir():
      return []
    return [
        self.compare_pair(path.name)
        for path in sorted(directory.iterdir())
        if path.is_dir() and not path.name.startswith('.')
    ]

  def load_pair_tensor(self, pair_id, role, mode='original'):
    """Read saved raw evidence or the exact declared comparison view."""
    if role not in ('ref', 'target') or mode not in ('original', 'comparison'):
      raise ValueError('invalid_explicit_pair_tensor_selection')
    report = self.compare_pair(pair_id)
    return self._load_validated_pair_tensor(report, role, mode)

  def explicit_pair_entries(self):
    """Enumerate declarations independently so each failure can be shown."""
    from .cross_runtime_pairs import read_pair_manifest

    directory = self.root / 'explicit_pairs'
    if not directory.is_dir():
      return
    for path in sorted(directory.iterdir()):
      if not path.is_dir() or path.name.startswith('.'):
        continue
      try:
        yield {
            'pair_id': path.name,
            'manifest': read_pair_manifest(directory, path.name),
        }
      except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
        yield {'pair_id': path.name, 'error': str(error)}

  def _load_validated_pair_tensor(self, report, role, mode='original'):
    """Internal read using a report already validated in this request."""
    pair_id = report['pair_id']
    run = report['runs'][role]
    value = load_tensor(
        self.root / 'explicit_pairs' / pair_id / role, run['original_tensor']
    )
    if mode == 'original':
      return value
    import numpy as np

    slices = tuple(
        slice(axis['start'], axis['stop'], axis['step']) for axis in run['view']
    )
    value = (
        value[slices]
        .transpose(run.get('axis_order', list(range(value.ndim))))
        .astype(np.float64)
    )
    if run.get('dequantization'):
      quant = run['dequantization']
      value = (value - quant['zero_point']) * quant['scale']
    if list(value.shape) != report['shape']:
      raise ValueError('explicit_pair_view_shape_mismatch')
    return value

  def load_resource(self, resource_id):
    """Load a boundary or KV resource without inventing a graph node."""
    record = self._resources.get(resource_id)
    if record is None:
      raise errors.UnknownResource('unknown_resource')
    try:
      tensor = load_tensor(self.root, record)
    except FileNotFoundError:
      # The client named a resource whose capture file is gone; the server path
      # is not part of the reply.
      raise errors.NotFound('resource_file_missing') from None
    path = (self.root / record['path']).resolve()
    stat = path.stat()
    identity = (str(path), stat.st_size, stat.st_mtime_ns, record.get('sha256'))
    if identity not in self._resource_hashes:
      if not record.get('sha256') or file_digest(path) != record['sha256']:
        raise ValueError('resource_checksum_mismatch')
      self._resource_hashes.add(identity)
    return tensor

  def compare_resources(self, reference, target):
    """Compare proven same-context resources at their full recorded layout."""
    ref, other = self._resources.get(reference), self._resources.get(target)
    if ref is None or other is None:
      raise errors.UnknownResource('unknown_resource')
    result = {
        'reference': reference,
        'target': target,
        'status': 'unavailable',
        'metrics': {},
        'shape': ref['shape'],
    }
    from .runtime.litert_storage import require_storage, NativeStorageError

    try:
      require_storage(ref)
      require_storage(other)
    except NativeStorageError as error:
      result.update(status=error.status, reason=error.reason)
      return result
    if ref.get('runtime') != other.get('runtime'):
      result['status'] = 'runtime_mismatch'
    elif not ref.get('model_sha256') or ref['model_sha256'] != other.get(
        'model_sha256'
    ):
      result['status'] = 'model_mismatch'
    elif ref.get('scope') != other.get('scope'):
      result['status'] = 'resource_identity_mismatch'
    elif not ref.get('sample') or ref['sample'] != other.get('sample'):
      result['status'] = 'sample_mismatch'
    elif ref['scope'] == 'kv':
      if any(
          ref.get(key) != other.get(key) for key in ('moment', 'layer', 'kind')
      ):
        result['status'] = 'resource_identity_mismatch'
      elif any(
          row.get('state') != 'available'
          or row.get('preparation_status') == 'failed'
          or (
              row.get('moment') == 'terminal'
              and row.get('terminal_status') != 'completed'
          )
          for row in (ref, other)
      ):
        pass
      elif not ref.get('layout') or ref['layout'] != other.get('layout'):
        result['status'] = 'layout_mismatch'
      elif any(
          ref.get(key) is None or ref.get(key) != other.get(key)
          for key in (
              'logical_start',
              'logical_end',
              'valid_length',
              'processed_token_count',
          )
      ):
        result['status'] = 'logical_range_mismatch'
      else:
        result['status'] = 'ready'
    elif ref.get('output_path') != other.get('output_path') or ref.get(
        'when'
    ) != other.get('when'):
      result['status'] = 'resource_identity_mismatch'
    else:
      result['status'] = 'ready'
    if result['status'] == 'ready':
      try:
        x, y = self.load_resource(reference), self.load_resource(target)
        if ref['scope'] == 'kv':
          from .kv_storage import comparison_view

          if bool(ref.get('dequantization')) != bool(
              other.get('dequantization')
          ):
            raise ValueError('quantization_unavailable')
          x, y = comparison_view(x, ref), comparison_view(y, other)
        result.update(
            status='ok',
            metrics=compare(x, y),
            shape=list(x.shape),
            comparison_basis=(
                'same logical context; cumulative numerical difference'
            ),
        )
      except (
          OSError,
          ValueError,
          TypeError,
          KeyError,
          FloatingPointError,
          EOFError,
      ) as error:
        result.update(
            status=str(error)
            if isinstance(error, ValueError)
            else type(error).__name__,
            metrics={},
        )
    return result

  def load(self, record):
    execution = next(
        (e for e in self.execution['executions'] if e['id'] == record['run']),
        None,
    )
    graph = next(
        (
            g
            for g in (execution or {}).get('graphs', [])
            if g['id'] == record['graph']
        ),
        None,
    )
    node = next(
        (
            n
            for n in (graph or {}).get('nodes', [])
            if n['id'] == record['node']
        ),
        None,
    )
    if not any(
        p['id'] == record['output']
        for p in (node or {}).get('outputsMetadata', [])
    ):
      raise ValueError('execution_identity_mismatch')
    return load_tensor(self.root, record)

  def overview(self):
    batches = []
    for batch in self.session['batches']:
      result = self.compare_batch(batch['batch'])
      metrics = {}
      for key in KEYS:
        summaries = [layer['metrics'][key] for layer in result['layers']]
        values = [m['value'] for m in summaries if m['value'] is not None]
        metrics[key] = {
            'value': (
                (min(values) if key == 'CosSim' else max(values))
                if values
                else None
            ),
            'valid': sum(m['valid'] for m in summaries),
            'total': sum(m['total'] for m in summaries),
        }
      batches.append({'batch': batch['batch'], 'metrics': metrics})
    return {'batches': batches}

  def compare_batch(self, batch_id):
    batch = next(
        (b for b in self.session['batches'] if b['batch'] == batch_id), None
    )
    if batch is None:
      raise errors.UnknownResource('unknown_batch')
    rows = []
    for i, layer in enumerate(self.semantic['layers']):
      graph = self.semantic['semantic_graph'][layer['def']]
      for anchor in graph['anchors']:
        row = {
            'layer': i,
            'anchor': anchor['id'],
            'metrics': {},
            'status': 'missing_tensor',
            'reference': None,
            'target': None,
            'shape': None,
        }
        candidates = [
            t
            for t in self.tensors
            if t.get('layer') == i
            and t.get('anchor') == anchor['id']
            and all(
                t.get(k) == batch.get(k)
                for k in ('batch', 'turn', 'phase', 'step')
            )
        ]
        ref = [t for t in candidates if t.get('run') == 'ref']
        target = [t for t in candidates if t.get('run') == 'target']
        if len(ref) > 1 or len(target) > 1:
          row['status'] = 'ambiguous_tensor'
        elif ref and target:
          row.update(
              reference=ref[0].get('id'),
              target=target[0].get('id'),
              shape=ref[0].get('shape'),
          )
          if ref[0].get('sample') is None or ref[0]['sample'] != target[0].get(
              'sample'
          ):
            row['status'] = 'sample_mismatch'
          else:
            try:
              row['metrics'] = compare(self.load(ref[0]), self.load(target[0]))
              row['status'] = 'ok'
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                FloatingPointError,
                EOFError,
            ) as exc:
              row['status'] = (
                  str(exc)
                  if isinstance(exc, ValueError)
                  else type(exc).__name__
              )
        mapping = self.node_details.saved(i, batch_id, 'anchor:' + anchor['id'])
        if mapping['status'] == 'saved':
          try:
            row.update(self.node_details.compare(mapping['record']))
            row['mapping'] = 'manual'
          except (OSError, ValueError, TypeError, KeyError, FloatingPointError):
            row.update(status='mapping_comparison_failed', metrics={})
        elif mapping['status'].startswith('stale'):
          row.update(status=mapping['status'], metrics={}, mapping='stale')
        rows.append(row)
    summaries = []
    for i in range(len(self.semantic['layers'])):
      layer_rows = [r for r in rows if r['layer'] == i]
      metrics = {}
      for key in KEYS:
        values = [
            r['metrics'][key]['value']
            for r in layer_rows
            if key in r['metrics'] and r['metrics'][key]['status'] == 'ok'
        ]
        metrics[key] = {
            'value': (
                (min(values) if key == 'CosSim' else max(values))
                if values
                else None
            ),
            'valid': len(values),
            'total': len(layer_rows),
        }
      summaries.append({'layer': i, 'metrics': metrics})
    return {'batch': batch_id, 'rows': rows, 'layers': summaries}
