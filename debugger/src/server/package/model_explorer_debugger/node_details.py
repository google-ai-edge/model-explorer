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

"""Evidence-backed node details and explicit saved tensor selections."""

import ast
import json
import threading
from model_explorer_debugger import fsutil
from model_explorer_debugger import metrics as metrics_lib

file_digest = fsutil.file_digest


class NodeDetails:

  def __init__(self, store):
    self.store = store
    self.lock = threading.RLock()
    self.path = store.root / 'mappings.json'
    self.fingerprint = fsutil.digest(
        [store.session, store.semantic, store.execution, store.tensors]
    )

  def context(self, layer, batch, semantic):
    if type(layer) is not int or not 0 <= layer < len(
        self.store.semantic['layers']
    ):
      raise ValueError('unknown_layer')
    coordinates = next(
        (b for b in self.store.session['batches'] if b['batch'] == batch), None
    )
    if coordinates is None:
      raise ValueError('unknown_batch')
    instance = self.store.semantic['layers'][layer]
    graph = self.store.semantic['semantic_graph'][instance['def']]
    anchor = next(
        (
            a
            for a in graph.get('anchors', [])
            if 'anchor:' + a['id'] == semantic
        ),
        None,
    )
    node_id = anchor['of'].rsplit(':', 1)[0] if anchor else semantic
    node = next(
        (
            n
            for n in graph.get('nodes', []) + graph.get('inputs', [])
            if n['id'] == node_id
        ),
        None,
    )
    if node is None:
      raise ValueError('unknown_semantic_node')
    return coordinates, instance, graph, node, anchor

  def details(self, layer, batch, semantic):
    coordinates, instance, graph, node, anchor = self.context(
        layer, batch, semantic
    )
    snippets = []
    evidence_file = self.store.root / 'semantic_evidence.json'
    if evidence_file.is_file():
      evidence = json.loads(evidence_file.read_text())
      category = 'nodes' if node in graph.get('nodes', []) else 'inputs'
      index = graph[category].index(node)
      prefix = f'/semantic_graph/{instance["def"]}/{category}/{index}'
      layer_prefix = f'/layers/{layer}/attrs/{node["id"]}'
      seen = set()
      for claim in evidence.get('claims', []):
        pointer = claim.get('path', '')
        if not any(
            pointer == p or pointer.startswith(p + '/')
            for p in [prefix, layer_prefix]
        ):
          continue
        source = claim.get('source', {})
        key = (source.get('file'), source.get('start'), source.get('end'))
        if key in seen or not all(v is not None for v in key):
          continue
        seen.add(key)
        record = {
            'file': key[0],
            'start': key[1],
            'end': key[2],
            'reason': claim.get('reason', ''),
            'code': '',
            'status': 'source_unavailable',
        }
        path = (self.store.root / key[0]).resolve()
        expected = next(
            (
                s.get('sha256')
                for s in evidence.get('sources', [])
                if s['file'] == key[0]
            ),
            None,
        )
        if path.is_relative_to(self.store.root) and path.is_file() and expected:
          if fsutil.file_digest(path) != expected:
            record['status'] = 'source_hash_mismatch'
          else:
            lines = path.read_text().splitlines()
            if 1 <= key[1] <= key[2] <= len(lines):
              start, end = key[1], key[2]
              try:
                functions = [
                    n
                    for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.lineno <= start
                    and n.end_lineno >= end
                ]
                if functions:
                  function = min(
                      functions, key=lambda n: n.end_lineno - n.lineno
                  )
                  start, end = function.lineno, function.end_lineno
              except SyntaxError:
                pass
              record.update(
                  start=start,
                  end=end,
                  evidence_start=key[1],
                  evidence_end=key[2],
                  code='\n'.join(lines[start - 1 : end]),
                  status='verified',
              )
        snippets.append(record)
    records = [
        {k: v for k, v in t.items() if k != 'path'}
        for t in self.store.tensors
        if t.get('layer') == layer
        and all(
            t.get(k) == coordinates.get(k)
            for k in ('batch', 'turn', 'phase', 'step')
        )
    ]
    return {
        'node': node,
        'anchor': anchor,
        'parameters': instance.get('attrs', {}).get(node['id'], {}),
        'binding': instance.get('from', {}).get(node['id']),
        'sources': snippets,
        'tensors': records,
        'executions': self.store.execution['executions'],
        'saved': self.saved(layer, batch, semantic),
    }

  def selection(self, payload):
    batch, _, _, _, _ = self.context(
        payload['layer'], payload['batch'], payload['semantic']
    )
    records = []
    for run, key in [('ref', 'reference'), ('target', 'target')]:
      matches = [
          t
          for t in self.store.tensors
          if t.get('id') == payload[key] and t.get('run') == run
      ]
      if len(matches) != 1:
        raise ValueError('unknown_or_ambiguous_tensor')
      t = matches[0]
      if t.get('layer') != payload['layer'] or not all(
          t.get(k) == batch.get(k) for k in ('batch', 'turn', 'phase', 'step')
      ):
        raise ValueError('tensor_context_mismatch')
      records.append(t)
    if records[0].get('sample') is None or records[0]['sample'] != records[
        1
    ].get('sample'):
      raise ValueError('sample_mismatch')
    return records

  def compare(self, payload):
    ref, target = self.selection(payload)
    metrics = metrics_lib.compare(self.store.load(ref), self.store.load(target))
    return {
        'status': 'ok',
        'reference': ref['id'],
        'target': target['id'],
        'shape': ref['shape'],
        'metrics': metrics,
    }

  def entries(self):
    if not self.path.exists():
      return []
    return json.loads(self.path.read_text())['mappings']

  @staticmethod
  def matches(record, layer, batch, semantic):
    return (record['layer'], record['batch'], record['semantic']) == (
        layer,
        batch,
        semantic,
    )

  def saved(self, layer, batch, semantic):
    with self.lock:
      entry = next(
          (
              m
              for m in self.entries()
              if self.matches(m, layer, batch, semantic)
          ),
          None,
      )
    if not entry:
      return {'status': 'not_saved'}
    if entry['dataset'] != self.fingerprint:
      return {'status': 'stale_dataset', 'record': entry}
    try:
      records = self.selection(entry)
      for record, signature in zip(records, entry['tensors']):
        self.store.load(record)
        if (
            fsutil.digest(record) != signature['identity']
            or fsutil.file_digest(self.store.root / record['path'])
            != signature['sha256']
        ):
          return {'status': 'stale_tensor', 'record': entry}
    except (ValueError, OSError, KeyError):
      return {'status': 'stale_tensor', 'record': entry}
    return {'status': 'saved', 'record': entry}

  def save(self, payload):
    ref, target = self.selection(payload)
    # Validate paths and content before hashing; never trust a path from HTTP.
    self.store.load(ref)
    self.store.load(target)
    before = [
        fsutil.file_digest(self.store.root / r['path']) for r in (ref, target)
    ]
    result = self.compare(payload)
    signatures = [
        {
            'identity': fsutil.digest(r),
            'sha256': fsutil.file_digest(self.store.root / r['path']),
        }
        for r in (ref, target)
    ]
    if before != [s['sha256'] for s in signatures]:
      raise ValueError('tensor_changed_during_comparison')
    entry = {
        k: payload[k]
        for k in ('layer', 'batch', 'semantic', 'reference', 'target')
    }
    entry.update(dataset=self.fingerprint, tensors=signatures)
    with self.lock:
      entries = [
          m
          for m in self.entries()
          if not self.matches(
              m, payload['layer'], payload['batch'], payload['semantic']
          )
      ]
      entries.append(entry)
      self.write_entries(entries)
    return {'status': 'saved', 'record': entry, 'comparison': result}

  def remove(self, payload):
    self.context(payload['layer'], payload['batch'], payload['semantic'])
    with self.lock:
      entries = [
          m
          for m in self.entries()
          if not self.matches(
              m, payload['layer'], payload['batch'], payload['semantic']
          )
      ]
      self.write_entries(entries)
    return {'status': 'not_saved'}

  def write_entries(self, entries):
    fsutil.atomic_json(self.path, {'version': 1, 'mappings': entries})
