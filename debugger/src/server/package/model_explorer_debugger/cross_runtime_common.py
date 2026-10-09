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

"""Shared evaluation core for explicit cross-runtime comparisons.

This does not relax automatic session pairing. Every path is relative to a
caller-supplied root and hash bound. Views are required even for whole tensors.
The admitted LiteRT final token is pending, so only its actually processed
prefix is compared. See test_cross_runtime_pairs.py for a complete manifest.
Registration and reload
live in cross_runtime_pairs; KV and position variants import this module only.
"""

from collections.abc import Mapping
import copy
import hashlib
import json
import pathlib
import re
from typing import Any

from model_explorer_debugger import fsutil
from model_explorer_debugger import metrics
from model_explorer_debugger import tensor_io
import numpy as np
import safetensors

BASIS = (
    'same declared variant and input, different artifacts; not an isolated'
    ' runtime error'
)
SEMANTIC = 'layer.0.input_layernorm.output'


def _require(condition: Any, reason: str) -> None:
  if not condition:
    raise ValueError('cross_runtime_pair: ' + reason)


def _sha(value: Any) -> bool:
  return (
      isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
  )


def _digest(value: Any, *, ensure_ascii: bool = False) -> str:
  return fsutil.proof_digest(value, ensure_ascii=ensure_ascii)


class Evidence:
  """Resolves and hash-verifies capture files and tensors across run roots."""

  def __init__(self, roots: Mapping[str, str | pathlib.Path]) -> None:
    """Initializes verified roots for 'ref' and 'target' runs."""
    _require(set(roots) == {'ref', 'target'}, 'two explicit roots required')
    self.roots = {
        role: pathlib.Path(root).resolve() for role, root in roots.items()
    }
    self.files: dict[tuple[str, str], tuple[pathlib.Path, str]] = {}

  def file(self, role: str, reference: Mapping[str, Any]) -> pathlib.Path:
    """Verifies a relative file reference against its SHA-256 checksum."""
    _require(isinstance(reference, dict), 'file reference required')
    relative, checksum = reference.get('path'), reference.get(
        'sha256', reference.get('file_sha256')
    )
    _require(
        isinstance(relative, str)
        and relative
        and not pathlib.Path(relative).is_absolute()
        and '..' not in pathlib.Path(relative).parts
        and '\x00' not in relative
        and _sha(checksum),
        'invalid file reference',
    )
    root = self.roots[role]
    path = (root / relative).resolve()
    _require(
        path.is_relative_to(root) and path.is_file(),
        'file outside root or missing',
    )
    identity = role, relative
    if identity not in self.files:
      _require(fsutil.file_digest(path) == checksum, 'file checksum mismatch')
      self.files[identity] = (path, checksum)
    else:
      _require(
          self.files[identity][1] == checksum, 'inconsistent file checksum'
      )
    return path

  def json(self, role: str, reference: Mapping[str, Any]) -> Any:
    """Loads and parses a hash-verified JSON file reference."""
    return json.loads(self.file(role, reference).read_text())

  def tensor(self, role: str, reference: Mapping[str, Any]) -> np.ndarray:
    """Loads a tensor from a hash-verified shard and checks tensor_sha256."""
    self.file(role, reference)
    value = tensor_io.load_tensor(self.roots[role], dict(reference))
    if 'tensor_sha256' in reference:
      _require(
          hashlib.sha256(value.tobytes()).hexdigest()
          == reference['tensor_sha256'],
          'tensor checksum mismatch',
      )
    return value


def _view(value: np.ndarray, declaration: Mapping[str, Any]) -> np.ndarray:
  layout, axes = declaration.get('layout'), declaration.get('view')
  _require(
      isinstance(layout, list)
      and len(layout) == value.ndim
      and all(isinstance(label, str) and label for label in layout)
      and len(set(layout)) == len(layout),
      'raw tensor layout required',
  )
  _require(
      isinstance(axes, list) and len(axes) == value.ndim,
      'explicit view required for every axis',
  )
  slices = []
  for size, axis in zip(value.shape, axes):
    _require(
        isinstance(axis, dict) and set(axis) == {'start', 'stop', 'step'},
        'invalid explicit view',
    )
    start, stop, step = axis['start'], axis['stop'], axis['step']
    _require(
        all(type(number) is int for number in (start, stop, step))
        and 0 <= start < stop <= size
        and step == 1,
        'view exceeds raw tensor or changes stride',
    )
    slices.append(slice(start, stop, step))
  return value[tuple(slices)]


def _anchor(
    evidence: Evidence,
    role: str,
    run: Mapping[str, Any],
    *,
    position_pair: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
  index_ref, anchor = run.get('capture_index'), run.get('anchor', {})
  index = evidence.json(role, index_ref)
  _require(
      index.get('format_version') == 2
      and index.get('tensor_root') in ('run', 'export'),
      'unsupported capture index',
  )
  index_path = evidence.file(role, index_ref)
  index_root = (
      index_path.parent.parent
      if index['tensor_root'] == 'run'
      else index_path.parent
  )
  tensor = anchor.get('tensor', {})
  tensor_path = evidence.file(role, tensor)
  matches = []
  for record in index.get('tensors', []):
    relative = record.get('path')
    _require(
        isinstance(relative, str) and not pathlib.Path(relative).is_absolute(),
        'invalid indexed path',
    )
    path = (index_root / relative).resolve()
    _require(path.is_relative_to(index_root), 'indexed tensor escapes capture')
    if path == tensor_path and record.get('key') == tensor.get('key'):
      matches.append(record)
  _require(
      len(matches) == 1, 'anchor must identify exactly one indexed raw tensor'
  )
  record = matches[0]
  _require(
      all(
          record.get(key) == tensor.get(key)
          for key in ('format', 'shape', 'dtype')
      ),
      'anchor metadata differs from capture index',
  )
  selector = anchor.get('selector')
  required = (
      {'module_path', 'edge', 'forward_id'}
      if role == 'ref'
      else {'signature', 'op', 'output', 'tensor'}
  )
  _require(
      isinstance(selector, dict)
      and required <= set(selector)
      and all(record.get(key) == value for key, value in selector.items()),
      'semantic anchor selector mismatch',
  )
  expected_phase = (
      run.get('observation', {}).get('phase') if position_pair else 'prefill'
  )
  _require(
      record.get('phase') == expected_phase
      and expected_phase in ('prefill', 'decode'),
      'anchor phase mismatch',
  )
  if role == 'ref':
    _require(
        selector['edge'] == 'out'
        and type(selector['forward_id']) is int
        and selector['forward_id'] >= 0
        and (position_pair or selector['forward_id'] == 0)
        and record.get('module_type') == 'Gemma4RMSNorm'
        and selector['module_path']
        in (
            'model.layers.0.input_layernorm',
            'model.language_model.layers.0.input_layernorm',
        ),
        'unsupported PyTorch semantic anchor',
    )
  else:
    _require(
        str(selector['signature']).startswith('prefill_')
        or (position_pair and selector['signature'] == 'decode'),
        'unsupported LiteRT semantic anchor',
    )
    with safetensors.safe_open(tensor_path, framework='numpy') as shard:
      metadata = shard.metadata() or {}
    _require(
        metadata.get('signature') == record['signature']
        and metadata.get('step') == str(record['step']),
        'native anchor raw signature/step mismatch',
    )
    tap_sources = [
        evidence.json(role, item)
        for item in run.get('semantic_evidence', [])
        if item['path'].endswith('.json')
    ]
    matching_taps = [
        tap
        for source in tap_sources
        if source.get('tapped_sha256') == index.get('tapped_sha256')
        for tap in source.get('taps', [])
        if all(
            tap.get(key) == record.get(key)
            for key in ('signature', 'subgraph', 'op', 'output', 'tensor')
        )
        and tap.get('shape') == record.get('shape')
        and tap.get('tensor_name') == record.get('tensor_name')
        and 'layer_0/layer_0.pre_qkv/pre_attention_norm/'
        in str(tap.get('tensor_name'))
    ]
    _require(
        len(matching_taps) == 1,
        'native anchor lacks verified language layer0 RMSNorm tap evidence',
    )
  raw = evidence.tensor(role, tensor)
  return record, raw, _view(raw, anchor)


def _pytorch_inputs(
    evidence: Evidence,
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    index: Mapping[str, Any],
) -> tuple[dict[str, Any], list[int]]:
  proof = result.get('input_proof')
  _require(
      isinstance(proof, dict)
      and result.get('input_identity') == _digest(proof),
      'PyTorch input identity mismatch',
  )
  rows = index.get('forward_input_proofs', [])
  matched = [row for row in rows if row.get('forward_id') == 0]
  _require(
      len(matched) == 1
      and matched[0].get('input_identity') == result['input_identity']
      and matched[0].get('input_proof') == proof,
      'PyTorch forward proof mismatch',
  )
  forward = next(
      (row for row in index.get('forwards', []) if row.get('forward_id') == 0),
      {},
  )
  _require(
      forward.get('status') == 'completed'
      and forward.get('phase') == 'prefill'
      and forward.get('pos_offset') == 0
      and forward.get('turn') == 1,
      'PyTorch must be a completed first-turn Prefill',
  )
  context = proof.get('cache_context')
  _require(
      context
      == {'processed_token_count': 0, 'token_ids': [], 'attention_mask': []},
      'cached Prefill is not supported',
  )
  input_file = run.get('input_tensor_file')
  path = evidence.file('ref', input_file)
  _require(
      result.get('input_tensor_path')
      == str(path.relative_to(evidence.roots['ref'])),
      'input source path mismatch',
  )
  tensors = {}
  for key, metadata in proof.get('tensors', {}).items():
    record = {
        **input_file,
        'format': 'safetensors',
        'key': key,
        'shape': metadata.get('shape'),
        'dtype': metadata.get('dtype'),
    }
    value = evidence.tensor('ref', record)
    _require(
        value.tolist() == metadata.get('values'),
        'PyTorch proof differs from raw input',
    )
    tensors[key] = value
  boundary_files = run.get('input_boundary_files', [])
  by_path = {item.get('path'): item for item in boundary_files}
  _require(
      len(by_path) == len(boundary_files) and by_path,
      'raw input boundary file hashes required',
  )
  resources = {
      item.get('capture_key'): item
      for item in index.get('resources', [])
      if item.get('scope') == 'boundary' and item.get('forward_id') == 0
  }
  observed = set()
  for item in forward.get('inputs', []):
    output_path = item.get('output_path')
    _require(
        isinstance(output_path, list)
        and len(output_path) == 2
        and output_path[0] == 'kwargs',
        'unsupported PyTorch boundary input tree',
    )
    key = output_path[1]
    resource = resources.get(item.get('key'), {})
    raw_file = by_path.get(resource.get('path'))
    _require(
        key in tensors
        and raw_file is not None
        and key not in observed
        and item.get('shape') == resource.get('shape')
        and str(item.get('dtype')).removeprefix('torch.')
        == resource.get('dtype'),
        'executed PyTorch input boundary mismatch',
    )
    observed.add(key)
    boundary = evidence.tensor(
        'ref', {**resource, 'sha256': raw_file['sha256']}
    )
    _require(
        boundary.dtype == tensors[key].dtype
        and np.array_equal(boundary, tensors[key]),
        'executed PyTorch input differs from its preparation proof',
    )
  _require(observed == set(tensors), 'PyTorch input proof is incomplete')
  ids, mask = tensors.get('input_ids'), tensors.get('attention_mask')
  _require(
      ids is not None
      and ids.ndim == 2
      and ids.shape[0] == 1
      and np.issubdtype(ids.dtype, np.integer)
      and np.all(ids >= 0),
      'invalid PyTorch input IDs',
  )
  _require(
      mask is not None and mask.shape == ids.shape and np.all(mask == 1),
      'only unpadded PyTorch input is supported',
  )
  position = tensors.get('position_ids', tensors.get('cache_position'))
  _require(
      position is not None
      and np.array_equal(position.reshape(-1), np.arange(ids.shape[1])),
      'PyTorch position proof mismatch',
  )
  return proof, ids.reshape(-1).tolist()


def _litert_inputs(
    evidence: Evidence, result: Mapping[str, Any]
) -> tuple[dict[str, Any], list[int], int]:
  proof = result.get('token_replay_proof')
  _require(
      isinstance(proof, dict)
      and proof.get('format_version') == 1
      and proof.get('mode') == 'first_prefill_exact_token_ids',
      'native replay proof required',
  )
  payload = {
      key: value for key, value in proof.items() if key != 'proof_sha256'
  }
  _require(
      proof.get('proof_sha256') == _digest(payload, ensure_ascii=True),
      'native replay proof hash mismatch',
  )
  ids = proof.get('actual_token_ids')
  _require(
      isinstance(ids, list)
      and 2 <= len(ids) <= 128
      and all(type(token) is int and 0 <= token <= 2147483647 for token in ids),
      'invalid admitted native IDs',
  )
  n = len(ids) - 1
  _require(
      proof.get('accepted_token_count') == len(ids)
      and proof.get('graph_valid_length') == n
      and proof.get('graph_token_ids') == ids[:-1]
      and proof.get('pending_token_id') == ids[-1]
      and proof.get('pending_token_position') == n,
      'native processed/pending token mismatch',
  )
  admitted = evidence.tensor('target', proof.get('input_resource', {}))
  _require(
      admitted.dtype == np.int32 and admitted.tolist() == [ids],
      'native admission readback mismatch',
  )
  inputs = proof.get('graph_inputs', {})
  _require(
      set(inputs) == {'position_ids', 'attention_mask'},
      'native graph inputs required',
  )
  positions = evidence.tensor('target', inputs['position_ids'])
  mask = evidence.tensor('target', inputs['attention_mask'])
  for row in inputs.values():
    with safetensors.safe_open(
        evidence.file('target', row), framework='numpy'
    ) as shard:
      metadata = shard.metadata() or {}
    _require(
        metadata.get('signature') == row.get('signature')
        and metadata.get('step') == str(row.get('step')),
        'native raw input signature/step mismatch',
    )
  _require(
      positions.dtype == np.int32
      and positions.ndim == 1
      and positions.size >= n
      and np.array_equal(positions[:n], np.arange(n))
      and np.all(positions[n:] == 0),
      'native positions mismatch',
  )
  _require(
      mask.dtype == np.bool_
      and mask.ndim == 4
      and mask.shape[:2] == (1, 1)
      and mask.shape[2] == positions.size
      and mask.shape[3] >= n,
      'unsupported native mask layout',
  )
  expected = np.zeros(mask.shape, dtype=np.bool_)
  expected[0, 0, :n, :n] = np.tri(n, dtype=np.bool_)
  _require(
      np.array_equal(mask, expected),
      'native mask is not the declared causal prefix',
  )
  _require(
      all(
          row.get('valid_length') == n and row.get('step') == len(ids)
          for row in inputs.values()
      )
      and inputs['position_ids'].get('signature')
      == inputs['attention_mask'].get('signature'),
      'native graph input coordinates mismatch',
  )
  return proof, ids, n


def _source_chain(
    evidence: Evidence,
    runs: Mapping[str, Any],
    indices: Mapping[str, Any],
) -> None:
  """Bind declared IT variants to recorded HF and LiteRT artifact provenance."""
  documents = {
      role: [
          evidence.json(role, item)
          for item in run['model']['source_chain']
          if item['path'].endswith('.json')
      ]
      for role, run in runs.items()
  }
  model = runs['ref']['model']
  sources = [
      doc
      for doc in documents['ref']
      if doc.get('repo_id') == 'google/gemma-4-E2B-it'
      and doc.get('revision') == model.get('revision')
  ]
  _require(len(sources) == 1, 'HF IT revision source evidence required')
  files = {item.get('path'): item for item in sources[0].get('files', [])}
  _require(
      files.get('model.safetensors', {}).get('local_sha256')
      == model.get('checkpoint_sha256')
      and files.get('config.json', {}).get('local_sha256')
      == model.get('config_sha256'),
      'HF checkpoint/config source evidence mismatch',
  )
  hubs = [
      doc
      for doc in documents['target']
      if doc.get('repo') == 'litert-community/gemma-4-E2B-it-litert-lm'
  ]
  _require(
      len(hubs) == 1
      and isinstance(hubs[0].get('revision'), str)
      and re.fullmatch('[0-9a-f]{40}', hubs[0]['revision']),
      'LiteRT IT release source evidence required',
  )
  hub = hubs[0]
  candidates = [
      item
      for item in hub.get('files', [])
      if item.get('filename') == 'gemma-4-E2B-it.litertlm'
  ]
  original = hub.get('local_original_sha256')
  _require(
      _sha(original)
      and len(candidates) == 1
      and candidates[0].get('lfs', {}).get('sha256') == original,
      'LiteRT original container differs from release source',
  )
  conversions = [
      doc
      for doc in documents['target']
      if doc.get('original_container_sha256') == original
      and doc.get('tapped_container_sha256')
      == runs['target']['model']['artifact_sha256']
      and doc.get('source_sha256') == indices['target'].get('tapped_sha256')
  ]
  _require(
      len(conversions) == 1, 'LiteRT original/tapped container chain required'
  )
  profile = runs['target']['model'].get('quantization_profile', 'unknown')
  _require(
      profile == 'unknown' or profile == conversions[0].get('observed_profile'),
      'quantization declaration differs from source inspection',
  )


def _evaluate_prefill(
    manifest: Mapping[str, Any], roots: Mapping[str, str | pathlib.Path]
) -> tuple[dict[str, Any], Evidence]:
  _require(
      isinstance(manifest, dict)
      and manifest.get('format_version') == 1
      and manifest.get('kind') == 'explicit_cross_runtime_prefill',
      'unsupported manifest schema',
  )
  runs = manifest.get('runs', {})
  _require(set(runs) == {'ref', 'target'}, 'two runs required')
  observation = manifest.get('observation', {})
  _require(
      type(observation.get('turn')) is int
      and observation['turn'] == 1
      and observation.get('phase') == 'prefill',
      'explicit observation turn and phase required',
  )
  mapping = manifest.get('mapping', {})
  _require(
      mapping.get('semantic') == SEMANTIC
      and mapping.get('mask_relation') == 'unpadded_causal_prefix',
      'unsupported explicit semantic mapping',
  )
  evidence = Evidence(roots)
  results, indices, anchors = {}, {}, {}
  for role, runtime in (('ref', 'PyTorch'), ('target', 'LiteRT-LM')):
    run = runs[role]
    _require(run.get('runtime') == runtime, 'runtime role mismatch')
    result = results[role] = evidence.json(role, run.get('result'))
    _require(
        result.get('turn_sequence') == 1,
        'source result must record the first turn',
    )
    indices[role] = evidence.json(role, run.get('capture_index'))
    model = run.get('model', {})
    _require(
        model.get('family') == 'gemma-4-E2B'
        and model.get('variant') == 'it'
        and _sha(model.get('artifact_sha256'))
        and result.get('model_sha256') == model['artifact_sha256'],
        'declared model variant or artifact identity mismatch',
    )
    source_chain = model.get('source_chain')
    _require(
        isinstance(source_chain, list) and source_chain,
        'model source chain required',
    )
    for reference in source_chain:
      evidence.file(role, reference)
    semantic_evidence = run.get('semantic_evidence')
    _require(
        isinstance(semantic_evidence, list) and semantic_evidence,
        'semantic mapping evidence required',
    )
    for reference in semantic_evidence:
      evidence.file(role, reference)
    anchors[role] = _anchor(evidence, role, run)
  _source_chain(evidence, runs, indices)
  ref_proof, ref_ids = _pytorch_inputs(
      evidence, runs['ref'], results['ref'], indices['ref']
  )
  target_proof, admitted_ids, n = _litert_inputs(evidence, results['target'])
  _require(
      ref_ids == admitted_ids, 'admitted token IDs differ from PyTorch input'
  )
  ref_model = runs['ref']['model']
  requested = target_proof.get('requested_source', {})
  _require(
      isinstance(ref_model.get('revision'), str)
      and re.fullmatch('[0-9a-f]{40}', ref_model['revision'])
      and _sha(ref_model.get('checkpoint_sha256'))
      and _sha(ref_model.get('config_sha256')),
      'HF revision and checkpoint/config hashes required',
  )
  _require(
      requested.get('source_model_sha256') == ref_model['checkpoint_sha256']
      and requested.get('source_revision') == ref_model['revision']
      and requested.get('config_sha256') == ref_model['config_sha256'],
      'replay source declaration differs from HF source',
  )
  files = results['ref'].get('model_files', [])
  _require(
      isinstance(files, list)
      and files
      and _digest(files, ensure_ascii=True) == ref_model['artifact_sha256'],
      'loaded HF file identity mismatch',
  )
  by_name = {row.get('path'): row.get('sha256') for row in files}
  _require(
      by_name.get('model.safetensors') == ref_model['checkpoint_sha256']
      and by_name.get('config.json') == ref_model['config_sha256'],
      'loaded HF checkpoint/config identity mismatch',
  )
  if requested.get('tokenizer') != ref_proof.get('tokenizer') or requested.get(
      'serialization'
  ) != ref_proof.get('serialization'):
    # The worker hashes its vocabulary and serialized template value; the
    # replay request can instead name original HF files. These are distinct
    # hash domains and must be bridged through the worker's loaded file list.
    tokenizer = requested.get('tokenizer', {})
    serialization = requested.get('serialization', {})
    _require(
        tokenizer.get('tokenizer_json_sha256') == by_name.get('tokenizer.json')
        and _sha(tokenizer.get('tokenizer_json_sha256'))
        and tokenizer.get('tokenizer_config_sha256')
        == by_name.get('tokenizer_config.json')
        and _sha(tokenizer.get('tokenizer_config_sha256'))
        and serialization.get('chat_template_sha256')
        == by_name.get('chat_template.jinja')
        and _sha(serialization.get('chat_template_sha256'))
        and by_name.get('model.safetensors') == ref_model['checkpoint_sha256']
        and by_name.get('config.json') == ref_model['config_sha256'],
        'tokenizer/template replay file identity mismatch',
    )
    text = results['ref'].get('serialized_input')
    _require(
        isinstance(text, str)
        and serialization.get('serialized_text') == text
        and hashlib.sha256(text.encode()).hexdigest()
        == serialization.get('serialized_text_sha256'),
        'serialized input text identity mismatch',
    )
  _require(
      _sha(ref_proof.get('tokenizer', {}).get('vocab_sha256'))
      and _sha(ref_proof.get('serialization', {}).get('template_sha256')),
      'tokenizer/template hashes required',
  )
  native = target_proof.get('native', {})
  library = runs['target'].get('native_library', {})
  source_lock = runs['target'].get('native_source_lock')
  evidence.file('target', library)
  _require(
      library.get('sha256') == native.get('library_sha256')
      and evidence.json('target', source_lock) == native.get('source_lock'),
      'native build identity mismatch',
  )
  _require(
      anchors['target'][0].get('signature')
      == target_proof['graph_inputs']['position_ids']['signature']
      and anchors['target'][0].get('step') == len(admitted_ids),
      'anchor belongs to a different native graph run',
  )
  for role, (_, raw, viewed) in anchors.items():
    declaration = runs[role]['anchor']
    _require(
        declaration['layout'] == ['batch', 'sequence', 'hidden']
        and raw.ndim == 3
        and viewed.shape == (1, n, raw.shape[2])
        and declaration['view']
        == [
            {'start': 0, 'stop': 1, 'step': 1},
            {'start': 0, 'stop': n, 'step': 1},
            {'start': 0, 'stop': raw.shape[2], 'step': 1},
        ],
        'comparison must explicitly select the complete processed prefix and'
        ' hidden width',
    )
  ref, target = anchors['ref'][2], anchors['target'][2]
  _require(ref.shape == target.shape, 'comparison view shape mismatch')
  report = {
      'format_version': 1,
      'pair_id': _digest(manifest),
      'status': 'ok',
      'scope': 'module',
      'semantic': SEMANTIC,
      'observation': copy.deepcopy(observation),
      'comparison_basis': BASIS,
      'weight_equivalence': 'unverified',
      'quantization_profile': (
          runs['target']['model'].get('quantization_profile', 'unknown')
      ),
      'shape': list(ref.shape),
      'processed_token_ids': admitted_ids[:n],
      'compared_position_range': [0, n],
      'accepted_token_count': len(admitted_ids),
      'processed_token_count': n,
      'pending_token_id': admitted_ids[-1],
      'pending_token_position': n,
      'mask_basis': (
          'PyTorch unpadded inputs with declared causal execution; captured'
          ' LiteRT boolean causal mask'
      ),
      'metrics': metrics.compare(ref, target),
      'runs': {
          role: {
              'runtime': run['runtime'],
              'model': copy.deepcopy(run['model']),
              'original_tensor': copy.deepcopy(run['anchor']['tensor']),
              'layout': run['anchor']['layout'],
              'view': run['anchor']['view'],
              'observation': {
                  'turn': observation['turn'],
                  'phase': 'prefill',
                  **(
                      {'forward_id': 0, 'step': 0, 'pos_offset': 0}
                      if role == 'ref'
                      else {
                          'signature': anchors['target'][0]['signature'],
                          'step': len(admitted_ids),
                      }
                  ),
                  'compared_position_range': [0, n],
              },
          }
          for role, run in runs.items()
      },
  }
  return report, evidence
