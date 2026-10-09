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

"""Import captured runtime coordinates, preserving original tensor shards."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from model_debugger_contracts import schema
from model_explorer_debugger.runtime import litert_dump_inference
from safetensors import safe_open

from .capture_telemetry import (
    append_telemetry,
    empty_telemetry,
    pair_forward_evidence,
)
from .fsutil import atomic_json, digest, file_digest, now
from .store import SessionStore
from .tensor_io import load_tensor
from .token_sources import link_token_batches


def trace_tokens(path):
  """Retain every released ID; chunk text does not establish token text."""
  tokens, ordinal = [], 0
  if not path or not path.is_file():
    return tokens
  for line in path.read_text().splitlines():
    record = json.loads(line)
    ids = record.get('token_ids', [])
    texts = record.get('texts', [])
    # Only a single response candidate is supported. Never flatten beams.
    if (
        not isinstance(ids, list)
        or len(ids) != 1
        or not isinstance(ids[0], list)
    ):
      continue
    if any(type(token) is not int or token < 0 for token in ids[0]):
      raise ValueError('Invalid recorded native token ID')
    aligned = (
        len(ids[0]) == 1
        and isinstance(texts, list)
        and len(texts) == 1
        and isinstance(texts[0], str)
    )
    for offset, token_id in enumerate(ids[0]):
      tokens.append(
          dict(
              id=token_id,
              text=texts[0] if aligned else None,
              step=ordinal + offset,
          )
      )
    ordinal += len(ids[0])
  return tokens


def normalize_tensors(index, runtime):
  """Give runtime records concrete display identities, inferring no edges."""
  if runtime not in ('LiteRT-LM', 'PyTorch'):
    raise ValueError('Unsupported capture runtime')
  if index.get('runtime', 'LiteRT-LM') != runtime:
    raise ValueError('Capture runtime differs from configured runtime')
  records, seen = [], set()
  complete = index.get('export_scope') == 'all'
  for original in index['tensors']:
    tensor = deepcopy(original)
    if tensor.get('phase') not in (
        ('prefill', 'decode', 'unknown') if complete else ('prefill', 'decode')
    ) or (
        not (
            complete
            and tensor.get('phase') == 'unknown'
            and tensor.get('step') is None
        )
        and (type(tensor.get('step')) is not int or tensor['step'] < 0)
    ):
      raise ValueError('Invalid captured invocation coordinates')
    if runtime == 'PyTorch':
      module, edge = tensor.get('module_path'), tensor.get('edge')
      invocation, output = tensor.get('invocation'), tensor.get('output')
      if (
          not isinstance(module, str)
          or not module
          or edge not in ('in', 'out')
          or type(invocation) is not int
          or invocation < 0
          or type(output) is not int
          or output < 0
          or type(tensor.get('call_seq')) is not int
          or tensor['call_seq'] < 0
          or (
              tensor.get('layer') is not None
              and (type(tensor['layer']) is not int or tensor['layer'] < 0)
          )
      ):
        raise ValueError('Invalid captured module coordinates')
      if not complete and (tensor['phase'] != 'prefill' or tensor['step'] != 0):
        raise ValueError(
            'PyTorch capture currently supports one Prefill forward'
        )
      if complete and (
          type(tensor.get('forward_id')) is not int or tensor['forward_id'] < 0
      ):
        raise ValueError('Missing captured forward identity')
      blocks_path = index.get('topology', {}).get('blocks_path')
      prefix = (
          f"{blocks_path}.{tensor['layer']}"
          if isinstance(blocks_path, str)
          and blocks_path
          and tensor.get('layer') is not None
          else None
      )
      label = module
      if module == prefix:
        label = 'block'
      elif prefix and module.startswith(prefix + '.'):
        label = module[len(prefix) + 1 :]
      graph = (
          f"pytorch/eager/forward/{tensor['forward_id']}"
          if complete
          else 'pytorch/eager/prefill'
      )
      tensor.update(
          graph=graph,
          node='module/' + digest([module, invocation]),
          module_output_index=output,
          output=f'{edge}.{output}',
          tensor_name=module,
          module_label=label,
          capture_group=graph,
      )
      identity = tensor['graph'], tensor['node'], tensor['output']
      if identity in seen:
        raise ValueError('Duplicate captured module boundary')
      seen.add(identity)
    else:
      tensor.update(
          graph=f"section-{tensor['section_offset']}/{tensor['signature']}",
          node=f"sg{tensor['subgraph']}/op{tensor['op']}",
          output=str(tensor['output']),
          capture_group=tensor['signature'],
      )
      if complete:
        if (
            type(tensor.get('forward_id')) is not int
            or tensor['forward_id'] < 0
        ):
          raise ValueError('Missing native forward identity')
        tensor['capture_group'] = f"litert/forward/{tensor['forward_id']}"
    tensor['runtime'] = runtime
    records.append(tensor)
  return records


def module_boundary(tensor):
  node = (
      'pytorch.module.'
      + digest([tensor['module_path'], tensor['invocation']])[:24]
  )
  port = f"{tensor['edge']}.{tensor['module_output_index']}"
  return node, port, node + '.' + port


def prepare_semantic(semantic, indexed, runtimes, artifacts):
  """Retain reviewed graphs; add only directly observed PyTorch boundaries."""
  for runtime in ('LiteRT-LM', 'PyTorch'):
    run_ids = [
        run
        for run, value in runtimes.items()
        if value == runtime and indexed[run]
    ]
    if not run_ids:
      continue
    reviewed = next(
        (
            artifacts[run]['semantic']
            for run in run_ids
            if artifacts[run].get('semantic')
        ),
        None,
    )
    existing = [
        layer
        for layer in semantic['layers']
        if layer.get('runtime', 'LiteRT-LM') == runtime
    ]
    if reviewed and not existing:
      profile = json.loads(Path(reviewed).read_text())
      offset = len(semantic['semantic_graph'])
      if not offset:
        semantic.update({
            key: deepcopy(value)
            for key, value in profile.items()
            if key not in ('semantic_graph', 'layers')
        })
      if runtime == 'LiteRT-LM':
        for definition in profile['semantic_graph']:
          if any(
              n['id'] == 'norm.attn.in' for n in definition['nodes']
          ) and not any(
              a['id'] == 'pre_attention_norm' for a in definition['anchors']
          ):
            definition['anchors'].append({
                'id': 'pre_attention_norm',
                'of': 'norm.attn.in:0',
                'semantic': 'pre-attention RMSNorm output',
            })
      semantic['semantic_graph'].extend(profile['semantic_graph'])
      semantic['layers'].extend(
          {
              **layer,
              'def': layer['def'] + offset,
              'runtime': runtime,
              'source': 'reviewed',
              'runtime_layer': i,
          }
          for i, layer in enumerate(profile['layers'])
      )
    if runtime != 'PyTorch' or reviewed:
      continue
    observed = [
        tensor
        for run in run_ids
        for tensor in indexed[run]
        if tensor.get('layer') == 0
    ]
    if not observed:
      continue
    instance = next(
        (
            layer
            for layer in semantic['layers']
            if layer.get('source') == 'observed_module_boundaries'
            and layer.get('runtime') == runtime
        ),
        None,
    )
    if instance is None:
      instance = {
          'def': len(semantic['semantic_graph']),
          'runtime': 'PyTorch',
          'runtime_layer': 0,
          'source': 'observed_module_boundaries',
          'attrs': {},
      }
      semantic['layers'].append(instance)
      semantic['semantic_graph'].append({
          'kind': 'PyTorch module boundaries',
          'description': (
              'Observed module inputs and outputs from layer 0; no inferred'
              ' operations or dataflow.'
          ),
          'inputs': [],
          'nodes': [],
          'anchors': [],
      })
    graph = semantic['semantic_graph'][instance['def']]
    for tensor in observed:
      node_id, port, anchor_id = module_boundary(tensor)
      node = next(
          (node for node in graph['nodes'] if node['id'] == node_id), None
      )
      if node is None:
        node = {
            'id': node_id,
            'namespace': (
                tensor['module_path'].rsplit('.', 1)[0]
                if '.' in tensor['module_path']
                else ''
            ),
            'module_path': tensor['module_path'],
            'module_type': tensor.get('module_type'),
            'invocation': tensor['invocation'],
            'incomingEdges': [],
        }
        graph['nodes'].append(node)
      node['label'] = tensor['module_label']
      anchor = next(
          (anchor for anchor in graph['anchors'] if anchor['id'] == anchor_id),
          None,
      )
      if anchor is None:
        anchor = {
            'id': anchor_id,
            'of': node_id + ':' + port,
            'semantic': (
                f"{tensor['module_path']}"
                f" {'input' if tensor['edge'] == 'in' else 'output'}"
                f"[{tensor['module_output_index']}]"
                f" invocation {tensor['invocation']}"
            ),
        }
        graph['anchors'].append(anchor)
      anchor.update(
          label=(
              f"{tensor['module_label']} ·"
              f" {tensor['edge']}[{tensor['module_output_index']}]"
          ),
          edge=tensor['edge'],
      )
  return semantic


def tensor_anchor(semantic, tensor):
  runtime = tensor['runtime']
  if runtime == 'PyTorch':
    anchor_id = module_boundary(tensor)[2]
    for i, layer in enumerate(semantic['layers']):
      if (
          layer.get('source') == 'observed_module_boundaries'
          and layer.get('runtime') == runtime
          and tensor.get('layer') == layer.get('runtime_layer')
          and any(
              a['id'] == anchor_id
              for a in semantic['semantic_graph'][layer['def']]['anchors']
          )
      ):
        return i, anchor_id
    # A supplied semantic graph alone does not prove a module-to-node mapping.
    return None, None
  if (
      '/layer_0/' in tensor['tensor_name']
      and '/pre_attention_norm/' in tensor['tensor_name']
  ):
    for i, layer in enumerate(semantic['layers']):
      if (
          layer.get('runtime', 'LiteRT-LM') == runtime
          and layer.get('runtime_layer', 0) == 0
          and any(
              a['id'] == 'pre_attention_norm'
              for a in semantic['semantic_graph'][layer['def']]['anchors']
          )
      ):
        return i, 'pre_attention_norm'
  return None, None


def matching_inputs(runtimes, results):
  if runtimes.get('ref') != runtimes.get('target'):
    return False
  ref, target = results['ref'], results['target']
  if not ref.get('model_sha256') or ref['model_sha256'] != target.get(
      'model_sha256'
  ):
    return False
  if runtimes['ref'] == 'PyTorch':
    return (
        bool(ref.get('input_identity'))
        and ref['input_identity'] == target.get('input_identity')
        and bool(ref.get('input_proof'))
        and ref['input_proof'] == target.get('input_proof')
    )
  return (
      ref['input'] == target['input']
      and ref['messages'] == target['messages']
      and ref.get('effective', {}).get('contextLength')
      == target.get('effective', {}).get('contextLength')
  )


def save_input_evidence(directory, result, temporary, name):
  """Keep each turn's exact PyTorch input proof independent of job folders."""
  evidence = {
      key: deepcopy(result[key])
      for key in ('input_identity', 'input_proof')
      if key in result
  }
  proof = result.get('input_proof')
  if proof is not None:
    if (
        not isinstance(proof, dict)
        or not isinstance(proof.get('tensors'), dict)
        or not proof['tensors']
    ):
      raise ValueError('Invalid PyTorch input proof')
    identity = hashlib.sha256(
        json.dumps(
            proof, sort_keys=True, separators=(',', ':'), ensure_ascii=False
        ).encode()
    ).hexdigest()
    if result.get('input_identity') != identity:
      raise ValueError('PyTorch input proof identity mismatch')
  relative = result.get('input_tensor_path')
  if relative is not None:
    if not proof:
      raise ValueError('Missing PyTorch input tensor proof')
    for key, metadata in proof['tensors'].items():
      tensor = load_tensor(
          directory,
          {
              'format': 'safetensors',
              'path': relative,
              'key': key,
              'shape': metadata['shape'],
              'dtype': metadata['dtype'],
          },
      )
      if tensor.tolist() != metadata['values']:
        raise ValueError('PyTorch input proof differs from captured inputs')
    with safe_open(directory / relative, framework='numpy') as captured:
      if set(captured.keys()) != set(proof['tensors']):
        raise ValueError('PyTorch input proof omits captured inputs')
    # All published references use the saved session as their root.
    path = 'inputs/' + name + '.safetensors'
    (temporary / 'inputs').mkdir(exist_ok=True)
    shutil.copy2(directory / relative, temporary / path)
    evidence.update(
        input_tensor_path=path,
        input_tensor_sha256=file_digest(temporary / path),
    )
  return evidence


def publish_capture(registry, record, job_dir, results, artifacts):
  identity = job_dir.name
  destination = (
      registry.root / 'sessions' / record['id'] / 'captures' / identity
  )
  temporary = destination.with_name(identity + '.pending')
  temporary.mkdir(parents=True)
  try:
    return _publish_capture(
        registry, record, job_dir, results, artifacts, temporary, destination
    )
  except BaseException:
    shutil.rmtree(temporary)
    raise


def save_tokenizer(root, info, temporary, session, run_id):
  """Keep one copy per tokenizer digest in `tokenizers/`; link the run to it."""
  if (
      not isinstance(info, dict)
      or info.get('kind') != 'sentencepiece'
      or not re.fullmatch(r'[0-9a-f]{64}', str(info.get('sha256')))
  ):
    raise ValueError('Invalid native tokenizer description')
  relative = info.get('path')
  source = (root / relative).resolve() if isinstance(relative, str) else None
  if (
      source is None
      or Path(relative).is_absolute()
      or not source.is_relative_to(root)
      or source.suffix != '.model'
      or not source.is_file()
  ):
    raise ValueError('Invalid native tokenizer path')
  if file_digest(source) != info['sha256']:
    raise ValueError('Native tokenizer bytes differ from their declared digest')
  path = f"tokenizers/{info['sha256']}.model"
  target = temporary / path
  if not target.exists():
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(source, target)
  next(saved for saved in session['runs'] if saved['id'] == run_id)[
      'tokenizer'
  ] = dict(
      kind='sentencepiece',
      path=path,
      sha256=info['sha256'],
      digest_scope=info.get('digest_scope'),
  )


def label_missing_texts(tokens, tokenizer_path):
  """Give unspelled control tokens their piece text from the saved tokenizer.

  These are the control tokens the Runner could not spell (BOS, stop tokens).
  """
  missing = [
      token
      for token in tokens or []
      if token.get('text') is None and type(token.get('id')) is int
  ]
  if (
      not missing
      or tokenizer_path is None
      or not Path(tokenizer_path).is_file()
  ):
    return
  try:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(
        model_proto=Path(tokenizer_path).read_bytes()
    )
  except Exception:
    return
  for token in missing:
    identity = token['id']
    if 0 <= identity < processor.GetPieceSize():
      piece = processor.IdToPiece(identity)
      token['text'] = (
          piece
          if processor.IsControl(identity) or processor.IsByte(identity)
          else piece.replace('\u2581', ' ')
      )
      token['text_basis'] = 'saved tokenizer'


def save_inferred_evidence(root, index, temporary, name):
  """Retain the released-token log an inferred native capture derives from."""
  if index.get(
      'capture_scope'
  ) != litert_dump_inference.CAPTURE_SCOPE or not isinstance(
      index.get('inference'), dict
  ):
    raise ValueError('Inferred native evidence requires its inference record')
  evidence = dict(
      generation_basis=index['inference'].get('basis'),
      inference=deepcopy(index['inference']),
      output_token_count=len(index.get('tokens', [])),
      token_count_basis=index.get('generation', {}).get('token_count_basis'),
  )
  source = (root / 'raw/generated_tokens.jsonl').resolve()
  if source.is_file() and source.is_relative_to(root):
    path = f'inputs/{name}-generated-tokens.jsonl'
    (temporary / 'inputs').mkdir(exist_ok=True)
    shutil.copy2(source, temporary / path)
    evidence['released_tokens_path'] = path
  return evidence


def save_native_evidence(root, index, temporary, name):
  trace = index.get('native_trace')
  if not isinstance(trace, dict):
    raise ValueError('Native execution telemetry requires its original trace')
  relative = trace.get('path')
  if not isinstance(relative, str) or Path(relative).is_absolute():
    raise ValueError('Invalid native trace path')
  source = (root / relative).resolve()
  if (
      not source.is_relative_to(root)
      or source.suffix != '.jsonl'
      or file_digest(source) != trace.get('sha256')
  ):
    raise ValueError('Native trace identity mismatch')
  path = f'inputs/{name}-runtime-trace.jsonl'
  (temporary / 'inputs').mkdir(exist_ok=True)
  shutil.copy2(source, temporary / path)
  return dict(native_trace_path=path, native_trace_sha256=trace['sha256'])


def _publish_capture(
    registry, record, job_dir, results, artifacts, temporary, destination
):
  identity = job_dir.name
  (temporary / 'tensors').mkdir()
  previous = (
      registry.capture_store(record['id']) if record.get('capture') else None
  )
  session = (
      deepcopy(previous.session)
      if previous
      else dict(
          model=record['model'],
          runs=[],
          turns=[],
          phases=[
              {'id': 'prefill', 'label': 'Prefill'},
              {'id': 'decode', 'label': 'Decode'},
          ],
          batches=[],
          layers=[],
          anchors=[],
          conversation=[],
      )
  )
  session['generation'] = deepcopy(record.get('generation', {}))
  runtimes = {run['id']: run['runtime'] for run in record['runs']}
  notice = (
      'Local LiteRT-LM capture. Prefill comparisons require identical model and'
      ' conversation input. Decode input equivalence is not established; decode'
      ' tensors are retained without automatic numeric pairing.'
  )
  if set(runtimes.values()) == {'PyTorch'}:
    notice = (
        'PyTorch eager execution capture. Module boundaries, KV snapshots, and'
        ' sampling events retain their own identities. Comparisons require the'
        ' same model and recorded logical input context.'
    )
  elif len(set(runtimes.values())) > 1:
    notice = (
        'Mixed-runtime capture. Each runtime retains its own execution and'
        ' batch coordinates; automatic cross-runtime tensor pairing is'
        ' unavailable.'
    )
  session.update(
      name=record['name'], created_at=now(), capture_id=identity, notice=notice
  )
  # Text-only successful Turns are not represented by fake capture rows.
  # Use the orchestration Turn identity, preserving gaps in Debug history.
  turn = record.get(
      'publication_turn',
      max((item['n'] for item in session['turns']), default=0) + 1,
  )
  if type(turn) is not int or turn <= max(
      (item['n'] for item in session['turns']), default=0
  ):
    raise ValueError('Capture Turn must advance beyond the last captured Turn')
  tensors = deepcopy(previous.tensors) if previous else []
  telemetry = previous.telemetry() if previous else empty_telemetry()
  retained = set()
  previous_paths = [tensor['path'] for tensor in tensors]
  previous_paths.extend(
      item['input_tensor_path']
      for item in session['conversation']
      if item.get('input_tensor_path')
  )
  previous_paths.extend(
      item['native_trace_path']
      for item in session['conversation']
      if item.get('native_trace_path')
  )
  previous_paths.extend(
      item['released_tokens_path']
      for item in session['conversation']
      if item.get('released_tokens_path')
  )
  previous_paths.extend(
      run['tokenizer']['path']
      for run in session.get('runs', [])
      if run.get('tokenizer')
  )
  if previous and (previous.root / 'explicit_pairs').is_dir():
    shutil.copytree(
        previous.root / 'explicit_pairs', temporary / 'explicit_pairs'
    )
  previous_paths.extend(item['path'] for item in telemetry['resources'])
  previous_paths.extend(
      item['storage_source']['path']
      for item in telemetry['resources']
      if item.get('storage_source')
  )
  for relative in previous_paths:
    if relative in retained:
      continue
    source = (previous.root / relative).resolve()
    if (
        Path(relative).is_absolute()
        or not source.is_relative_to(previous.root)
        or (
            source.suffix != '.safetensors'
            and not (
                str(relative).startswith('inputs/')
                and source.suffix == '.jsonl'
            )
            and not (
                str(relative).startswith('tokenizers/')
                and source.suffix == '.model'
            )
        )
    ):
      raise ValueError('Invalid saved tensor path')
    target = temporary / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    retained.add(relative)
  semantic = (
      deepcopy(previous.semantic)
      if previous
      else {'semantic_graph': [], 'layers': []}
  )
  execution = deepcopy(previous.execution) if previous else {'executions': []}
  session['runs'] = [
      {
          **run,
          'precision': (
              (
                  (results[run['id']].get('effective') or {}).get('precision')
                  or 'Unknown'
              )
              if run['runtime'] == 'PyTorch'
              else 'Model default'
          ),
          'artifact_name': artifacts[run['id']]['name'],
          'provenance': deepcopy(results[run['id']]),
      }
      for run in record['runs']
  ]
  indexed = {}
  tensor_roots = {}
  current_forwards = {}
  complete_runs = set()
  inferred_runs = set()
  input_counts = {}
  copied = {}
  for run in record['runs']:
    run_id = run['id']
    runtime = runtimes[run_id]
    directory = job_dir / run_id
    result = results[run_id]
    export = directory / 'export/capture_index.json'
    index = (
        json.loads(export.read_text()) if export.exists() else {'tensors': []}
    )
    storage_root = index.get('tensor_root', 'export')
    if export.exists() and index.get('format_version') != 2:
      raise ValueError('Unsupported capture index version')
    if export.exists():
      schema.validate('capture-index-v2', index)
    if storage_root not in ('export', 'run'):
      raise ValueError('Invalid exported tensor root')
    tensor_roots[run_id] = (
        directory if storage_root == 'run' else directory / 'export'
    ).resolve()
    if (
        runtime == 'LiteRT-LM'
        and export.exists()
        and index.get('tokenizer') is not None
    ):
      save_tokenizer(
          tensor_roots[run_id], index['tokenizer'], temporary, session, run_id
      )
    trace = (
        directory / result['trace_log_path']
        if result.get('trace_log_path')
        else None
    )
    tokens = (
        deepcopy(result.get('tokens', []))
        if runtime == 'PyTorch'
        else trace_tokens(trace)
    )
    evidence = (
        save_input_evidence(
            directory, result, temporary, f'{identity}-{run_id}'
        )
        if runtime == 'PyTorch'
        else {}
    )
    if runtime == 'LiteRT-LM' and index.get('export_scope') == 'all':
      tokens = deepcopy(index.get('tokens', []))
      if index.get('capture_scope') == litert_dump_inference.CAPTURE_SCOPE:
        inferred_runs.add(run_id)
        evidence = save_inferred_evidence(
            tensor_roots[run_id], index, temporary, f'{identity}-{run_id}'
        )
        if isinstance(index.get('input_tokens'), list):
          evidence['input_tokens'] = deepcopy(index['input_tokens'])
        if isinstance(index.get('serialized_input'), str):
          evidence['serialized_input'] = index['serialized_input']
        tokenizer_file = (
            (tensor_roots[run_id] / index['tokenizer']['path'])
            if isinstance(index.get('tokenizer'), dict)
            else None
        )
        label_missing_texts(tokens, tokenizer_file)
        label_missing_texts(evidence.get('input_tokens'), tokenizer_file)
      else:
        evidence = save_native_evidence(
            tensor_roots[run_id], index, temporary, f'{identity}-{run_id}'
        )
      admitted = index.get('admitted_input_ids')
      if admitted is not None:
        if not isinstance(admitted, list) or any(
            type(token) is not int or token < 0 for token in admitted
        ):
          raise ValueError('Invalid native admitted input IDs')
        input_counts[run_id] = len(admitted)
        evidence.update(
            admitted_input_ids=admitted,
            input_token_count=len(admitted),
            output_token_count=len(tokens),
            token_count_basis='actual native admission and sampler IDs',
        )
    input_context = {
        key: deepcopy(result[key])
        for key in ('serialized_input', 'messages', 'stop_reason')
        if key in result
    }
    session['conversation'].append(
        dict(
            turn=turn,
            run=run_id,
            input=result['input'],
            output=result['output'],
            tokens=tokens,
            **input_context,
            **evidence,
        )
    )
    if evidence:
      next(saved for saved in session['runs'] if saved['id'] == run_id)[
          'provenance'
      ] = {**deepcopy(result), **evidence}
    if runtime == 'PyTorch' and index.get('input_identity') != result.get(
        'input_identity'
    ):
      raise ValueError('PyTorch tensor capture input identity mismatch')
    indexed[run_id] = normalize_tensors(index, runtime)
    if index.get('export_scope') == 'all':
      complete_runs.add(run_id)
    current_forwards[run_id] = append_telemetry(
        telemetry,
        index,
        tensor_roots[run_id],
        temporary,
        identity,
        run_id,
        turn,
        result,
        copied,
    )
    for tensor in indexed[run_id]:
      if run_id in complete_runs:
        forward = current_forwards[run_id].get(tensor['forward_id'])
        if forward is None or any(
            tensor.get(key) != forward.get(key) for key in ('phase', 'step')
        ):
          raise ValueError('Module tensor differs from its forward identity')
    existing = next(
        (e for e in execution['executions'] if e['id'] == run_id), None
    )
    if existing is None:
      existing = {'id': run_id, 'runtime': runtime, 'graphs': []}
      execution['executions'].append(existing)
    elif existing['runtime'] != runtime:
      raise ValueError(
          'Saved execution runtime differs from configured runtime'
      )
    for tensor in indexed[run_id]:
      graph_id = tensor['graph']
      graph = next((g for g in existing['graphs'] if g['id'] == graph_id), None)
      if graph is None:
        graph = {'id': graph_id, 'nodes': []}
        existing['graphs'].append(graph)
      node_id = tensor['node']
      node = next((n for n in graph['nodes'] if n['id'] == node_id), None)
      if node is None:
        node = {
            'id': node_id,
            'label': tensor['tensor_name'],
            'namespace': (
                tensor['module_path']
                if runtime == 'PyTorch'
                else tensor['signature']
            ),
            'incomingEdges': [],
            'outputsMetadata': [],
        }
        graph['nodes'].append(node)
      if not any(
          output['id'] == tensor['output'] for output in node['outputsMetadata']
      ):
        attrs = (
            [
                {'key': 'edge', 'value': tensor['edge']},
                {
                    'key': 'module_output_index',
                    'value': str(tensor['module_output_index']),
                },
            ]
            if runtime == 'PyTorch'
            else [{'key': 'tensor', 'value': str(tensor['tensor'])}]
        )
        node['outputsMetadata'].append({'id': tensor['output'], 'attrs': attrs})
  semantic = prepare_semantic(semantic, indexed, runtimes, artifacts)
  pair_forward_evidence(telemetry, current_forwards, runtimes, results, turn)
  for run_id, items in indexed.items():
    if runtimes[run_id] == 'LiteRT-LM' and run_id in complete_runs:
      for tensor in items:
        sample = current_forwards[run_id][tensor['forward_id']].get('sample')
        tensor['capture_group'] = (
            'litert/context/' + sample
            if sample
            else f"litert/{run_id}/forward/{tensor['forward_id']}"
        )

  # Forward IDs preserve actual call order even when a later call is Prefill.
  def coordinate_order(coordinate):
    runtime, phase, step, group = coordinate
    if group.startswith(('pytorch/eager/forward/', 'litert/forward/')):
      return (runtime, 0, int(group.rsplit('/', 1)[1]), 0, group)
    return (
        runtime,
        1,
        phase != 'prefill',
        step if step is not None else -1,
        group,
    )

  coordinates = sorted(
      {
          (t['runtime'], t['phase'], t['step'], t['capture_group'])
          for items in indexed.values()
          for t in items
      },
      key=coordinate_order,
  )
  next_batch = max((b['batch'] for b in session['batches']), default=-1) + 1
  phase_indexes = {'prefill': 0, 'decode': 0, 'unknown': 0}
  same_input = matching_inputs(runtimes, results)
  for runtime, phase, step, group in coordinates:
    batch_id = next_batch
    next_batch += 1
    phase_indexes[phase] += 1
    basis = (
        'same model and exact recorded inputs'
        if runtime == 'PyTorch'
        else 'same model and serialized conversation'
    )
    batch = dict(
        batch=batch_id,
        turn=turn,
        phase=phase,
        step=step,
        index=phase_indexes[phase],
        runtime=runtime,
        comparison_basis=basis
        if phase == 'prefill' and same_input
        else 'input equivalence not established',
    )
    if runtime == 'LiteRT-LM':
      batch['signature'] = next(
          t['signature']
          for items in indexed.values()
          for t in items
          if t['capture_group'] == group
      )
      if group.startswith('litert/'):
        forward_ids = {
            run_id: tensor['forward_id']
            for run_id, items in indexed.items()
            for tensor in items
            if tensor['capture_group'] == group
        }
        batch['forward_ids'] = forward_ids
        if len(set(forward_ids.values())) == 1:
          batch['forward_id'] = next(iter(forward_ids.values()))
        paired = [
            current_forwards[run_id][forward_id]
            for run_id, forward_id in forward_ids.items()
        ]
        batch['comparison_basis'] = (
            'same logical context; cumulative numerical difference'
            if len(paired) == 2 and all(row.get('sample') for row in paired)
            else 'input equivalence not established'
        )
    else:
      batch['graph'] = group
    if group.startswith(('pytorch/eager/forward/', 'litert/forward/')):
      batch['forward_id'] = int(group.rsplit('/', 1)[1])
      paired = [
          rows.get(batch['forward_id'], {})
          for rows in current_forwards.values()
      ]
      batch['comparison_basis'] = (
          'same logical context; cumulative numerical difference'
          if len(paired) == 2 and all(row.get('sample') for row in paired)
          else 'input equivalence not established'
      )
    session['batches'].append(batch)
    for run_id, items in indexed.items():
      for tensor in [
          t
          for t in items
          if (t['runtime'], t['phase'], t['step'], t['capture_group'])
          == (runtime, phase, step, group)
      ]:
        if tensor.get('format') != 'safetensors':
          raise ValueError('Unsupported exported tensor format')
        root = tensor_roots[run_id]
        source = (root / tensor['path']).resolve()
        if (
            Path(tensor['path']).is_absolute()
            or not source.is_relative_to(root)
            or source.suffix != '.safetensors'
        ):
          raise ValueError('Invalid exported tensor path')
        if not isinstance(tensor.get('key'), str) or not tensor['key']:
          raise ValueError('Missing exported tensor key')
        if source not in copied:
          name = f'tensors/{identity}-{run_id}-{len(tensors)}.safetensors'
          shutil.copy2(source, temporary / name)
          copied[source] = name, file_digest(temporary / name)
        name, checksum = copied[source]
        layer, anchor = tensor_anchor(semantic, tensor)
        sample = None
        if run_id in complete_runs:
          sample = current_forwards[run_id][tensor['forward_id']].get('sample')
        elif phase == 'prefill' and same_input:
          proof = (
              results['ref']['input_identity']
              if runtime == 'PyTorch'
              else [results['ref']['input'], results['ref']['messages']]
          )
          sample = digest([
              turn,
              runtime,
              group,
              step,
              results['ref']['model_sha256'],
              proof,
          ])
        tensors.append({
            **tensor,
            'id': f'{identity}-{run_id}-{len(tensors)}',
            'run': run_id,
            'runtime_layer': tensor.get('layer'),
            'layer': layer,
            'anchor': anchor,
            'batch': batch_id,
            'turn': turn,
            'runtime_step': step,
            'graph_invocation': phase_indexes[phase],
            'token_range': None,
            'sample': sample,
            'format': 'safetensors',
            'path': name,
            'source': str(source.relative_to(registry.root)),
            'raw_source': (
                (
                    str(
                        Path(tensor['source'])
                        .resolve()
                        .relative_to(registry.root)
                    )
                    if Path(tensor['source'])
                    .resolve()
                    .is_relative_to(registry.root)
                    else str(Path(tensor['source']).resolve())
                )
                if tensor.get('source')
                else None
            ),
            'sha256': checksum,
        })
  steps = [c[2] for c in coordinates if c[2] is not None]
  link_token_batches(session, telemetry, turn, runtimes)
  turn_tokens = [
      c['tokens'] for c in session['conversation'] if c['turn'] == turn
  ]
  for run_id, forwards in current_forwards.items():
    if run_id in input_counts:
      continue
    prefill = next(
        (
            row
            for _, row in sorted(forwards.items())
            if row['phase'] == 'prefill' and row['status'] == 'completed'
        ),
        None,
    )
    ids = (
        (prefill or {})
        .get('input_proof', {})
        .get('tensors', {})
        .get('input_ids', {})
        .get('values')
    )
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
      # The proof was checked against the stored model boundary. This is
      # the actual uncached suffix, not a retokenized message or history.
      input_counts[run_id] = len(ids[0])
  for conversation in session['conversation']:
    if conversation['turn'] == turn and conversation['run'] in input_counts:
      conversation['input_token_count'] = input_counts[conversation['run']]
  same_count = (
      set(input_counts) == set(runtimes)
      and len(set(input_counts.values())) == 1
  )
  session['turns'].append(
      dict(
          n=turn,
          step_start=min(steps) if steps else None,
          step_end=max(steps) if steps else None,
          prefill_tokens=next(iter(input_counts.values()))
          if same_count
          else None,
          decode_tokens=len(turn_tokens[0])
          if turn_tokens and len(turn_tokens[0]) == len(turn_tokens[-1])
          else None,
      )
  )
  if set(runtimes.values()) == {'LiteRT-LM'} and complete_runs:
    session['notice'] = (
        'Native LiteRT-LM capture. Batches and KV observations retain original'
        ' per-run invocation IDs. Automatic numerical pairing requires matching'
        ' observed logical token context; differing Prefill chunk sizes are'
        ' preserved. Token counts include actual sampler stop IDs. Captured'
        ' logits precede optional sampling constraints.'
    )
    if inferred_runs:
      session['notice'] = (
          'Native LiteRT-LM capture with inferred generation evidence.'
          ' Token-to-invocation bindings and logical contexts are derived from'
          ' the runtime dump under greedy decoding (each released token'
          ' re-checked against its Decode logits argmax); they are not a'
          ' recorded native trace. KV snapshots exist only after Prefill and at'
          ' generation end, with derived logical contexts; GPU KV bytes are'
          ' normalized on the Server with the pinned WebGPU conversion.'
      )
  session['anchors'] = (
      semantic['semantic_graph'][0]['anchors']
      if semantic['semantic_graph']
      else []
  )
  if any(tensor['phase'] == 'unknown' for tensor in tensors) and not any(
      p['id'] == 'unknown' for p in session['phases']
  ):
    session['phases'].append({'id': 'unknown', 'label': 'Unknown'})
  for name, value in [
      ('session.json', session),
      ('semantic.json', semantic),
      ('execution.json', execution),
      ('tensor_index.json', {'tensors': tensors}),
      ('telemetry.json', telemetry),
  ]:
    atomic_json(temporary / name, value)
  store = SessionStore(temporary)
  for tensor in tensors:
    store.load(tensor)
  for resource in telemetry['resources']:
    store.load_resource(resource['id'])
  # Persist evidence before committing the publication directory name.
  for path in temporary.rglob('*'):
    if path.is_file():
      with path.open('rb') as stream:
        os.fsync(stream.fileno())
  for path in sorted(
      [temporary, *(p for p in temporary.rglob('*') if p.is_dir())],
      key=lambda p: len(p.parts),
      reverse=True,
  ):
    descriptor = os.open(path, os.O_RDONLY)
    try:
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
  temporary.rename(destination)
  descriptor = os.open(destination.parent, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)
  return str(destination.relative_to(registry.root))
