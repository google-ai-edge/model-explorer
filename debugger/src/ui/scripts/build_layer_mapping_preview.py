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

"""Rebuild bundled Transformer Mapping previews from a LiteRT Model Explorer export.

Usage: python3 src/ui/scripts/build_layer_mapping_preview.py
/path/to/export.json Only identical preview runtimes are paired; compiler-scope
semantic bindings are static presentation metadata, not evidence of a captured
session correspondence.
"""

import copy
import json
import pathlib
import re
import sys
from typing import Any


def region(
    out: pathlib.Path,
    catalog: dict[str, Any],
    scope: str,
    collection_index: int,
    graph: dict[str, Any],
    selected_nodes: list[dict[str, Any]],
    label: str,
    stage: str = 'any',
    note: str = '',
) -> None:
  """Writes one region preview JSON file and records it in `catalog`."""
  selected_ids = {node['id'] for node in selected_nodes}
  if not selected_ids:
    raise ValueError(f'No selected nodes for {scope} in {graph["id"]}.')
  by_id = {node['id']: node for node in graph['nodes']}
  nodes = []
  inputs = {}
  consts = {}

  def boundary(source_node: dict[str, Any], port_id: str) -> str:
    key = (source_node['id'], port_id)
    if key not in inputs:
      meta = copy.deepcopy(
          next(
              meta
              for meta in source_node['outputsMetadata']
              if meta['id'] == port_id
          )
      )
      port = str(len(inputs))
      meta['id'] = port
      inputs[key] = meta
    return inputs[key]['id']

  for orig in graph['nodes']:
    if orig['id'] not in selected_ids:
      continue
    node = copy.deepcopy(orig)
    node['subgraphIds'] = []
    node['attrs'] = [
        *node.get('attrs', []),
        {'key': 'source_node_id', 'value': node['id']},
        {'key': 'source_namespace', 'value': node.get('namespace', '')},
    ]
    edges = []
    for edge in node.get('incomingEdges', []):
      src = by_id[edge['sourceNodeId']]
      if src['label'] == 'pseudo_const':
        consts[src['id']] = src
        continue
      if src['id'] not in selected_ids:
        edge = {
            **edge,
            'sourceNodeId': 'boundary:inputs',
            'sourceNodeOutputId': boundary(src, edge['sourceNodeOutputId']),
        }
      edges.append(edge)
    node['incomingEdges'] = edges
    node['inputsMetadata'] = [
        meta
        for meta in node.get('inputsMetadata', [])
        if any(edge['targetNodeInputId'] == meta['id'] for edge in edges)
    ]
    node['attrs'].append({
        'key': 'weights_and_constants',
        'value': str(
            sum(
                by_id[edge['sourceNodeId']]['label'] == 'pseudo_const'
                for edge in orig.get('incomingEdges', [])
            )
        ),
    })
    # Use genuine source scopes; remove only the common root and leaf op name.
    node['namespace'] = '/'.join(node.get('namespace', '').split('/')[:-1])
    nodes.append(node)
  used = {
      (edge['sourceNodeId'], edge['sourceNodeOutputId'])
      for node in graph['nodes']
      for edge in node.get('incomingEdges', [])
      if node['id'] not in selected_ids and edge['sourceNodeId'] in selected_ids
  }
  all_used = {
      (edge['sourceNodeId'], edge['sourceNodeOutputId'])
      for target_node in graph['nodes']
      for edge in target_node.get('incomingEdges', [])
  }
  for node in nodes:
    for meta in node['outputsMetadata']:
      if (node['id'], meta['id']) not in all_used:
        used.add((node['id'], meta['id']))
  if inputs:
    nodes.insert(
        0,
        {
            'id': 'boundary:inputs',
            'label': 'Inputs',
            'namespace': '',
            'attrs': [{'key': 'role', 'value': 'External activation inputs'}],
            'incomingEdges': [],
            'outputsMetadata': list(inputs.values()),
        },
    )
  if used:
    nodes.append({
        'id': 'boundary:outputs',
        'label': 'Outputs',
        'namespace': '',
        'attrs': [],
        'incomingEdges': [
            {
                'sourceNodeId': node_id,
                'sourceNodeOutputId': port_id,
                'targetNodeInputId': str(index),
            }
            for index, (node_id, port_id) in enumerate(sorted(used))
        ],
        'outputsMetadata': [],
    })
  paths = [node['namespace'].split('/') for node in nodes if node['namespace']]
  prefix = []
  if paths:
    for parts in zip(*paths):
      if len(set(parts)) != 1:
        break
      prefix.append(parts[0])
  for node in nodes:
    if node['namespace']:
      node['namespace'] = '/'.join(node['namespace'].split('/')[len(prefix) :])
  key = f'c{collection_index}-{graph["id"]}-{scope}'
  data = {
      'id': key,
      'label': label,
      'runtime': 'LiteRT-LM',
      'stage': stage,
      'sourceGraph': graph['id'],
      'collection': collection_index,
      'note': note,
      'nodes': nodes,
      'constants': list(consts.values()),
  }
  ids = {node['id'] for node in nodes}
  if not all(
      edge['sourceNodeId'] in ids
      for node in nodes
      for edge in node.get('incomingEdges', [])
  ):
    raise ValueError(f'Dangling incoming edge in {key}.')
  file_stem = key.replace('-', '_')
  (out / (file_stem + '.json')).write_text(
      json.dumps(data, separators=(',', ':'))
  )
  catalog['scopes'].setdefault(scope, []).append({
      'key': key,
      'label': label,
      'stage': stage,
      'path': 'mapping_data/' + file_stem + '.json',
      'opCount': len(selected_ids),
      'note': note,
      **(
          {'mappingRegion': 'lookup' if stage == 'any' else 'projection'}
          if scope == 'ple_preparation'
          else {}
      ),
  })


# Match Decode semantic bindings exactly; Prefill uses the same compiler
# scope/opcode and ordered occurrence only when both scope/opcode groups have
# the same cardinality.
def signature(node: dict[str, Any]) -> tuple[str, str]:
  namespace = next(
      (
          attr['value']
          for attr in node['attrs']
          if attr['key'] == 'source_namespace'
      ),
      '',
  )
  return (
      re.sub(
          r'layer_\d+',
          'layer_N',
          re.sub(
              r'^LanguageModel\.(decode|prefill)_graph/',
              '',
              namespace.rsplit('/', 1)[0],
          ),
      ),
      node['label'],
  )


def attrs(port: dict[str, Any]) -> dict[str, str]:
  return {attr['key']: attr['value'] for attr in port.get('attrs', [])}


def main() -> None:
  """Rebuilds bundled Transformer Mapping previews."""
  root = pathlib.Path(__file__).resolve().parents[1] / (
      'public/graph_arch_preview'
  )
  source = pathlib.Path(sys.argv[1])
  collections = json.loads(source.read_text())['graphCollections']
  sem = json.loads(
      (root / 'semantic_graph.js')
      .read_text()
      .split('window.SEMANTIC_MODEL=', 1)[1]
      .rstrip(';\n')
  )
  out = root / 'mapping_data'
  catalog: dict[str, Any] = {'scopes': {}}

  exports = {}
  reports = []
  for graph in collections[8]['graphs']:
    if graph['id'] not in ['decode', 'prefill_128', 'prefill_1024']:
      continue
    dynamic = [
        node
        for node in graph['nodes']
        if node['label'] not in ('pseudo_const', 'GraphInputs', 'GraphOutputs')
    ]
    starts = sorted([
        (
            int(node['id']),
            int(re.search(r'/layer_(\d+)/', node['namespace'])[1]),
        )
        for node in dynamic
        if '/pre_attention_norm/' in node.get('namespace', '')
    ])
    by_id = {node['id']: node for node in dynamic}
    head = [
        node
        for node in dynamic
        if 'decode_softmax/' in node.get('namespace', '')
    ]
    head_ids = {node['id'] for node in head}
    final_norm = [
        int(edge['sourceNodeId'])
        for node in head
        for edge in node.get('incomingEdges', [])
        if edge['sourceNodeId'] not in head_ids
        and by_id.get(edge['sourceNodeId'], {}).get('label')
        == 'stablehlo_composite'
    ]
    for idx, (start, layer) in enumerate(starts):
      # Compiler operators are topologically ordered. Layer boundaries are
      # verified norm entry markers.
      end = (
          starts[idx + 1][0]
          if idx + 1 < len(starts)
          else min(
              final_norm, default=max(int(node['id']) for node in dynamic) + 1
          )
      )
      selected = [
          node
          for node in dynamic
          if start <= int(node['id']) < end
          and 'decode_softmax/' not in node.get('namespace', '')
      ]
      region(
          out,
          catalog,
          f'decoder_layer_{layer}',
          8,
          graph,
          selected,
          graph['id'] + f' · Layer {layer}',
          'decode' if graph['id'] == 'decode' else 'prefill',
          'Static preview source region; the final exported Prefill layer may'
          ' be partial.',
      )
      path = out / f'c8_{graph["id"]}_decoder_layer_{layer}.json'
      data = json.loads(path.read_text())
      exports[(graph['id'], layer)] = (data, path)

  for (stage, layer), (graph, path) in exports.items():
    decode_graph = exports[('decode', 0)][0]
    decode_nodes = [
        node
        for node in decode_graph['nodes']
        if not node['id'].startswith('boundary:')
    ]
    graph_nodes = [
        node
        for node in graph['nodes']
        if not node['id'].startswith('boundary:')
    ]
    semantic = sem['semantic_graph'][sem['layers'][layer]['def']]
    anchors = {}
    missing = []
    for op in semantic['nodes']:
      anchor = next(
          attr['value']
          for attr in op['attrs']
          if attr['key'] == 'semantic_output'
      )
      index = sem['runtime_bindings']['0'].get(op['id'])
      match = [
          (node, port)
          for node in decode_nodes
          for port in node['outputsMetadata']
          if attrs(port).get('tensor_index') == str(index)
      ]
      if len(match) != 1:
        missing.append(anchor)
        continue
      matched_node, matched_port = match[0]
      if stage != 'decode' or layer != 0:
        decode_group = [
            cand
            for cand in decode_nodes
            if signature(cand) == signature(matched_node)
        ]
        target_group = [
            cand
            for cand in graph_nodes
            if signature(cand) == signature(matched_node)
        ]
        if len(decode_group) != len(target_group):
          if anchor == 'attention_value':
            consumers = [
                cand
                for cand in graph_nodes
                if cand['label'] == 'reshape'
                and signature(cand)[0].endswith('/attn_vec_einsum')
            ]
            edge = (
                consumers[0]['incomingEdges'][0]
                if len(consumers) == 1
                else None
            )
            producer = next(
                (
                    cand
                    for cand in graph_nodes
                    if edge and cand['id'] == edge['sourceNodeId']
                ),
                None,
            )
            port = (
                next(
                    (
                        out_port
                        for out_port in producer['outputsMetadata']
                        if out_port['id'] == edge['sourceNodeOutputId']
                    ),
                    None,
                )
                if producer
                else None
            )
            if producer and port:
              anchors[anchor] = {
                  'outputNode': producer['id'],
                  'outputPort': port['id'],
                  'expandedGroups': (
                      [producer['namespace']] if producer['namespace'] else []
                  ),
                  'evidence': (
                      'Execution edge feeding the attention output reshape'
                  ),
              }
              continue
          if (
              anchor == 'q_rope_out'
              and layer >= 15
              and len(target_group) == 1
              and len(decode_group) == 2
          ):
            decode_group = [matched_node]
          else:
            missing.append(anchor)
            continue
        matched_node = target_group[decode_group.index(matched_node)]
        matched_port = next(
            (
                out_port
                for out_port in matched_node['outputsMetadata']
                if out_port['id'] == matched_port['id']
            ),
            None,
        )
        if not matched_port:
          missing.append(anchor)
          continue
      anchors[anchor] = {
          'outputNode': matched_node['id'],
          'outputPort': matched_port['id'],
          'expandedGroups': (
              [matched_node['namespace']] if matched_node['namespace'] else []
          ),
          'evidence': (
              'Exact Decode tensor index'
              if stage == 'decode' and layer == 0
              else 'Matching compiler scope, opcode and occurrence'
          ),
      }
    graph['mappingPresentation'] = {
        'anchors': anchors,
        'grouping': (
            'Source compiler namespaces; original node and output-port'
            ' identities preserved.'
        ),
    }
    path.write_text(json.dumps(graph, separators=(',', ':')))
    reports.append((stage, layer, len(anchors), missing))
  print('graphs', len(exports), 'coverage', reports[:3])
  print(
      'incomplete',
      [(stg, lyr, cnt, miss) for stg, lyr, cnt, miss in reports if miss],
  )


if __name__ == '__main__':
  main()
