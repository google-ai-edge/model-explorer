# Copyright 2024 The AI Edge Model Explorer Authors.
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

from typing import Dict

import pydot
from model_explorer import (
    Adapter,
    AdapterMetadata,
    ModelExplorerGraphs,
    graph_builder,
)


class GraphVizAdapter(Adapter):
  """Add GraphViz dot support to Model Explorer.

  This is an example adapter and only supports a subset of features in dot,
  including node label, edge connections, and subgraphs/clusters.
  """

  metadata = AdapterMetadata(
      id='model_explorer_graphviz_dot_adapter',
      name='GraphViz Dot adapter',
      description='Add GraphViz (dot) support to Model Explorer',
      source_repo='https://github.com/user/graphviz_dot_adapter',
      fileExts=['dot'],
  )

  # This is required.
  def __init__(self):
    super().__init__()

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
    gv_graphs = pydot.graph_from_dot_file(model_path)
    if gv_graphs is None:
      return {'graphs': []}

    graphs: list[graph_builder.Graph] = []

    for gv_graph in gv_graphs:
      graph_id = gv_graph.get_name()
      graph = graph_builder.Graph(id=graph_id)
      graphs.append(graph)

      # Generate all graph nodes.
      seen_gv_nodes = set()
      node_to_graph_node = {}
      self._gen_graph_nodes(gv_graph, seen_gv_nodes, node_to_graph_node, '')
      graph.nodes.extend(node_to_graph_node.values())

      # Generate all edges.
      self._gen_graph_edges(gv_graph, node_to_graph_node)

    return {'graphs': graphs}

  def _gen_graph_nodes(
      self,
      cur_gv_graph,
      seen_gv_nodes: set,
      node_to_graph_node: Dict,
      cur_namespace: str,
  ):
    # Get nodes from the node list.
    for gv_node in cur_gv_graph.get_nodes():
      node_name = gv_node.get_name()

      # Skip the special 'node' node.
      if node_name == 'node':
        continue

      if node_name in seen_gv_nodes:
        node_to_graph_node[node_name].namespace = cur_namespace
        continue

      seen_gv_nodes.add(node_name)
      node_label = node_name
      if 'label' in gv_node.obj_dict['attributes']:
        node_label = gv_node.obj_dict['attributes']['label']
      self._create_graph_node(
          node_name, node_label, node_to_graph_node, cur_namespace
      )

    # Get nodes from the edge list.
    gv_edges = cur_gv_graph.get_edges()
    for gv_edge in gv_edges:
      source_node_name = gv_edge.get_source()
      target_node_name = gv_edge.get_destination()

      if source_node_name not in seen_gv_nodes:
        seen_gv_nodes.add(source_node_name)
        self._create_graph_node(
            source_node_name,
            source_node_name,
            node_to_graph_node,
            cur_namespace,
        )
      else:
        node_to_graph_node[source_node_name].namespace = cur_namespace

      if target_node_name not in seen_gv_nodes:
        seen_gv_nodes.add(target_node_name)
        self._create_graph_node(
            target_node_name,
            target_node_name,
            node_to_graph_node,
            cur_namespace,
        )
      else:
        node_to_graph_node[target_node_name].namespace = cur_namespace

    # Recursive on subgraphs.
    subgraphs = cur_gv_graph.get_subgraphs()
    for subgraph in subgraphs:
      subgraph_id = subgraph.get_name()
      if 'label' in subgraph.obj_dict['attributes']:
        subgraph_id = subgraph.obj_dict['attributes']['label'].strip('"')
      new_namespace = f'{cur_namespace}/{subgraph_id}'
      if cur_namespace == '':
        new_namespace = subgraph_id
      self._gen_graph_nodes(
          subgraph, seen_gv_nodes, node_to_graph_node, new_namespace
      )

  def _gen_graph_edges(self, cur_gv_graph, node_to_graph_node: Dict):
    for gv_edge in cur_gv_graph.get_edges():
      source_node = gv_edge.get_source()
      target_node = gv_edge.get_destination()
      target_graph_node = node_to_graph_node[target_node]
      target_graph_node.incomingEdges.append(
          graph_builder.IncomingEdge(sourceNodeId=source_node)
      )

    # Recursive on subgraphs.
    subgraphs = cur_gv_graph.get_subgraphs()
    for subgraph in subgraphs:
      self._gen_graph_edges(subgraph, node_to_graph_node)

  def _create_graph_node(
      self,
      node_name: str,
      node_label: str,
      node_to_graph_node: Dict,
      namespace: str,
  ):
    graph_node = graph_builder.GraphNode(
        node_name, node_label, namespace=namespace
    )
    node_to_graph_node[node_name] = graph_node
