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

from model_explorer import (
    Adapter,
    AdapterMetadata,
    ModelExplorerGraphs,
    graph_builder,
)


class MyAdapter(Adapter):
  """A simple adapter that returns a hard-coded graph.

  See more info at:
  https://github.com/google-ai-edge/model-explorer/wiki/6.-Develop-Adapter-Extension
  """

  metadata = AdapterMetadata(
      id='my-adapter',
      name='My first adapter',
      description='My first adapter!',
      source_repo='https://github.com/user/my_adapter',
      fileExts=['test'],
  )

  # This is required.
  def __init__(self):
    super().__init__()

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:

    # Create a graph for my road trip.
    graph = graph_builder.Graph(id='road_trip')

    ###################
    # Add nodes.

    # Create start and end node.
    #
    # They are located at root level hence the empty `namespace` parameter.
    vancouver = graph_builder.GraphNode(
        id='vancouver', label='Vancouver', namespace=''
    )
    la = graph_builder.GraphNode(id='la', label='Los Angeles', namespace='')

    # Create a node for Seattle and put it into a 'coastal drive' layer.
    seattle = graph_builder.GraphNode(
        id='seattle', label='Seattle', namespace='CoastalDrive'
    )

    # Create San Franciso as a sublayer of the CoastalDrive layer and add some
    # tourist sites there.
    sf_golden_gate_bridge = graph_builder.GraphNode(
        id='sf_golden_gate_bridge',
        label='Golden gate bridge',
        namespace='CoastalDrive/SanFrancisco',
    )
    sf_pier_39 = graph_builder.GraphNode(
        id='sf_pier_39', label='PIER 39', namespace='CoastalDrive/SanFrancisco'
    )

    # Create another two cities and put them into an 'inland drive' layer.
    salt_lake_city = graph_builder.GraphNode(
        id='salt_lake_city', label='Salt lake city', namespace='InlandDrive'
    )
    las_vegas = graph_builder.GraphNode(
        id='las_vegas', label='Las Vegas', namespace='InlandDrive'
    )

    # Add all the nodes into graph.
    graph.nodes.extend([
        vancouver,
        la,
        seattle,
        sf_golden_gate_bridge,
        sf_pier_39,
        salt_lake_city,
        las_vegas,
    ])

    ###################
    # Add edges.

    # Connect edges along the cities for coastal drive.
    la.incomingEdges.append(
        graph_builder.IncomingEdge(sourceNodeId='sf_pier_39')
    )
    sf_pier_39.incomingEdges.append(
        graph_builder.IncomingEdge(sourceNodeId='sf_golden_gate_bridge')
    )
    sf_golden_gate_bridge.incomingEdges.append(
        graph_builder.IncomingEdge(sourceNodeId='seattle')
    )
    seattle.incomingEdges.append(
        graph_builder.IncomingEdge(sourceNodeId='vancouver')
    )

    # Connect edges along the cities for inland drive.
    #
    # LA has two incoming edges from pier_39 and las_vegas.
    # We use targetNodeInputId to identify these two edges. pier_39 goes into
    # input id '0' (default), and las_vegas goes into input id '1'.
    la.incomingEdges.append(
        graph_builder.IncomingEdge(
            sourceNodeId='las_vegas', targetNodeInputId='1'
        )
    )
    las_vegas.incomingEdges.append(
        graph_builder.IncomingEdge(sourceNodeId='salt_lake_city')
    )

    # Vancouver has two outgoing edges to seattle and salt_lake_city.
    # We use sourceNodeOutputId to identify these two edges. Vancouver's output
    # id '0' (default) goes to seattle, and its output id '1' goes to salt_lake_city.
    salt_lake_city.incomingEdges.append(
        graph_builder.IncomingEdge(
            sourceNodeId='vancouver', sourceNodeOutputId='1'
        )
    )

    #######################
    # Add node attributes.

    temperatures = ['52F', '74F', '55F', '64F', '65F', '62F', '90F']
    for i, node in enumerate(graph.nodes):
      node.attrs.append(
          graph_builder.KeyValue(key='temperature', value=temperatures[i])
      )

    #######################
    # Add outputs metadata.

    # This is the edge from vancouver to seattle.
    vancouver.outputsMetadata.append(
        graph_builder.MetadataItem(
            # This identifies which output id the metadata attached to.
            #
            # From the "add edges" section we know that output id "0" connects to
            # seattle.
            id='0',
            attrs=[
                graph_builder.KeyValue(key='distance', value='230 km'),
                # "__tensor_tag" is a special metadata key whose value will be
                # used as output name in side panel.
                graph_builder.KeyValue(key='__tensor_tag', value='coastal'),
            ],
        )
    )

    # This is the edge from vancouver to salt_lake_city.
    vancouver.outputsMetadata.append(
        graph_builder.MetadataItem(
            # From the "add edges" section we know that output id "1" connects to
            # salt_lake_city.
            id='1',
            attrs=[
                graph_builder.KeyValue(key='distance', value='1554 km'),
                graph_builder.KeyValue(key='__tensor_tag', value='inland'),
            ],
        )
    )

    # Add other distances
    def add_distance_output_metadata(
        from_node: graph_builder.GraphNode, distance: str
    ):
      from_node.outputsMetadata.append(
          graph_builder.MetadataItem(
              id='0',
              attrs=[graph_builder.KeyValue(key='distance', value=distance)],
          )
      )

    add_distance_output_metadata(salt_lake_city, '677 km')
    add_distance_output_metadata(las_vegas, '439 km')
    add_distance_output_metadata(seattle, '1310 km')
    add_distance_output_metadata(sf_golden_gate_bridge, '10 km')
    add_distance_output_metadata(sf_pier_39, '613 km')

    #######################
    # Add inputs metadata.

    la.inputsMetadata.append(
        graph_builder.MetadataItem(
            id='0',
            attrs=[graph_builder.KeyValue(key='__tensor_tag', value='coastal')],
        )
    )
    la.inputsMetadata.append(
        graph_builder.MetadataItem(
            id='1',
            attrs=[graph_builder.KeyValue(key='__tensor_tag', value='inland')],
        )
    )

    return {'graphs': [graph]}
