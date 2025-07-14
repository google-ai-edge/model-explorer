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

from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class Graph:
  """A Model Explorer graph."""

  # The id of the graph.
  id: str

  # A list of nodes in the graph.
  nodes: list['GraphNode'] = field(default_factory=list)

  # Attributes for group nodes.
  #
  # It is displayed in the side panel when the group is selected.
  groupNodeAttributes: Union['GroupNodeAttributes', None] = None

  # The data for various tasks that provide extra data to be visualized, such
  # as node data, edge overlay, etc.
  tasksData: Union['TasksData', None] = None

  # Layout-related options.
  layoutConfigs: Union['LayoutConfigs', None] = None


@dataclass
class GraphNode:
  """A node in a graph."""

  # The unique id of the node. It only needs to be unique within the graph.
  id: str

  # The label of the node, displayed on the node in the model graph.
  label: str

  # Define the node's namespace hierarchy in the form of a "path" (e.g., a/b/c).
  # Don't include the node label as the last component of the namespace.
  # This enables the visualizer to create nested layers for easier navigation.
  #
  # For example, for three nodes with the following label and namespace data:
  # - N1: a/b
  # - N2: a/b
  # - N3: a
  #
  # The visualizer will first show a collapsed layer labeled 'a'. After the
  # layer is expanded (by user clicking on it), it will reveal node N3, and
  # another collapsed layer labeled 'b'. After the layer 'b' is expanded, it
  # will reveal two nodes N1 and N2 inside.
  #
  # (optional)
  namespace: str = ''

  # Ids of subgraphs that this node goes into.
  #
  # The ids should point to other Graph objects returned by your adapter
  # extension.  Once set, users will be able to click this node, pick a subgraph
  # from a drop-down list, and see the visualization for the selected subgraph.
  #
  # (optional)
  subgraphIds: list[str] = field(default_factory=list)

  # The attributes of the node.
  #
  # They will be displayd in the side panel when a node is selected:
  #
  # - attr1_key
  #   attr1_value
  #
  # - attr2_key
  #   attr2_value
  # ...
  #
  # (optional)
  attrs: list['KeyValue'] = field(default_factory=list)

  # A list of incoming edges.
  incomingEdges: list['IncomingEdge'] = field(default_factory=list)

  # Metadata for outputs.
  #
  # Typically an adapter would store metadata for output tensors here.
  #
  # They will be displayed in the side panel when a node is selected:
  #
  # Output 0:
  #   - attr1_key: attr1_value
  #   - attr2_key: attr2_value
  #
  # Output 1:
  #   - attr1_key: attr1_value
  #   - attr2_key: attr2_value
  # ...
  #
  # (optional)
  outputsMetadata: list['MetadataItem'] = field(default_factory=list)

  # Metadata for inputs.
  #
  # The following data is stored in this field:
  #
  # - __tensor_tag: identify the name of a node's input tensors.
  # - quantization: the quantization parameter for tflite models when using
  #   tflite direct adapter.
  #
  # (optional)
  inputsMetadata: list['MetadataItem'] = field(default_factory=list)

  # The style of the node.
  #
  # This will overwrite the default node styles.
  #
  # (optional)
  style: Union['GraphNodeStyle', None] = None

  # The custom configs for the node.
  config: Union['GraphNodeConfig', None] = None


@dataclass
class IncomingEdge:
  """An incoming edge of a graph node."""

  # The id of the source node.
  sourceNodeId: str

  # The id of the output from the source node that this edge goes out of.
  #
  # (optional)
  sourceNodeOutputId: str = '0'

  # The id of the input from the target node that this edge connects to.
  #
  # (optional)
  targetNodeInputId: str = '0'


# A "node ids" node attribute value.
#
# Clicking on a node id will jump to the corresponding node in the graph.
@dataclass
class NodeIdsNodeAttributeValue:
  type: str = 'node_ids'
  nodeIds: list[str] = field(default_factory=list)


SpecialNodeAttributeValue = NodeIdsNodeAttributeValue
NodeAttributeValue = Union[str, SpecialNodeAttributeValue]


@dataclass
class KeyValue:
  """A key-value pair"""

  key: str
  value: NodeAttributeValue


@dataclass
class MetadataItem:
  """A single output metadata item for a specific output id."""

  # The id of the input/output this metadata item belongs to.
  id: str

  # The attributes for the input/output.
  #
  # (optional)
  attrs: list[KeyValue] = field(default_factory=list)


@dataclass
class GraphNodeStyle:
  """Style of the node."""

  # The background color of the node.
  #
  # (optional)
  backgroundColor: str = ''

  # The border color of the node.
  #
  # (optional)
  borderColor: str = ''

  # The border color of the node when hovered.
  #
  # (optional)
  hoveredBorderColor: str = ''


@dataclass
class GraphNodeConfig:
  """Custom configs for the node."""

  # Whether to pin the node to the top of the group it belongs to.
  #
  # (optional)
  pinToGroupTop: bool = False


# From group's namespace to its attribuets (key-value pairs).
#
# Use empty group namespace for the graph-level attributes (i.e. shown in
# side panel when no node is selected).
GroupNodeAttributes = Dict[str, Dict[str, str]]


@dataclass
class GraphCollection:
  # The label of the collection.
  #
  # It will be appended to the file name to distinguish different graph
  # collections in the graph selector.
  label: str

  # A list of graphs in this collection.
  graphs: list[Graph] = field(default_factory=list)


@dataclass
class Edge:
  """An edge in the overlay."""

  # The id of the source node. Op node only.
  sourceNodeId: str

  # The id of the target node. Op node only.
  targetNodeId: str

  # Label shown on the edge.
  label: Union[str, None] = None


@dataclass
class EdgeOverlay:
  """An edge overlay."""

  # The name displayed in the UI to identify this overlay.
  name: str

  # The color of the overlay edges.
  #
  # They are rendered in this color when any of the nodes in this overlay is
  # selected.
  edgeColor: str

  # The edges that define the overlay.
  edges: list[Edge] = field(default_factory=list)

  # The width of the overlay edges. Default to 2.
  edgeWidth: Union[int, None] = 2

  # The font size of the edge labels. Default to 7.5.
  edgeLabelFontSize: Union[float, None] = 7.5

  # If set to `true`, only edges that are directly connected to the currently
  # selected node (i.e., edges that either start from or end at the selected
  # node) will be displayed for this overlay. All other edges within this
  # overlay will be hidden.
  showEdgesConnectedToSelectedNodeOnly: Union[bool, None] = None


@dataclass
class EdgeOverlaysData:
  """The data for edge overlays."""

  # The name of this set of overlays, for UI display purposes.
  name: str

  type: str = 'edge_overlays'

  # A list of edge overlays.
  overlays: list[EdgeOverlay] = field(default_factory=list)


@dataclass
class TasksData:
  """Data for various tasks that provide extra data to be visualized."""

  # List of data for edge overlays that will be applied to the left pane
  # (2-pane view) or the only pane (1-pane view).
  edgeOverlaysDataListLeftPane: list[EdgeOverlaysData] = field(
      default_factory=list
  )

  # List of data for edge overlays that will be applied to the right pane.
  edgeOverlaysDataListRightPane: list[EdgeOverlaysData] = field(
      default_factory=list
  )


@dataclass
class LayoutConfigs:
  """Layout-related configs."""

  # Number of pixels that separate nodes horizontally in the layout.
  #
  # Default is 20.
  nodeSep: Union[int, None] = None

  # Number of pixels between each rank in the layout.
  #
  # A rank is a vertical layer that layout algorithm assigns nodes to, forming
  # a hierarchical structure to optimize graph layout and minimize edge
  # crossings.
  #
  # Default is 50.
  rankSep: Union[int, None] = None

  # Number of pixels that separate edges horizontally in the layout.
  #
  # Default is 20.
  edgeSep: Union[int, None] = None
