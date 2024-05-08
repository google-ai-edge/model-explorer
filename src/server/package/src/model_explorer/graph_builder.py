from dataclasses import dataclass, field
from typing import Union


@dataclass
class Graph:
  """A Model Explorer graph."""

  # The id of the graph.
  id: str

  # A list of nodes in the graph.
  nodes: list['GraphNode'] = field(default_factory=list)


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


@dataclass
class KeyValue:
  """A key-value pair"""
  key: str
  value: str


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
class GraphCollection:
  # The label of the collection.
  #
  # It will be appended to the file name to distinguish different graph
  # collections in the graph selector.
  label: str

  # A list of graphs in this collection.
  graphs: list[Graph] = field(default_factory=list)
