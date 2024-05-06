import json
import os
from typing import Any, Callable, Tuple, TypedDict, Union
from urllib.parse import quote

import torch
from typing_extensions import NotRequired

from .node_data_builder import GraphNodeData, ModelNodeData, NodeData
from .pytorch_exported_program_adater_impl import \
    PytorchExportedProgramAdapterImpl
from .types import ModelExplorerGraphs

ModelSource = TypedDict(
    'ModelSource', {'url': str, 'adapterId': NotRequired[str]})

EncodedUrlData = TypedDict(
    'EncodedUrlData', {
        'models': list[ModelSource],
        'nodeData': NotRequired[list[str]]})


class ModelExplorerConfig:
  """Stores the data to be visualized in Model Explorer."""

  def __init__(self) -> None:
    self.model_sources: list[ModelSource] = []
    self.graphs_list: list[ModelExplorerGraphs] = []
    self.node_data_sources: list[str] = []
    self.node_data_list: list[NodeData] = []

  def add_model_from_path(self, path: str, adapterId: str = '') -> 'ModelExplorerConfig':
    """Adds a model path to the config.

    Args:
      path: the model path to add.
      adapterId: the id of the adapter to use. Default is empty string meaning
          it will use the default adapter.
    """
    # Get the absolute path (after expanding home dir path "~").
    abs_model_path = os.path.abspath(os.path.expanduser(path))

    # Construct model source and add it.
    model_source: ModelSource = {'url': abs_model_path}
    if adapterId != '':
      model_source['adapterId'] = adapterId
    self.model_sources.append(model_source)

    return self

  def add_model_from_pytorch(self,
                             name: str,
                             model: Callable,
                             inputs: Tuple[Any, ...]) -> 'ModelExplorerConfig':
    """Adds the given pytorch model.

    Args:
      name: the name of the model for display purpose.
      model: the callable to trace.
      inputs: Example positional inputs.
    """
    # Convert the given model to model explorer graphs.
    print('Converting pytorch model to model explorer graphs...')
    exported = torch.export.export(model, inputs)
    adapter = PytorchExportedProgramAdapterImpl(exported)
    graphs = adapter.convert()
    graphs_index = len(self.graphs_list)
    self.graphs_list.append(graphs)

    # Construct model source.
    #
    # The model source has a special format, in the form of:
    # graphs://{name}/{graphs_index}
    model_source: ModelSource = {'url': f'graphs://{name}/{graphs_index}'}
    self.model_sources.append(model_source)

    return self

  def add_node_data_from_path(self, path: str) -> 'ModelExplorerConfig':
    """Adds node data file to the config.

    Args:
      path: the path of the node data json file to add.
    """
    # Get the absolute path (after expanding home dir path "~").
    abs_model_path = os.path.abspath(os.path.expanduser(path))

    self.node_data_sources.append(abs_model_path)

    return self

  def add_node_data(self, name: str, node_data: NodeData) -> 'ModelExplorerConfig':
    """Adds the given node data object.

    Args:
      name: the name of the NodeData for display purpose.
      node_data: the NodeData object to add.
    """
    node_data_index = len(self.node_data_list)
    self.node_data_list.append(node_data)

    # Construct sources.
    #
    # The node data source has a special format, in the form of:
    # node_data://{name}//{index}
    self.node_data_sources.append(f'node_data://{name}/{node_data_index}')
    return self

  def to_url_param_value(self) -> str:
    """Converts the config to the url param value."""
    # Construct url data.
    encoded_url_data: EncodedUrlData = {
        'models': self.model_sources
    }

    if self.node_data_sources:
      encoded_url_data['nodeData'] = self.node_data_sources

    # Return its json string.
    return quote(json.dumps(encoded_url_data))

  def get_model_explorer_graphs(self, index: int) -> ModelExplorerGraphs:
    return self.graphs_list[index]

  def get_node_data(self, index: int) -> NodeData:
    return self.node_data_list[index]

  def has_data_to_encode_in_url(self) -> bool:
    return len(self.model_sources) > 0 or len(self.node_data_sources) > 0
