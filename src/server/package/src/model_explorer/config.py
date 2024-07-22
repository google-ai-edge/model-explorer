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

import json
import os
from typing import Any, Callable, Dict, Tuple, TypedDict, Union
from urllib.parse import quote

import torch
from typing_extensions import NotRequired

from .consts import DEFAULT_SETTINGS
from .node_data_builder import GraphNodeData, ModelNodeData, NodeData
from .pytorch_exported_program_adater_impl import PytorchExportedProgramAdapterImpl
from .types import ModelExplorerGraphs

ModelSource = TypedDict(
    'ModelSource', {'url': str, 'adapterId': NotRequired[str]}
)

EncodedUrlData = TypedDict(
    'EncodedUrlData',
    {'models': list[ModelSource],
     'nodeData': NotRequired[list[str]],
     'nodeDataTargets': NotRequired[list[str]]},
)


class ModelExplorerConfig:
  """Stores the data to be visualized in Model Explorer."""

  def __init__(self) -> None:
    self.model_sources: list[ModelSource] = []
    self.graphs_list: list[ModelExplorerGraphs] = []
    self.node_data_sources: list[str] = []
    # List of model names to apply node data to. For the meaning of
    # "model name", see comments in `add_node_data_from_path` method below.
    self.node_data_target_models: list[str] = []
    self.node_data_list: list[NodeData] = []

  def add_model_from_path(
      self, path: str, adapterId: str = ''
  ) -> 'ModelExplorerConfig':
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

  def add_model_from_pytorch(
      self,
      name: str,
      exported_program: torch.export.ExportedProgram,
      settings=DEFAULT_SETTINGS,
  ) -> 'ModelExplorerConfig':
    """Adds the given pytorch model.

    Args:
      name: the name of the model for display purpose.
      exported_program: the ExportedProgram from torch.export.export.
      settings: The settings that config the visualization.
    """
    # Convert the given model to model explorer graphs.
    print('Converting pytorch model to model explorer graphs...')
    adapter = PytorchExportedProgramAdapterImpl(exported_program, settings)
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

  def add_node_data_from_path(
          self,
          path: str,
          model_name: Union[str, None] = None) -> 'ModelExplorerConfig':
    """Adds node data file to the config.

    Args:
      path: the path of the node data json file to add.
      model_name: the name of the model. If not set, the node data will be
          applied to the first model added to the config by default.

          To set this parameter:
          For non-pytorch model, this should be the name of the model file
          (e.g. model.tflite). For pytorch model, it should be the `name`
          parameter used to call the `add_model_from_pytorch` api.
    """
    # Get the absolute path (after expanding home dir path "~").
    abs_model_path = os.path.abspath(os.path.expanduser(path))

    self.node_data_sources.append(abs_model_path)
    if model_name is None:
      self.node_data_target_models.append('')
    else:
      self.node_data_target_models.append(model_name)

    return self

  def add_node_data(
      self,
      name: str,
      node_data: NodeData,
      model_name: Union[str, None] = None
  ) -> 'ModelExplorerConfig':
    """Adds the given node data object.

    Args:
      name: the name of the NodeData for display purpose.
      node_data: the NodeData object to add.
      model_name: the name of the model. If not set, the node data will be
          applied to the first model added to the config by default.

          To set this parameter:
          For non-pytorch model, this should be the name of the model file
          (e.g. model.tflite). For pytorch model, it should be the `name`
          parameter used to call the `add_model_from_pytorch` api.
    """
    node_data_index = len(self.node_data_list)
    self.node_data_list.append(node_data)

    # Construct sources.
    #
    # The node data source has a special format, in the form of:
    # node_data://{name}//{index}
    self.node_data_sources.append(f'node_data://{name}/{node_data_index}')
    if model_name is None:
      self.node_data_target_models.append('')
    else:
      self.node_data_target_models.append(model_name)
    return self

  def to_url_param_value(self) -> str:
    """Converts the config to the url param value."""
    # Construct url data.
    encoded_url_data: EncodedUrlData = {'models': self.model_sources}

    if self.node_data_sources:
      encoded_url_data['nodeData'] = self.node_data_sources
    if self.node_data_target_models:
      encoded_url_data['nodeDataTargets'] = self.node_data_target_models

    # Return its json string.
    return quote(json.dumps(encoded_url_data))

  def get_model_explorer_graphs(self, index: int) -> ModelExplorerGraphs:
    return self.graphs_list[index]

  def get_node_data(self, index: int) -> NodeData:
    return self.node_data_list[index]

  def has_data_to_encode_in_url(self) -> bool:
    return len(self.model_sources) > 0 or len(self.node_data_sources) > 0
