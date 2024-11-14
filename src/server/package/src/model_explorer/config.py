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
from typing import Dict, TypedDict, Union
from urllib.parse import quote

import requests
from typing_extensions import NotRequired

try:
  import torch
except ImportError:
  torch = None

from .consts import DEFAULT_HOST, DEFAULT_SETTINGS
from .node_data_builder import NodeData
from .types import ModelExplorerGraphs
from .utils import convert_adapter_response

if torch is not None:
  from .pytorch_exported_program_adater_impl import PytorchExportedProgramAdapterImpl

ModelSource = TypedDict(
    'ModelSource', {'url': str, 'adapterId': NotRequired[str]}
)

EncodedUrlData = TypedDict(
    'EncodedUrlData',
    {
        'models': list[ModelSource],
        'nodeData': NotRequired[list[str]],
        'nodeDataTargets': NotRequired[list[str]],
    },
)


class ModelExplorerConfig:
  """Stores the data to be visualized in Model Explorer."""

  def __init__(self) -> None:
    self.model_sources: list[ModelSource] = []
    # Array of ModelExplorerGraphs or json string.
    self.graphs_list: list[Union[ModelExplorerGraphs, str]] = []
    self.node_data_sources: list[str] = []
    # List of model names to apply node data to. For the meaning of
    # "model name", see comments in `add_node_data_from_path` method below.
    self.node_data_target_models: list[str] = []
    self.node_data_list: list[Union[NodeData, str]] = []
    self.reuse_server_host = ''
    self.reuse_server_port = -1

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
      exported_program: 'torch.export.ExportedProgram',
      settings=DEFAULT_SETTINGS,
  ) -> 'ModelExplorerConfig':
    """Adds the given pytorch model.

    Args:
      name: the name of the model for display purpose.
      exported_program: the ExportedProgram from torch.export.export.
      settings: The settings that config the visualization.
    """

    if torch is None:
      raise ImportError(
          '`torch` not found. Please install it via `pip install torch`, '
          'and restart the Model Explorer server.'
      )

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
      self, path: str, model_name: Union[str, None] = None
  ) -> 'ModelExplorerConfig':
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
      node_data: Union[NodeData, str],
      model_name: Union[str, None] = None,
  ) -> 'ModelExplorerConfig':
    """Adds the given node data object.

    Args:
      name: the name of the NodeData for display purpose.
      node_data: the NodeData object or node data json string to add.
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

  def set_reuse_server(
      self,
      server_host: str = DEFAULT_HOST,
      server_port: Union[int, None] = None,
  ):
    """Makes it to reuse the existing server instead of starting a new one.

    Args:
      server_host: the host of the server to reuse.
      server_port: the port of the server to reuse. If unspecified, it will try
          to find a running server from port 8080 to 8099.
    """
    # Find the server to reuse.
    self.reuse_server_host = server_host
    if server_port is None or server_port < 0:
      self.reuse_server_port = self._find_running_server_port(host=server_host)
    else:
      if self._check_running_server(host=server_host, port=server_port):
        self.reuse_server_port = server_port

    if self.reuse_server_port > 0:
      print(
          'Re-using running server at'
          f' http://{self.reuse_server_host}:{self.reuse_server_port}'
      )
    else:
      print(f'No running server found. Will start a new server.')

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

  def get_model_explorer_graphs(
      self, index: int
  ) -> Union[ModelExplorerGraphs, str]:
    return self.graphs_list[index]

  def get_node_data(self, index: int) -> Union[NodeData, str]:
    return self.node_data_list[index]

  def has_data_to_encode_in_url(self) -> bool:
    return len(self.model_sources) > 0 or len(self.node_data_sources) > 0

  def get_transferrable_data(self) -> Dict:
    # Convert the graphs list to a list of strings.
    graphs_list = []
    for graph in self.graphs_list:
      if isinstance(graph, str):
        graphs_list.append(graph)
      else:
        graphs_list.append(json.dumps(convert_adapter_response(graph)))

    return {
        'graphs_list': json.dumps(graphs_list),
        'model_sources': self.model_sources,
        'node_data_sources': self.node_data_sources,
        'node_data_target_models': self.node_data_target_models,
    }

  def set_transferrable_data(self, data: Dict):
    self.graphs_list = json.loads(data['graphs_list'])
    self.model_sources = data['model_sources']
    self.node_data_sources = data['node_data_sources']
    self.node_data_target_models = data['node_data_target_models']

  def _find_running_server_port(self, host: str = DEFAULT_HOST) -> int:
    for port in range(8080, 8100):
      if self._check_running_server(host=host, port=port):
        return port
    return -1

  def _check_running_server(self, host: str, port: int) -> bool:
    try:
      resp = requests.get(f'http://{host}:{port}/check_health')
      return resp.status_code == 200 and resp.text == 'model_explorer_ok'
    except:
      return False
