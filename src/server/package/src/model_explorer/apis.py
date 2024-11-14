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

from typing import TypedDict, Union

from typing_extensions import NotRequired

from . import server
from .config import ModelExplorerConfig, NodeData
from .consts import (
    DEFAULT_COLAB_HEIGHT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SETTINGS,
)

try:
  import torch
except ImportError:
  torch = None

NodeDataInfo = TypedDict(
    'NodeDataInfo',
    {
        # The name of the node data for display purpose.
        'name': str,
        # The NodeData object of node data json string to add.
        #
        # This field takes precedence over node_data_path field below when they
        # are both set.
        'node_data': NotRequired[Union[NodeData, str]],
        # The path of the node data json file to add.
        'node_data_path': NotRequired[str],
        # The name of the model to apply the node data to. If not set, the node
        # data will be applied to the first model by default.
        #
        # To set this field, use the name of the model file (e.g. model.tflite).
        'model_name': NotRequired[str],
    },
)


def config() -> ModelExplorerConfig:
  """Create a new config object."""
  return ModelExplorerConfig()


def visualize(
    model_paths: Union[str, list[str]] = [],
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    extensions: list[str] = [],
    node_data: Union[NodeDataInfo, list[NodeDataInfo]] = [],
    colab_height=DEFAULT_COLAB_HEIGHT,
    reuse_server: bool = False,
    reuse_server_host: str = DEFAULT_HOST,
    reuse_server_port: Union[int, None] = None,
) -> None:
  """Starts the ME local server and visualizes the models by the given paths.

  Args:
    model_paths: A model path or a list of model paths to visualize.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    extensions: List of extension names to be run with model explorer.
    node_data: The node data or a list of node data to display.
    colab_height: The height of the embedded iFrame when running in colab.
    reuse_server: Whether to reuse the current server/browser tab(s) to
        visualize.
    reuse_server_host: the host of the server to reuse. Default to localhost.
    reuse_server_port: the port of the server to reuse. If unspecified, it will
        try to find a running server from port 8080 to 8099.
  """
  # Construct config.
  cur_config = config()
  model_paths_list = model_paths

  if isinstance(model_paths, str):
    model_paths_list = [model_paths]
  for model_path in model_paths_list:
    cur_config.add_model_from_path(path=model_path)

  _add_node_data_to_config(node_data=node_data, config=cur_config)

  if reuse_server:
    cur_config.set_reuse_server(
        server_host=reuse_server_host, server_port=reuse_server_port
    )

  # Start server.
  server.start(
      host=host,
      port=port,
      config=cur_config,
      colab_height=colab_height,
      extensions=extensions,
  )


def visualize_pytorch(
    name: str,
    exported_program: 'torch.export.ExportedProgram',
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    extensions: list[str] = [],
    node_data: Union[NodeDataInfo, list[NodeDataInfo]] = [],
    colab_height=DEFAULT_COLAB_HEIGHT,
    settings=DEFAULT_SETTINGS,
    reuse_server: bool = False,
    reuse_server_host: str = DEFAULT_HOST,
    reuse_server_port: Union[int, None] = None,
) -> None:
  """Visualizes a pytorch model.

  Args:
    name: The name of the model for display purpose.
    exported_program: The ExportedProgram from torch.export.export.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    extensions: List of extension names to be run with model explorer.
    node_data: The node data or a list of node data to display.
    colab_height: The height of the embedded iFrame when running in colab.
    settings: The settings that config the visualization.
    reuse_server: Whether to reuse the current server/browser tab(s) to
        visualize.
    reuse_server_host: the host of the server to reuse. Default to localhost.
    reuse_server_port: the port of the server to reuse. If unspecified, it will
        try to find a running server from port 8080 to 8099.
  """
  # Construct config.
  cur_config = config()
  cur_config.add_model_from_pytorch(
      name, exported_program=exported_program, settings=settings
  )

  _add_node_data_to_config(node_data=node_data, config=cur_config)

  if reuse_server:
    cur_config.set_reuse_server(
        server_host=reuse_server_host, server_port=reuse_server_port
    )

  # Start server.
  server.start(
      host=host,
      port=port,
      config=cur_config,
      colab_height=colab_height,
      extensions=extensions,
  )


def visualize_from_config(
    config: Union[ModelExplorerConfig, None] = None,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    extensions: list[str] = [],
    cors_host: Union[str, None] = None,
    no_open_in_browser: bool = False,
    colab_height=DEFAULT_COLAB_HEIGHT,
) -> None:
  """Visualizes with a config.

  Args:
    config: the object that stores the models to be visualized.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    extensions: List of extension names to be run with model explorer.
    cors_host: The value of the Access-Control-Allow-Origin header. The header
      won't be present if it is None.
    no_open_in_browser: Don't open the web app in browser after server starts.
    colab_height: The height of the embedded iFrame when running in colab.
  """
  # Start server.
  server.start(
      host=host,
      port=port,
      config=config,
      extensions=extensions,
      cors_host=cors_host,
      no_open_in_browser=no_open_in_browser,
      colab_height=colab_height,
  )


def _add_node_data_to_config(
    node_data: Union[NodeDataInfo, list[NodeDataInfo]],
    config: ModelExplorerConfig,
):
  # Convert NodeDataInfo to [NodeDataInfo] if necessary.
  node_data_list: list[NodeDataInfo] = []
  if isinstance(node_data, list):
    node_data_list = node_data
  else:
    node_data_list = [node_data]

  for node_data_info in node_data_list:
    name = node_data_info.get('name', 'node data')
    node_data_path = node_data_info.get('node_data_path')
    node_data_obj = node_data_info.get('node_data')
    model_name = node_data_info.get('model_name')
    if node_data_obj:
      config.add_node_data(
          name=name, node_data=node_data_obj, model_name=model_name
      )
    elif node_data_path:
      config.add_node_data_from_path(path=node_data_path, model_name=model_name)
