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

from typing import Union

import torch

from . import server
from .config import ModelExplorerConfig
from .consts import DEFAULT_COLAB_HEIGHT, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_SETTINGS


def config() -> ModelExplorerConfig:
  """Create a new config object."""
  return ModelExplorerConfig()


def visualize(
    model_paths: Union[str, list[str]] = [],
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    extensions: list[str] = [],
    colab_height=DEFAULT_COLAB_HEIGHT,
) -> None:
  """Starts the ME local server and visualizes the models by the given paths.

  Args:
    model_paths: A model path or a list of model paths to visualize.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    extensions: List of extension names to be run with model explorer.
    colab_height: The height of the embedded iFrame when running in colab.
  """
  # Construct config.
  cur_config = config()
  model_paths_list = model_paths
  if isinstance(model_paths, str):
    model_paths_list = [model_paths]
  for model_path in model_paths_list:
    cur_config.add_model_from_path(path=model_path)

  # Start server.
  server.start(
      host=host, port=port, config=cur_config, colab_height=colab_height, extensions=extensions
  )


def visualize_pytorch(
    name: str,
    exported_program: torch.export.ExportedProgram,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    extensions: list[str] = [],
    colab_height=DEFAULT_COLAB_HEIGHT,
    settings=DEFAULT_SETTINGS,
) -> None:
  """Visualizes a pytorch model.

  Args:
    name: The name of the model for display purpose.
    exported_program: The ExportedProgram from torch.export.export.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    extensions: List of extension names to be run with model explorer.
    colab_height: The height of the embedded iFrame when running in colab.
    settings: The settings that config the visualization.
  """
  # Construct config.
  cur_config = config()
  cur_config.add_model_from_pytorch(
      name, exported_program=exported_program, settings=settings
  )

  # Start server.
  server.start(
      host=host, port=port, config=cur_config, colab_height=colab_height, extensions=extensions
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
