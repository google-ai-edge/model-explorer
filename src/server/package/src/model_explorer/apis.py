from typing import Any, Callable, List, Tuple, Union

from . import server
from .config import ModelExplorerConfig
from .consts import DEFAULT_COLAB_HEIGHT, DEFAULT_HOST, DEFAULT_PORT


def config() -> ModelExplorerConfig:
  """Create a new config object."""
  return ModelExplorerConfig()


def visualize(
        model_paths: Union[str, list[str]] = [],
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        colab_height=DEFAULT_COLAB_HEIGHT) -> None:
  """Starts the ME local server and visualizes the models by the given paths.

  Args:
    model_paths: A model path or a list of model paths to visualize.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
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
      host=host,
      port=port,
      config=cur_config,
      colab_height=colab_height)


def visualize_pytorch(
        name: str,
        model: Callable,
        inputs: Tuple[Any, ...],
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        colab_height=DEFAULT_COLAB_HEIGHT) -> None:
  """Visualizes a pytorch model.

  Args:
    name: The name of the model for display purpose.
    model: The callable to trace.
    inputs: Example positional inputs.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    colab_height: The height of the embedded iFrame when running in colab.
  """
  # Construct config.
  cur_config = config()
  cur_config.add_model_from_pytorch(name, model, inputs)

  # Start server.
  server.start(
      host=host,
      port=port,
      config=cur_config,
      colab_height=colab_height)


def visualize_from_config(
        config: Union[ModelExplorerConfig, None] = None,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        cors_host: Union[str, None] = None,
        no_open_in_browser: bool = False,
        colab_height=DEFAULT_COLAB_HEIGHT) -> None:
  """Visualizes with a config.

  Args:
    config: the object that stores the models to be visualized.
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
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
      cors_host=cors_host,
      no_open_in_browser=no_open_in_browser,
      colab_height=colab_height)
