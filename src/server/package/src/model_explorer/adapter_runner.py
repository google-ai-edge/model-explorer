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

from collections.abc import Callable
import inspect
import os
from typing import Any

from .registered_extension import RegisteredExtension
from .types import AdapterCommand, ModelExplorerGraphs, ProgressCallback
from .utils import get_instance_method


def _accepts_on_progress(fn: Callable[..., Any]) -> bool:
  """Whether `fn` declares a parameter literally named `on_progress`.

  A `**kwargs` parameter doesn't count: adapters that forward their keyword
  arguments elsewhere would break on an unexpected `on_progress`.

  Args:
    fn: The adapter method.

  Returns:
    Whether `fn` opted into progress reporting.
  """
  try:
    params = inspect.signature(fn).parameters.values()
  except (TypeError, ValueError):
    return False
  return any(
      param.name == 'on_progress'
      and param.kind != inspect.Parameter.VAR_KEYWORD
      for param in params
  )


class AdapterRunner:
  """A runner to run adapter extension."""

  def run_adapter(
      self,
      extension: RegisteredExtension,
      cmd: AdapterCommand,
      on_progress: ProgressCallback | None = None,
  ) -> ModelExplorerGraphs:
    """Runs the adapter method named by `cmd['cmdId']`.

    Args:
      extension: The adapter extension to run.
      cmd: The adapter command.
      on_progress: Optional progress callback. It is passed as the `on_progress`
        keyword argument only to adapter methods that declare a parameter of
        that name, so 2-argument `(model_path, settings)` and `**kwargs`
        adapters keep working unchanged.

    Returns:
      The converted graphs.
    """
    # Get extension class and create an instance.
    extension_class = extension.ext_class
    instance = extension_class()

    # Get the method by the cmdId.
    fn = get_instance_method(instance, cmd['cmdId'])
    if fn is None:
      raise Exception(
          f'Method {cmd["cmdId"]} not implemented in the adapter extension'
          f' "{extension.metadata.name}"'
      )
    model_path = os.path.expanduser(cmd['modelPath'])
    if on_progress is not None and _accepts_on_progress(fn):
      return fn(model_path, cmd['settings'], on_progress=on_progress)
    return fn(model_path, cmd['settings'])
