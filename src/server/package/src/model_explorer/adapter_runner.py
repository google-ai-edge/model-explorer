import os
from .types import AdapterCommand, ModelExplorerGraphs

from .registered_extension import RegisteredExtension
from .utils import get_instance_method


class AdapterRunner:
  """A runner to run adapter extension."""

  def run_adapter(self, extension: RegisteredExtension, cmd: AdapterCommand) -> ModelExplorerGraphs:
    # Get extension class and create an instance.
    extension_class = extension.ext_class
    instance = extension_class()

    # Get the method by the cmdId.
    fn = get_instance_method(instance, cmd['cmdId'])
    if fn is None:
      raise Exception(
          f'Method {cmd["cmdId"]} not implemented in the adapter extension "{extension.metadata.name}"')
    modelPath = os.path.expanduser(cmd['modelPath'])
    return fn(modelPath, cmd['settings'])
