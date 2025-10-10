import os
from .types import ModelExplorerGraphs, NdpCommand

from .node_data_provider import NodeDataProviderResult
from .registered_extension import RegisteredExtension
from .utils import get_instance_method
from dataclasses import asdict


class NdpRunner:
  """A runner to run node data provider extension."""

  def run_ndp(
      self, extension: RegisteredExtension, cmd: NdpCommand
  ) -> dict:
    # Get extension class and create an instance.
    extension_class = extension.ext_class
    instance = extension_class()

    # Get the method by the cmdId.
    fn = get_instance_method(instance, cmd['cmdId'])
    if fn is None:
      raise Exception(
          f'Method {cmd["cmdId"]} not implemented in the node data provider extension'
          f' "{extension.metadata.name}"'
      )
    modelPath = os.path.expanduser(cmd['modelPath'])
    return asdict(fn(extension.metadata.id, modelPath, cmd['configValues']))
