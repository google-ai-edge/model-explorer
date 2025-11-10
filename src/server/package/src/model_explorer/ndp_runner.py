# Copyright 2025 The AI Edge Model Explorer Authors.
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

import os
from .types import NdpCommand

from .graph_builder import Graph
from .registered_extension import RegisteredExtension
from .utils import get_instance_method
from dacite import from_dict
from dataclasses import asdict
from typing import Union


class NdpRunner:
  """A runner to run node data provider extension."""

  def run_ndp(self, extension: RegisteredExtension, cmd: NdpCommand) -> dict:
    # Get extension class and create an instance.
    extension_class = extension.ext_class
    instance = extension_class()

    # Get the method by the cmdId.
    cmd_id = cmd['cmdId']
    fn = get_instance_method(instance, cmd_id)
    if fn is None:
      raise Exception(
          f'Method {cmd_id} not implemented in the node data provider extension'
          f' "{extension.metadata.name}"'
      )

    # Run it.
    if cmd_id == 'run':
      model_path = os.path.expanduser(cmd['modelPath'])
      graph_id = cmd['graphId']
      graph: Union[Graph, None] = None
      if 'graph' in cmd:
        graph_json = cmd['graph']
        graph = from_dict(data_class=Graph, data=graph_json)
      return asdict(
          fn(
              provider_id=extension.metadata.id,
              model_path=model_path,
              graph_id=graph_id,
              config_values=cmd['configValues'],
              graph=graph,
          )
      )
    elif cmd_id == 'get_config_editors':
      return asdict(fn(provider_ids=extension.metadata.id))
    else:
      return {}
