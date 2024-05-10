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
