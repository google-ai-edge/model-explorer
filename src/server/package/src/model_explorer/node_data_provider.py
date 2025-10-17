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


from dataclasses import dataclass, field
from abc import abstractmethod
from typing import Union

from .config_editor import ConfigEditor
from .extension_base import Extension
from .extension_matadata import ExtensionMetadata
from .node_data_builder import GraphNodeData


@dataclass
class NodeDataProviderMetadata(ExtensionMetadata):
  """Metadata for a node data provider extension."""

  # The icon of the task, for display purpose.
  #
  # You can pick an icon from https://fonts.google.com/icons?icon.set=Material+Icons
  # and use the icon name from the side panel.
  icon: str = 'extension'

  #
  adapterExtensionIds: list[str] = field(default_factory=list)


@dataclass
class NodeDataProviderResult:
  """The result of a node data provider extension."""

  # The UI will consider a node data provider run is in progress when this field
  # is None (and the error field below is empty).
  result: Union[GraphNodeData, None] = None

  # Error message.
  error: str= ''


@dataclass
class GetConfigEditorsResult:
  """The result of the get_config_editors method."""

  # Config editors.
  configEditors: Union[list[ConfigEditor], None] = None

  # Error message.
  error: str = ''



class NodeDataProvider(Extension):
  """The base class of a node data provider extension."""

  # Subclass needs to override this class variable to provide its metadata.
  metadata: Union[NodeDataProviderMetadata,
                  list[NodeDataProviderMetadata]] = NodeDataProviderMetadata()

  def __init__(self):
    Extension.__init__(self)

  @abstractmethod
  def get_config_editors(self, extension_id: str) -> list[ConfigEditor]:
    """Returns the config editors for the given extension.

    When user clicks a NDP extension to run, the UI will show a dialog to let
    users fill in some data config for this run. The dialog will generate the
    proper UI elements for each config editor and will pass the config data
    (indexed by config editor id) to `NodeDataProvider.run` below.

    Args:
      extension_id: The id of the extension.
    """
    pass

  @abstractmethod
  def run(self, extension_id: str, model_path: str, configs: dict) -> NodeDataProviderResult:
    """Starts the node data calculation.

    Args:
      extension_id: The id of the extension.
      model_path: The path to the model file.
      configs: key-value pairs for the config values that users entered in the
          UI.
    """
    pass
