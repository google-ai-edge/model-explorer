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


from dataclasses import dataclass
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

  # Conditions that must be met for this extension to appear in the UI.
  #
  # If unset, this extension applies to all graphs.
  filter: Union['NodeDataProviderFilter', None] = None


@dataclass
class NodeDataProviderFilter:
  """
  The filter conditions that determine whether a node data provider
  extension should be available or visible in the UI.

  The filter conditions (modelFileExts and adapterIds) are combined using
  an AND logic. For the extension to be selected, both conditions
  must be met (if set).
  """

  # Extensions (e.g. 'tflite', 'json') of the model files that this extension
  # supports. Don't include ".".
  #
  # Unset (None) means the extension applies to all model file extensions.
  modelFileExts: Union[list[str], None] = None

  # IDs of the adapter(s) used to convert the model graph that this extension
  # supports.
  #
  # Unset (None) means the extension applies to all adapter IDs.
  adapterIds: Union[list[str], None] = None


@dataclass
class NodeDataProviderResult:
  """The result of a node data provider extension."""

  # The UI will consider a node data provider run is in progress when this field
  # is None (and the error field below is empty).
  result: Union[GraphNodeData, None] = None

  # Error message.
  error: str = ''


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
  def run(self, extension_id: str, model_path: str, config_values: dict) -> NodeDataProviderResult:
    """Calculates the node data.

    Args:
      extension_id: The id of the extension.
      model_path: The path to the model file.
      config_values: key-value pairs for the config values that users entered in
          the UI.
    """
    pass
