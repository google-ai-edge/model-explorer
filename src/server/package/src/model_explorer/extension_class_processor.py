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

from typing import Union, Dict, List, TypedDict, Any

from .types import ExtensionClassType


class ExtensionInfo(TypedDict):
  """Information about a registered extension."""

  type: str
  cls: ExtensionClassType
  metadata: Dict[str, Any]


class ExtensionClassProcessor(type):
  """
  Processes the class related info for the corresponding extension. This class
  is used as metaclass for the base Extension class and its __init__ function
  will be called when the module that contains any Extension based class is
  imported.
  """

  # To add a new extension type:
  #
  # 1. Add a base class that extends Extension
  # 2. Add the class name in the ignore list here.
  # 3. Add a if branch in __init__ below.
  IGNORE_CLASS_NAMES = ['Extension', 'Adapter', 'NodeDataProvider']

  extension_registry: Dict[str, ExtensionInfo] = {}

  def __init__(cls, name, bases, attrs):
    super().__init__(cls)

    extension_id = f'{cls.__module__}.{name}'

    if name not in ExtensionClassProcessor.IGNORE_CLASS_NAMES:
      # Get the extension type by checking its base class.
      ext_type = ''
      for base in bases:
        base_name = base.__name__
        if base_name == 'Adapter':
          ext_type = 'adapter'
          break
        elif base_name == 'NodeDataProvider':
          ext_type = 'node_data_provider'
          break
      if not ext_type:
        ext_type = 'unknown'

      if ExtensionClassProcessor.has_extension(extension_id):
        raise Exception(
            f'Extension with id "{extension_id}" has already been registered.'
        )
      ExtensionClassProcessor.extension_registry[extension_id] = {
          'type': ext_type,
          'cls': cls,
          'metadata': cls.metadata,
      }

  @classmethod
  def get_registry(cls) -> Dict[str, ExtensionInfo]:
    return cls.extension_registry

  @classmethod
  def has_extension(cls, extension_id: str) -> bool:
    return extension_id in cls.extension_registry
