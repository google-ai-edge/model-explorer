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

from .types import ExtensionClassType


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
  IGNORE_CLASS_NAMES = ['Extension', 'Adapter']

  extension_class: Union[ExtensionClassType, None] = None
  extension_type: str = ''

  def __init__(cls, name, bases, attrs):
    super().__init__(cls)

    if name not in ExtensionClassProcessor.IGNORE_CLASS_NAMES:
      # Get the extension type by checking its base class.
      for base in bases:
        base_name = base.__name__
        if base_name == 'Adapter':
          ExtensionClassProcessor.extension_type = 'adapter'
          break

      # Store its class.
      ExtensionClassProcessor.extension_class = cls
