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
