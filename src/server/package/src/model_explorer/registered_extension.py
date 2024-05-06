from .extension_matadata import ExtensionMetadata
from .types import ExtensionClassType


class RegisteredExtension(object):
  """A registered extension in extension manager."""

  def __init__(self, metadata: ExtensionMetadata, type: str, ext_class: ExtensionClassType):
    self.metadata = metadata
    self.type = type
    self.ext_class = ext_class
