from .extension_class_processor import ExtensionClassProcessor
from .extension_matadata import ExtensionMetadata


class Extension(object, metaclass=ExtensionClassProcessor):
  """The base class for all extensions."""

  # Base class needs to override this class variable to provide its metadata.
  metadata = ExtensionMetadata()

  pass
