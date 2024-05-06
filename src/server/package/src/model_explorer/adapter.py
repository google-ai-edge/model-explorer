from dataclasses import dataclass, field
from typing import Dict, Any

from .extension_base import Extension
from .extension_matadata import ExtensionMetadata
from .types import ModelExplorerGraphs


@dataclass
class AdapterMetadata(ExtensionMetadata):
  """Metadata for a adapter extension."""

  # Supported model file extensions.
  fileExts: list[str] = field(default_factory=list)


class Adapter(Extension):
  """The base class of a adapter extension."""

  # Base class needs to override this class variable to provide its metadata.
  metadata = AdapterMetadata()

  def __init__(self):
    Extension.__init__(self)

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
    """Converts the given model with the given settings.

    Args:
      model_path: the absolute path of the model.
      settings: settings set from the frontend (TBD).

    Returns:
      A list of graphs, or a list of graph collections.

      {
        graphs,
        graphCollections,
      }
    """
    return {}
