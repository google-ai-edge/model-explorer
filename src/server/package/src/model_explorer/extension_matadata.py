from dataclasses import dataclass


@dataclass
class ExtensionMetadata:
  """Class for storing extension metadata"""

  id: str = ''
  name: str = ''
  description: str = ''
  source_repo: str = ''
