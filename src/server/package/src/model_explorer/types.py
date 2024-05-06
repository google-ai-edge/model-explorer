from typing import Dict, Type, TypedDict
from typing_extensions import NotRequired

from .extension_matadata import ExtensionMetadata
from .graph_builder import Graph, GraphCollection


class ClassWithMetadata:
  metadata: ExtensionMetadata


ExtensionClassType = Type[ClassWithMetadata]

AdapterCommand = TypedDict(
    'AdapterCommand',
    {'cmdId': str,
     'modelPath': str,
     'settings': Dict,
     'deleteAfterConversion': bool})

ModelExplorerGraphs = TypedDict(
    'ModelExplorerGraphs',
    {'graphs': NotRequired[list[Graph]],
     'graphCollections': NotRequired[list[GraphCollection]]})
