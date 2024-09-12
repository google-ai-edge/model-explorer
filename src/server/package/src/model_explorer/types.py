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

from typing import Dict, Type, TypedDict
from typing_extensions import NotRequired

from .extension_matadata import ExtensionMetadata
from .graph_builder import Graph, GraphCollection


class ClassWithMetadata:
  metadata: ExtensionMetadata


ExtensionClassType = Type[ClassWithMetadata]

AdapterCommand = TypedDict(
    'AdapterCommand',
    {
        'cmdId': str,
        'modelPath': str,
        'settings': Dict,
        'deleteAfterConversion': bool,
    },
)

ModelExplorerGraphs = TypedDict(
    'ModelExplorerGraphs',
    {
        'graphs': NotRequired[list[Graph]],
        'graphCollections': NotRequired[list[GraphCollection]],
    },
)
