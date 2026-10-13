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

from collections.abc import Callable
from typing import Any, TypedDict

from typing_extensions import NotRequired

from .extension_metadata import ExtensionMetadata
from .graph_builder import Graph, GraphCollection


class ClassWithMetadata:
  metadata: ExtensionMetadata


ExtensionClassType = type[ClassWithMetadata]

DiagnosticsSummary = TypedDict(
    'DiagnosticsSummary',
    {
        'warningCount': int,
        'quantMismatches': int,
        'missingOpDefs': int,
        'details': NotRequired[dict[str, int]],
    },
)

ProgressReport = TypedDict(
    'ProgressReport',
    {
        'taskId': NotRequired[str],
        'stage': str,
        'stageId': NotRequired[int],
        'currentStep': NotRequired[int],
        'totalSteps': NotRequired[int],
        'progressFraction': NotRequired[float],
        'message': NotRequired[str],
        'subgraphName': NotRequired[str],
        'currentOp': NotRequired[str],
        'currentOpLabel': NotRequired[str],
        'elapsedTimeMs': NotRequired[int],
        'payloadJson': NotRequired[str],
        'diagnostics': NotRequired[DiagnosticsSummary],
    },
)

# Receives one conversion progress dictionary (stage, stageId, currentStep,
# totalSteps, progressFraction, message, ...). Returning `False` requests
# cooperative cancellation; any other value continues the conversion.
ProgressCallback = Callable[[dict[str, Any]], bool | None]

AdapterCommand = TypedDict(
    'AdapterCommand',
    {
        'cmdId': str,
        'modelPath': str,
        'settings': dict[str, Any],
        'deleteAfterConversion': bool,
    },
)

NdpCommand = TypedDict(
    'NdpCommand',
    {
        'cmdId': str,
        'modelPath': str,
        'graphId': str,
        'configValues': dict[str, Any],
        'graph': NotRequired[dict[str, Any]],
    },
)


ModelExplorerGraphs = TypedDict(
    'ModelExplorerGraphs',
    {
        'graphs': NotRequired[list[Graph]],
        'graphCollections': NotRequired[list[GraphCollection]],
    },
)
