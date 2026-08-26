# Copyright 2026 The AI Edge Model Explorer Authors.
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
"""Backward-compatibility shim redirecting to ai_edge_model_explorer_adapter."""

import ai_edge_model_explorer_adapter

VisualizeConfig = ai_edge_model_explorer_adapter.VisualizeConfig
ConvertSavedModelToJson = ai_edge_model_explorer_adapter.ConvertSavedModelToJson
ConvertFlatbufferToJson = ai_edge_model_explorer_adapter.ConvertFlatbufferToJson
ConvertFlatbufferDirectlyToJson = (
    ai_edge_model_explorer_adapter.ConvertFlatbufferDirectlyToJson
)
ConvertSavedModelDirectlyToJson = (
    ai_edge_model_explorer_adapter.ConvertSavedModelDirectlyToJson
)
ConvertGraphDefDirectlyToJson = (
    ai_edge_model_explorer_adapter.ConvertGraphDefDirectlyToJson
)
ConvertMlirToJson = ai_edge_model_explorer_adapter.ConvertMlirToJson

__all__ = [
    "VisualizeConfig",
    "ConvertSavedModelToJson",
    "ConvertFlatbufferToJson",
    "ConvertFlatbufferDirectlyToJson",
    "ConvertSavedModelDirectlyToJson",
    "ConvertGraphDefDirectlyToJson",
    "ConvertMlirToJson",
]
