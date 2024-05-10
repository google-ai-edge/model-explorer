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

from typing import Dict

from ai_edge_model_explorer_adapter import \
    _pywrap_convert_wrapper as convert_wrapper  # type: ignore

from .adapter import Adapter, AdapterMetadata
from .types import ModelExplorerGraphs
from .utils import convert_builtin_resp


class BuiltinTfliteMlirAdapter(Adapter):
  """Built-in tflite adapter using MLIR."""

  metadata = AdapterMetadata(id='builtin_tflite_mlir',
                             name='TFLite adapter (MLIR)',
                             description='A built-in adapter that converts a TFLite model to Model Explorer format through MLIR.',
                             fileExts=['tflite'])

  def __init__(self):
    super().__init__()

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
    # Construct config.
    config = convert_wrapper.VisualizeConfig()
    if 'const_element_count_limit' in settings:
      config.const_element_count_limit = settings['const_element_count_limit']

    # Run
    resp_json_str = convert_wrapper.ConvertFlatbufferToJson(
        config, model_path, True)
    return {'graphCollections': convert_builtin_resp(resp_json_str)}
