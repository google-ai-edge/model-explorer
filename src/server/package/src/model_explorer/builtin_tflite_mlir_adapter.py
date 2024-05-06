from typing import Dict

from model_explorer_adapter import \
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
