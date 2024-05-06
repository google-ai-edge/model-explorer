from typing import Dict

from model_explorer_adapter import \
    _pywrap_convert_wrapper as convert_wrapper  # type: ignore

from .adapter import Adapter, AdapterMetadata
from .types import ModelExplorerGraphs
from .utils import convert_builtin_resp


class BuiltinGraphdefAdapter(Adapter):
  """Built-in graphdef adapter."""

  metadata = AdapterMetadata(id='builtin_graphdef',
                             name='GraphDef adapter',
                             description='A built-in adapter that converts GraphDef file to Model Explorer format.',
                             fileExts=['pb', 'pbtxt', 'graphdef'])

  def __init__(self):
    super().__init__()

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
    # Construct config.
    config = convert_wrapper.VisualizeConfig()
    if 'const_element_count_limit' in settings:
      config.const_element_count_limit = settings['const_element_count_limit']

    # Run
    resp_json_str = convert_wrapper.ConvertGraphDefDirectlyToJson(
        config, model_path)
    return {'graphCollections': convert_builtin_resp(resp_json_str)}
