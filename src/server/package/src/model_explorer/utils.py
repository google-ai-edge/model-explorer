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

import json
import os
import tempfile
from dataclasses import asdict
from typing import Any, Union

from .types import GraphCollection, ModelExplorerGraphs


def get_instance_method(instance: object, fn_name: str) -> Union[Any, None]:
  """Gets the given method from the given class instance."""
  method = getattr(instance, fn_name, None)
  if not callable(method):
    return None
  return method


def convert_builtin_resp(resp_json_str: str) -> list[GraphCollection]:
  """Converts the json string response from the built-in adapters."""
  resp = json.loads(resp_json_str)
  return [
      GraphCollection(label=item['label'], graphs=item['subgraphs'])
      for item in resp
  ]


def convert_adapter_response(resp: ModelExplorerGraphs):
  """Converts the given adapter convert response to python object."""
  if 'graphs' in resp:
    return {'graphs': [asdict(x) for x in resp['graphs']]}
  if 'graphCollections' in resp:
    return {'graphCollections': [asdict(x) for x in resp['graphCollections']]}


def ensure_tf_model_name(model_path: str) -> str:
  """Renames the model file to saved_model.pb by symlinking it to a temp file."""
  file_name = os.path.basename(model_path)
  cur_model_path = model_path

  if file_name != 'saved_model.pb':
    # Create a temp dir.
    tmp_dir = tempfile.mkdtemp()
    # Symlink the given model file to a "saved_model.pb" file in that temp dir.
    target_path = os.path.join(tmp_dir, 'saved_model.pb')
    os.symlink(model_path, target_path)
    cur_model_path = target_path

  return os.path.dirname(cur_model_path)


def remove_none(d: Any) -> Any:
  if isinstance(d, list):
    return [remove_none(x) for x in d if x is not None]
  elif isinstance(d, dict):
    return dict((k, remove_none(v)) for k, v in d.items() if v is not None)
  else:
    return d
