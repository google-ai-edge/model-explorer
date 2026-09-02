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

import os
import shutil

try:
  import torch
except ImportError:
  torch = None

from dataclasses import asdict
from importlib import import_module
from typing import Any, Dict, Union

from .adapter_runner import AdapterRunner
from .ndp_runner import NdpRunner
from .consts import MODULE_NAME
from .extension_class_processor import ExtensionClassProcessor
from .registered_extension import RegisteredExtension
from .singleton import Singleton
from .types import AdapterCommand
from .utils import convert_adapter_response


class ExtensionManager(object, metaclass=Singleton):
  BUILTIN_ADAPTER_MODULES: list[str] = (
      [
          '.builtin_tflite_flatbuffer_adapter',
          '.builtin_tflite_mlir_adapter',
          '.builtin_tf_mlir_adapter',
          '.builtin_tf_direct_adapter',
          '.builtin_graphdef_adapter',
      ]
      + (['.builtin_pytorch_exportedprogram_adapter'] if torch else [])
      + [
          '.builtin_mlir_adapter',
      ]
  )

  CACHED_REGISTERED_EXTENSIONS: Dict[str, list[RegisteredExtension]] = {}

  def __init__(self, custom_extension_modules: list[str] = []):
    # Don't load extensions from backend adapter if it is not available.
    try:
      import ai_edge_model_explorer_adapter  # type: ignore
    except ImportError:
      ExtensionManager.BUILTIN_ADAPTER_MODULES = (
          ['.builtin_pytorch_exportedprogram_adapter']
          if torch is not None
          else []
      )

    # For custom extensions (i.e. non-built-in extensions), we load their "main"
    # module by default.
    self.custom_extension_modules = [
        f'{x}.main' for x in custom_extension_modules
    ]
    self.extensions: list[RegisteredExtension] = []
    self.adapter_runner: AdapterRunner = AdapterRunner()
    self.ndp_runner: NdpRunner = NdpRunner()
    self.uploaded_model_dirs_to_delete: list[str] = []

  def load_extensions(self) -> None:
    """Loads all extensions."""
    self.extensions = []
    self._import_extensions()

  def get_extensions_metadata(self) -> list:
    """Get metadata for all extensions."""
    exts = [
        {**asdict(ext.metadata), 'type': ext.type} for ext in self.extensions
    ]
    exts.append({
        'fileExts': ['json'],
        'type': 'adapter',
        'id': 'builtin_json',
        'name': 'JSON adapter',
        'description': 'Convert graphs json data file or tfjs model.',
    })
    return exts

  def run_cmd(self, cmd: Any) -> Any:
    """Runs the given command."""
    # Get the extension.
    extension_id = cmd['extensionId']
    extension = self._get_extension_by_id(extension_id)
    if extension is None:
      raise Exception(f'Extension "{extension_id}" not found')

    # Run.
    #
    # Adapter.
    if extension.type == 'adapter':
      resp = self.adapter_runner.run_adapter(extension=extension, cmd=cmd)
      return convert_adapter_response(resp)
    # Node data provider.
    elif extension.type == 'node_data_provider':
      resp = self.ndp_runner.run_ndp(extension=extension, cmd=cmd)
      return resp

    return {}

  def cleanup(self, cmd: Any):
    # Get the extension.
    extension_id = cmd['extensionId']
    extension = self._get_extension_by_id(extension_id)

    if extension is not None:
      # Adapter
      if extension.type == 'adapter':
        # Store the file marked to be deleted. We will delete them when the
        # server is stopped.
        adapter_cmd: AdapterCommand = cmd
        if adapter_cmd['deleteAfterConversion']:
          model_path = adapter_cmd['modelPath']
          model_dir = os.path.dirname(model_path)
          self.uploaded_model_dirs_to_delete.append(model_dir)

  def delete_uploaded_model_dirs(self):
    for model_dir in self.uploaded_model_dirs_to_delete:
      shutil.rmtree(model_dir, ignore_errors=True)

  def _import_extensions(self):
    # Built-in pywrapped c++ extensions + custom extensions.
    for module in (
        ExtensionManager.BUILTIN_ADAPTER_MODULES + self.custom_extension_modules
    ):
      module_full_name = (
          f'{MODULE_NAME}{module}' if module.startswith('.') else module
      )

      # Get the registered extension from cache if it has already been
      # registered.
      if module_full_name in ExtensionManager.CACHED_REGISTERED_EXTENSIONS:
        self.extensions.extend(
            ExtensionManager.CACHED_REGISTERED_EXTENSIONS[module_full_name]
        )
      # Import the extension module if it has not been registered.
      else:
        before_keys = set(ExtensionClassProcessor.get_registry().keys())
        try:
          import_module(module, MODULE_NAME if module.startswith('.') else None)
        except Exception as err:
          print(f'! Failed to load extension module "{module}":')
          print(err)
          print()
          continue

        registry = ExtensionClassProcessor.get_registry()
        new_keys = set(registry.keys()) - before_keys
        matching_exts = [
            ext
            for k, ext in registry.items()
            if k in new_keys or ext['cls'].__module__ == module_full_name
        ]

        module_extensions: list[RegisteredExtension] = []
        for ext in matching_exts:
          metadata = ext['metadata']
          metadata_list = metadata if isinstance(metadata, list) else [metadata]
          for m in metadata_list:
            extension = RegisteredExtension(
                metadata=m, type=ext['type'], ext_class=ext['cls']
            )
            module_extensions.append(extension)

        ExtensionManager.CACHED_REGISTERED_EXTENSIONS[module_full_name] = (
            module_extensions
        )
        self.extensions.extend(module_extensions)

  def _get_extension_by_id(self, id: str) -> Union[RegisteredExtension, None]:
    matches = [ext for ext in self.extensions if ext.metadata.id == id]
    if len(matches) > 0:
      return matches[0]
    return None
