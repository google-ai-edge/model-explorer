import os
import shutil
from dataclasses import asdict
from importlib import import_module
from typing import Any, Dict, Union

from .adapter_runner import AdapterRunner
from .consts import PACKAGE_NAME
from .extension_class_processor import ExtensionClassProcessor
from .registered_extension import RegisteredExtension
from .singleton import Singleton
from .types import AdapterCommand
from .utils import convert_adapter_response


class ExtensionManager(object, metaclass=Singleton):
  BUILTIN_ADAPTER_MODULES: list[str] = [
      '.builtin_tflite_flatbuffer_adapter',
      '.builtin_tflite_mlir_adapter',
      '.builtin_tf_mlir_adapter',
      '.builtin_tf_direct_adapter',
      '.builtin_graphdef_adapter',
      '.builtin_pytorch_exportedprogram_adapter',
  ]

  CACHED_REGISTERED_EXTENSIONS: Dict[str, RegisteredExtension] = {}

  def __init__(self, custom_extension_modules: list[str] = []):
    # For custom extensions (i.e. non-built-in extensions), we load their "main"
    # module by default.
    self.custom_extension_modules = [
        f'{x}.main' for x in custom_extension_modules]
    self.extensions: list[RegisteredExtension] = []
    self.adapter_runner: AdapterRunner = AdapterRunner()

  def load_extensions(self) -> None:
    """Loads all extensions."""
    self.extensions = []
    self._import_extensions()

  def get_extensions_metadata(self) -> list:
    """Get metadata for all extensions."""
    exts = [{**asdict(ext.metadata), 'type': ext.type}
            for ext in self.extensions]
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
      resp = self.adapter_runner.run_adapter(
          extension=extension, cmd=cmd)
      return convert_adapter_response(resp)

    return {}

  def cleanup(self, cmd: Any):
    # Get the extension.
    extension_id = cmd['extensionId']
    extension = self._get_extension_by_id(extension_id)

    if extension is not None:
      # Adapter
      if extension.type == 'adapter':
        # Delete the file if it is marked "deleteAfterConversion".
        adapter_cmd: AdapterCommand = cmd
        if adapter_cmd['deleteAfterConversion']:
          model_path = adapter_cmd['modelPath']
          model_dir = os.path.dirname(model_path)
          shutil.rmtree(model_dir, ignore_errors=True)

  def _import_extensions(self):
    # Built-in pywrapped c++ extensions + custom extensions.
    for module in ExtensionManager.BUILTIN_ADAPTER_MODULES + self.custom_extension_modules:
      module_full_name = f'{PACKAGE_NAME}{module}'

      # Get the registered extension from cache if it has already been
      # registered.
      if module_full_name in ExtensionManager.CACHED_REGISTERED_EXTENSIONS:
        self.extensions.append(
            ExtensionManager.CACHED_REGISTERED_EXTENSIONS[module_full_name])
      # Import the extension module if it has not been registered.
      else:
        try:
          import_module(module, PACKAGE_NAME)
        except Exception as err:
          print(f'! Failed to load extension module "{module}":')
          print(err)
          print()
          continue

        if ExtensionClassProcessor.extension_class is not None:
          extension = RegisteredExtension(metadata=ExtensionClassProcessor.extension_class.metadata,
                                          type=ExtensionClassProcessor.extension_type,
                                          ext_class=ExtensionClassProcessor.extension_class)
          self.extensions.append(extension)
          ExtensionManager.CACHED_REGISTERED_EXTENSIONS[module_full_name] = extension

  def _get_extension_by_id(self, id: str) -> Union[RegisteredExtension, None]:
    matches = [ext for ext in self.extensions if ext.metadata.id == id]
    if len(matches) > 0:
      return matches[0]
    return None
