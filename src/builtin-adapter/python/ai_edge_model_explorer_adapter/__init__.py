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
"""Model Explorer Adapter Python API (100% backward-compatible)."""

from __future__ import annotations

import ctypes
from typing import Any

from . import _ffi

__version__ = "0.2.0"


class VisualizeConfig:
  """Configuration for model graph visualization."""

  def __init__(self, const_element_count_limit: int = 16):
    self.const_element_count_limit = const_element_count_limit
    self.add_tensor_name_attribute = False

  def ToCStruct(self) -> _ffi.AdapterVisualizeConfig:
    """Converts this config into a C FFI struct."""
    return _ffi.AdapterVisualizeConfig(
        const_element_count_limit=self.const_element_count_limit,
        add_tensor_name_attribute=self.add_tensor_name_attribute,
    )


def _InvokeConverter(func: Any, config: VisualizeConfig, *args: Any) -> str:
  """Invokes the underlying C converter function and returns the JSON string."""
  lib = _ffi.GetLib()
  c_config = config.ToCStruct()
  out_json = ctypes.c_char_p()
  out_error = ctypes.c_char_p()

  status = func(
      ctypes.byref(c_config),
      *args,
      ctypes.byref(out_json),
      ctypes.byref(out_error),
  )

  try:
    if status != 0:
      error_msg = (
          out_error.value.decode("utf-8")
          if out_error.value is not None
          else "Unknown conversion error"
      )
      raise RuntimeError(error_msg)
    return out_json.value.decode("utf-8") if out_json.value is not None else ""
  finally:
    if out_json.value is not None:
      lib.AdapterFreeString(out_json)
    if out_error.value is not None:
      lib.AdapterFreeString(out_error)


def ConvertSavedModelToJson(config: VisualizeConfig, model_path: str) -> str:
  """Converts a SavedModel to visualizer JSON string."""
  lib = _ffi.GetLib()
  return _InvokeConverter(
      lib.AdapterConvertSavedModelToJson, config, model_path
  )


def ConvertFlatbufferToJson(
    config: VisualizeConfig, model_path: str, is_modelpath: bool = True
) -> str:
  """Converts a Flatbuffer to visualizer JSON string through tfl MLIR."""
  lib = _ffi.GetLib()
  return _InvokeConverter(
      lib.AdapterConvertFlatbufferToJson, config, model_path, is_modelpath
  )


def ConvertFlatbufferDirectlyToJson(
    config: VisualizeConfig, model_path: str
) -> str:
  """Converts a Flatbuffer directly to visualizer JSON string."""
  lib = _ffi.GetLib()
  return _InvokeConverter(
      lib.AdapterConvertFlatbufferDirectlyToJson, config, model_path
  )


def ConvertSavedModelDirectlyToJson(
    config: VisualizeConfig, model_path: str
) -> str:
  """Converts a SavedModel directly to visualizer JSON string."""
  lib = _ffi.GetLib()
  return _InvokeConverter(
      lib.AdapterConvertSavedModelDirectlyToJson, config, model_path
  )


def ConvertGraphDefDirectlyToJson(
    config: VisualizeConfig, model_path: str
) -> str:
  """Converts a GraphDef directly to visualizer JSON string."""
  lib = _ffi.GetLib()
  return _InvokeConverter(
      lib.AdapterConvertGraphDefDirectlyToJson, config, model_path
  )


def ConvertMlirToJson(config: VisualizeConfig, model_path: str) -> str:
  """Converts a MLIR textual/bytecode file to visualizer JSON string."""
  lib = _ffi.GetLib()
  return _InvokeConverter(lib.AdapterConvertMlirToJson, config, model_path)


__all__ = [
    "_ffi",
    "VisualizeConfig",
    "ConvertSavedModelToJson",
    "ConvertFlatbufferToJson",
    "ConvertFlatbufferDirectlyToJson",
    "ConvertSavedModelDirectlyToJson",
    "ConvertGraphDefDirectlyToJson",
    "ConvertMlirToJson",
]
