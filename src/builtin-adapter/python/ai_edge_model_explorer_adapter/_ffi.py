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
"""Low-level C library loading and FFI signatures for Model Explorer Adapter."""

from __future__ import annotations

import ctypes
from importlib import resources
import os
import sys


class CStringParam(ctypes.c_char_p):
  """Custom ctypes type that automatically encodes Python strings to UTF-8 bytes."""

  @classmethod
  def from_param(cls, obj):
    if obj is None:
      return None
    if isinstance(obj, str):
      return obj.encode("utf-8")
    return obj


class AdapterVisualizeConfig(ctypes.Structure):
  _fields_ = [
      ("const_element_count_limit", ctypes.c_int32),
      ("add_tensor_name_attribute", ctypes.c_bool),
  ]


_LIB: ctypes.CDLL | None = None


def get_lib() -> ctypes.CDLL:
  """Loads and returns the Model Explorer Adapter C shared library."""
  global _LIB
  if _LIB is not None:
    return _LIB

  if sys.platform == "win32":
    lib_name = "ai_edge_model_explorer_adapter.dll"
  elif sys.platform == "darwin":
    lib_name = "libai_edge_model_explorer_adapter.dylib"
  else:
    lib_name = "libai_edge_model_explorer_adapter.so"

  # 1. Try loading via importlib.resources from installed wheel package
  try:
    ref = resources.files(__package__) / lib_name
    with resources.as_file(ref) as path:
      if path.exists():
        _LIB = ctypes.CDLL(str(path))
  except (ImportError, FileNotFoundError, TypeError):
    pass

  # 2. Try loading from package directory
  if _LIB is None:
    candidate_path = os.path.join(os.path.dirname(__file__), lib_name)
    if os.path.exists(candidate_path):
      _LIB = ctypes.CDLL(candidate_path)

  # 3. Fallback to direct path in runfiles for local development / Bazel
  if _LIB is None:
    path = os.path.join(os.path.dirname(__file__), "../../c", lib_name)
    if os.path.exists(path):
      _LIB = ctypes.CDLL(path)

  if _LIB is None:
    raise RuntimeError(
        f"Could not find {lib_name}. Ensure it is built and included in the"
        " package or runfiles."
    )

  _setup_signatures(_LIB)
  return _LIB


def _setup_signatures(lib: ctypes.CDLL) -> None:
  """Configures argument and return types for the C library."""
  lib.adapter_free_string.argtypes = [ctypes.c_void_p]
  lib.adapter_free_string.restype = None

  lib.adapter_get_default_config.argtypes = [
      ctypes.POINTER(AdapterVisualizeConfig)
  ]
  lib.adapter_get_default_config.restype = None

  converter_funcs = [
      lib.adapter_convert_saved_model_to_json,
      lib.adapter_convert_flatbuffer_directly_to_json,
      lib.adapter_convert_saved_model_directly_to_json,
      lib.adapter_convert_graph_def_directly_to_json,
      lib.adapter_convert_mlir_to_json,
  ]

  for func in converter_funcs:
    func.argtypes = [
        ctypes.POINTER(AdapterVisualizeConfig),
        CStringParam,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    func.restype = ctypes.c_int

  lib.adapter_convert_flatbuffer_to_json.argtypes = [
      ctypes.POINTER(AdapterVisualizeConfig),
      CStringParam,
      ctypes.c_bool,
      ctypes.POINTER(ctypes.c_char_p),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.adapter_convert_flatbuffer_to_json.restype = ctypes.c_int
