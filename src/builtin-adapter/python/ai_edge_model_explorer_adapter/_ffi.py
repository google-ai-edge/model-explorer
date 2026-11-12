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

from collections.abc import Callable
import ctypes
import enum
from importlib import resources
import os
import sys
from typing import Any, TypedDict


class CStringParam(ctypes.c_char_p):
  """Custom ctypes type that automatically encodes Python strings to UTF-8 bytes."""

  @classmethod
  def from_param(cls, obj):
    if obj is None:
      return None
    if isinstance(obj, str):
      return obj.encode("utf-8")
    return obj


class AdapterStatusCode(enum.IntEnum):
  """Status codes returned by the C adapter entry points."""

  OK = 0
  ERROR = 1
  INVALID_ARGUMENT = 2
  NOT_FOUND = 3
  CANCELLED = 4


class AdapterModelFormat(enum.IntEnum):
  """Model formats supported by `adapter_convert`."""

  UNKNOWN = 0
  FLATBUFFER = 1
  FLATBUFFER_DIRECT = 2
  SAVED_MODEL = 3
  SAVED_MODEL_DIRECT = 4
  GRAPH_DEF_DIRECT = 5
  MLIR = 6


class AdapterVisualizeConfig(ctypes.Structure):
  """C FFI struct mirroring `adapter_visualize_config_t`."""

  _fields_ = [
      ("const_element_count_limit", ctypes.c_int32),
      ("add_tensor_name_attribute", ctypes.c_bool),
  ]


class AdapterProgressReport(ctypes.Structure):
  """C FFI struct mirroring `adapter_progress_report_t`."""

  _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("stage", ctypes.c_char_p),
      ("stage_id", ctypes.c_int32),
      ("current_step", ctypes.c_int64),
      ("total_steps", ctypes.c_int64),
      ("progress_fraction", ctypes.c_double),
      ("message", ctypes.c_char_p),
      ("subgraph_name", ctypes.c_char_p),
      ("current_op", ctypes.c_char_p),
      ("elapsed_time_ms", ctypes.c_int64),
      ("payload_json", ctypes.c_char_p),
  ]


AdapterProgressCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(AdapterProgressReport), ctypes.c_void_p
)


class AdapterConvertOptions(ctypes.Structure):
  """C FFI struct mirroring `adapter_convert_options_t`."""

  _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("format", ctypes.c_int32),
      ("model_path", ctypes.c_char_p),
      ("config", AdapterVisualizeConfig),
      ("progress_callback", AdapterProgressCallback),
      ("user_data", ctypes.c_void_p),
      ("is_modelpath", ctypes.c_bool),
  ]


# A copy of `AdapterProgressReport`. Keys are camelCase because servers forward
# the dict verbatim to the web app.
ProgressReport = TypedDict(
    "ProgressReport",
    {
        "stage": str,
        "stageId": int,
        "currentStep": int,
        "totalSteps": int,
        "progressFraction": float,
        "message": str,
        "subgraphName": str,
        "currentOp": str,
        "elapsedTimeMs": int,
        "payloadJson": str,
    },
)

# Receives one progress report per C++ progress report. Returning `False`
# requests cancellation; any other value continues the conversion.
ProgressCallback = Callable[[ProgressReport], bool | None]


def decode_c_string(value: bytes | None) -> str:
  """Decodes a nullable UTF-8 C string; NULL decodes to an empty string."""
  return value.decode("utf-8", errors="replace") if value is not None else ""


def progress_report_to_dict(report: AdapterProgressReport) -> ProgressReport:
  """Copies a C progress report into a Python dict before the C call returns.

  The report and its strings are only valid for the duration of the callback,
  so every field is copied out eagerly.

  Args:
    report: The progress report passed to the C callback.

  Returns:
    The progress report handed to the Python progress callback.
  """
  return {
      "stage": decode_c_string(report.stage),
      "stageId": report.stage_id,
      "currentStep": report.current_step,
      "totalSteps": report.total_steps,
      "progressFraction": report.progress_fraction,
      "message": decode_c_string(report.message),
      "subgraphName": decode_c_string(report.subgraph_name),
      "currentOp": decode_c_string(report.current_op),
      "elapsedTimeMs": report.elapsed_time_ms,
      "payloadJson": decode_c_string(report.payload_json),
  }


def make_progress_callback(
    on_progress: ProgressCallback, callback_errors: list[BaseException]
) -> Any:
  """Wraps `on_progress` in an `AdapterProgressCallback` C function pointer.

  ctypes runs CFUNCTYPE callbacks with the GIL held: its trampoline brackets
  every call with `PyGILState_Ensure()`/`PyGILState_Release()`, which also
  registers foreign C++ threads with the interpreter.

  Args:
    on_progress: The Python progress callback.
    callback_errors: Receives an exception raised by `on_progress`. The
      exception is stored instead of propagated because it must never unwind
      through C++ frames.

  Returns:
    The C function pointer. Callers must keep it referenced for as long as C++
    may invoke it.
  """

  def _c_callback(report_ptr: Any, user_data: Any) -> int:
    del user_data  # Unused; the closure carries all Python state.
    try:
      should_continue = on_progress(
          progress_report_to_dict(report_ptr.contents)
      )
    except BaseException as e:  # pylint: disable=broad-exception-caught
      callback_errors.append(e)
      return 1  # Abort the conversion; `Convert` re-raises `e`.
    # Only `False` cancels: `None` (no return value) continues.
    return 1 if should_continue is not None and not should_continue else 0

  return AdapterProgressCallback(_c_callback)


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

  lib.adapter_convert.argtypes = [
      ctypes.POINTER(AdapterConvertOptions),
      ctypes.POINTER(ctypes.c_char_p),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.adapter_convert.restype = ctypes.c_int

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
