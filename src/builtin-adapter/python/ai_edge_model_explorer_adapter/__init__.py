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
"""Model Explorer Adapter Python API."""

from __future__ import annotations

import ctypes

from . import _ffi
from . import version

__version__ = version.__version__

AdapterModelFormat = _ffi.AdapterModelFormat
ProgressReport = _ffi.ProgressReport
ProgressCallback = _ffi.ProgressCallback


class ConversionCancelledError(RuntimeError):
  """Raised when a conversion is cancelled through its progress callback."""


class VisualizeConfig:
  """Configuration for model graph visualization."""

  def __init__(
      self,
      const_element_count_limit: int = 16,
      add_tensor_name_attribute: bool = False,
  ):
    """Initializes the visualization configuration.

    Args:
      const_element_count_limit: Maximum number of constant tensor elements to
        serialize inline.
      add_tensor_name_attribute: Whether to attach tensor names as node
        attributes.
    """
    self.const_element_count_limit = const_element_count_limit
    self.add_tensor_name_attribute = add_tensor_name_attribute

  def to_c_struct(self) -> _ffi.AdapterVisualizeConfig:
    """Converts this config into a C FFI struct."""
    return _ffi.AdapterVisualizeConfig(
        const_element_count_limit=self.const_element_count_limit,
        add_tensor_name_attribute=self.add_tensor_name_attribute,
    )


def Convert(
    config: VisualizeConfig,
    model_path: str,
    model_format: _ffi.AdapterModelFormat,
    *,
    on_progress: ProgressCallback | None = None,
    is_modelpath: bool = True,
) -> str:
  """Converts a model to visualizer JSON string through `adapter_convert`.

  Args:
    config: Visualization settings.
    model_path: Path to the model file or directory (or the serialized model
      when `is_modelpath` is False for flatbuffer conversion).
    model_format: The conversion pipeline to invoke.
    on_progress: Optional callback receiving one progress dictionary per
      throttled C++ progress report. Return `False` to request cancellation; any
      other return value (including `None`) continues the conversion. An
      exception raised by the callback aborts the conversion and is re-raised
      from this function.
    is_modelpath: Whether `model_path` is a filesystem path (flatbuffer only).

  Returns:
    The visualizer JSON string.

  Raises:
    ConversionCancelledError: If `on_progress` requested cancellation.
    RuntimeError: If the conversion failed.
  """
  lib = _ffi.get_lib()
  # Holds an exception raised by `on_progress` so that it can be re-raised
  # after control returns from C++; exceptions must never unwind through C++.
  callback_errors: list[BaseException] = []
  # The CFUNCTYPE object owns the C trampoline. This local (and `options`,
  # which ctypes makes keep a reference) outlives the `adapter_convert` call,
  # so the trampoline cannot be freed while C++ may still invoke it.
  progress_callback = (
      _ffi.make_progress_callback(on_progress, callback_errors)
      if on_progress is not None
      else _ffi.AdapterProgressCallback()  # NULL: no progress reporting.
  )
  options = _ffi.AdapterConvertOptions(
      struct_size=ctypes.sizeof(_ffi.AdapterConvertOptions),
      format=model_format,
      model_path=model_path.encode("utf-8"),
      config=config.to_c_struct(),
      progress_callback=progress_callback,
      user_data=None,
      is_modelpath=is_modelpath,
  )
  out_json = ctypes.c_char_p()
  out_error = ctypes.c_char_p()

  # `lib` is a ctypes.CDLL, so the GIL is released for the duration of the
  # call and other Python threads keep running while C++ converts.
  status = lib.adapter_convert(
      ctypes.byref(options), ctypes.byref(out_json), ctypes.byref(out_error)
  )

  try:
    if callback_errors:
      raise callback_errors[0]
    if status == _ffi.AdapterStatusCode.CANCELLED:
      raise ConversionCancelledError(
          _ffi.decode_c_string(out_error.value)
          or "Conversion cancelled by user"
      )
    if status != _ffi.AdapterStatusCode.OK:
      raise RuntimeError(
          _ffi.decode_c_string(out_error.value) or "Unknown conversion error"
      )
    return _ffi.decode_c_string(out_json.value)
  finally:
    # Check pointer truthiness (`bool`) rather than `.value is not None` so
    # ctypes does not allocate and copy the C string buffer a second time.
    if bool(out_json):
      lib.adapter_free_string(out_json)
    if bool(out_error):
      lib.adapter_free_string(out_error)


def ConvertSavedModelToJson(
    config: VisualizeConfig,
    model_path: str,
    on_progress: ProgressCallback | None = None,
) -> str:
  """Converts a SavedModel to visualizer JSON string."""
  return Convert(
      config,
      model_path,
      _ffi.AdapterModelFormat.SAVED_MODEL,
      on_progress=on_progress,
  )


def ConvertFlatbufferToJson(
    config: VisualizeConfig,
    model_path: str,
    is_modelpath: bool = True,
    on_progress: ProgressCallback | None = None,
) -> str:
  """Converts a Flatbuffer to visualizer JSON string through tfl MLIR."""
  return Convert(
      config,
      model_path,
      _ffi.AdapterModelFormat.FLATBUFFER,
      on_progress=on_progress,
      is_modelpath=is_modelpath,
  )


def ConvertFlatbufferDirectlyToJson(
    config: VisualizeConfig,
    model_path: str,
    on_progress: ProgressCallback | None = None,
) -> str:
  """Converts a Flatbuffer directly to visualizer JSON string."""
  return Convert(
      config,
      model_path,
      _ffi.AdapterModelFormat.FLATBUFFER_DIRECT,
      on_progress=on_progress,
  )


def ConvertSavedModelDirectlyToJson(
    config: VisualizeConfig,
    model_path: str,
    on_progress: ProgressCallback | None = None,
) -> str:
  """Converts a SavedModel directly to visualizer JSON string."""
  return Convert(
      config,
      model_path,
      _ffi.AdapterModelFormat.SAVED_MODEL_DIRECT,
      on_progress=on_progress,
  )


def ConvertGraphDefDirectlyToJson(
    config: VisualizeConfig,
    model_path: str,
    on_progress: ProgressCallback | None = None,
) -> str:
  """Converts a GraphDef directly to visualizer JSON string."""
  return Convert(
      config,
      model_path,
      _ffi.AdapterModelFormat.GRAPH_DEF_DIRECT,
      on_progress=on_progress,
  )


def ConvertMlirToJson(
    config: VisualizeConfig,
    model_path: str,
    on_progress: ProgressCallback | None = None,
) -> str:
  """Converts a MLIR textual/bytecode file to visualizer JSON string."""
  return Convert(
      config, model_path, _ffi.AdapterModelFormat.MLIR, on_progress=on_progress
  )


__all__ = [
    "__version__",
    "_ffi",
    "AdapterModelFormat",
    "ConversionCancelledError",
    "ProgressCallback",
    "ProgressReport",
    "VisualizeConfig",
    "Convert",
    "ConvertSavedModelToJson",
    "ConvertFlatbufferToJson",
    "ConvertFlatbufferDirectlyToJson",
    "ConvertSavedModelDirectlyToJson",
    "ConvertGraphDefDirectlyToJson",
    "ConvertMlirToJson",
]
