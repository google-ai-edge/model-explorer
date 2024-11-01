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

try:
  import torch
except ImportError:
  torch = None

from .adapter import Adapter, AdapterMetadata
from .types import ModelExplorerGraphs

if torch is not None:
  from .pytorch_exported_program_adater_impl import PytorchExportedProgramAdapterImpl


class BuiltinPytorchExportedProgramAdapter(Adapter):
  """Built-in pytorch adapter using ExportedProgram."""

  metadata = AdapterMetadata(
      id='builtin_pytorch_exportedprogram',
      name='Pytorch adapter (exported program)',
      description=(
          'A built-in adapter that converts a Pytorch exported program to Model'
          ' Explorer format.'
      ),
      fileExts=['pt2'],
  )

  def __init__(self):
    super().__init__()

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
    if torch is None:
      raise ImportError(
          'Please install the `torch` package, e.g. via `pip install torch`, '
          'and restart the Model Explorer server.'
      )
    ep = torch.export.load(model_path)
    return PytorchExportedProgramAdapterImpl(ep, settings).convert()
