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

"""Built-in adapter for LogicGraph (`model-explorer-hfgraph`) files.

This module must stay importable on a base install (Python 3.9, no `torch`,
`transformers`, or `pydantic`): it only depends on `hfgraph.main`, which is
standard-library only. Running a `.hfrun` spec needs the `hfgraph` extra, and
`hfgraph.main` raises an error naming the install command when it is missing.
"""

from __future__ import annotations

from typing import Any

from . import adapter
from . import types as server_types
from .hfgraph import main as hfgraph_main


class BuiltinHfGraphAdapter(adapter.Adapter):
  """Built-in adapter for Hugging Face logic-complete graphs (LogicGraph)."""

  metadata = adapter.AdapterMetadata(
      id='builtin_hfgraph',
      name='HF logic graph adapter (LogicGraph)',
      description=(
          'A built-in adapter that loads source-structured, logic-complete'
          ' graphs of Hugging Face models: module hierarchy, every source'
          ' branch and its coverage status, folded loops, and the source line'
          ' per op. Opening a .hfrun spec runs the pipeline first and requires'
          f' `{hfgraph_main.HFGRAPH_EXTRA_INSTALL_HINT}`.'
      ),
      fileExts=['hfgraph', 'hfrun'],
  )

  def convert(
      self, model_path: str, settings: dict[str, Any]
  ) -> server_types.ModelExplorerGraphs:
    del settings  # Unused.
    return hfgraph_main.convert(model_path)
