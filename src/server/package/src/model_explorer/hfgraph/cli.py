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

"""Command-line interface for the model-explorer-hfgraph pipeline.

This module imports none of the `hfgraph` extra (torch, transformers, pydantic)
at load time: the `model-explorer-hfgraph` script is installed even without the
extra, so it checks for the extra first and prints the install command instead
of failing with `ModuleNotFoundError`.
"""

from __future__ import annotations

import argparse
import sys

from . import main as hfgraph_main


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
  """Parses the `model-explorer-hfgraph` command line."""
  parser = argparse.ArgumentParser(
      prog="model-explorer-hfgraph",
      description=(
          "Build a LogicGraph (source-structured, logic-complete graph) of a"
          " Hugging Face model for Model Explorer."
      ),
  )
  parser.add_argument(
      "model", help="Hugging Face model id or local checkpoint directory"
  )
  parser.add_argument("out", help="output directory")
  parser.add_argument(
      "--full-coverage",
      action="store_true",
      help="trace every static prefill length the shape conditions suggest",
  )
  parser.add_argument(
      "--tiny",
      action="store_true",
      help="shrunken config with random weights (structure only)",
  )
  parser.add_argument(
      "--primary",
      default=None,
      help="trace whose shapes the merged graph shows",
  )
  parser.add_argument(
      "--draft",
      action="store_true",
      help="also run torch.export.draft_export",
  )
  parser.add_argument(
      "--skip-trace",
      action="store_true",
      help="reuse traces/*.json already in out",
  )
  parser.add_argument(
      "--decode",
      action="store_true",
      help="add a cached one-token decode trace",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  """Execute the model-explorer-hfgraph pipeline from command-line arguments.

  Args:
    argv: Arguments without the program name; defaults to `sys.argv[1:]`.

  Raises:
    SystemExit: With an error message if the `hfgraph` extra is missing or a
      trace fails.
  """
  args = _parse_args(argv)
  extra = hfgraph_main.hfgraph_extra_status()
  if not extra.ok:
    sys.exit(extra.message("model-explorer-hfgraph"))
  # Imported here so that the check above runs before torch is imported.
  from . import pipeline  # pylint: disable=g-import-not-at-top

  try:
    pipeline.run(pipeline.Options(**vars(args)))
  except RuntimeError as err:
    sys.exit(str(err))


if __name__ == "__main__":
  main()
