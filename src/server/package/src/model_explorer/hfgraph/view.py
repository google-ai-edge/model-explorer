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

"""model-explorer-hfgraph-view <out_dir> [--port 8085].

Serves model_explorer.json through Model Explorer's in-memory graph source.
"""

from __future__ import annotations

import argparse
import json
import os

from .. import apis
from . import main as hfgraph_main


def main() -> None:
  """Serve model_explorer.json locally using Model Explorer in-memory source."""
  parser = argparse.ArgumentParser()
  parser.add_argument("out")
  parser.add_argument("--port", type=int, default=8085)
  parser.add_argument("--open", action="store_true")
  args = parser.parse_args()

  path = (
      os.path.join(args.out, "model_explorer.json")
      if os.path.isdir(args.out)
      else args.out
  )

  with open(path, encoding="utf-8") as f:
    coll = json.load(f)

  cfg = apis.config()
  graphs = {
      "graphCollections": [
          hfgraph_main.to_collection(
              coll, os.path.basename(os.path.dirname(path))
          )
      ]
  }
  idx = len(cfg.graphs_list)
  cfg.graphs_list.append(graphs)
  cfg.model_sources.append(
      {"url": f"graphs://{coll.get('label', 'graph')}/{idx}"}
  )
  print(
      f"URL: http://localhost:{args.port}/?data={cfg.to_url_param_value()}",
      flush=True,
  )
  apis.visualize_from_config(
      cfg, port=args.port, no_open_in_browser=not args.open
  )


if __name__ == "__main__":
  main()
