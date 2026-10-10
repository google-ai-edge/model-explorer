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

"""Loads LogicGraph (`model-explorer-hfgraph`) outputs into Model Explorer.

Two file types:
  *.hfgraph: a Model Explorer GraphCollection JSON written by
    `model-explorer-hfgraph <model> out/` (out/model_explorer.json)
  *.hfrun: a small JSON spec that runs model-explorer-hfgraph first, then
    loads the result:
    {"model": "Qwen/Qwen3.5-2B", "out": "out/qwen3_5_2b", "tiny": false,
    "decode": true, "full_coverage": false}
    `out` must be a directory below the spec's directory; each run replaces
    it. The result is cached there and reused until an output-affecting field
    or the Model Explorer version changes, or `"rerun": true` is set.

Both are handled by the built-in `builtin_hfgraph` adapter, so no
`--extensions` flag is needed:
  model-explorer path/to/file.hfgraph

This module only uses the standard library, so viewing an existing `.hfgraph`
file works on a base install. Running a `.hfrun` spec requires the `hfgraph`
extra: pip install "ai-edge-model-explorer[hfgraph]".
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import importlib.util
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from .. import graph_builder as gb
from .. import types as server_types

HFGRAPH_EXTRA_INSTALL_HINT = 'pip install "ai-edge-model-explorer[hfgraph]"'
HFGRAPH_EXTRA_MODULES = ("torch", "transformers", "pydantic")
_HFGRAPH_EXTRA_MODULES = HFGRAPH_EXTRA_MODULES
_MIN_PYTHON = (3, 10)


def _node(node_dict: dict[str, Any]) -> gb.GraphNode:
  """Build a Model Explorer GraphNode from a dictionary specification."""
  node = gb.GraphNode(
      id=node_dict["id"],
      label=node_dict["label"],
      namespace=node_dict.get("namespace", ""),
      subgraphIds=list(node_dict.get("subgraphIds", [])),
      attrs=[
          gb.KeyValue(key=a["key"], value=str(a["value"]))
          for a in node_dict.get("attrs", [])
      ],
      incomingEdges=[
          gb.IncomingEdge(
              sourceNodeId=e["sourceNodeId"],
              sourceNodeOutputId=str(e.get("sourceNodeOutputId", "0")),
              targetNodeInputId=str(e.get("targetNodeInputId", "0")),
          )
          for e in node_dict.get("incomingEdges", [])
      ],
      outputsMetadata=[
          gb.MetadataItem(
              id=str(m["id"]),
              attrs=[
                  gb.KeyValue(key=a["key"], value=str(a["value"]))
                  for a in m.get("attrs", [])
              ],
          )
          for m in node_dict.get("outputsMetadata", [])
      ],
  )
  style = node_dict.get("style")
  if style:
    node.style = gb.GraphNodeStyle(
        backgroundColor=style.get("backgroundColor", ""),
        borderColor=style.get("borderColor", ""),
        hoveredBorderColor=style.get("hoveredBorderColor", ""),
    )
  return node


def _graph(graph_dict: dict[str, Any]) -> gb.Graph:
  """Build a Model Explorer Graph from a dictionary specification."""
  graph = gb.Graph(
      id=graph_dict["id"],
      nodes=[_node(n) for n in graph_dict.get("nodes", [])],
  )
  if graph_dict.get("groupNodeAttributes"):
    group_attrs = {}
    for k, v in graph_dict["groupNodeAttributes"].items():
      group_attrs[k] = {kk: str(vv) for kk, vv in v.items()}
    graph.groupNodeAttributes = group_attrs
  if graph_dict.get("groupNodeConfigs"):
    graph.groupNodeConfigs = [
        gb.GroupNodeConfig(**config)
        for config in graph_dict["groupNodeConfigs"]
    ]
  if graph_dict.get("layoutConfigs"):
    graph.layoutConfigs = gb.LayoutConfigs(**graph_dict["layoutConfigs"])
  return graph


def to_collection(
    coll: dict[str, Any], fallback_label: str
) -> gb.GraphCollection:
  """Convert a JSON GraphCollection into graph_builder dataclasses.

  Args:
    coll: Dictionary representing a Model Explorer GraphCollection.
    fallback_label: Default label if not present in coll.

  Returns:
    A graph_builder.GraphCollection dataclass.
  """
  return gb.GraphCollection(
      label=coll.get("label", fallback_label),
      graphs=[_graph(g) for g in coll.get("graphs", [])],
  )


def convert(model_path: str) -> server_types.ModelExplorerGraphs:
  """Converts a `.hfgraph` or `.hfrun` file into ModelExplorerGraphs.

  Args:
    model_path: Path to a `.hfgraph` GraphCollection JSON, or to a `.hfrun` spec
      that is executed first (requires the `hfgraph` extra).

  Returns:
    ModelExplorerGraphs with a single graph collection.
  """
  if model_path.endswith(".hfrun"):
    model_path = run_hfrun(model_path)
  with open(model_path, encoding="utf-8") as f:
    coll = json.load(f)
  return {
      "graphCollections": [to_collection(coll, os.path.basename(model_path))]
  }


def missing_hfgraph_extra_modules() -> list[str]:
  """Returns the `hfgraph` extra modules that cannot be imported."""
  missing = []
  for name in _HFGRAPH_EXTRA_MODULES:
    try:
      spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
      spec = None
    if spec is None:
      missing.append(name)
  return missing


@dataclasses.dataclass(frozen=True)
class HfgraphExtraStatus:
  """Whether the `hfgraph` pipeline can run in this interpreter.

  The pipeline needs the `hfgraph` extra (torch, transformers, pydantic), which
  is only installable on Python >= 3.10.

  Attributes:
    missing_modules: Modules of the extra that cannot be imported.
    python_too_old: Whether the interpreter is older than Python 3.10.
  """

  missing_modules: list[str]
  python_too_old: bool

  @property
  def ok(self) -> bool:
    """Returns True if the Python version and extra modules are available."""
    return not self.missing_modules and not self.python_too_old

  def message(self, subject: str) -> str:
    """Formats the missing-extra install hint for `subject`."""
    needs = list(self.missing_modules)
    if self.python_too_old:
      needs.append("Python >= 3.10")
    return (
        f"{subject} requires {', '.join(needs)}. Install the hfgraph extra on"
        f" Python >= 3.10: {HFGRAPH_EXTRA_INSTALL_HINT}"
    )


def hfgraph_extra_status() -> HfgraphExtraStatus:
  """Checks for the `hfgraph` extra without importing it."""
  return HfgraphExtraStatus(
      missing_modules=missing_hfgraph_extra_modules(),
      python_too_old=sys.version_info[:2] < _MIN_PYTHON,
  )


# ---- .hfrun: run the pipeline, then load out/<dir>/model_explorer.json
# Boolean spec keys, passed to `model-explorer-hfgraph` as `--<key>` flags.
_BOOL_FLAGS = ("tiny", "decode", "full_coverage", "draft")
# Records which spec produced an output directory; see `_spec_stamp`.
_SPEC_STAMP_FILE = "hfrun_spec.json"


def _load_spec(spec_path: str) -> dict[str, Any]:
  """Reads a `.hfrun` spec, which must be a JSON object."""
  with open(spec_path, encoding="utf-8") as f:
    spec = json.load(f)
  if not isinstance(spec, dict):
    raise ValueError(
        f"{spec_path} must contain a JSON object; got {type(spec).__name__}"
    )
  return spec


def _resolve_out_dir(spec_path: str, spec: dict[str, Any]) -> str:
  """Returns the absolute output directory, confined to the spec's directory.

  Args:
    spec_path: Path to the `.hfrun` spec.
    spec: The parsed spec.

  Raises:
    ValueError: If `out` is not a string or does not resolve (after `..` and
      symlinks) to a directory strictly below the one that contains the spec.
      A spec may come from an untrusted source, and the output directory is
      replaced on every run.
  """
  base = os.path.realpath(os.path.dirname(os.path.abspath(spec_path)))
  default = os.path.splitext(os.path.basename(spec_path))[0] + ".out"
  out = spec.get("out")
  if out is not None and not isinstance(out, str):
    raise ValueError(f"'out' in {spec_path} must be a string; got {out!r}")
  out = out or default
  resolved = os.path.realpath(os.path.join(base, out))
  norm_base, norm_resolved = os.path.normcase(base), os.path.normcase(resolved)
  if (
      norm_resolved == norm_base
      or os.path.commonpath([norm_resolved, norm_base]) != norm_base
  ):
    raise ValueError(
        f"'out' in {spec_path} must be a directory under {base} (not that"
        f" directory itself); got {out!r}"
    )
  return resolved


def _validated_model(spec_path: str, spec: dict[str, Any]) -> str:
  """Returns `spec["model"]`, rejecting values the CLI would read as options."""
  model = spec.get("model")
  if not isinstance(model, str) or not model or model.startswith("-"):
    raise ValueError(
        f"'model' in {spec_path} must be a Hugging Face model id or a local"
        f" checkpoint directory that does not start with '-'; got {model!r}"
    )
  return model


def _validated_primary(spec_path: str, spec: dict[str, Any]) -> str | None:
  """Returns `spec["primary"]` (a trace id) or None if it is not set."""
  primary = spec.get("primary")
  if primary is None:
    return None
  if not isinstance(primary, str):
    raise ValueError(
        f"'primary' in {spec_path} must be a string; got {primary!r}"
    )
  return primary or None


def _package_version() -> str:
  """Returns the installed Model Explorer version, or 'unknown'."""
  package = importlib.import_module(__package__.rpartition(".")[0])
  return str(getattr(package, "__version__", "unknown"))


def _spec_stamp(spec: dict[str, Any], model: str, primary: str | None) -> str:
  """Returns the canonical form of everything that changes the pipeline output."""
  fields = {flag: bool(spec.get(flag)) for flag in _BOOL_FLAGS}
  fields["model"] = model
  fields["primary"] = primary
  fields["version"] = _package_version()
  return json.dumps(fields, sort_keys=True)


def _is_cached(out_dir: str, result: str, stamp: str) -> bool:
  """Returns whether `out_dir` holds a result produced from a spec with `stamp`.

  Symlinked files are never trusted: they may come with an untrusted spec.

  Args:
    out_dir: The resolved output directory.
    result: The `model_explorer.json` path inside `out_dir`.
    stamp: The current spec's `_spec_stamp`.
  """
  stamp_path = os.path.join(out_dir, _SPEC_STAMP_FILE)
  for path in (result, stamp_path):
    if os.path.islink(path) or not os.path.isfile(path):
      return False
  with open(stamp_path, encoding="utf-8") as f:
    return f.read() == stamp


def _pipeline_executable() -> str:
  """Finds the `model-explorer-hfgraph` executable."""
  exe = shutil.which("model-explorer-hfgraph")
  if exe:
    return exe
  # `sys.executable` is empty or None in embedded interpreters.
  if sys.executable:
    exe = os.path.join(
        os.path.dirname(sys.executable), "model-explorer-hfgraph"
    )
    if os.path.exists(exe):
      return exe
  raise RuntimeError(
      "model-explorer-hfgraph is not installed in this environment"
      f" ({HFGRAPH_EXTRA_INSTALL_HINT}; requires Python >= 3.10)"
  )


def _pipeline_command(
    exe: str, spec: dict[str, Any], model: str, primary: str | None
) -> list[str]:
  """Builds the `model-explorer-hfgraph` command line, minus the out dir."""
  cmd = [exe]
  for flag in _BOOL_FLAGS:
    if spec.get(flag):
      cmd.append("--" + flag.replace("_", "-"))
  if primary:
    # `--opt=value` keeps a value that starts with '-' from parsing as an
    # option.
    cmd.append(f"--primary={primary}")
  # `--` ends option parsing, so the positionals can never act as options.
  return cmd + ["--", model]


def _remove(path: str) -> None:
  """Deletes a file, symlink or directory tree without following symlinks."""
  if os.path.isdir(path) and not os.path.islink(path):
    shutil.rmtree(path)
  else:
    os.remove(path)


def _swap_in(staging: str, out_dir: str) -> None:
  """Replaces `out_dir` (whatever it is) with the `staging` directory."""
  if not os.path.lexists(out_dir):
    os.replace(staging, out_dir)
    return
  previous = staging + ".previous"
  if os.path.lexists(previous):
    _remove(previous)
  os.replace(out_dir, previous)
  try:
    os.replace(staging, out_dir)
  except OSError:
    os.replace(previous, out_dir)
    raise
  _remove(previous)


def _run_pipeline(cmd: list[str], out_dir: str, stamp: str) -> None:
  """Runs `cmd <staging dir>` and swaps the finished output in as `out_dir`.

  The pipeline writes into a fresh, empty directory next to `out_dir`, so
  nothing already in `out_dir` (e.g. symlinks shipped with an untrusted spec)
  is ever written through, and `out_dir` only changes once the run succeeds.

  Args:
    cmd: The pipeline command line without the output directory.
    out_dir: The resolved output directory.
    stamp: The spec stamp to record next to the result.
  """
  # A failed run must not leave the previous result looking current.
  with contextlib.suppress(FileNotFoundError, NotADirectoryError):
    os.remove(os.path.join(out_dir, _SPEC_STAMP_FILE))
  parent = os.path.dirname(out_dir)
  os.makedirs(parent, exist_ok=True)
  staging = tempfile.mkdtemp(
      prefix=f".{os.path.basename(out_dir)}.", suffix=".tmp", dir=parent
  )
  try:
    logging.warning(
        "[model-explorer-hfgraph] running: %s", shlex.join(cmd + [staging])
    )
    subprocess.run(cmd + [staging], check=True)
    with open(
        os.path.join(staging, _SPEC_STAMP_FILE), "w", encoding="utf-8"
    ) as f:
      f.write(stamp)
    _swap_in(staging, out_dir)
  finally:
    if os.path.lexists(staging):
      shutil.rmtree(staging, ignore_errors=True)


def run_hfrun(spec_path: str) -> str:
  """Runs the pipeline described by a `.hfrun` spec unless already cached.

  The result is cached in `out` (resolved relative to the spec file and
  confined below its directory) and reused while the output-affecting spec
  fields (`model`, `primary`, the boolean flags) and the Model Explorer version
  stay the same, unless `rerun` is set. Each run replaces `out` as a whole.

  Args:
    spec_path: Path to the `.hfrun` JSON spec.

  Returns:
    Path to the resulting `model_explorer.json`.

  Raises:
    ValueError: If the spec is not a JSON object or has an invalid `model`,
      `out` or `primary`.
    ImportError: If the `hfgraph` extra (torch, transformers, pydantic) is not
      installed and the result is not cached.
    RuntimeError: If the `model-explorer-hfgraph` executable cannot be found.
    subprocess.CalledProcessError: If the pipeline fails; `out` keeps its
      previous content but is no longer treated as cached.
  """
  spec = _load_spec(spec_path)
  out_dir = _resolve_out_dir(spec_path, spec)
  model = _validated_model(spec_path, spec)
  primary = _validated_primary(spec_path, spec)
  result = os.path.join(out_dir, "model_explorer.json")
  stamp = _spec_stamp(spec, model, primary)
  if not spec.get("rerun", False) and _is_cached(out_dir, result, stamp):
    return result
  extra = hfgraph_extra_status()
  if not extra.ok:
    raise ImportError(
        extra.message("Running a .hfrun spec")
        + "; then restart the Model Explorer server."
    )
  cmd = _pipeline_command(_pipeline_executable(), spec, model, primary)
  _run_pipeline(cmd, out_dir, stamp)
  return result
