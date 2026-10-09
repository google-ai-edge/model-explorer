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

"""LogicGraph data model v3 (see the "Output" section of hfgraph/README.md).

Three parts:
  templates: every source construct once: a class forward, a helper function, a
    branch, a region.
  paths: where things are, as one string:
    "layers.0/if@888#then/linear_attn/torch_chunk_gated_delta_rule/for@399#body"
    Segments are module instances, function names, and region ids; the same
    string is the Model Explorer namespace.
  traces: one file per torch.export: ops carry their path; the trace also lists
    every path it executed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

SCHEMA_VERSION = 3

CondKind = Literal[
    "config", "shape", "value", "guarded", "python", "unknown", "loop"
]
BranchKind = Literal["if", "for", "while", "ternary"]
RegionLabel = Literal["then", "else", "body"]
Static = Literal["taken", "pruned", "guarded", "undecided"]
Status = Literal[
    "covered", "pruned", "guarded", "uncovered", "data_dependent", "conflict"
]


class Meta(BaseModel):
  """Metadata for a staged model skeleton."""

  schema_version: int = SCHEMA_VERSION
  hf_id: str
  config_hash: str
  torch: str
  transformers: str
  source_file_hashes: dict[str, str] = Field(default_factory=dict)


# --- Templates (one per source construct) ------------------------------------
class FunctionTemplate(BaseModel):
  """A class forward (kind=forward) or a helper function (kind=function)."""

  id: str  # forward: '<Class>.forward'; function: '<qualname root>'
  kind: Literal["forward", "function"]
  cls: str = ""  # class name for forwards
  file: str  # relative to the transformers package
  line_start: int
  line_end: int
  calls: list[str] = Field(
      default_factory=list
  )  # function template ids called from here
  # Callee (function id or 'self.<attr>') -> region id it is called from ('' =
  # body).
  call_sites: dict[str, str] = Field(default_factory=dict)


class Region(BaseModel):
  """A branch destination or body region (e.g. #then, #else, #body)."""

  # '<branch id>#then' | '#else' | '#body' (branch id is 'if@<line>' etc.,
  # unique within its template).
  id: str
  label: RegionLabel
  line_start: int
  line_end: int


class Branch(BaseModel):
  """A source branch (if, loop, or ternary) within a FunctionTemplate."""

  id: str  # 'if@888', 'for@399', 'ternary@1729'
  template: str  # owning FunctionTemplate id
  kind: BranchKind
  cond_src: str
  cond_kind: CondKind
  cond_hash: str
  file: str
  line_start: int
  line_end: int
  # Enclosing region id within the same template.
  parent_region: str | None = None
  regions: list[Region] = Field(default_factory=list)


class Instance(BaseModel):
  """A statically known path: a module instance, call site, or branch region."""

  path: str
  kind: Literal["module", "function", "region"]
  # FunctionTemplate id for module/function paths; region id for region paths.
  template: str
  # Branch id for region paths.
  branch: str | None = None
  static: Static = "undecided"
  reason: str = ""


class Skeleton(BaseModel):
  """Static model graph skeleton with templates, branches, and instances."""

  meta: Meta
  functions: list[FunctionTemplate]
  branches: list[Branch]
  instances: list[Instance]


# ---------------------------------------------------------------- traces
class Specialized(BaseModel):
  """Trace specialization evidence for a guarded branch."""

  branch_path: str  # path of the branch (without the region suffix)
  kind: Literal["guard", "shape_eval", "draft_failure"]
  evidence: str


class Tensor(BaseModel):
  """Tensor metadata including name, shape, and datatype."""

  name: str
  shape: list[int]
  dtype: str


class Op(BaseModel):
  """An executed operation node within a trace."""

  id: str  # '<path>|<file>:<line>|<ordinal>'
  path: str  # full containment path, the Model Explorer namespace
  fx_name: str
  aten: str
  label: str
  file: str
  line: int
  loop_iter: int | None = None
  inputs: list[Tensor] = Field(default_factory=list)
  outputs: list[Tensor] = Field(default_factory=list)
  attrs: dict[str, str] = Field(default_factory=dict)


class Edge(BaseModel):
  """A directed edge connecting two operation nodes."""

  src: str
  src_out: int = 0
  dst: str
  dst_in: int = 0


class PathRecord(BaseModel):
  """Execution record tracking path traversal and folded iterations."""

  path: str
  ops: int = 0
  iterations: int | None = (
      None  # loop regions: how many unrolled iterations were folded
  )


class TraceInfo(BaseModel):
  """Metadata, configuration, and inputs for an execution trace."""

  id: str
  inputs: dict[str, str]
  plan: dict[str, int] = Field(default_factory=dict)
  config_overrides: dict[str, str] = Field(default_factory=dict)
  strict: bool = True
  primary: bool = False
  specialized: list[Specialized] = Field(default_factory=list)
  stats: dict[str, int] = Field(default_factory=dict)


class Trace(BaseModel):
  """Execution trace containing ops, edges, and traversed paths."""

  trace: TraceInfo
  ops: list[Op]
  edges: list[Edge]
  paths: list[PathRecord] = Field(default_factory=list)


# --- Coverage (derived) -----------------------------------------------------
class PathCoverage(BaseModel):
  """Coverage status and classification for a hierarchy path."""

  path: str
  branch: str | None = None
  status: Status
  reason: str = ""
  evidence: str = ""
  covered_by: list[str] = Field(default_factory=list)
  needs: str | None = None


class Coverage(BaseModel):
  """Aggregate coverage collection across all hierarchy paths."""

  paths: list[PathCoverage]
  summary: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------- path helpers
def region_seg(region_id: str) -> bool:
  """Determine whether the given segment represents a branch region.

  Args:
    region_id: The segment string to check.

  Returns:
    True if region_id contains '@' and '#', False otherwise.
  """
  return "@" in region_id and "#" in region_id


def join_path(*segs: str) -> str:
  """Join non-empty path segments with forward slashes.

  Args:
    *segs: Path segments to join.

  Returns:
    The joined path string.
  """
  return "/".join(s for s in segs if s)


def owner_path(path: str) -> str:
  """Extract the owning module or function path preceding branch regions.

  Args:
    path: Forward-slash delimited path string.

  Returns:
    The path without trailing region segments.
  """
  parts = path.split("/")
  while parts and region_seg(parts[-1]):
    parts.pop()
  return "/".join(parts)


# Models whose JSON schemas ship under hfgraph/schema/ for tools that read the
# pipeline's output files without importing this package.
SCHEMA_MODELS = (Skeleton, Trace, Coverage)
SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schema")


def schema_file_name(model: type[BaseModel]) -> str:
  """Returns the shipped schema file name, e.g. 'trace.schema.json'."""
  return f"{model.__name__.lower()}.schema.json"


def write_json_schemas(out_dir: str) -> None:
  """Writes the JSON schema of every model in `SCHEMA_MODELS` to `out_dir`."""
  os.makedirs(out_dir, exist_ok=True)
  for model in SCHEMA_MODELS:
    with open(
        os.path.join(out_dir, schema_file_name(model)), "w", encoding="utf-8"
    ) as f:
      json.dump(model.model_json_schema(), f, indent=1)


def main(argv: list[str]) -> None:
  """Rewrites the JSON schemas: `python -m model_explorer.hfgraph.schema [dir]`.

  Args:
    argv: Optional output directory (default: the shipped `schema/`). A relative
      directory is resolved against `BUILD_WORKING_DIRECTORY` when set, i.e.
      where `bazel run` was invoked, since it starts the binary elsewhere.
  """
  schema_dir = argv[0] if argv else SCHEMA_DIR
  invoked_from = os.environ.get("BUILD_WORKING_DIRECTORY")
  if invoked_from and not os.path.isabs(schema_dir):
    schema_dir = os.path.join(invoked_from, schema_dir)
  write_json_schemas(schema_dir)
  print("wrote schemas to", schema_dir)


if __name__ == "__main__":
  main(sys.argv[1:])
