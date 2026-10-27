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

"""The model-explorer-hfgraph pipeline: stages A to F for one model.

Requires the `hfgraph` extra (torch, transformers, pydantic); `cli.py` checks
for it before importing this module. Progress lines are printed per stage
because this runs as a command-line tool.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from typing import Any

from . import emit
from . import merge
from . import models
from . import plan as plan_lib
from . import prune
from . import schema
from . import skeleton as skeleton_lib
from . import trace as trace_lib


@dataclasses.dataclass(frozen=True)
class Options:
  """Command-line options of `model-explorer-hfgraph`.

  Attributes:
    model: Hugging Face model id or local checkpoint directory.
    out: Output directory.
    full_coverage: Trace every static prefill length the shape conditions
      suggest (implies `decode`).
    tiny: Use a shrunken config with random weights.
    primary: Trace whose shapes the merged graph shows (default: the first).
    draft: Also run `torch.export.draft_export`.
    skip_trace: Reuse trace files that already exist in `out`.
    decode: Add a cached one-token decode trace.
  """

  model: str
  out: str
  full_coverage: bool = False
  tiny: bool = False
  primary: str | None = None
  draft: bool = False
  skip_trace: bool = False
  decode: bool = False


@dataclasses.dataclass
class _TraceEvidence:
  """Traces plus the per-trace evidence parsed from their side files."""

  traces: list[schema.Trace] = dataclasses.field(default_factory=list)
  # Trace id -> {"file:line": guard expression}.
  guards: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
  # Trace id -> [(file, line, function)] frames from the draft_export report.
  drafts: dict[str, list[tuple[str, int, str]]] = dataclasses.field(
      default_factory=dict
  )


def _with_suffix(path: str, suffix: str) -> str:
  """Replaces the extension of `path` (only the last one) with `suffix`."""
  return os.path.splitext(path)[0] + suffix


def _trace_path(out_dir: str, trace_id: str) -> str:
  return os.path.join(out_dir, "traces", f"{trace_id}.json")


def run_trace(
    hf_id: str,
    out_dir: str,
    planned: plan_lib.PlannedTrace,
    tiny: bool,
    draft: bool,
) -> tuple[str, str]:
  """Run the trace subprocess for a planned trace configuration.

  Args:
    hf_id: Hugging Face model identifier.
    out_dir: Target output directory.
    planned: The trace to export.
    tiny: Whether to execute with tiny mock tensors.
    draft: Whether to enable draft export.

  Returns:
    A tuple of (trace_json_path, guards_log_path).

  Raises:
    RuntimeError: If the trace subprocess execution fails.
  """
  trace_path = _trace_path(out_dir, planned.id)
  guards_log = _with_suffix(trace_path, ".guards.log")
  os.makedirs(os.path.dirname(trace_path), exist_ok=True)
  cmd = [
      sys.executable,
      "-m",
      "model_explorer.hfgraph.trace",
      hf_id,
      os.path.join(out_dir, "skeleton.json"),
      trace_path,
      "--seq",
      str(planned.seq_len),
      "--trace-id",
      planned.id,
  ]
  if tiny:
    cmd.append("--tiny")
  if planned.decode:
    cmd += ["--decode", str(planned.decode)]
  if draft:
    cmd.append("--draft")
  env = dict(os.environ, TORCH_LOGS="guards")
  with open(guards_log, "w", encoding="utf-8") as log_file:
    completed = subprocess.run(
        cmd, env=env, stderr=log_file, stdout=subprocess.DEVNULL, check=False
    )
  if completed.returncode != 0:
    with open(guards_log, encoding="utf-8") as log_file:
      tail = log_file.read().splitlines()[-15:]
    raise RuntimeError(f"trace {planned.id} failed:\n" + "\n".join(tail))
  return trace_path, guards_log


def _write_text(path: str, text: str) -> None:
  with open(path, "w", encoding="utf-8") as f:
    f.write(text)


def _read_text(path: str) -> str | None:
  """Returns the file content, or None if the file does not exist."""
  if not os.path.exists(path):
    return None
  with open(path, encoding="utf-8") as f:
    return f.read()


# ---------------------------------------------------------------- stages
def _build_skeleton(options: Options) -> tuple[schema.Skeleton, dict[str, Any]]:
  """Stages A + B on a meta-device model (no weights needed)."""
  model, text_config = models.build_text_model(
      options.model, options.tiny, "meta"
  )
  skeleton, registry = skeleton_lib.build_skeleton(
      model, options.model, text_config, models.config_hash(text_config)
  )
  skeleton = prune.expand_instances(skeleton, registry, model)
  _write_text(
      os.path.join(options.out, "skeleton.json"),
      skeleton.model_dump_json(indent=1),
  )
  n_regions = sum(1 for i in skeleton.instances if i.kind == "region")
  n_pruned = sum(1 for i in skeleton.instances if i.static == "pruned")
  print(
      f"[A/B] {skeleton_lib.summarize(skeleton)}; region paths: {n_regions}"
      f" (pruned {n_pruned})"
  )
  return skeleton, registry


def _plan_traces(
    skeleton: schema.Skeleton, options: Options
) -> list[plan_lib.PlannedTrace]:
  """Stage C."""
  trace_plan = plan_lib.plan(skeleton, options.full_coverage, options.decode)
  dropped = (
      f" (dropped {trace_plan.dropped_seq_lens})"
      if trace_plan.dropped_seq_lens
      else ""
  )
  print(f"[C] trace plan: {[t.id for t in trace_plan.traces]}{dropped}")
  return trace_plan.traces


def _trace_all(
    planned_traces: list[plan_lib.PlannedTrace], options: Options
) -> _TraceEvidence:
  """Stage D: exports (or reuses) every planned trace and parses evidence."""
  evidence = _TraceEvidence()
  for planned in planned_traces:
    trace_path = _trace_path(options.out, planned.id)
    guards_log = _with_suffix(trace_path, ".guards.log")
    if not (options.skip_trace and os.path.exists(trace_path)):
      started = time.time()
      trace_path, guards_log = run_trace(
          options.model, options.out, planned, options.tiny, options.draft
      )
      print(f"[D] {planned.id} done in {time.time() - started:.1f}s")
    with open(trace_path, encoding="utf-8") as f:
      trace = schema.Trace.model_validate_json(f.read())
    trace_id = trace.trace.id
    evidence.traces.append(trace)
    guards_text = _read_text(guards_log)
    evidence.guards[trace_id] = (
        merge.parse_guards(guards_text) if guards_text else {}
    )
    draft_text = _read_text(trace_lib.draft_path(trace_path))
    evidence.drafts[trace_id] = (
        merge.parse_draft(draft_text) if draft_text else []
    )
    max_diff = trace.trace.stats.get("max_abs_diff_e9", 0) / 1e9
    print(
        f"[D] {trace_id}: ops={len(trace.ops)}"
        f" guards-on-branches={len(evidence.guards[trace_id])}"
        f" max|diff|={max_diff:.1e}"
    )
  return evidence


def _fold_all(
    skeleton: schema.Skeleton, traces: list[schema.Trace]
) -> dict[str, int]:
  """Stage E, part 1: folds loops; returns {loop path: max iterations}."""
  iterations: dict[str, int] = {}
  for trace in traces:
    trace.trace.specialized = []
    folded = merge.fold_loops(skeleton, trace)
    for loop_path, count in folded.items():
      iterations[loop_path] = max(iterations.get(loop_path, 0), count)
    merge.record_paths(trace, folded)
    print(
        f"[E] {trace.trace.id}: ops {trace.trace.stats['ops']} →"
        f" {len(trace.ops)} after folding {len(folded)} loops; paths:"
        f" {len(trace.paths)}"
    )
  return iterations


def _merge(
    skeleton: schema.Skeleton,
    registry: dict[str, Any],
    evidence: _TraceEvidence,
    options: Options,
) -> tuple[schema.Coverage, dict[str, int], str]:
  """Stage E: loop folding and coverage; writes traces and coverage.json.

  Args:
    skeleton: Model skeleton.
    registry: Function template registry from `build_skeleton`.
    evidence: Traces and their parsed guard / draft evidence.
    options: Command-line options.

  Returns:
    (coverage report, {loop path: iterations}, primary trace id).
  """
  iterations = _fold_all(skeleton, evidence.traces)
  primary = options.primary or evidence.traces[0].trace.id
  for trace in evidence.traces:
    trace.trace.primary = trace.trace.id == primary
  coverage_report = merge.coverage(
      skeleton, evidence.traces, registry, evidence.guards, evidence.drafts
  )
  for trace in evidence.traces:
    _write_text(
        _trace_path(options.out, trace.trace.id), trace.model_dump_json()
    )
  _write_text(
      os.path.join(options.out, "coverage.json"),
      coverage_report.model_dump_json(indent=1),
  )
  print(f"[E] coverage: {coverage_report.summary}")
  return coverage_report, iterations, primary


def _emit(
    skeleton: schema.Skeleton,
    coverage_report: schema.Coverage,
    iterations: dict[str, int],
    traces: list[schema.Trace],
    primary: str,
    options: Options,
) -> None:
  """Stage F: writes model_explorer.json and prints the self-check."""
  graph_collection = emit.Emitter(skeleton, coverage_report, iterations).emit(
      traces, primary
  )
  out_path = os.path.join(options.out, "model_explorer.json")
  _write_text(out_path, json.dumps(graph_collection))
  problems = emit.check(graph_collection, coverage_report)
  merged_graph = graph_collection["graphs"][0]
  n_placeholders = sum(
      1 for node in merged_graph["nodes"] if node["id"].startswith("ph:")
  )
  print(
      f"[F] {out_path}: graphs={len(graph_collection['graphs'])}"
      f" merged nodes={len(merged_graph['nodes'])}"
      f" groups={len(merged_graph['groupNodeAttributes'])}"
      f" placeholders={n_placeholders}"
      f"  check: {len(problems)} problems"
  )
  for problem in problems[:10]:
    print("   -", problem)


def coverage_markdown(hf_id: str, coverage_report: schema.Coverage) -> str:
  """Renders the coverage report as the `coverage.md` table."""
  lines = [
      f"# coverage {hf_id}",
      "",
      json.dumps(coverage_report.summary),
      "",
      "<!-- mdformat off(preserve GFM table layout) -->",
      "| path | status | traces | needs |",
      "|---|---|---|---|",
  ]
  for row in coverage_report.paths:
    lines.append(
        f"| `{row.path}` | {row.status} | {','.join(row.covered_by)} |"
        f" {row.needs or ''} |"
    )
  lines.append("<!-- mdformat on -->")
  return "\n".join(lines) + "\n"


def run(options: Options) -> None:
  """Runs the whole pipeline and writes the output directory.

  Args:
    options: Command-line options.

  Raises:
    RuntimeError: If a trace subprocess fails.
  """
  os.makedirs(options.out, exist_ok=True)
  started = time.time()
  skeleton, registry = _build_skeleton(options)
  planned_traces = _plan_traces(skeleton, options)
  evidence = _trace_all(planned_traces, options)
  coverage_report, iterations, primary = _merge(
      skeleton, registry, evidence, options
  )
  _emit(
      skeleton,
      coverage_report,
      iterations,
      evidence.traces,
      primary,
      options,
  )
  _write_text(
      os.path.join(options.out, "coverage.md"),
      coverage_markdown(options.model, coverage_report),
  )
  print(f"done in {time.time() - started:.1f}s")
