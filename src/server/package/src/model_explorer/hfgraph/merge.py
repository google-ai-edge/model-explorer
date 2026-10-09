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

"""Stage E: loop folding, executed-path records, evidence, coverage."""

from __future__ import annotations

import collections
import math
import re
from typing import Any

from . import prune
from . import schema

GUARD_RE = re.compile(
    r"\+- ([A-Z_]+): (.*?)  # (.*?)  # (transformers/)?([^ ]+):(\d+) in (\w+)"
)
CACHE_WORDS = (
    "cache",
    "past_key_values",
    "use_precomputed_states",
    "initial_state",
)


# ---------------------------------------------------------------- loop folding
def _period(signatures: list[Any]) -> int | None:
  """Find minimum repeating period length in a sequence of op signatures."""
  n = len(signatures)
  for period in range(1, n // 2 + 1):
    if all(signatures[i] == signatures[i % period] for i in range(n)):
      return period
  return None


def fold_loops(
    skeleton: schema.Skeleton, trace: schema.Trace
) -> dict[str, int]:
  """Keep iteration 0 of every unrolled loop body.

  Args:
    skeleton: Model skeleton representing static structure.
    trace: Execution trace containing ops and edges; folded in place.

  Returns:
    Dictionary of {loop region path: iterations}.
  """
  loop_regions = {
      f"{branch.id}#body"
      for branch in skeleton.branches
      if branch.kind in ("for", "while")
  }
  loop_paths = set()
  for op in trace.ops:
    segments = op.path.split("/")
    for i, segment in enumerate(segments):
      if segment in loop_regions:
        loop_paths.add("/".join(segments[: i + 1]))
  folded: dict[str, int] = {}
  removed: set[str] = set()
  for loop_path in sorted(loop_paths, key=len):  # outer loops first
    indices = [
        i
        for i, op in enumerate(trace.ops)
        if op.id not in removed
        and (op.path == loop_path or op.path.startswith(loop_path + "/"))
    ]
    signatures = [
        (trace.ops[i].path, trace.ops[i].line, trace.ops[i].aten)
        for i in indices
    ]
    period = _period(signatures)
    if period is None or len(signatures) < 2 * period:
      continue
    n_iterations = math.ceil(len(signatures) / period)
    folded[loop_path] = n_iterations
    for position, i in enumerate(indices):
      if position < period:
        trace.ops[i].loop_iter = 0
        trace.ops[i].attrs["iterations"] = str(n_iterations)
      else:
        removed.add(trace.ops[i].id)
  if removed:
    kept = [op for op in trace.ops if op.id not in removed]
    ordinal, new_id = collections.Counter(), {}
    for op in kept:
      key = op.id.rsplit("|", 1)[0]
      new_id[op.id] = f"{key}|{ordinal[key]}"
      ordinal[key] += 1
      op.id = new_id[op.id]
    trace.ops = kept
    trace.edges = [
        edge
        for edge in trace.edges
        if edge.src in new_id and edge.dst in new_id
    ]
    for edge in trace.edges:
      edge.src, edge.dst = new_id[edge.src], new_id[edge.dst]
  trace.trace.stats["ops_after_fold"] = len(trace.ops)
  return folded


def record_paths(trace: schema.Trace, folded: dict[str, int]) -> None:
  """Record every executed path with op counts and loop iterations.

  Args:
    trace: Execution trace containing ops to record.
    folded: Mapping of loop region path to folded iteration count.
  """
  op_counts = collections.Counter(op.path for op in trace.ops)
  seen = {}
  for path in op_counts:
    segments = path.split("/")
    for i in range(1, len(segments) + 1):
      seen.setdefault("/".join(segments[:i]), 0)
  seen.update(op_counts)
  trace.paths = [
      schema.PathRecord(path=path, ops=count, iterations=folded.get(path))
      for path, count in sorted(seen.items())
  ]


# ---------------------------------------------------------------- evidence
def parse_guards(log_text: str) -> dict[str, str]:
  """Parse guard lines mentioning branch lines into a 'file:line' mapping.

  Args:
    log_text: Log string from torch compile / Dynamo guards.

  Returns:
    Dictionary mapping 'file:line' to formatted guard expressions.
  """
  guards = {}
  for match in GUARD_RE.finditer(log_text):
    kind, expr, code, _, file, line, _ = match.groups()
    if (
        code.strip().startswith(("if ", "elif ", "for ", "while "))
        or " if " in code
    ):
      guards.setdefault(f"{file}:{line}", f"{kind}: {expr[:160]}")
  return guards


def parse_draft(txt: str) -> list[tuple[str, int, str]]:
  """Parse stack trace entries from draft export output.

  Args:
    txt: Raw draft export report output.

  Returns:
    List of (file, line, func) tuples.
  """
  frames = []
  for match in re.finditer(
      r"File (\S+?transformers/)([^,]+), lineno (\d+), in (\w+)", txt or ""
  ):
    frames.append((match.group(2), int(match.group(3)), match.group(4)))
  for match in re.finditer(
      r'File "(\S+?transformers/)([^"]+)", line (\d+), in (\w+)', txt or ""
  ):
    frames.append((match.group(2), int(match.group(3)), match.group(4)))
  return frames


# ---------------------------------------------------------------- coverage
def _enclosing_region(path: str) -> str | None:
  """Find the nearest enclosing region path."""
  segments = path.split("/")[:-1]
  for i in range(len(segments), 0, -1):
    if "#" in segments[i - 1]:
      return "/".join(segments[:i])
  return None


def _guard_evidence(
    branch: schema.Branch | None, guards: dict[str, dict[str, str]]
) -> str:
  """Returns the first Dynamo guard on the branch's source line, or ''."""
  if branch is None:
    return ""
  key = f"{branch.file}:{branch.line_start}"
  return next((g[key] for g in guards.values() if key in g), "")


def _is_data_dependent(
    branch: schema.Branch | None, draft_lines: set[tuple[str, int]]
) -> bool:
  """Returns whether the condition depends on tensor values.

  True for `value` conditions and for branches containing a line that
  draft_export reported as a failure.

  Args:
    branch: The region's branch, if known.
    draft_lines: (file, line) pairs from all draft_export reports.
  """
  if branch is None:
    return False
  return branch.cond_kind == "value" or any(
      file == branch.file and branch.line_start <= line <= branch.line_end
      for file, line in draft_lines
  )


def _shape_decisions(
    branch: schema.Branch | None,
    registry: dict[str, Any],
    traces: list[schema.Trace],
    branch_path: str,
) -> dict[str, bool]:
  """Evaluates a shape condition per trace; records it as specialization.

  Args:
    branch: The region's branch, if known.
    registry: Function template registry (template id -> raw function).
    traces: All traces; their `specialized` lists are extended.
    branch_path: Path of the branch (the region path without its last segment).

  Returns:
    {trace id: condition value} for the traces where it is decidable.
  """
  if (
      branch is None
      or branch.cond_kind != "shape"
      or branch.template not in registry
  ):
    return {}
  decided: dict[str, bool] = {}
  for trace in traces:
    value = prune.evaluate(
        branch.cond_src,
        registry[branch.template],
        None,
        prune.shape_env(trace.trace.plan),
    )
    if value is None:
      continue
    decided[trace.trace.id] = value
    trace.trace.specialized.append(
        schema.Specialized(
            branch_path=branch_path,
            kind="shape_eval",
            evidence=(
                f"{branch.cond_src[:60]} -> {value} with {trace.trace.plan}"
            ),
        )
    )
  return decided


def _needs(
    branch: schema.Branch | None,
    region: schema.Instance,
    label: str,
    traces: list[schema.Trace],
) -> str:
  """Returns a description of what input would take an uncovered region."""
  cond_src = branch.cond_src if branch else ""
  region_segments = " ".join(s for s in region.path.split("/") if "#" in s)
  cond_chain = f"{cond_src} {region_segments}"
  mentions_cache = any(word in cond_chain for word in CACHE_WORDS)
  wanted = "True" if label == "then" else "False"
  if (
      branch is not None
      and branch.kind in ("for", "while")
      and not mentions_cache
  ):
    return (
        "loop body never executed in any trace (trip count:"
        f" {branch.cond_src[:60]})"
    )
  if mentions_cache and not any(t.trace.plan.get("cache") for t in traces):
    return "cache state (decode step with past key/values): run with --decode"
  if branch is not None and branch.cond_kind == "shape":
    return f"an input where ({branch.cond_src[:80]}) is {wanted}"
  if branch is not None and branch.cond_kind == "unknown":
    return "unclassified condition; inspect manually"
  return f"({(branch.cond_src if branch else '?')[:80]}) == {wanted}"


def _region_coverage(
    region: schema.Instance,
    branch: schema.Branch | None,
    executed_in: list[str],
    decided: dict[str, bool],
    taken_by: list[str],
    data_dependent: bool,
    traces: list[schema.Trace],
) -> tuple[str, str, str | None]:
  """Decides a region's (status, reason, needs) from the collected evidence.

  Args:
    region: The region instance.
    branch: Its branch, if known.
    executed_in: Ids of the traces with ops in the region.
    decided: Shape-evaluated condition value per trace id.
    taken_by: Traces where shape evaluation says the region is taken.
    data_dependent: Whether the condition depends on tensor values.
    traces: All traces.

  Returns:
    (status, reason, needs); `needs` is set for uncovered regions only.
  """
  label = region.template.split("#")[-1]
  if data_dependent:
    return "data_dependent", "condition depends on tensor values", None
  if region.static == "guarded":
    return "guarded", region.reason, None
  if region.static == "pruned":
    if executed_in:
      return (
          "conflict",
          f"statically pruned but executed in {executed_in}: {region.reason}",
          None,
      )
    return "pruned", region.reason, None
  if executed_in:
    contradicting = [
        t
        for t in executed_in
        if t in decided and decided[t] != (label == "then")
    ]
    if contradicting:
      return (
          "conflict",
          (
              f"executed in {contradicting} but the shape evaluator says this"
              " region is not taken there"
          ),
          None,
      )
    return "covered", region.reason, None
  if region.static == "taken" or taken_by:
    return "covered", (region.reason or "taken") + " (no tensor ops)", None
  return "uncovered", region.reason, _needs(branch, region, label, traces)


def coverage(
    skeleton: schema.Skeleton,
    traces: list[schema.Trace],
    registry: dict[str, Any],
    guards: dict[str, dict[str, str]] | None = None,
    drafts: dict[str, Any] | None = None,
) -> schema.Coverage:
  """Compute coverage across all traces against the skeleton.

  Args:
    skeleton: Model skeleton.
    traces: Executed trace list; guard and shape-evaluation evidence is appended
      to their `trace.specialized` lists.
    registry: Function template registry.
    guards: Parsed Dynamo guards per trace.
    drafts: Parsed draft export logs per trace.

  Returns:
    Coverage summary object.
  """
  guards = guards or {}
  drafts = drafts or {}
  templates = prune.Templates(skeleton)
  executed: dict[str, set[str]] = collections.defaultdict(set)  # path -> ids
  for trace in traces:
    for record in trace.paths:
      executed[record.path].add(trace.trace.id)
  draft_lines: set[tuple[str, int]] = set()
  for frames in drafts.values():
    for file, line, _ in frames:
      draft_lines.add((file, line))
  rows = []
  for region in skeleton.instances:
    if region.kind != "region":
      continue
    branch = templates.branch_at(region.path)
    branch_path = region.path.rsplit("/", 1)[0]
    label = region.template.split("#")[-1]
    executed_in = sorted(executed.get(region.path, ()))
    evidence = _guard_evidence(branch, guards)
    if evidence:
      for trace in traces:
        trace.trace.specialized.append(
            schema.Specialized(
                branch_path=branch_path, kind="guard", evidence=evidence
            )
        )
    decided = _shape_decisions(
        branch,
        registry,
        traces,
        branch_path,
    )
    # A region can only be taken in a trace that executed its enclosing region
    # (or its owner, at top level).
    enclosing = _enclosing_region(region.path)
    taken_by = [
        t
        for t, value in decided.items()
        if (value if label == "then" else not value)
        and (enclosing is None or t in executed.get(enclosing, ()))
    ]
    status, reason, needs = _region_coverage(
        region,
        branch,
        executed_in,
        decided,
        taken_by,
        _is_data_dependent(branch, draft_lines),
        traces,
    )
    rows.append(
        schema.PathCoverage(
            path=region.path,
            branch=region.branch,
            status=status,
            reason=reason,
            evidence=evidence or (f"shape_eval {decided}" if decided else ""),
            covered_by=executed_in or (taken_by if status == "covered" else []),
            needs=needs,
        )
    )
  return schema.Coverage(
      paths=rows, summary=dict(collections.Counter(r.status for r in rows))
  )
