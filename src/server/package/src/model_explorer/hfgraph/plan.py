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

"""Stage C: derive the list of (static) trace inputs from shape branches."""

from __future__ import annotations

import dataclasses
import re

from . import schema

MAX_TRACES = 8
DEFAULT_SEQ = 8
CHUNK = 64


@dataclasses.dataclass(frozen=True)
class PlannedTrace:
  """One static-shape trace to export.

  Attributes:
    id: Trace id; also names the trace files (`traces/<id>.json`).
    seq_len: Static sequence length of the exported input.
    decode: If positive, this many tokens are prefilled eagerly and one cached
      decode step of `seq_len` tokens is exported instead of a prefill.
  """

  id: str
  seq_len: int
  decode: int = 0


@dataclasses.dataclass(frozen=True)
class TracePlan:
  """The traces to export.

  Attributes:
    traces: Planned traces, prefill traces first.
    dropped_seq_lens: Candidate prefill lengths not traced because of the
      `MAX_TRACES` limit.
  """

  traces: list[PlannedTrace]
  dropped_seq_lens: list[int] = dataclasses.field(default_factory=list)


def plan(
    skeleton: schema.Skeleton, full_coverage: bool, decode: bool = False
) -> TracePlan:
  """Generate trace plans for model inputs.

  Args:
    skeleton: The model skeleton.
    full_coverage: Whether to generate traces for all shape boundaries.
    decode: Whether to include a cached decode trace.

  Returns:
    The planned traces and the candidate lengths dropped by `MAX_TRACES`.
  """
  base = [PlannedTrace(id=f"seq{DEFAULT_SEQ}", seq_len=DEFAULT_SEQ)]

  decode_traces = (
      [PlannedTrace(id="decode1", seq_len=1, decode=DEFAULT_SEQ)]
      if (decode or full_coverage)
      else []
  )
  if not full_coverage:
    return TracePlan(traces=base + decode_traces)
  candidates = {DEFAULT_SEQ}
  for branch in skeleton.branches:
    if branch.cond_kind != "shape" and branch.kind not in ("for", "while"):
      continue
    src = branch.cond_src
    if re.search(
        r"seq_len(?:gth)?\s*==\s*1|shape\[1\]\s*==\s*1|q_length\s*==\s*1", src
    ):
      candidates.update({1, 2})
    if "num_chunks" in src or "chunk_size" in src:
      candidates.update({CHUNK, CHUNK + 1, 2 * CHUNK + 1})
    for match in re.finditer(r"[<>]=?\s*(\d{2,})", src):
      bound = int(match.group(1))
      if bound <= 4096:
        candidates.update({bound, bound + 1})
  seqs = sorted(candidates)
  prefill = [PlannedTrace(id=f"seq{s}", seq_len=s) for s in seqs[:MAX_TRACES]]
  return TracePlan(
      traces=prefill + decode_traces, dropped_seq_lens=seqs[MAX_TRACES:]
  )
