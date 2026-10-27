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

"""Stage B: static evaluation of conditions and instance path expansion.

After this stage `skeleton.instances` holds every statically known path: module
instances (placed under the region that calls them), helper-function call
sites, and the regions of every branch under every instance, each with its
static decision (taken / pruned / guarded / undecided).
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch

from . import plan as plan_lib
from . import schema
from . import skeleton as skeleton_lib

EXPORT_CONSTS = {
    "is_tracing": lambda *a, **k: True,
    "is_torchdynamo_compiling": lambda *a, **k: True,
    "is_torchdynamo_exporting": lambda *a, **k: True,
    "is_jit_tracing": lambda *a, **k: False,
    "_is_torch_available": True,
}
CALL_DEFAULTS = {  # values of forward kwargs under a plain-forward export
    "past_key_values": None,
    "cache_params": None,
    "use_cache": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "labels": None,
    "return_dict": True,
    "logits_to_keep": 0,
    "position_ids": None,
    "inputs_embeds": None,
    "encoder_hidden_states": None,
    "pixel_values": None,
    "initial_state": None,
    "layer_idx": None,
    "or_mask_function": None,
    "and_mask_function": None,
    "packed_sequence_mask": None,
    "block_sequence_ids": None,
    "position_bias": None,
    "early_exit": False,
    "output_final_state": False,
    "use_qk_l2norm_in_kernel": True,
    "use_precomputed_states": False,
    "mask_functions": (lambda: None, lambda: None),
    "dimensions": (0, 1, 2, 3),
    "kwargs": {},
}
CACHE_KEYS = frozenset({
    "cache_params",
    "past_key_values",
    "cache_position",
    "use_precomputed_states",
    "initial_state",
})


def _params(fn: Any) -> list[inspect.Parameter]:
  """Extract parameter signatures from a callable."""
  try:
    return list(inspect.signature(fn).parameters.values())
  except (TypeError, ValueError):
    return []


def evaluate(
    cond_src: str,
    raw_fn: Any,
    self_obj: Any = None,
    extra: dict[str, Any] | None = None,
    is_root: bool = False,
) -> bool | None:
  """Evaluate condition string. Return None if not statically decidable.

  Args:
    cond_src: Python condition source expression.
    raw_fn: Function or method reference providing globals.
    self_obj: Bound module instance if evaluating a method.
    extra: Additional local variable bindings.
    is_root: Whether evaluation is at the root model module.

  Returns:
    Boolean result if decidable, or None if dynamic / undecidable.
  """
  g = dict(getattr(raw_fn, "__globals__", {}))
  g.update(EXPORT_CONSTS)
  g["math"] = math
  g["torch"] = torch
  env = {}
  if (
      is_root and getattr(raw_fn, "__name__", "") == "forward"
  ):  # only the ROOT forward's kwargs are under our control
    for p in _params(raw_fn):
      if p.default is not inspect.Parameter.empty:
        env[p.name] = p.default
  env.update(CALL_DEFAULTS)
  if self_obj is not None:
    env["self"] = self_obj
    env["config"] = getattr(self_obj, "config", None)
  if extra:
    none_defaults = {p.name for p in _params(raw_fn) if p.default is None}
    force = CACHE_KEYS if extra.get("cache_params") is not None else set()
    env.update(
        {k: v for k, v in extra.items() if k not in none_defaults or k in force}
    )
  # `cond_src` is a condition from the installed transformers source that
  # `build_skeleton` parsed in this process (never read back from an output
  # file), evaluated with the globals of the function it appears in; this runs
  # the same code that tracing the model runs anyway. That holds only because
  # `models.py` loads every config and model with `trust_remote_code=False`, so
  # no checkpoint-supplied modeling code is ever parsed or evaluated here.
  # `ast.literal_eval` cannot express these conditions (`self.config.x`,
  # `hasattr(...)`, `len(...)`).
  # Any exception (NameError for runtime-only locals, AttributeError,
  # TypeError, torch errors on meta tensors, SyntaxError for a truncated
  # condition, ...) means "not statically decidable".
  try:
    # pylint: disable=eval-used,broad-exception-caught
    val = eval(compile(cond_src, "<cond>", "eval"), g, env)
  except Exception:
    return None
  if isinstance(val, bool):
    return val
  if isinstance(val, torch.Tensor):
    return None
  if val is None or isinstance(val, (int, float, str, tuple, list, dict)):
    return bool(val)
  return None


def shape_env(plan: dict[str, Any], hidden: int = 1) -> dict[str, Any]:
  """Concrete-input environment for shape conditions of one trace."""
  s = int(plan.get("seq_len", 8))
  cached = int(plan.get("cache", 0))
  env = {
      "seq_len": s,
      "seq_length": s,
      "sequence_length": s,
      "q_len": s,
      "q_length": s,
      "kv_length": s,
      "kv_len": s,
      "past_seen_tokens": 0,
      "q_offset": 0,
      "kv_offset": 0,
      "batch_size": 1,
      "chunk_size": plan_lib.CHUNK,
      "num_chunks": math.ceil(s / plan_lib.CHUNK),
      "input_ids": torch.zeros(1, s, dtype=torch.long),
      "attention_mask": torch.ones(1, s, dtype=torch.long),
      "inputs_embeds": torch.zeros(1, s, hidden),
      "cache_position": torch.arange(s),
      "padding_mask": None,
      "local_attention_size": None,
      "local_size": None,
      "n_rep": 2,
      "hidden_states": torch.zeros(1, s, hidden),
      "key": torch.zeros(1, 2, s, 64),
      "value": torch.zeros(1, 2, s, 64),
      "query": torch.zeros(1, 4, s, 64),
  }
  if cached:
    env.update({
        "past_seen_tokens": cached,
        "kv_length": s + cached,
        "kv_len": s + cached,
        "use_precomputed_states": True,
        "cache_params": object(),
        "past_key_values": object(),
        "cache_position": torch.arange(cached, cached + s),
        "initial_state": torch.zeros(1),
    })
  return env


# --- Region chains -----------------------------------------------------------
class Templates:
  """Lookup helpers over a skeleton's templates."""

  def __init__(self, skeleton: schema.Skeleton):
    self.functions = {f.id: f for f in skeleton.functions}
    self.branches: dict[tuple[str, str], schema.Branch] = {
        (branch.template, branch.id): branch for branch in skeleton.branches
    }
    self._branch_by_id: dict[str, schema.Branch] = {}
    self.by_template: dict[str, list[schema.Branch]] = {}
    for branch in skeleton.branches:
      self._branch_by_id.setdefault(branch.id, branch)
      self.by_template.setdefault(branch.template, []).append(branch)
    # Module instance path -> forward template id (first instance wins).
    self.module_template: dict[str, str] = {}
    for instance in skeleton.instances:
      if instance.kind == "module":
        self.module_template.setdefault(instance.path, instance.template)

  def owner_template(self, path: str) -> str | None:
    """Returns the template that owns the trailing region segments of `path`.

    The owner is the last non-region segment: a helper function (its segment
    is the template id) or a module instance.

    Args:
      path: An instance path, e.g. 'layers.0/if@12#else/for@17#body'.

    Returns:
      The owning template id, or None if it is not known (e.g. regions of the
      root module's forward, which has no module instance path).
    """
    owner = schema.owner_path(path)
    if not owner:
      return None
    owner_segment = owner.rsplit("/", 1)[-1]
    if owner_segment in self.functions:
      return owner_segment
    return self.module_template.get(owner)

  def branch_at(self, region_path: str) -> schema.Branch | None:
    """Returns the Branch whose region is the last segment of `region_path`."""
    branch_id = region_path.rsplit("/", 1)[-1].split("#")[0]
    template = self.owner_template(region_path)
    if template is None:
      return self._branch_by_id.get(branch_id)
    return self.branches.get((template, branch_id))

  def region_chain(self, template: str, region_id: str | None) -> list[str]:
    """Returns outer -> inner region chain for a region id inside template."""
    chain = []
    while region_id:
      chain.append(region_id)
      bid = region_id.split("#")[0]
      branch = self.branches.get((template, bid))
      region_id = branch.parent_region if branch else None
    return list(reversed(chain))

  def regions_at(self, template: str, line: int) -> list[str]:
    """Return region chain (outer → inner) containing a source line."""
    best, best_span = None, None
    for branch in self.by_template.get(template, ()):
      for region in branch.regions:
        if region.line_start <= line <= region.line_end:
          span = region.line_end - region.line_start
          if best is None or span < best_span:
            best, best_span = region.id, span
    return self.region_chain(template, best) if best else []


# --- Instance expansion -----------------------------------------------------
def expand_instances(
    skeleton: schema.Skeleton,
    registry: dict[str, Any],
    model: torch.nn.Module,
) -> schema.Skeleton:
  """Expand static instance paths and evaluate static branch decisions.

  Args:
    skeleton: The static model skeleton.
    registry: Mapping from template ID to raw callable.
    model: Instantiated PyTorch module.

  Returns:
    The skeleton updated with concrete Instance records.
  """
  templates = Templates(skeleton)
  mods = {fqn: m for fqn, m in model.named_modules()}
  out: list[schema.Instance] = []

  def add_regions(
      template: str, base: str, self_obj: Any, is_root: bool
  ) -> None:
    for branch in templates.by_template.get(template, ()):
      chain = templates.region_chain(template, branch.parent_region)
      prefix = schema.join_path(base, *chain)
      raw = registry.get(template)
      val = None
      if (
          branch.cond_kind in ("config", "python", "guarded")
          and raw is not None
      ):
        val = evaluate(branch.cond_src, raw, self_obj, None, is_root)
      for region in branch.regions:
        static, reason = "undecided", ""
        if branch.cond_kind == "guarded":
          static, reason = (
              "guarded",
              (
                  "value-dependent, forced by is_tracing()/is_compiling() under"
                  " export"
              ),
          )
        elif val is not None:
          taken = (
              val if region.label == "then" else (not val)
          ) or region.label == "body"
          static = "taken" if taken else "pruned"
          reason = (
              f"static {branch.cond_kind}: ({branch.cond_src[:80]}) == {val}"
          )
        out.append(
            schema.Instance(
                path=schema.join_path(prefix, region.id),
                kind="region",
                template=region.id,
                branch=branch.id,
                static=static,
                reason=reason,
            )
        )

  def add_function_calls(
      template: str, base: str, depth: int, seen: frozenset[str]
  ) -> None:
    f = templates.functions[template]
    for cid in f.calls:
      if (
          cid not in templates.functions
          or templates.functions[cid].kind != "function"
          or cid in seen
          or depth > skeleton_lib.MAX_DEPTH
      ):
        continue
      site = f.call_sites.get(cid, "")
      chain = templates.region_chain(template, site or None)
      path = schema.join_path(base, *chain, cid)
      out.append(schema.Instance(path=path, kind="function", template=cid))
      add_regions(cid, path, None, False)
      add_function_calls(cid, path, depth + 1, seen | {cid})

  # Module instances, placed under the region of the parent forward that
  # calls them.
  path_of_fqn: dict[str, str] = {"": ""}
  for fqn, m in mods.items():
    if not fqn:
      continue
    parent_fqn = fqn.rsplit(".", 1)[0] if "." in fqn else ""
    parent_path = path_of_fqn.get(parent_fqn, "")
    attr = fqn.rsplit(".", 1)[-1]
    chain: list[str] = []
    ptid = (
        f"{type(mods[parent_fqn]).__name__}.forward"
        if parent_fqn in mods
        else None
    )
    if ptid in templates.functions and not attr.isdigit():
      site = templates.functions[ptid].call_sites.get(f"self.{attr}", "")
      chain = templates.region_chain(ptid, site or None)
    seg = skeleton_lib.module_segment(fqn)
    if attr.isdigit():  # ModuleList child: merge into parent's segment
      path = schema.join_path(
          parent_path.rsplit("/", 1)[0] if "/" in parent_path else "", seg
      )
    else:
      path = schema.join_path(parent_path, *chain, seg)
    path_of_fqn[fqn] = path
    tid = f"{type(m).__name__}.forward"
    if tid in templates.functions:
      out.append(schema.Instance(path=path, kind="module", template=tid))
  # Regions and helper calls under every module instance (root included,
  # base '').
  for fqn, m in mods.items():
    tid = f"{type(m).__name__}.forward"
    if tid not in templates.functions:
      continue
    base = path_of_fqn[fqn]
    add_regions(tid, base, m, is_root=(not fqn))
    add_function_calls(tid, base, 0, frozenset())
  skeleton.instances = out
  return skeleton
