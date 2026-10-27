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

"""Stage A: templates from the AST, plus static module instances.

Extracts class forwards, helper functions, branches, and regions from the AST.
Every source construct is stored once. Instances (paths) are generated in stage
B once the static evaluation is known.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import inspect
import logging
import os
import re
import textwrap
from typing import Any

import torch
import transformers

from . import schema

TF_ROOT = os.path.dirname(transformers.__file__)
EXCLUDE_FILES = (
    "utils/",
    "integrations/accelerate.py",
    "integrations/hub_kernels.py",
    "activations.py",
    "configuration_utils.py",
    "modeling_utils.py",
    "modeling_layers.py",
    "generation/",
)
MAX_DEPTH = 3

# --- cond_kind rules (see "Branch classification" in hfgraph/README.md)
GUARD_NAMES = frozenset({
    "is_tracing",
    "is_compiling",
    "is_torchdynamo_exporting",
    "is_torchdynamo_compiling",
    "is_jit_tracing",
})
VALUE_METHODS = frozenset({
    "item",
    "tolist",
    "any",
    "all",
    "nonzero",
    "topk",
    "max",
    "min",
    "sum",
    "argmax",
    "numel",
})
SHAPE_ATTRS = frozenset({"shape", "size", "ndim", "dim", "dtype"})
SHAPE_NAMES = frozenset({
    "seq_len",
    "seq_length",
    "q_len",
    "kv_length",
    "q_length",
    "num_chunks",
    "cache_position",
    "batch_size",
    "hidden_shape",
    "sequence_length",
    "past_seen_tokens",
    "kv_len",
    "chunk_size",
    "n_rep",
    "padding_length",
    "kv_offset",
    "q_offset",
    "local_attention_size",
    "local_size",
})
PYTHON_NAMES = frozenset({
    "return_dict",
    "output_attentions",
    "output_hidden_states",
    "labels",
    "logits_to_keep",
    "output_router_logits",
    "use_cache",
    "return_legacy_cache",
    "kwargs",
    "early_exit",
    "use_qk_l2norm_in_kernel",
    "output_final_state",
    "is_causal",
    "use_precomputed_states",
    "allow_is_causal_skip",
    "allow_is_bidirectional_skip",
    "use_gqa_in_sdpa",
    "dimensions",
    "mask_functions",
    "activation",
})
TENSOR_ARGS = frozenset({
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "input_ids",
    "cache_params",
    "cache_position",
    "position_embeddings",
    "initial_state",
    "encoder_hidden_states",
    "pixel_values",
    "padding_mask",
    "position_bias",
    "packed_sequence_mask",
    "block_sequence_ids",
    "layer_idx",
})
MODULE_CONSTS = re.compile(
    r"^_?is_[a-z0-9_]+(available|than_\d+_\d+)$|^_is_torch|^use_vmap$"
)
MODULELIST_LOOP = re.compile(
    r"self\.(layers|blocks|h|encoder_layers|decoder_layers)\b"
)


def _names(node: ast.AST) -> set[str]:
  out = set()
  for n in ast.walk(node):
    if isinstance(n, ast.Name):
      out.add(n.id)
    elif isinstance(n, ast.Attribute):
      out.add(n.attr)
  return out


def _chains(node: ast.AST) -> list[str]:
  chains = []
  for n in ast.walk(node):
    if isinstance(n, ast.Attribute):
      parts, cur = [], n
      while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
      if isinstance(cur, ast.Name):
        parts.append(cur.id)
      chains.append(".".join(reversed(parts)))
  return chains


def _calls(node: ast.AST) -> tuple[set[str], set[str]]:
  methods, builtins = set(), set()
  for n in ast.walk(node):
    if isinstance(n, ast.Call):
      f = n.func
      (methods if isinstance(f, ast.Attribute) else builtins).add(
          f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "?")
      )
  return methods, builtins


def classify(cond: ast.AST) -> str:
  """Classifies an AST condition node into a branch category."""
  names, chains = _names(cond), _chains(cond)
  methods, builtins = _calls(cond)
  if (methods | builtins) & GUARD_NAMES:
    return "guarded" if methods & VALUE_METHODS else "config"
  if methods & VALUE_METHODS:
    return "value"
  if (
      any(MODULE_CONSTS.match(n) for n in names)
      or "_attn_implementation" in names
  ):
    return "config"
  if builtins & {"hasattr", "isinstance", "callable"} or (
      builtins & {"all", "any"} and not methods
  ):
    return "python"
  if isinstance(cond, ast.Call) or (
      isinstance(cond, ast.UnaryOp) and isinstance(cond.operand, ast.Call)
  ):
    return "shape"
  if any(
      c.startswith("self.config") or c == "self.training" for c in chains
  ) and not (names & (SHAPE_NAMES | SHAPE_ATTRS)):
    return "config"
  if names & SHAPE_ATTRS or names & SHAPE_NAMES:
    return "shape"
  if names & PYTHON_NAMES:
    return "python"
  has_identity_compare = False
  for n in ast.walk(cond):
    if isinstance(n, ast.Compare):
      for c in n.ops:
        if isinstance(c, (ast.Is, ast.IsNot)):
          has_identity_compare = True
          break
      if has_identity_compare:
        break
  if has_identity_compare:
    return "shape" if names & TENSOR_ARGS else "python"
  if chains and all(c.startswith("self.") for c in chains):
    return "config"
  return "unknown"


# --- AST walk of one function
def _span(stmts: list[ast.stmt]) -> tuple[int, int]:
  """Returns the (first, last) source line of a statement list."""
  return (
      stmts[0].lineno,
      max(getattr(s, "end_lineno", s.lineno) for s in stmts),
  )


class _Collector(ast.NodeVisitor):
  """Collects branches, function calls, and regions from a template AST."""

  # pylint: disable=invalid-name

  def __init__(self, template_id: str, relfile: str):
    self.tid, self.file = template_id, relfile
    self.branches: list[schema.Branch] = []
    self.calls: set[str] = set()
    self.call_sites: dict[str, str] = {}
    self.region_stack: list[str] = []

  def _branch(
      self,
      node: ast.stmt | ast.expr,
      kind: schema.BranchKind,
      cond: ast.expr,
      regions: list[tuple[schema.RegionLabel, int, int]],
  ) -> schema.Branch:
    """Creates and records a Branch from an AST branch node."""
    bid = f"{kind}@{node.lineno}"
    src = ast.unparse(cond)[:200]
    b = schema.Branch(
        id=bid,
        template=self.tid,
        kind=kind,
        cond_src=src,
        cond_kind="loop" if kind in ("for", "while") else classify(cond),
        cond_hash=hashlib.sha1(f"{self.tid}|{src}".encode()).hexdigest()[:10],
        file=self.file,
        line_start=node.lineno,
        line_end=getattr(node, "end_lineno", node.lineno),
        parent_region=self.region_stack[-1] if self.region_stack else None,
        regions=[
            schema.Region(
                id=f"{bid}#{lab}", label=lab, line_start=s, line_end=e
            )
            for lab, s, e in regions
        ],
    )
    self.branches.append(b)
    return b

  def _visit_regions(self, pairs):
    for region, stmts in pairs:
      self.region_stack.append(region.id)
      for s in stmts:
        self.visit(s)
      self.region_stack.pop()

  def visit_If(self, n):
    """Visits an If statement and extracts then/else regions."""
    regions = [("then",) + _span(n.body)]
    if n.orelse:
      regions.append(("else",) + _span(n.orelse))
    b = self._branch(n, "if", n.test, regions)
    self.visit(n.test)
    self._visit_regions(list(zip(b.regions, [n.body, n.orelse])))

  def visit_IfExp(self, n):
    """Visits a ternary IfExp and extracts then/else regions."""
    self._branch(
        n,
        "ternary",
        n.test,
        [
            (
                "then",
                n.body.lineno,
                getattr(n.body, "end_lineno", n.body.lineno),
            ),
            (
                "else",
                n.orelse.lineno,
                getattr(n.orelse, "end_lineno", n.orelse.lineno),
            ),
        ],
    )
    self.generic_visit(n)

  def _loop(self, n, kind, it):
    """Extracts a loop branch and body region unless iterating ModuleList."""
    if MODULELIST_LOOP.search(
        ast.unparse(it)
    ):  # iterating a ModuleList is hierarchy, not a branch
      self.visit(it)
      for s in n.body + n.orelse:
        self.visit(s)
      return
    b = self._branch(n, kind, it, [("body",) + _span(n.body)])
    self.visit(it)
    self._visit_regions([(b.regions[0], n.body)])
    for s in n.orelse:
      self.visit(s)

  def visit_For(self, n):
    """Visits a For loop."""
    self._loop(n, "for", n.iter)

  def visit_While(self, n):
    """Visits a While loop."""
    self._loop(n, "while", n.test)

  def visit_Call(self, n):
    """Visits function/method call to record callees and call sites."""
    f = n.func
    site = self.region_stack[-1] if self.region_stack else ""
    if isinstance(f, ast.Attribute):
      self.calls.add(f.attr)
      if (
          isinstance(f.value, ast.Name) and f.value.id == "self"
      ):  # self.<attr>(...): a submodule or method
        self.call_sites.setdefault(f"self.{f.attr}", site)
    else:
      name = getattr(f, "id", None)
      if name:
        self.calls.add(name)
        self.call_sites.setdefault(name, site)
    self.generic_visit(n)


def _unwrap_all(fn: Any) -> Any:
  """Unwrap nested function decorators down to the underlying callable."""
  seen = set()
  while hasattr(fn, "__wrapped__") and id(fn) not in seen:
    seen.add(id(fn))
    fn = fn.__wrapped__
  return fn


def _relfile(fn: Any) -> str | None:
  try:
    f = inspect.getsourcefile(fn) or ""
  except TypeError:
    return None
  return os.path.relpath(f, TF_ROOT) if f.startswith(TF_ROOT) else None


def is_excluded(rel: str | None) -> bool:
  """Returns whether `rel` is excluded from skeleton and stack frame tracking."""
  return rel is None or any(
      rel.startswith(x) or ("/" + x) in rel for x in EXCLUDE_FILES
  )


_excluded = is_excluded


def _parse_function(
    fn: Any, tid: str
) -> tuple[_Collector, Any, int, int] | None:
  """Parse a function into an AST Collector and line ranges."""
  raw = _unwrap_all(fn)
  rel = _relfile(raw)
  if rel is None:
    return None
  src, start = inspect.getsourcelines(raw)
  tree = ast.parse(textwrap.dedent("".join(src)))
  ast.increment_lineno(tree, start - 1)
  c = _Collector(tid, rel)
  c.visit(tree)
  return c, raw, start, start + len(src) - 1


def _callee(name: str, raw_fn: Any) -> Any:
  obj = getattr(raw_fn, "__globals__", {}).get(name)
  if obj is None or not inspect.isfunction(obj):
    return None
  return None if is_excluded(_relfile(_unwrap_all(obj))) else obj


def _decorator_wrappers(fn: Any) -> list[Any]:
  out, cur, seen = [], fn, set()
  while hasattr(cur, "__wrapped__") and id(cur) not in seen:
    seen.add(id(cur))
    if (
        inspect.isfunction(cur)
        and not _excluded(_relfile(cur))
        and cur is not _unwrap_all(fn)
    ):
      out.append(cur)
    cur = cur.__wrapped__
  return out


def module_segment(fqn: str) -> str:
  """Extract the leaf segment from an FQN (e.g. 'layers.0')."""
  parts = fqn.split(".")
  if len(parts) >= 2 and parts[-1].isdigit():
    return f"{parts[-2]}.{parts[-1]}"
  return parts[-1]


def module_path(fqn: str) -> str:
  """Convert an FQN into a slash-separated path merging numeric children."""
  segs = []
  for p in fqn.split("."):
    if not p:
      continue
    if segs and p.isdigit():
      segs[-1] = f"{segs[-1]}.{p}"
    else:
      segs.append(p)
  return "/".join(segs)


# --- driver
def build_skeleton(
    model: torch.nn.Module, hf_id: str, tcfg: Any, config_hash: str
) -> tuple[schema.Skeleton, dict[str, Any]]:
  """Build a skeleton without region instances.

  Args:
    model: The PyTorch module to analyze.
    hf_id: HuggingFace model identifier.
    tcfg: Text configuration instance.
    config_hash: Hash prefix of the configuration.

  Returns:
    A tuple of (Skeleton, registry_dict).
  """
  # pylint: disable=g-import-not-at-top,broad-exception-caught
  functions: dict[str, schema.FunctionTemplate] = {}
  branches: list[schema.Branch] = []
  registry: dict[str, tuple[Any, ...]] = {}
  file_hashes: dict[str, str] = {}

  def add_function(
      fn: Any, depth: int, tid: str, cls_name: str = ""
  ) -> str | None:
    if tid in functions:
      return tid
    parsed = _parse_function(fn, tid)
    if parsed is None:
      return None
    c, raw, start, end = parsed
    registry[tid] = raw
    functions[tid] = schema.FunctionTemplate(
        id=tid,
        kind="forward" if cls_name else "function",
        cls=cls_name,
        file=c.file,
        line_start=start,
        line_end=end,
        call_sites=dict(c.call_sites),
    )
    branches.extend(c.branches)
    if c.file not in file_hashes:
      with open(os.path.join(TF_ROOT, c.file), "rb") as f:
        file_hashes[c.file] = hashlib.sha1(f.read()).hexdigest()[:12]
    if depth < MAX_DEPTH:
      for name in sorted(n for n in c.calls if n):
        callee = _callee(name, raw)
        if callee is not None:
          cid = add_function(callee, depth + 1, name)
          if cid and cid not in functions[tid].calls:
            functions[tid].calls.append(cid)
            if name in c.call_sites:
              functions[tid].call_sites[cid] = c.call_sites[name]
    return tid

  # class forwards, once per class, plus decorator wrappers
  instances: list[schema.Instance] = []
  seen_cls = set()
  for fqn, mod in model.named_modules():
    cls = type(mod)
    fwd = getattr(cls, "forward", None)
    tid = f"{cls.__name__}.forward"
    if cls not in seen_cls:
      seen_cls.add(cls)
      if fwd and not _excluded(_relfile(_unwrap_all(fwd))):
        add_function(fwd, 0, tid, cls.__name__)
        for w in _decorator_wrappers(fwd):
          wid = w.__qualname__.split(".<locals>")[0].split(".")[-1]
          if add_function(w, 1, wid):
            if tid in functions and wid not in functions[tid].calls:
              functions[tid].calls.append(wid)
    if tid in functions and fqn:
      instances.append(
          schema.Instance(path=module_path(fqn), kind="module", template=tid)
      )

  # attention / mask / experts interfaces resolved from config
  from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
  from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

  impl = getattr(tcfg, "_attn_implementation", None) or "sdpa"
  iface = {}
  for key, table in (
      ("attention_interface", ALL_ATTENTION_FUNCTIONS),
      ("mask_interface", ALL_MASK_ATTENTION_FUNCTIONS),
  ):
    if impl in table:
      fn = table[impl]
      fid = add_function(fn, 1, _unwrap_all(fn).__name__)
      if fid:
        iface[key] = fid
  if getattr(tcfg, "num_experts", None):
    try:
      from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

      fn = ALL_EXPERTS_FUNCTIONS["batched_mm"]
      fid = add_function(fn, 1, _unwrap_all(fn).__name__)
      for f in functions.values():
        if (
            f.kind == "forward"
            and f.cls.endswith("Experts")
            and fid
            and fid not in f.calls
        ):
          f.calls.append(fid)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("experts interface not resolved: %s", e)
  for f in list(functions.values()):
    src = inspect.getsource(registry[f.id])
    for key, fid in iface.items():
      if f"{key}(" in src and fid not in f.calls:
        f.calls.append(fid)
        f.call_sites[fid] = f.call_sites.get(key, "")

  meta = schema.Meta(
      hf_id=hf_id,
      config_hash=config_hash,
      torch=torch.__version__,
      transformers=transformers.__version__,
      source_file_hashes=file_hashes,
  )
  sk = schema.Skeleton(
      meta=meta,
      functions=list(functions.values()),
      branches=branches,
      instances=instances,
  )
  return sk, registry


def summarize(sk: schema.Skeleton) -> str:
  """Summarize template and branch counts across the skeleton.

  Args:
    sk: The static model skeleton.

  Returns:
    A human-readable summary string of skeleton metrics.
  """
  kinds = collections.Counter(b.cond_kind for b in sk.branches)
  nfwd = sum(1 for f in sk.functions if f.kind == "forward")
  real = sum(v for k, v in kinds.items() if k != "loop")
  unk = kinds.get("unknown", 0)
  return (
      f"templates: {nfwd} forwards, {len(sk.functions) - nfwd} functions;"
      f" branches: {len(sk.branches)} ({dict(kinds)}); unknown ratio"
      f" {unk / max(1, real):.1%}; module instances:"
      f" {sum(1 for i in sk.instances if i.kind == 'module')}"
  )
