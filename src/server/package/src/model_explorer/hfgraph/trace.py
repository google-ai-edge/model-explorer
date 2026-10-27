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

"""Stage D: static-shape strict torch.export to ops with containment paths.

Run as a subprocess (the CLI sets TORCH_LOGS=guards and captures stderr as the
guard log):
    python -m model_explorer.hfgraph.trace <hf_id> <skeleton.json> <out.json>
    --seq 8 [--tiny] [--decode N] [--draft]
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import operator
import os
import re
import sys
import time
from typing import Any
import warnings

import torch

from . import models
from . import prune
from . import schema
from . import skeleton as skeleton_lib

NOISE = frozenset({
    "sym_size",
    "_assert_tensor_metadata",
    "_assert_scalar",
    "_check",
    "_assert_async",
    "sym_constrain_range",
    "sym_constrain_range_for_size",
    "_local_scalar_dense",
    "sym_sum",
    "sym_max",
    "sym_min",
    "detach",
})
COLLAPSE = frozenset(
    {"Linear", "Embedding", "Conv1d", "SiLUActivation", "SiLU", "GELU"}
)
FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')


# --- export stand-ins (dense equivalents for data-dependent code)
def _qsa_indexer_dense_forward(
    self: Any,
    hidden_states: Any,
    position_embeddings: Any,
    attention_mask: torch.Tensor,
    past_key_values: Any = None,
) -> torch.Tensor:
  """Qwen sparse attention indexer stand-in.

  The real code picks top-k key blocks per query with nonzero()/topk
  (data-dependent, not exportable). This stand-in selects every visible token,
  i.e. dense attention with the same mask shape.

  Args:
    self: Indexer instance.
    hidden_states: Input tensor of hidden states.
    position_embeddings: Position embeddings tensor.
    attention_mask: Mask tensor indicating token visibility.
    past_key_values: Optional cached key/values.

  Returns:
    Dense attention mask with identical shape and semantics.
  """
  del self, hidden_states, position_embeddings, past_key_values
  visible = (
      attention_mask
      if attention_mask.dtype == torch.bool
      else attention_mask == 0
  )
  if attention_mask.is_floating_point():
    return torch.where(
        visible,
        attention_mask.new_zeros(()),
        torch.finfo(attention_mask.dtype).min,
    )
  return visible


STAND_INS = {"Qwen4ExpTextQSAIndexer": _qsa_indexer_dense_forward}


def _apply_stand_ins(lm: torch.nn.Module) -> list[str]:
  """Apply dense stand-in forward methods for unexportable dynamic kernels."""
  applied = []
  for _, mod in lm.named_modules():
    fn = STAND_INS.get(type(mod).__name__)
    if fn is not None and type(mod).__name__ not in applied:
      type(mod).forward = fn
      applied.append(type(mod).__name__)
  return applied


class DecodeStep(torch.nn.Module):
  """One decode step over a cache pre-filled by an eager prefill.

  Exported instead of the model for the `--decode` trace: the prefill runs
  eagerly, so the exported graph is a single cached step and covers the
  decode-only branches. The cache tensors are lifted as constants, so the graph
  is meant for coverage, not inference (see "Models & limits" in
  hfgraph/README.md).
  """

  def __init__(self, model: torch.nn.Module, cache: Any):
    super().__init__()
    self.model = model
    self.cache = cache

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.model(
        input_ids=input_ids, past_key_values=self.cache, use_cache=True
    ).logits


# ---------------------------------------------------------------- fx helpers
def _op_name(t: Any) -> str:
  """Return normalized operation name string from target."""
  # pylint: disable=protected-access
  if t is operator.getitem:
    return "getitem"
  if isinstance(t, torch._ops.OpOverload):
    return "aten." + t.overloadpacket.__name__
  if isinstance(t, torch._ops.HigherOrderOperator):
    return "hop." + t.__name__
  return getattr(t, "__name__", str(t))


def _tinfo(name: str, v: Any) -> schema.Tensor | None:
  """Extract Tensor metadata from a PyTorch tensor object."""
  if isinstance(v, torch.Tensor):
    return schema.Tensor(
        name=name,
        shape=[int(d) if isinstance(d, int) else -1 for d in v.shape],
        dtype=str(v.dtype).replace("torch.", ""),
    )
  return None


def _outs_of(name: str, v: Any) -> list[schema.Tensor | None]:
  """Extract Tensor output metadata from a node value."""
  if isinstance(v, (tuple, list)):
    return [_tinfo(f"{name}:{i}", x) for i, x in enumerate(v)]
  return [_tinfo(f"{name}:0", v)]


def _frames_of(st: str | None) -> list[tuple[str, int, str, bool]]:
  """Extract model stack frames outer → inner.

  Args:
    st: Raw stack trace string, or None.

  Returns:
    A list of (relfile, line, func, excluded) tuples.
  """
  out = []
  for p, line, fn in FRAME_RE.findall(st or ""):
    if (
        "/torch/" in p
        or "/torch_" in p
        or not p.startswith(skeleton_lib.TF_ROOT)
    ):
      continue
    rel = os.path.relpath(p, skeleton_lib.TF_ROOT)
    out.append((rel, int(line), fn, skeleton_lib.is_excluded(rel)))
  return out


# --- path computation
class PathBuilder:
  """Builds hierarchical region and module namespaces from stack frames."""

  def __init__(self, sk: schema.Skeleton, strip_prefix: str = ""):
    self.templates = prune.Templates(sk)
    self.strip = strip_prefix
    self.by_file: dict[str, list[schema.FunctionTemplate]] = (
        collections.defaultdict(list)
    )
    for f in sk.functions:
      self.by_file[f.file].append(f)

  def template_at(self, file: str, line: int) -> schema.FunctionTemplate | None:
    """Find the most specific function template containing a source line."""
    best = None
    for f in self.by_file.get(file, ()):
      if f.line_start <= line <= f.line_end and (
          best is None
          or f.line_end - f.line_start < best.line_end - best.line_start
      ):
        best = f
    return best

  def modules_of(self, stack: Any) -> list[tuple[str, str]]:
    """Extract [(fqn, cls)] outer → inner with the decode wrapper stripped."""
    mods = []
    for fqn, cls in (stack or {}).values():
      if fqn.startswith("_empty_nn_module_stack"):
        continue
      cls = (
          cls
          if isinstance(cls, str)
          else f"{cls.__module__}.{cls.__qualname__}"
      )
      cls = cls.rsplit(".", 1)[-1]
      if self.strip:
        if fqn == self.strip.rstrip("."):
          fqn = ""
        elif fqn.startswith(self.strip):
          fqn = fqn[len(self.strip) :]
        elif not fqn:
          continue  # the wrapper itself
      if mods and mods[-1][0] == fqn:
        continue
      mods.append((fqn, cls))
    return mods

  def path(
      self, frames: list[tuple[str, int, str, bool]], stack: Any
  ) -> tuple[str, str]:
    """Compute (path, innermost module fqn) from frames and module stack."""
    mods = self.modules_of(stack)
    segs: list[str] = []
    j = 0
    fqn_inner = mods[-1][0] if mods else ""
    for file, line, fn, excluded in frames:
      t = self.template_at(file, line) if not excluded else None
      if fn == "forward":
        cls = t.cls if t and t.kind == "forward" else None
        k = j
        while k < len(mods) and (cls is None or mods[k][1] != cls):
          if cls is None:
            break
          k += 1
        if k < len(mods) and (cls is None or mods[k][1] == cls):
          for m in mods[
              j : k + 1
          ]:  # modules skipped over (torch built-ins) still get their segment
            if m[0]:
              segs.append(skeleton_lib.module_segment(m[0]))
          j = k + 1
        if t:
          segs.extend(self.templates.regions_at(t.id, line))
        continue
      if excluded:
        continue
      if t and t.kind == "function":
        segs.append(t.id)
        segs.extend(self.templates.regions_at(t.id, line))
      elif (
          t and t.kind == "forward"
      ):  # a method of the module class other than forward (e.g. _norm)
        segs.append(fn)
        segs.extend(self.templates.regions_at(t.id, line))
      else:
        segs.append(fn)
    for m in mods[
        j:
    ]:  # innermost modules whose forward frames were not recorded
      if m[0]:
        segs.append(skeleton_lib.module_segment(m[0]))
    return schema.join_path(*segs), fqn_inner


@dataclasses.dataclass
class _RawNode:
  """Intermediate FX node record before path collapsing and edge resolution."""

  name: str
  op: str
  meta: dict[str, Any]
  ins: list[tuple[str, int]]
  attrs: dict[str, str]
  path: str = ""
  fqn: str = ""


def convert(
    exported_program: Any,
    skeleton: schema.Skeleton,
    strip_prefix: str = "",
) -> tuple[list[schema.Op], list[schema.Edge]]:
  """Convert an ExportedProgram into hfgraph Op and Edge lists.

  Args:
    exported_program: PyTorch ExportedProgram instance.
    skeleton: Model skeleton representing static structure.
    strip_prefix: Module prefix to strip from path namespaces.

  Returns:
    A tuple of (ops, edges) extracted from the graph.
  """
  # pylint: disable=protected-access
  sig = exported_program.graph_signature
  param_of = dict(sig.inputs_to_parameters)
  buffer_of = dict(sig.inputs_to_buffers)
  const_of = dict(getattr(sig, "inputs_to_lifted_tensor_constants", {}) or {})
  lifted = set(param_of) | set(buffer_of) | set(const_of)
  outputs: dict[str, list[schema.Tensor | None] | None] = {}
  alias: dict[str, tuple[str, int]] = {}
  raw: list[_RawNode] = []
  path_builder = PathBuilder(skeleton, strip_prefix)

  def resolve(node: torch.fx.Node, prefix: str) -> tuple[str, int]:
    key, idx, hops = prefix + node.name, 0, 0
    while key in alias:
      key, idx = alias[key]
      hops += 1
      if hops > 10000:
        raise RuntimeError("alias cycle")
    return key, idx

  def walk(
      graph: torch.fx.Graph,
      prefix: str,
      in_map: dict[str, tuple[str, int]],
      sink: dict[int, tuple[str, int]],
  ) -> None:
    for node in graph.nodes:
      name = prefix + node.name
      if node.op == "placeholder":
        if prefix:
          alias[name] = in_map[node.name]
          continue
        outputs[name] = _outs_of(name, node.meta.get("val"))
        if node.name in lifted or outputs[name][0] is None:
          continue
        raw.append(
            _RawNode(
                name=name, op="placeholder", meta=node.meta, ins=[], attrs={}
            )
        )
        continue
      if node.op == "output":
        args = (
            node.args[0]
            if isinstance(node.args[0], (tuple, list))
            else [node.args[0]]
        )
        if prefix:
          for i, arg in enumerate(args):
            if isinstance(arg, torch.fx.Node):
              sink[i] = resolve(arg, prefix)
          continue
        for i, arg in enumerate(args):
          if (
              isinstance(arg, torch.fx.Node)
              and sig.output_specs[i].kind.name == "USER_OUTPUT"
          ):
            raw.append(
                _RawNode(
                    name=f"output_{i}",
                    op="output",
                    meta=arg.meta,
                    ins=[resolve(arg, prefix)],
                    attrs={},
                )
            )
        continue
      if node.op == "get_attr":
        continue
      op = _op_name(node.target)
      if op == "getitem":
        if name not in alias:
          src, _ = resolve(node.args[0], prefix)
          alias[name] = (src, node.args[1])
        continue
      if isinstance(node.target, torch._ops.HigherOrderOperator):
        sub = next(
            arg
            for arg in node.args
            if isinstance(arg, torch.fx.Node) and arg.op == "get_attr"
        )
        subgraph_mod = exported_program.graph_module.get_submodule(sub.target)
        operands = [
            arg
            for arg in node.args
            if isinstance(arg, torch.fx.Node) and arg.op != "get_attr"
        ]
        placeholders = [
            p for p in subgraph_mod.graph.nodes if p.op == "placeholder"
        ]
        sub_outputs: dict[int, tuple[str, int]] = {}
        walk(
            subgraph_mod.graph,
            name + "__",
            {
                p.name: resolve(o, prefix)
                for p, o in zip(placeholders, operands)
            },
            sub_outputs,
        )
        outputs[name] = None
        for user in node.users:
          if _op_name(user.target) == "getitem" and user.args[1] in sub_outputs:
            alias[prefix + user.name] = sub_outputs[user.args[1]]
        continue
      if op.replace("aten.", "") in NOISE:
        continue
      outputs[name] = _outs_of(name, node.meta.get("val"))
      ins: list[tuple[str, int]] = []
      attrs: dict[str, str] = {}
      for arg in node.all_input_nodes:
        if arg.op == "placeholder" and arg.name in lifted:
          fq = (
              param_of.get(arg.name)
              or buffer_of.get(arg.name)
              or const_of.get(arg.name)
          )
          tensor = (outputs.get(arg.name) or [None])[0]
          key = "buffer" if arg.name in buffer_of else "weight"
          attrs[f"{key}:{fq.rsplit('.', 1)[-1]}"] = (
              f"{fq} {tensor.dtype}{tensor.shape}" if tensor else fq
          )
          continue
        resolved = resolve(arg, prefix)
        if resolved[0] in outputs and outputs[resolved[0]] is not None:
          ins.append(resolved)
      for k, v in node.kwargs.items():
        if not isinstance(v, torch.fx.Node) and v is not None:
          attrs[k] = str(v)
      raw.append(
          _RawNode(name=name, op=op, meta=node.meta, ins=ins, attrs=attrs)
      )

  walk(exported_program.graph, "", {}, {})

  # paths first, so leaf-module collapsing can count ops per module
  for record in raw:
    if record.op in ("placeholder", "output"):
      record.path, record.fqn = record.op + "s", ""
      continue
    record.path, record.fqn = path_builder.path(
        _frames_of(record.meta.get("stack_trace")),
        record.meta.get("nn_module_stack"),
    )
  per_mod = collections.Counter(
      record.fqn for record in raw if record.op not in ("placeholder", "output")
  )
  cls_of = {}
  for record in raw:
    stack = record.meta.get("nn_module_stack") or {}
    if stack:
      _, cls = list(stack.values())[-1]
      cls_of[record.fqn] = (
          cls if isinstance(cls, str) else cls.__qualname__
      ).rsplit(".", 1)[-1]

  ops, edges, ordinal, ids = [], [], collections.Counter(), {}
  for record in raw:
    meta, op = record.meta, record.op
    frames = [f for f in _frames_of(meta.get("stack_trace")) if not f[3]]
    file, line = (frames[-1][0], frames[-1][1]) if frames else ("", 0)
    outs = [t for t in outputs.get(record.name) or [] if t]
    path = record.path
    if op == "placeholder":
      label, aten, file, line = record.name, "Placeholder", "", 0
    elif op == "output":
      src, idx = record.ins[0]
      outs = [outputs[src][idx].model_copy(update={"name": "logits"})]
      label, aten, file, line = "logits", "Output", "", 0
    else:
      if not outs:
        continue
      label, aten = op.replace("aten.", ""), op
      cls = cls_of.get(record.fqn, "")
      if cls in COLLAPSE and per_mod[record.fqn] == 1 and "/" in path:
        label, aten = (
            path.rsplit("/", 1)[-1],
            cls,
        )  # nn.Linear etc. -> one node named after the attribute
        path = path.rsplit("/", 1)[0]
        record.attrs["module_class"] = cls
    key = (path, file, line)
    op_id = f"{path}|{file}:{line}|{ordinal[key]}"
    ordinal[key] += 1
    ids[record.name] = op_id
    input_tensors = []
    for src, idx in record.ins:
      lst = outputs.get(src) or []
      tensor = lst[idx] if idx < len(lst) else None
      if tensor is None or src not in ids:
        continue
      input_tensors.append(tensor)
      edges.append(
          schema.Edge(
              src=ids[src],
              src_out=idx,
              dst=op_id,
              dst_in=len(input_tensors) - 1,
          )
      )
    ops.append(
        schema.Op(
            id=op_id,
            path=path,
            fx_name=record.name,
            aten=aten,
            label=label,
            file=file,
            line=line,
            inputs=input_tensors,
            outputs=outs,
            attrs=record.attrs,
        )
    )
  return ops, edges


def _draft_report_text(exported_program: Any) -> str:
  """Returns the draft_export failure report of an ExportedProgram, or ''.

  `torch.export.draft_export` has no public accessor for its report: torch
  attaches it as the private `_report` attribute and its own warning tells
  users to `print(ep._report)`. Read it defensively so a torch version that
  renames it degrades to an empty report instead of failing the trace.

  Args:
    exported_program: The ExportedProgram returned by `draft_export`.
  """
  return str(getattr(exported_program, "_report", "") or "")


def draft_path(trace_path: str) -> str:
  """Returns the draft_export log path for a trace JSON path."""
  return os.path.splitext(trace_path)[0] + ".draft.txt"


# ---------------------------------------------------------------- entry point
def main() -> None:
  """CLI entrypoint for exporting a single model trace."""
  # pylint: disable=protected-access,g-import-not-at-top,broad-exception-caught
  warnings.filterwarnings(
      "ignore",
      category=UserWarning,
      module=r"^(torch|transformers)(\.|$)",
  )
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("skeleton")
  parser.add_argument("out")
  parser.add_argument("--seq", type=int, default=8)
  parser.add_argument("--tiny", action="store_true")
  parser.add_argument("--draft", action="store_true")
  parser.add_argument("--trace-id", default=None)
  parser.add_argument(
      "--decode",
      type=int,
      default=0,
      help="prefill N tokens eagerly, then export one cached decode step",
  )
  args = parser.parse_args()
  trace_id = args.trace_id or f"seq{args.seq}"
  with open(args.skeleton, encoding="utf-8") as f:
    skeleton = schema.Skeleton.model_validate_json(f.read())
  t0 = time.time()
  lm, tcfg = models.build_text_model(args.model, args.tiny)
  lm.config.use_cache = bool(args.decode)
  overrides = {"use_cache": str(bool(args.decode)), "tiny": str(args.tiny)}
  if args.decode:
    overrides["decode_after_prefill"] = str(args.decode)
  stand_ins = _apply_stand_ins(lm)
  if stand_ins:
    overrides["stand_ins"] = ",".join(stand_ins)
  if getattr(
      tcfg, "num_experts", None
  ):  # MoE: dense-equivalent experts so all experts are in the graph
    lm.config._experts_implementation = "batched_mm"
    tcfg._experts_implementation = "batched_mm"
    overrides["_experts_implementation"] = "batched_mm"
  ids = torch.randint(0, tcfg.vocab_size, (1, args.seq))
  mask = torch.ones_like(ids)
  target, export_args, kwargs = lm, (ids,), {"attention_mask": mask}
  if args.decode:
    from transformers.cache_utils import DynamicCache

    lm.requires_grad_(False)
    cache = DynamicCache(config=tcfg)
    pre = torch.randint(0, tcfg.vocab_size, (1, args.decode))
    with torch.no_grad():
      lm(input_ids=pre, past_key_values=cache, use_cache=True)
    ids = torch.randint(0, tcfg.vocab_size, (1, 1))
    target, export_args, kwargs = DecodeStep(lm, cache), (ids,), {}
  print(
      f"[trace {trace_id}] model ready {time.time() - t0:.1f}s", file=sys.stderr
  )
  plan = {"seq_len": 1 if args.decode else args.seq, "cache": args.decode}
  if args.decode:
    inputs = {"input_ids": f"int64[1, {plan['seq_len']}]"}
  else:
    inputs = {
        "input_ids": f"int64[1, {args.seq}]",
        "attention_mask": f"int64[1, {args.seq}]",
    }
  draft_txt = ""
  if args.draft:
    draft_txt = _draft_report_text(
        torch.export.draft_export(target, export_args, kwargs=kwargs)
    )
  t0 = time.time()
  exported_program, strict_used, failures = None, True, []
  for strict in (True, False):
    try:
      exported_program = torch.export.export(
          target, export_args, kwargs=kwargs, strict=strict
      )
      strict_used = strict
      break
    # torch.export raises many unrelated types (UserError, Unsupported,
    # GuardOnDataDependentSymNode, RuntimeError, AssertionError, ...); any
    # failure is recorded and the next mode is tried.
    except Exception as e:
      failures.append(f"strict={strict}: {type(e).__name__}: {str(e)[:600]}")
      print(
          f"[trace {trace_id}] export failed {failures[-1][:200]}",
          file=sys.stderr,
      )
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  if exported_program is None:
    try:
      draft_txt = _draft_report_text(
          torch.export.draft_export(target, export_args, kwargs=kwargs)
      )
    # Same as above: only the failure text is kept for the report.
    except Exception as e:
      draft_txt = f"draft_export failed: {type(e).__name__}: {str(e)[:600]}"
    info = schema.TraceInfo(
        id=trace_id,
        inputs=inputs,
        plan=plan,
        config_overrides=overrides,
        strict=False,
        stats={"fx_call_function": 0, "ops": 0, "edges": 0, "export_failed": 1},
    )
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(schema.Trace(trace=info, ops=[], edges=[]).model_dump_json())
    with open(draft_path(args.out), "w", encoding="utf-8") as f:
      f.write("\n\n".join(failures) + "\n\n" + draft_txt)
    print(
        f"[trace {trace_id}] EXPORT FAILED in both modes; wrote empty trace +"
        " failure report",
        file=sys.stderr,
    )
    return
  print(
      f"[trace {trace_id}] export strict={strict_used} {time.time() - t0:.1f}s",
      file=sys.stderr,
  )
  ops, edges = convert(
      exported_program,
      skeleton,
      strip_prefix="model." if args.decode else "",
  )
  if args.decode:
    diff = -1.0  # cache is mutated in place; eager comparison is not meaningful
  else:
    with torch.no_grad():
      ref = lm(ids, attention_mask=mask).logits
      out = exported_program.module()(ids, attention_mask=mask).logits
    diff = float((ref - out).abs().max())
  if failures:
    draft_txt = "\n\n".join(failures) + (
        "\n\n" + draft_txt if draft_txt else ""
    )
  info = schema.TraceInfo(
      id=trace_id,
      inputs=inputs,
      plan=plan,
      config_overrides=overrides,
      strict=strict_used,
      stats={
          "fx_call_function": sum(
              1 for n in exported_program.graph.nodes if n.op == "call_function"
          ),
          "ops": len(ops),
          "edges": len(edges),
          "max_abs_diff_e9": int(diff * 1e9) if diff >= 0 else -1,
      },
  )
  with open(args.out, "w", encoding="utf-8") as f:
    f.write(schema.Trace(trace=info, ops=ops, edges=edges).model_dump_json())
  if draft_txt:
    with open(draft_path(args.out), "w", encoding="utf-8") as f:
      f.write(draft_txt)
  print(
      f"[trace {trace_id}] ops={len(ops)} edges={len(edges)}"
      f" max|eager-export|={diff:.2e} -> {args.out}",
      file=sys.stderr,
  )


if __name__ == "__main__":
  main()
