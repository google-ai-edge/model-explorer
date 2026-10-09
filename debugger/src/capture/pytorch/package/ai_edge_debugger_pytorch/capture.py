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

"""CaptureRun context manager and PyTorch activation hooks."""

from collections.abc import Callable, Iterator
import contextlib
import dataclasses
import pathlib
import types
from typing import Any, TypedDict

from ai_edge_debugger_pytorch import kv_cache
from ai_edge_debugger_pytorch import writer
import torch
import torch.nn as nn


class ModuleManifestRow(TypedDict, total=False):
  scope: str
  shard: str
  slot: str
  key: str
  shape: list[int]
  dtype: str
  path: str
  cls: str
  when: str
  invocation: int
  call_seq: int
  index: int
  layer: int
  kind: str
  phase: str
  step: int
  forward_id: int
  module_call_id: int
  turn: int
  pos_offset: int


class BoundaryManifestRow(TypedDict, total=False):
  scope: str
  shard: str
  slot: str
  key: str
  shape: list[int]
  dtype: str
  edge: str
  when: str
  output_path: list[str]
  forward_id: int


@dataclasses.dataclass
class _ForwardState:
  """Execution tracking state for the active forward pass."""

  forward_id: int
  phase: str
  step: int
  pos_offset: int
  turn: int
  tokens_in_forward: int
  module_tensors: dict[str, torch.Tensor] = dataclasses.field(
      default_factory=dict
  )
  module_rows: list[ModuleManifestRow] = dataclasses.field(default_factory=list)
  boundary_rows: list[BoundaryManifestRow] = dataclasses.field(
      default_factory=list
  )
  inputs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  outputs: list[dict[str, Any]] = dataclasses.field(default_factory=list)


class Topology:
  """Topology description for causal language models.

  Attributes:
    blocks_path: Dot-delimited module path to the transformer block stack.
    n_layers: Total number of transformer layers in the model.
    width: Hidden layer dimension or embedding width.
  """

  def __init__(
      self, blocks_path: str, n_layers: int, width: int | None
  ) -> None:
    self.blocks_path = blocks_path
    self.n_layers = n_layers
    self.width = width


class Site:
  """A capture site module within a layer.

  Attributes:
    path: Dot-delimited module path to the hooked site.
    module: The PyTorch neural network module instance.
    layer: Zero-based layer index.
    kind: Scope classification ('layer' or 'sublayer').
  """

  def __init__(
      self, path: str, module: nn.Module, layer: int, kind: str
  ) -> None:
    self.path = path
    self.module = module
    self.layer = layer
    self.kind = kind


def detect_topology(
    model: nn.Module,
) -> tuple[str, nn.ModuleList, int, int | None]:
  """Detects transformer block structure, layer count, and model width.

  Args:
    model: PyTorch model instance to inspect.

  Returns:
    A tuple of (blocks_path, blocks_module, n_layers, width).

  Raises:
    ValueError: If standard transformer block topologies cannot be detected.
  """
  config = getattr(model, "config", None)
  width = getattr(config, "hidden_size", None)
  n_layers = getattr(config, "num_hidden_layers", None)

  # Check standard block locations
  for candidate in (
      "model.layers",
      "layers",
      "transformer.h",
      "gpt_neox.layers",
      "decoder.layers",
  ):
    curr = model
    parts = candidate.split(".")
    found = True
    for part in parts:
      if hasattr(curr, part):
        curr = getattr(curr, part)
      else:
        found = False
        break
    if found and isinstance(curr, (nn.ModuleList, list)) and len(curr) > 0:
      if n_layers is None:
        n_layers = len(curr)
      return candidate, curr, n_layers, width

  # Dynamic search if not found in standard paths
  for name, module in model.named_modules():
    if isinstance(module, nn.ModuleList) and len(module) > 0:
      first_child = module[0]
      child_names = [n for n, _ in first_child.named_children()]
      if any(
          "attn" in n or "attention" in n or "mlp" in n for n in child_names
      ):
        if n_layers is None:
          n_layers = len(module)
        return name, module, n_layers, width

  raise ValueError("Unsupported module topology")


class CaptureRun:
  """Context manager orchestrating runtime tensor capture on PyTorch models."""

  def __init__(
      self,
      model: nn.Module,
      granularity: str = "sublayer",
      layers: str | int | list[int] = "0",
      outside: bool = False,
      out_dir: str | pathlib.Path | None = None,
      run: dict[str, Any] | None = None,
  ) -> None:
    """Initializes CaptureRun.

    Args:
      model: PyTorch neural network model to capture activations from.
      granularity: Capture granularity ('layer' or 'sublayer').
      layers: Target layer index or collection of layer indices to hook.
      outside: Protocol compatibility flag for outer scope capture.
      out_dir: Directory where tensor shards and manifest records are saved.
      run: Optional run configuration dictionary.
    """
    blocks_path, blocks_module, n_layers, width = detect_topology(model)
    self.model = model
    self.granularity = granularity
    # Protocol compatibility attributes preserved for caller contracts.
    self.outside = outside
    self.out_dir = (
        pathlib.Path(out_dir).resolve() if out_dir is not None else None
    )
    self.capture_dir = str(self.out_dir) if self.out_dir is not None else None
    self.run = run if run is not None else {}
    self.topology = Topology(blocks_path, n_layers, width)

    if isinstance(layers, int):
      layer_indices = [layers]
    elif isinstance(layers, str):
      layer_indices = [int(x.strip()) for x in layers.split(",") if x.strip()]
    else:
      layer_indices = list(layers)

    self.sites: list[Site] = []
    for layer_idx in layer_indices:
      if layer_idx < 0 or layer_idx >= len(blocks_module):
        continue
      layer_module = blocks_module[layer_idx]
      layer_path = f"{blocks_path}.{layer_idx}"
      self.sites.append(
          Site(
              path=layer_path,
              module=layer_module,
              layer=layer_idx,
              kind="layer",
          )
      )
      if granularity == "sublayer":
        for child_name, child_module in layer_module.named_children():
          child_path = f"{layer_path}.{child_name}"
          self.sites.append(
              Site(
                  path=child_path,
                  module=child_module,
                  layer=layer_idx,
                  kind="sublayer",
              )
          )

    # Sink object retained for compatibility with predecessor capture probes.
    self.sink = types.SimpleNamespace(skipped={})
    self.forwards: list[dict[str, Any]] = []
    self.kv_snapshots: list[dict[str, Any]] = []
    self.token_records: list[dict[str, Any]] = []
    self.generation: dict[str, Any] | None = None

    self._handles: list[Any] = []
    self._call_counter = 0
    self._call_seq = 0
    self._forward_id_counter = 0
    self._current_forward_info: _ForwardState | None = None
    self._module_call_map: dict[str, int] = {}
    self._pending_tokens: list[dict[str, Any]] = []
    self._last_past_key_values: Any = None
    self._kv_shard_counter = 0
    self._cumulative_boundary_tensors: dict[str, torch.Tensor] = {}
    self._boundaries_written = False

  def _next_call_seq(self) -> int:
    """Returns the next sequential hook invocation order counter."""
    seq = self._call_seq
    self._call_seq += 1
    return seq

  def _create_module_manifest_row(
      self,
      site: Site,
      tensor: torch.Tensor,
      when: str,
      slot: str,
      capture_key: str,
      call_seq: int,
      call_id: int,
  ) -> ModuleManifestRow:
    """Constructs a manifest row dictionary for a module activation.

    Args:
      site: Capture site metadata for the module.
      tensor: Cloned CPU activation tensor.
      when: Execution timing ('in' or 'out').
      slot: Unique safetensors tensor slot identifier.
      capture_key: Monotonic capture index lookup key.
      call_seq: Sequence number of the hook invocation.
      call_id: Sequential identifier of the module call.

    Returns:
      A dictionary populated with module manifest metadata.
    """
    f_info = self._current_forward_info
    if f_info is None:
      raise RuntimeError("No active forward pass state.")
    return {
        "scope": "module",
        "shard": "",  # populated on write
        "slot": slot,
        "key": capture_key,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "path": site.path,
        "cls": type(site.module).__name__,
        "when": when,
        "invocation": 0,
        "call_seq": call_seq,
        "index": 0,
        "layer": site.layer,
        "kind": site.kind,
        "phase": f_info.phase,
        "step": f_info.step,
        "forward_id": f_info.forward_id,
        "module_call_id": call_id,
        "turn": f_info.turn,
        "pos_offset": f_info.pos_offset,
    }

  @staticmethod
  def _extract_input_tensor(
      args: tuple[Any, ...], kwargs: dict[str, Any] | None
  ) -> torch.Tensor | None:
    """Extracts the first input activation tensor from hook args or kwargs."""
    if args and isinstance(args[0], torch.Tensor):
      return args[0]
    if (
        kwargs
        and "hidden_states" in kwargs
        and isinstance(kwargs["hidden_states"], torch.Tensor)
    ):
      return kwargs["hidden_states"]
    if args:
      for arg in args:
        if isinstance(arg, torch.Tensor):
          return arg
    if kwargs:
      for val in kwargs.values():
        if isinstance(val, torch.Tensor):
          return val
    return None

  @staticmethod
  def _extract_output_tensor(output: Any) -> torch.Tensor | None:
    """Extracts the primary output activation tensor from hook output."""
    if isinstance(output, torch.Tensor):
      return output
    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
      return output[0]
    return None

  def _record_module_tensor(
      self,
      site: Site,
      tensor: torch.Tensor,
      when: str,
      call_id: int,
  ) -> None:
    """Clones and records a module activation tensor into the forward state."""
    if self._current_forward_info is None or self.out_dir is None:
      return
    forward_id = self._current_forward_info.forward_id
    slot = f"{site.path.replace('.', '_')}__{when}__{forward_id}__0"
    capture_key = f"{site.path}:{when}:{forward_id}"
    call_seq = self._next_call_seq()
    t_cpu = kv_cache.clone_tensor_to_cpu(tensor)

    self._current_forward_info.module_tensors[slot] = t_cpu
    self._current_forward_info.module_rows.append(
        self._create_module_manifest_row(
            site=site,
            tensor=t_cpu,
            when=when,
            slot=slot,
            capture_key=capture_key,
            call_seq=call_seq,
            call_id=call_id,
        )
    )

  def _record_boundary_tensor(
      self,
      tensor: torch.Tensor,
      edge: str,
      name: str,
      output_path: list[str] | None = None,
  ) -> None:
    """Clones and records an input/output boundary tensor."""
    if self._current_forward_info is None or self.out_dir is None:
      return
    forward_id = self._current_forward_info.forward_id
    slot = f"boundary_{edge}_{forward_id}_{name}"
    key = f"boundary:{edge}:{forward_id}:{name}"
    resolved_path = output_path or (
        ["output", name] if edge == "out" else ["kwargs", name]
    )
    t_cpu = kv_cache.clone_tensor_to_cpu(tensor)
    self._cumulative_boundary_tensors[slot] = t_cpu
    self._current_forward_info.boundary_rows.append({
        "scope": "boundary",
        "shard": "boundaries.safetensors",
        "slot": slot,
        "key": key,
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "edge": edge,
        "when": edge,
        "output_path": resolved_path,
        "forward_id": forward_id,
    })
    entry = {
        "key": key,
        "output_path": resolved_path,
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
    }
    if edge == "out":
      self._current_forward_info.outputs.append(entry)
    else:
      self._current_forward_info.inputs.append(entry)

  def _make_pre_hook(
      self, site: Site
  ) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any] | None], None]:
    """Creates a forward pre-hook capturing input activations for a site."""

    def hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> None:
      if self._current_forward_info is None or self.out_dir is None:
        return
      tensor = self._extract_input_tensor(args, kwargs)
      if tensor is None:
        return
      call_id = self._call_counter
      self._call_counter += 1
      self._module_call_map[site.path] = call_id
      self._record_module_tensor(site, tensor, when="in", call_id=call_id)

    return hook

  def _make_post_hook(
      self, site: Site
  ) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
    """Creates a forward hook capturing output activations for a site."""

    def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
      if self._current_forward_info is None or self.out_dir is None:
        return
      tensor = self._extract_output_tensor(output)
      if tensor is None:
        return
      call_id = self._module_call_map.get(site.path, 0)
      self._record_module_tensor(site, tensor, when="out", call_id=call_id)

    return hook

  def _make_model_pre_hook(
      self,
  ) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any] | None], None]:
    """Creates a pre-hook capturing model input tensors at the boundary."""

    def hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> None:
      if self._current_forward_info is None:
        return
      ids = None
      if kwargs and "input_ids" in kwargs:
        ids = kwargs["input_ids"]
      elif args and isinstance(args[0], torch.Tensor):
        ids = args[0]

      if ids is not None and hasattr(ids, "shape") and len(ids.shape) >= 2:
        self._current_forward_info.tokens_in_forward = ids.shape[1]

      recorded_names: set[str] = set()
      if kwargs:
        for name, val in kwargs.items():
          if isinstance(val, torch.Tensor):
            self._record_boundary_tensor(
                val, edge="in", name=name, output_path=["kwargs", name]
            )
            recorded_names.add(name)
      if (
          ids is not None
          and isinstance(ids, torch.Tensor)
          and "input_ids" not in recorded_names
      ):
        self._record_boundary_tensor(
            ids,
            edge="in",
            name="input_ids",
            output_path=["kwargs", "input_ids"],
        )

    return hook

  def _make_model_post_hook(
      self,
  ) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
    """Creates a post-hook capturing logits and caching past_key_values."""

    def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
      past = getattr(output, "past_key_values", None)
      if past is not None:
        self._last_past_key_values = past

      if self._current_forward_info is None or self.out_dir is None:
        return

      logits = getattr(output, "logits", None)
      if logits is None and isinstance(output, torch.Tensor):
        logits = output

      if logits is not None and isinstance(logits, torch.Tensor):
        self._record_boundary_tensor(
            logits,
            edge="out",
            name="logits",
            output_path=["output", "logits"],
        )

    return hook

  def __enter__(self) -> "CaptureRun":
    if self.out_dir is not None:
      self.out_dir.mkdir(parents=True, exist_ok=True)
      (self.out_dir / "boundaries").mkdir(parents=True, exist_ok=True)
      (self.out_dir / "kv").mkdir(parents=True, exist_ok=True)
      (self.out_dir / "tokens.jsonl").touch(exist_ok=True)

    # Register hooks on sites
    for site in self.sites:
      h_pre = site.module.register_forward_pre_hook(
          self._make_pre_hook(site), with_kwargs=True
      )
      self._handles.append(h_pre)
      h_post = site.module.register_forward_hook(self._make_post_hook(site))
      self._handles.append(h_post)

    # Register hooks on model
    h_m_pre = self.model.register_forward_pre_hook(
        self._make_model_pre_hook(), with_kwargs=True
    )
    self._handles.append(h_m_pre)
    h_m_post = self.model.register_forward_hook(self._make_model_post_hook())
    self._handles.append(h_m_post)

    return self

  def _record_kv_snapshot(
      self,
      moment: str,
      forward_id: int,
      processed_token_count: int,
      phase: str = "prefill",
      step: int = 0,
      turn: int = 1,
  ) -> None:
    """Creates and persists a KV cache snapshot for the given moment.

    Args:
      moment: Lifecycle moment label ('prefill_post' or 'terminal').
      forward_id: Identifier of the forward pass associated with this snapshot.
      processed_token_count: Number of cumulative tokens processed up to this
        moment.
      phase: Execution phase ('prefill' or 'decode').
      step: Sequence step counter.
      turn: Multi-turn interaction index.
    """
    if self._last_past_key_values is None or self.out_dir is None:
      return
    snap_idx = len(self.kv_snapshots)
    shard_name = f"kv_{self._kv_shard_counter:05d}.safetensors"
    self._kv_shard_counter += 1
    snap, kv_tensors, kv_rows = kv_cache.create_kv_snapshot_with_storage(
        self._last_past_key_values,
        moment=moment,
        forward_id=forward_id,
        processed_token_count=processed_token_count,
        valid_length=processed_token_count,
        snapshot_idx=snap_idx,
        shard_name=shard_name,
    )
    snap["phase"] = phase
    snap["step"] = step
    snap["turn"] = turn
    self.kv_snapshots.append(snap)
    if kv_tensors:
      writer.write_kv_shard(self.out_dir, shard_name, kv_tensors, kv_rows)

  def _flush_boundary_shard(self) -> None:
    """Writes cumulative boundary tensors to boundaries.safetensors."""
    if (
        self.out_dir is not None
        and self._cumulative_boundary_tensors
        and not self._boundaries_written
    ):
      writer.write_boundary_shard(
          self.out_dir,
          "boundaries.safetensors",
          self._cumulative_boundary_tensors,
      )
      self._boundaries_written = True

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    for handle in self._handles:
      handle.remove()
    self._handles.clear()
    self._flush_boundary_shard()

  @contextlib.contextmanager
  def forward_context(
      self,
      phase: str = "prefill",
      step: int = 0,
      pos_offset: int = 0,
      turn: int = 1,
      **extra: Any,
  ) -> Iterator[None]:
    """Context manager scoping execution of a single model forward pass.

    Args:
      phase: Execution phase ('prefill' or 'decode').
      step: Sequence step counter.
      pos_offset: Starting token position offset in the sequence.
      turn: Multi-turn interaction index.
      **extra: Additional metadata attributes passed by callers.

    Yields:
      None during the active forward pass context.
    """
    forward_id = self._forward_id_counter
    self._forward_id_counter += 1

    # Update consumption of previous tokens
    for rec in self._pending_tokens:
      for c in rec["consumption"]:
        if c["consumed_by_forward_id"] is None:
          c["consumed_by_forward_id"] = forward_id
          c["position"] = pos_offset
    self._pending_tokens.clear()

    # If first forward (prefill), record prefill_pre snapshot
    if forward_id == 0 and phase == "prefill":
      pre_snap = {
          "snapshot_id": len(self.kv_snapshots),
          "moment": "prefill_pre",
          "phase": phase,
          "step": step,
          "turn": turn,
          "state": "not_allocated",
          "storage_complete": True,
          "forward_id": forward_id,
          "processed_token_count": pos_offset,
          "layers": [],
      }
      self.kv_snapshots.append(pre_snap)
      if self.out_dir is not None:
        writer.write_kv_index(self.out_dir, self.kv_snapshots)

    self._current_forward_info = _ForwardState(
        forward_id=forward_id,
        phase=phase,
        step=step,
        pos_offset=pos_offset,
        turn=turn,
        tokens_in_forward=1 if phase == "decode" else 0,
    )

    failed = False
    try:
      yield
    except BaseException:
      failed = True
      raise
    finally:
      forward_info = self._current_forward_info
      if forward_info is None:
        raise RuntimeError("No active forward pass state.")
      tokens_count = forward_info.tokens_in_forward
      if tokens_count == 0:
        tokens_count = 1 if phase == "decode" else 0

      cache_before = {
          "state": "not_allocated" if pos_offset == 0 else "available",
          "processed_token_count": pos_offset,
      }
      cache_after = {
          "state": "available",
          "processed_token_count": pos_offset + tokens_count,
      }
      forward_record = {
          "forward_id": forward_id,
          "phase": phase,
          "step": step,
          "turn": turn,
          "pos_offset": pos_offset,
          "status": "failed" if failed else "completed",
          "inputs": list(forward_info.inputs),
          "outputs": list(forward_info.outputs),
          "cache_before": cache_before,
          "cache_after": cache_after,
      }
      self.forwards.append(forward_record)

      if self.out_dir is not None:
        # Write module tensors
        if forward_info.module_tensors:
          shard_name = f"shard-{forward_id:05d}.safetensors"
          for row in forward_info.module_rows:
            row["shard"] = shard_name
          writer.write_module_shard(
              self.out_dir,
              shard_name,
              forward_info.module_tensors,
              forward_info.module_rows,
          )

        # Append boundary manifest rows incrementally
        if forward_info.boundary_rows:
          b_shard_name = "boundaries.safetensors"
          for row in forward_info.boundary_rows:
            row["shard"] = b_shard_name
          writer.append_boundary_manifest(
              self.out_dir, forward_info.boundary_rows
          )

        # If prefill, record prefill_post snapshot
        if phase == "prefill":
          self._record_kv_snapshot(
              moment="prefill_post",
              forward_id=forward_id,
              processed_token_count=pos_offset + tokens_count,
              phase=phase,
              step=step,
              turn=turn,
          )

        writer.write_forward_index(self.out_dir, self.forwards)
        writer.write_kv_index(self.out_dir, self.kv_snapshots)

      self._current_forward_info = None

  def observe_tokens(
      self,
      tokens: list[int],
      forward_id: int = 0,
      texts: list[str] | None = None,
      scores: list[float] | None = None,
  ) -> None:
    """Records newly generated tokens and schedules consumption tracking.

    Args:
      tokens: List of emitted token IDs.
      forward_id: Identifier of the forward pass that produced these tokens.
      texts: Optional decoded string representations of each token.
      scores: Optional logit/probability scores for each token.
    """
    record = {
        "forward_id": forward_id,
        "candidate": 0,
        "token_ids": list(tokens),
        "texts": list(texts) if texts is not None else None,
        "scores": list(scores) if scores is not None else None,
        "consumption": [
            {"consumed_by_forward_id": None, "position": None} for _ in tokens
        ],
    }
    self.token_records.append(record)
    self._pending_tokens.append(record)
    if self.out_dir is not None:
      writer.write_tokens(self.out_dir, self.token_records)

  def finish_generation(self, stop_reason: str) -> None:
    """Finalizes generation metadata and records terminal KV snapshot.

    Args:
      stop_reason: Reason generation halted (e.g. 'completed', 'cancelled',
        'error').
    """
    status = (
        "cancelled"
        if stop_reason == "cancelled"
        else ("failed" if stop_reason == "error" else "completed")
    )
    generated_count = sum(len(r["token_ids"]) for r in self.token_records)
    processed_count = (
        self.forwards[-1]["cache_after"]["processed_token_count"]
        if self.forwards
        else 0
    )
    pending_token_ids = [
        t_id
        for r in self.token_records
        for i, t_id in enumerate(r["token_ids"])
        if r["consumption"][i]["consumed_by_forward_id"] is None
    ]

    if status == "completed":
      last_forward = self.forwards[-1] if self.forwards else {}
      last_forward_id = last_forward.get("forward_id", 0)
      self._record_kv_snapshot(
          moment="terminal",
          forward_id=last_forward_id,
          processed_token_count=processed_count,
          phase=last_forward.get("phase", "decode"),
          step=last_forward.get("step", 0),
          turn=last_forward.get("turn", 1),
      )

    self._flush_boundary_shard()

    self.generation = {
        "status": status,
        "stop_reason": stop_reason,
        "generated_token_count": generated_count,
        "processed_token_count": processed_count,
        "pending_token_ids": pending_token_ids,
    }

    if self.out_dir is not None:
      writer.write_forward_index(self.out_dir, self.forwards)
      writer.write_tokens(self.out_dir, self.token_records)
      writer.write_kv_index(self.out_dir, self.kv_snapshots)
      writer.write_generation(self.out_dir, self.generation)
