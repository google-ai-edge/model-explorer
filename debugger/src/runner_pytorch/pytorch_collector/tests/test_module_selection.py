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

"""collect(model, modules=...): only the selected module boundaries, no operators."""

# pylint: disable=protected-access

import pytest
from pytorch_collector import CaptureError
from pytorch_collector import collect
from pytorch_collector import to_model_explorer
from pytorch_collector import unused_ops
import torch
from torch import nn
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack


class Block(nn.Module):
  """Toy block module for testing."""

  def __init__(self):
    super().__init__()
    self.norm = nn.LayerNorm(4)
    self.proj = nn.Linear(4, 4)

  def forward(self, x, *, scale):
    return {"y": self.proj(self.norm(x)) * scale, "skip": [x, None]}


class Net(nn.Module):
  """Toy network with embedding, blocks, and a linear head."""

  def __init__(self):
    super().__init__()
    self.embed = nn.Embedding(8, 4)
    self.blocks = nn.ModuleList([Block(), Block()])
    self.head = nn.Linear(4, 3, bias=False)

  def forward(self, ids):
    x = self.embed(ids)
    for block in self.blocks:
      x = block(x, scale=torch.tensor(0.5))["y"] + x
    return self.head(x)


def net():
  torch.manual_seed(7)
  return Net().eval()


def bits(tensor):
  return tensor.contiguous().reshape(-1).view(torch.uint8)


def assert_same_records(actual, expected):
  """Same order, place and dtype, and bit-identical values."""
  assert [
      (r.forward_id, r.module, r.edge, r.path, r.tensor.dtype, r.tensor.shape)
      for r in actual
  ] == [
      (r.forward_id, r.module, r.edge, r.path, r.tensor.dtype, r.tensor.shape)
      for r in expected
  ]
  for a, b in zip(actual, expected, strict=True):
    assert torch.equal(bits(a.tensor), bits(b.tensor))


def hook_counts(model):
  """Returns total forward hooks count per module in model."""
  return {
      path: len(module._forward_pre_hooks) + len(module._forward_hooks)
      for path, module in model.named_modules()
  }


def test_selected_modules_match_the_full_capture_bit_for_bit():
  """Tests that selected modules match full capture bit-for-bit."""
  model = net()
  calls = [torch.tensor([[1, 2, 3]]), torch.tensor([[4]])]
  with collect(model) as full:
    expected_outputs = [model(ids) for ids in calls]
  selected = ["blocks.0", "blocks.1.norm", model.head]
  stack = _get_current_dispatch_mode_stack()
  with collect(model, modules=selected) as capture:
    assert _get_current_dispatch_mode_stack() == stack  # No dispatch mode.
    counts = hook_counts(model)
    outputs = [model(ids) for ids in calls]
  assert capture.modules == ("blocks.0", "blocks.1.norm", "head")
  # The root is hooked only to number forwards; unselected modules not at all.
  assert {path for path, count in counts.items() if count} == {
      "",
      "blocks.0",
      "blocks.1.norm",
      "head",
  }
  assert all(count == 0 for count in hook_counts(model).values())
  for output, baseline in zip(outputs, expected_outputs, strict=True):
    assert torch.equal(bits(output), bits(baseline))

  wanted = {"blocks.0", "blocks.1.norm", "head"}
  assert_same_records(
      capture.records,
      [r for r in full.records if r.op is None and r.module in wanted],
  )
  assert {r.module for r in capture.records} == wanted
  assert all(
      r.op is None and r.module_call_id == r.call_id for r in capture.records
  )
  assert ("kwargs", "scale") in {
      r.path for r in capture.records if r.module == "blocks.0"
  }
  assert ("output", "skip", 0) in {
      r.path for r in capture.records if r.module == "blocks.0"
  }
  # One call id per module call, shared by its inputs and outputs, unique
  # across forwards.
  by_call = {}
  for row in capture.records:
    by_call.setdefault(row.call_id, set()).add(
        (row.forward_id, row.module, row.edge)
    )
  assert sorted(len(v) for v in by_call.values()) == [2] * 6
  for name in ("ops", "edges", "outputs"):
    with pytest.raises(
        CaptureError,
        match=f"{name} are not recorded when collect\\(\\) selects modules",
    ):
      getattr(capture, name)


def test_graph_export_refuses_a_module_only_capture():
  """Tests that graph export refuses a module-only capture."""
  model = net()
  with collect(model, modules=["head"]) as capture:
    model(torch.tensor([[1]]))
  with pytest.raises(
      ValueError,
      match="to_model_explorer needs every operator.*selected 1 modules",
  ):
    to_model_explorer(capture)
  with pytest.raises(ValueError, match="unused_ops needs every operator"):
    unused_ops(capture)


def test_selected_root_parameters_buffers_and_calls_outside_root():
  """Tests selected root parameters, buffers, and calls outside root."""

  class Model(nn.Module):

    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.arange(4.0))
      self.register_buffer("state", torch.ones(2))
      self.child = nn.Identity()

    def forward(self, x):
      return {
          "weight": self.weight,
          "view": self.weight.view(2, 2),
          "buffer": self.state,
          "derived": self.weight * 2,
          "x": self.child(x),
      }

  model = Model()
  with collect(model, modules=[""]) as root_only:
    model.child(torch.ones(1))  # Outside a root forward: nothing.
    model(torch.zeros(1))
  assert [(r.module, r.path) for r in root_only.records] == [
      ("", ("args", 0)),
      ("", ("output", "buffer")),
      ("", ("output", "derived")),
      ("", ("output", "x")),
  ]
  with collect(model, modules=["child"]) as child_only:
    model.child(torch.ones(1))
    model(torch.zeros(1))
    model(torch.zeros(1))
  assert [(r.forward_id, r.module, r.edge) for r in child_only.records] == [
      (0, "child", "in"),
      (0, "child", "out"),
      (1, "child", "in"),
      (1, "child", "out"),
  ]


def test_shared_module_alias_path_records_under_its_first_path():
  """Tests shared module alias path records under its first path."""
  shared = nn.ReLU()
  model = nn.Sequential(nn.Identity(), shared, nn.Identity(), shared)
  with collect(model, modules=["3"]) as capture:
    model(torch.tensor([-1.0, 2.0]))
  assert capture.modules == ("1",)
  assert [(r.module, r.edge, r.call_id) for r in capture.records] == [
      ("1", "in", 1),
      ("1", "out", 1),
      ("1", "in", 2),
      ("1", "out", 2),
  ]


def test_budget_counts_only_selected_snapshots_and_failure_releases_everything():
  """Tests budget counts only selected snapshots and releases on failure."""
  model = net()
  ids = torch.tensor([[1, 2]])
  with collect(model, modules=["head"]) as exact:
    model(ids)
  used = sum(r.tensor.numel() * r.tensor.element_size() for r in exact.records)
  assert used == (2 * 4 + 2 * 3) * 4
  with collect(model, modules=["head"], memory_limit_bytes=used):
    model(ids)
  with pytest.raises(MemoryError, match="Snapshot memory limit exceeded"):
    with collect(
        model, modules=["head"], memory_limit_bytes=used - 1
    ) as failed:
      model(ids)
  with pytest.raises(CaptureError, match="successful scope exit"):
    _ = failed.records
  assert all(count == 0 for count in hook_counts(model).values())
  assert torch.is_grad_enabled()


def test_model_exception_removes_hooks_and_leaves_nothing_readable():
  """Tests model exception removes hooks and leaves nothing readable."""
  model = net()
  with pytest.raises(IndexError):
    with collect(model, modules=["blocks.0.proj"]) as capture:
      model(torch.tensor([[1]]))
      model(torch.tensor([[99]]))
  with pytest.raises(CaptureError):
    _ = capture.records
  assert all(count == 0 for count in hook_counts(model).values())
  with collect(model, modules=["blocks.0.proj"]):  # Not still claimed.
    pass


def test_invalid_selections_fail_before_any_hook():
  """Tests invalid module selections fail before installing any hooks."""
  model = net()
  with pytest.raises(
      ValueError, match=r"Unknown module paths: \['blocks.2', 'tail'\]"
  ):
    collect(model, modules=["blocks.0", "blocks.2", "tail"])
  with pytest.raises(
      ValueError, match=r"Modules not registered in the model: \['Linear'\]"
  ):
    collect(model, modules=[nn.Linear(1, 1)])
  with pytest.raises(ValueError, match="at least one module"):
    collect(model, modules=[])
  for bad in ("head", model.head, 3, [3]):
    with pytest.raises(TypeError, match="modules must"):
      collect(model, modules=bad)
  assert all(count == 0 for count in hook_counts(model).values())


def test_selected_module_removed_before_entry_is_rejected():
  """Tests selected module removed before entry is rejected."""
  model = net()
  capture = collect(model, modules=["head"])
  model.head = nn.Linear(4, 3)
  with pytest.raises(
      CaptureError, match="changed between collect\\(\\) and scope entry"
  ):
    with capture:
      pytest.fail("entered")
  assert all(count == 0 for count in hook_counts(model).values())


def test_module_only_snapshots_stay_on_their_device():
  """Tests module-only snapshots stay on their respective device."""
  # The meta device runs everywhere; full captures stay CPU-only.
  model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).to("meta")
  with pytest.raises(ValueError, match="CPU tensors only"):
    with collect(model):
      pass
  with collect(model, modules=["0"]) as capture:
    model(torch.empty(2, 4, device="meta"))
  assert [(r.path, r.tensor.device.type) for r in capture.records] == [
      (("args", 0), "meta"),
      (("output",), "meta"),
  ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs an MPS device"
)
def test_module_only_capture_on_mps_matches_the_model_values():
  """Tests module-only capture on MPS matches the model values."""
  model = net().to("mps")
  ids = torch.tensor([[1, 2, 3]], device="mps")
  with collect(model, modules=["blocks.0.norm", "head"]) as capture:
    output = model(ids)
  rows = {(r.module, r.edge, r.path): r.tensor for r in capture.records}
  assert set(rows) == {
      ("blocks.0.norm", "in", ("args", 0)),
      ("blocks.0.norm", "out", ("output",)),
      ("head", "in", ("args", 0)),
      ("head", "out", ("output",)),
  }
  assert all(t.device.type == "mps" for t in rows.values())
  assert torch.equal(rows["head", "out", ("output",)], output)
  assert torch.equal(rows["blocks.0.norm", "in", ("args", 0)], model.embed(ids))


def test_tiny_gemma_anchors_match_the_full_capture_including_the_cache():
  """Tests TinyGemma anchors match full capture including the cache."""
  transformers = pytest.importorskip(
      "transformers", reason="install the integration dependency group"
  )
  from transformers import DynamicCache  # pylint: disable=g-import-not-at-top,g-importing-member

  previous_threads = torch.get_num_threads()
  torch.set_num_threads(2)
  try:
    torch.manual_seed(314)
    text = transformers.Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=32,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=32,
        num_kv_shared_layers=2,
        sliding_window=3,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    config = transformers.Gemma4Config(
        text_config=text, vision_config=None, audio_config=None
    )
    config._attn_implementation = "eager"
    model = (
        transformers.Gemma4ForConditionalGeneration(config)
        .to(torch.bfloat16)
        .eval()
    )

    def run(**options):
      cache = DynamicCache(config=model.config)
      ids, mask = torch.tensor([[2, 3, 4, 5, 6]]), torch.ones(
          (1, 5), dtype=torch.long
      )
      with collect(model, **options) as capture:
        for _ in range(3):
          logits = model(
              input_ids=ids,
              attention_mask=mask,
              past_key_values=cache,
              use_cache=True,
          ).logits
          ids = logits[:, -1:].argmax(-1)
          mask = torch.cat((mask, torch.ones_like(ids)), dim=-1)
      return capture

    suffixes = (
        "",
        ".pre_feedforward_layernorm",
        ".per_layer_input_gate",
    )
    anchors = []
    for i in range(4):
      for suffix in suffixes:
        anchors.append(f"model.language_model.layers.{i}{suffix}")
    full, selected = run(), run(modules=anchors)
  finally:
    torch.set_num_threads(previous_threads)
  assert selected.modules == tuple(anchors)
  assert_same_records(
      selected.records,
      [r for r in full.records if r.op is None and r.module in anchors],
  )
  assert {r.forward_id for r in selected.records} == {0, 1, 2}
  layer_inputs = {
      r.path
      for r in selected.records
      if r.module == "model.language_model.layers.0" and r.edge == "in"
  }
  assert ("kwargs", "past_key_values", "layers", 0, "keys") in layer_inputs
  assert ("args", 1) in layer_inputs  # per_layer_input
