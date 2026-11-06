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

"""The anchors and modules collect() captures of a tiny Gemma 4.

Against plain hooks of our own, in both of the observer's modes.
"""

# pylint: disable=redefined-outer-name

from typing import Any

from model_debugger.runner_pytorch.llm import anchor_observer
from model_debugger.runner_pytorch.testing import test_main
from model_debugger.runner_pytorch.testing import tiny_gemma4
import pytest
import torch
import transformers

_PROMPT = [tiny_gemma4.BOS, 10, 11, 12, 13, 14]


@pytest.fixture(scope='module')
def model(tmp_path_factory) -> torch.nn.Module:
  """The tiny model, on the CPU."""
  directory = tiny_gemma4.write(tmp_path_factory.mktemp('model'))
  return transformers.Gemma4ForConditionalGeneration.from_pretrained(
      directory, attn_implementation='eager'
  ).eval()


def _passes(model, run):
  """A prefill of the prompt but its last token, then two decode steps.

  Args:
      model: The model.
      run: Called as run(forward, cache, signature, step) for each pass,
        `forward` the pass itself; returns its logits.

  Returns:
      Each pass's logits, and the cache.
  """
  cache = transformers.DynamicCache(config=model.config)

  def forward(ids, first):
    def call():
      return model(
          input_ids=torch.tensor([ids]),
          position_ids=torch.arange(first, first + len(ids))[None],
          past_key_values=cache,
          use_cache=True,
          logits_to_keep=1,
      ).logits

    return call

  n = len(_PROMPT)
  logits = [run(forward(_PROMPT[:-1], 0), cache, 'prefill', n)]
  token = _PROMPT[-1]
  for position in (n - 1, n):
    logits.append(
        run(forward([token], position), cache, 'decode', position + 1)
    )
    token = int(logits[-1][0, -1].argmax())
  return logits, cache


# The modules asked for beside the anchors: what one returns, what one
# reads.
_MODULES = (
    anchor_observer.Module(
        'model.language_model.layers.1.self_attn.q_proj', 'out'
    ),
    anchor_observer.Module('model.language_model.layers.0.mlp.down_proj', 'in'),
)


def _hooked(model) -> tuple[list[Any], dict[Any, Any]]:
  """The same passes, the anchors and _MODULES taken by hooks of our own."""
  taken = {}
  handles = []
  layers = model.model.language_model.layers
  current = {}

  def keep(layer, anchor, value):
    taken[current['pass'], layer, anchor] = value.detach().clone()

  modules = dict(model.named_modules())
  for asked in _MODULES:
    if asked.edge == 'out':
      handles.append(
          modules[asked.module].register_forward_hook(
              lambda m, args, out, a=asked: keep(None, a, out)
          )
      )
    else:
      handles.append(
          modules[asked.module].register_forward_pre_hook(
              lambda m, args, a=asked: keep(None, a, args[0])
          )
      )

  for index, layer in enumerate(layers):
    handles += [
        layer.register_forward_pre_hook(
            lambda m, args, i=index: keep(i, 'input', args[0])
        ),
        layer.pre_feedforward_layernorm.register_forward_pre_hook(
            lambda m, args, i=index: keep(i, 'intermediate.1', args[0])
        ),
        layer.per_layer_input_gate.register_forward_pre_hook(
            lambda m, args, i=index: keep(i, 'intermediate.2', args[0])
        ),
        layer.register_forward_hook(
            lambda m, args, out, i=index: keep(i, 'output', out)
        ),
    ]

  def run(call, cache, signature, step):
    del cache  # Unused.
    current['pass'] = (signature, step)
    return call()

  try:
    with torch.no_grad():
      logits, _ = _passes(model, run)
  finally:
    for handle in handles:
      handle.remove()
  return logits, taken


@pytest.mark.parametrize('full', [False, True])
def test_the_anchors_are_the_layers_tensors_there(model, full):
  """Each anchor and module is what our own hooks see there, bit for bit."""
  expected_logits, expected = _hooked(model)
  observer = anchor_observer.Observer(model, 1 << 30, _MODULES, full)
  assert observer.mode == (
      anchor_observer.FULL if full else anchor_observer.MODULES
  )
  with torch.no_grad():
    logits, _ = _passes(model, observer.forward)
  found = observer.finish()
  assert not observer.reason
  assert observer.snapshot_bytes > 0
  for got, want in zip(logits, expected_logits, strict=True):
    assert torch.equal(got, want)
  layers = len(tiny_gemma4.LAYER_TYPES)
  assert len(found) == 3 * (layers * 4 + len(_MODULES))
  asked = [t for t in found if not t.anchor]
  found = [t for t in found if t.anchor]
  for tensor in asked:
    module = anchor_observer.Module(tensor.module, tensor.edge)
    want = expected[(tensor.signature, tensor.step), None, module]
    assert torch.equal(tensor.tensor, want), tensor
    assert tensor.layer is None
    assert tensor.path == ('output' if tensor.edge == 'out' else 'args.0')
    assert tensor.key == f'post_module_{tensor.module}_{tensor.edge}'
  for tensor in found:
    want = expected[
        (tensor.signature, tensor.step), tensor.layer, tensor.anchor
    ]
    assert torch.equal(tensor.tensor, want), tensor
    rows = len(_PROMPT) - 1 if tensor.signature == 'prefill' else 1
    assert tuple(tensor.tensor.shape) == (1, rows, 64)
    assert tensor.key == f'post_anchor_{tensor.layer}_{tensor.anchor}'
  first = found[0]
  assert (first.signature, first.layer, first.anchor) == (
      'prefill',
      0,
      'input',
  )
  assert first.module == 'model.language_model.layers.0'
  assert (first.edge, first.path) == ('in', 'args.0')
  gate = next(t for t in found if t.anchor == 'intermediate.2')
  assert gate.module == 'model.language_model.layers.0.per_layer_input_gate'
  # A layer's output is what the next one reads.
  by_key = {(t.step, t.layer, t.anchor): t.tensor for t in found}
  for (step, layer, anchor), value in by_key.items():
    if anchor == 'output' and layer + 1 < layers:
      assert torch.equal(value, by_key[step, layer + 1, 'input'])


def test_a_module_only_capture_exports_no_graph(model):
  """Only a full capture has the operators the graph is drawn from."""
  observer = anchor_observer.Observer(model, 1 << 30)
  with torch.no_grad():
    _passes(model, observer.forward)
  assert observer.finish({'prefill', 'decode'})
  assert observer.graph is None and not observer.graph_reason


def test_a_module_only_capture_snapshots_less(model):
  """The selected modules alone: fewer bytes than every module's."""
  sizes = {}
  for full in (False, True):
    observer = anchor_observer.Observer(model, 1 << 30, full=full)
    with torch.no_grad():
      _passes(model, observer.forward)
    assert observer.finish()
    sizes[full] = observer.snapshot_bytes
  assert 0 < sizes[False] < sizes[True]


def test_the_modules_asked_for_are_checked(model):
  """parse_modules() takes the model's modules, each with an edge."""
  found = anchor_observer.parse_modules(
      model,
      [{'module': m.module, 'edge': m.edge} for m in _MODULES + _MODULES],
  )
  assert found == _MODULES
  with pytest.raises(ValueError, match='has no module layers.9'):
    anchor_observer.parse_modules(
        model, [{'module': 'layers.9', 'edge': 'out'}]
    )
  with pytest.raises(ValueError, match='"in" or "out"'):
    anchor_observer.parse_modules(
        model, [{'module': _MODULES[0].module, 'edge': 'up'}]
    )
  with pytest.raises(ValueError, match='must be a list'):
    anchor_observer.parse_modules(model, {'module': 'x'})


def test_the_execution_graph_is_the_collectors_of_each_first_pass(model):
  """The collector's graph of the first pass of each signature asked for."""
  observer = anchor_observer.Observer(model, 1 << 30, full=True)
  with torch.no_grad():
    _passes(model, observer.forward)
  assert observer.finish({'prefill', 'decode'})
  graph = observer.graph
  assert not observer.graph_reason
  assert graph['format'] == 1 and graph['hide_unused']
  assert [g['id'] for g in graph['graphs']] == ['prefill', 'decode']
  n = len(_PROMPT)
  assert graph['steps'] == {'prefill': n, 'decode': n}
  modules = graph['modules']
  layer = 'model.language_model.layers.0'
  assert modules[layer] == 'Gemma4TextDecoderLayer'
  assert modules[f'{layer}.input_layernorm'] == 'Gemma4RMSNorm'
  assert modules[f'{layer}.self_attn.q_proj'] == 'Linear'
  assert modules[f'{layer}.mlp.act_fn'] == 'GELUTanh'
  decode = graph['graphs'][1]['nodes']
  namespace = layer.replace('.', '/')
  in_layer = [n for n in decode if n['namespace'] == namespace]
  # The layer scales its output in place, last: the output's producer.
  assert [n['label'] for n in in_layer].count('aten.mul_') == 1
  assert in_layer[-1]['label'] == 'aten.mul_'
  # A KV cache the decode pass reads is an input of its graph.
  assert any(n['id'] == 'kwargs.past_key_values.layers.0.keys' for n in decode)


def test_no_graph_of_a_signature_the_turn_did_not_run(model):
  """None for signatures no pass ran; nothing asked, nothing exported."""
  observer = anchor_observer.Observer(model, 1 << 30, full=True)
  with torch.no_grad():
    _passes(model, observer.forward)
  assert observer.finish({'verify'})
  assert observer.graph is None and not observer.graph_reason
  plain = anchor_observer.Observer(model, 1 << 30, full=True)
  with torch.no_grad():
    _passes(model, plain.forward)
  assert plain.finish()
  assert plain.graph is None


@pytest.mark.parametrize('full', [False, True])
def test_a_budget_exceeded_leaves_the_turn_as_it_would_be(model, full):
  """The pass that exceeds the budget runs again, from the cache before it."""
  expected_logits, _ = _hooked(model)
  with torch.no_grad():
    _, clean_cache = _passes(model, lambda call, *_: call())
  whole = anchor_observer.Observer(model, 1 << 30, full=full)
  with torch.no_grad():
    _passes(model, whole.forward)
  assert whole.finish()
  # A byte short of the turn: the last snapshot of the last decode step,
  # its output, fails, after the step extended the cache.
  observer = anchor_observer.Observer(
      model, whole.snapshot_bytes - 1, full=full
  )
  with torch.no_grad():
    logits, cache = _passes(model, observer.forward)
  assert not observer.finish({'prefill', 'decode'})
  assert observer.graph is None
  assert observer.reason.startswith(
      f'collect() failed at decode step {len(_PROMPT) + 1}: '
  )
  assert 'MemoryError: Snapshot memory limit exceeded' in observer.reason
  for got, want in zip(logits, expected_logits, strict=True):
    assert torch.equal(got, want)
  for got, want in zip(cache.layers, clean_cache.layers, strict=True):
    assert torch.equal(got.keys, want.keys)
    assert torch.equal(got.values, want.values)
  # The hooks are gone: the model can be observed again.
  again = anchor_observer.Observer(model, 1 << 30)
  with torch.no_grad():
    _passes(model, again.forward)
  assert again.finish() and not again.reason


def _on_mps(model):
  """The model on MPS, or a skip without it."""
  if not torch.backends.mps.is_available():
    pytest.skip('MPS is not available here.')
  return model.to('mps')


def _mps_prefill(model, observer):
  """One prefill pass on MPS, observed; its logits and cache."""
  cache = transformers.DynamicCache(config=model.config)
  with torch.no_grad():
    logits = observer.forward(
        lambda: (
            model(
                input_ids=torch.tensor([_PROMPT], device='mps'),
                past_key_values=cache,
                use_cache=True,
            ).logits
        ),
        cache,
        'prefill',
        len(_PROMPT) + 1,
    )
  return logits, cache


def test_a_module_only_capture_observes_on_mps(model):
  """The selected modules are snapshotted on MPS, where the model runs."""
  on_gpu = _on_mps(model)
  try:
    observer = anchor_observer.Observer(on_gpu, 1 << 30, _MODULES)
    logits, _ = _mps_prefill(on_gpu, observer)
    found = observer.finish()
    assert not observer.reason
    assert len(found) == len(tiny_gemma4.LAYER_TYPES) * 4 + len(_MODULES)
    assert {t.tensor.device.type for t in found} == {'mps'}
    assert logits.device.type == 'mps'
  finally:
    model.to('cpu')


def test_a_full_capture_on_no_cpu_is_not_observed(model):
  """A full collect() refuses what is not on the CPU; the turn runs on."""
  on_gpu = _on_mps(model)
  try:
    observer = anchor_observer.Observer(on_gpu, 1 << 30, full=True)
    logits, cache = _mps_prefill(on_gpu, observer)
    assert logits.device.type == 'mps'
    assert not observer.finish()
    assert observer.reason == (
        'collect() did not start: ValueError: Snapshots support CPU '
        'tensors only'
    )
    assert cache.layers[2].keys.shape[-2] == len(_PROMPT)
  finally:
    model.to('cpu')


if __name__ == '__main__':
  test_main.main()
