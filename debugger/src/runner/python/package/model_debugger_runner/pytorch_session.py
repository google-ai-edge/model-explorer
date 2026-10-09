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

"""One loaded HF model and a verified consumed-token cache prefix.

The cache prefix is tracked per session run.
"""

from collections.abc import Callable, Mapping
import copy
import gc
import os
import pathlib
import platform
import sys
import time
from typing import Any
import uuid
from model_debugger_contracts import errors as contract_errors
from model_debugger_runner import pytorch_worker as worker


def _ensure_capture_package_on_path() -> None:
  """Ensures the first-party ai_edge_debugger_pytorch package is importable.

  In development checkouts, the capture package lives in
  `src/capture/pytorch/package` rather than site-packages. This helper lazily
  resolves and prepends that directory to sys.path before model loading and
  generation, preserving clean package isolation for processes that do not
  require runtime activation capture. An explicit AI_EDGE_CAPTURE_PYTORCH_PATH
  environment variable can override the directory lookup.
  """
  env_path = os.environ.get('AI_EDGE_CAPTURE_PYTORCH_PATH')
  if env_path:
    capture_root = pathlib.Path(env_path).resolve()
  else:
    capture_root = (
        pathlib.Path(__file__).resolve().parents[4] / 'capture/pytorch/package'
    )
  if capture_root.is_dir() and str(capture_root) not in sys.path:
    sys.path.insert(0, str(capture_root))


class PyTorchResidentRuntime:
  """Maintains one loaded Hugging Face causal LM and its KV cache prefix."""

  def __init__(self) -> None:
    """Initializes an unloaded PyTorch resident runtime instance."""
    self.identity = str(uuid.uuid4())
    self.model_load_count = 0
    self.model: Any = None
    self.tokenizer: Any = None
    self.past: Any = None
    self.generator: Any = None
    self.config: dict[str, Any] | None = None
    self.base: dict[str, Any] | None = None
    self.torch: Any = None
    self.backend: str | None = None
    self.device: str | None = None
    self.context_limit: int = 0
    self.messages: list[dict[str, Any]] = []
    self.turn = 0
    self.consumed_ids: Any = None
    self.consumed_mask: Any = None
    self.token_history: list[int] = []
    self.processed_token_count: int | None = None
    self.state = 'closed'

  @staticmethod
  def configuration(request: Mapping[str, Any]) -> dict[str, Any]:
    """Extracts the canonical session and generation configuration dictionary."""
    return copy.deepcopy({
        key: request.get(key)
        for key in (
            'session_id',
            'pytorch_root',
            'run',
            'model',
            'model_sha256',
            'model_files',
            'generation',
        )
    })

  @classmethod
  def model_configuration(cls, request: Mapping[str, Any]) -> dict[str, Any]:
    """Extracts the model-level configuration excluding per-chat sampler settings."""
    value = cls.configuration(request)
    value.pop('generation', None)
    return value

  def reset_chat(self) -> None:
    """Release only this Conversation's cache; model ownership is unchanged."""
    invalidated = self.state == 'invalidated'
    self.past = self.generator = None
    self.consumed_ids = self.consumed_mask = None
    self.messages = []
    self.token_history = []
    self.processed_token_count = None
    self.turn = 0
    self.state = 'invalidated' if invalidated else 'chat_closed'

  def close(self) -> None:
    """Releases the loaded model, tokenizer, KV cache, and device memory."""
    self.model = self.tokenizer = self.past = self.generator = None
    self.config = self.base = None
    self.consumed_ids = self.consumed_mask = None
    self.messages = []
    self.token_history = []
    self.processed_token_count = None
    self.state = 'closed'
    gc.collect()
    torch = self.torch
    if torch is not None:
      if self.backend == 'CUDA' and torch.cuda.is_available():
        torch.cuda.empty_cache()
      elif self.backend == 'MPS' and torch.backends.mps.is_available():
        torch.mps.empty_cache()

  def initialize(
      self, request: Mapping[str, Any], emit: Callable[..., None]
  ) -> None:
    """Loads the Hugging Face model if needed and initializes a fresh chat."""
    if self.state == 'invalidated':
      raise ValueError(
          'PyTorch runtime execution failed; this Session cannot create another'
          ' Chat'
      )
    if request.get('messages'):
      raise ValueError('A new Chat starts with empty history')
    if self.model is not None:
      if self.state != 'chat_closed':
        raise ValueError('End the previous Chat before initializing another')
      old = dict(self.config or {})
      old.pop('generation', None)
      if old != self.model_configuration(request):
        raise ValueError('Model configuration changed; create a new Session')
      self._initialize_chat(request, emit)
      return
    sampler = worker._settings(request)
    run = request['run']
    backend = run.get('backend') or 'CPU'
    if backend not in ('CPU', 'MPS', 'CUDA'):
      raise ValueError('PyTorch backend must be CPU, MPS or CUDA')
    precision = run.get('precision') or 'default'
    if precision not in ('default', 'float32', 'float16', 'bfloat16'):
      raise ValueError('Unsupported PyTorch precision')
    root = pathlib.Path(request['pytorch_root']).resolve()
    if str(root) not in sys.path:
      sys.path.insert(0, str(root))
    _ensure_capture_package_on_path()
    import torch
    import transformers
    import ai_edge_debugger_pytorch
    from model_debugger_contracts import model_identity

    self.torch, self.backend, self.device = torch, backend, backend.lower()
    model_path = pathlib.Path(request['model']).resolve()
    model_identity.verify_model(
        model_path, request['model_files'], request['model_sha256']
    )
    worker._cancelled(request['output'])
    if backend == 'MPS' and not torch.backends.mps.is_available():
      raise ValueError('PyTorch MPS backend is unavailable')
    if backend == 'CUDA' and not torch.cuda.is_available():
      raise ValueError('PyTorch CUDA backend is unavailable')
    if run.get('cpuThreads') not in (None, ''):
      if (
          backend != 'CPU'
          or isinstance(run['cpuThreads'], bool)
          or not str(run['cpuThreads']).isdigit()
          or int(run['cpuThreads']) < 1
      ):
        raise ValueError('cpuThreads requires a positive CPU thread count')
      torch.set_num_threads(int(run['cpuThreads']))
    emit('loading', backend=backend)
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    worker._cancelled(request['output'])
    self.model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype='auto'
            if precision == 'default'
            else getattr(torch, precision),
        )
        .eval()
        .to(self.device)
    )
    self.model_load_count += 1
    worker._cancelled(request['output'])
    if getattr(self.model.config, 'is_encoder_decoder', False):
      raise ValueError('Only text causal language models are supported')
    self.context_limit = worker._context_limit(self.model, self.tokenizer, run)
    with ai_edge_debugger_pytorch.CaptureRun(
        self.model, granularity='sublayer', layers='0', outside=False
    ):
      pass
    self.config = self.configuration(request)
    self.base = dict(
        runtime='PyTorch',
        model_sha256=request['model_sha256'],
        model_files=request['model_files'],
        backend_requested=backend,
        backend_effective=backend,
        requested=copy.deepcopy(run),
        effective=dict(
            backend=backend,
            precision=str(next(self.model.parameters()).dtype).removeprefix(
                'torch.'
            ),
            contextLength=self.context_limit,
            sampler=sampler,
            use_cache=True,
            torch_compile=False,
            attention_implementation=getattr(
                self.model.config, '_attn_implementation', None
            ),
        ),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        runtime_root=str(root),
        runtime_instance=self.identity,
        worker_pid=os.getpid(),
        model_load_count=self.model_load_count,
        debug_enabled=True,
        verified_taps=False,
        capture_scope='all_observed_forwards',
        published_scope='all_observed_forwards',
        raw_capture=None,
        decode_capture=False,
        kv_capture=False,
        captured_tensors=0,
    )
    self._initialize_chat(request, emit)

  def _initialize_chat(
      self, request: Mapping[str, Any], emit: Callable[..., None]
  ) -> None:
    sampler = worker._settings(request)
    self.reset_chat()
    self.config = self.configuration(request)
    if self.base is not None:
      self.base['effective']['sampler'] = sampler
    self.torch.manual_seed(sampler['seed'])
    self.generator = self.torch.Generator(device='cpu').manual_seed(
        sampler['seed']
    )
    self.state = 'ready'
    emit('initialized', debug_enabled=True, verified_taps=False)

  def preflight(
      self, request: Mapping[str, Any]
  ) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Tokenizes and validates context length before running generation."""
    if (
        self.state != 'ready'
        or self.model is None
        or self.configuration(request) != self.config
    ):
      raise ValueError(
          'PyTorch session is unavailable or configuration changed. Initialize'
          ' again.'
      )
    messages = worker._messages({**request, 'messages': self.messages})
    encoded, serialized, serialization = worker._tokenize(
        self.tokenizer, messages
    )
    ids = encoded.get('input_ids')
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
      raise contract_errors.InputRejected(
          'Tokenizer must produce one nonempty input sequence'
      )
    sampler = worker._settings(request)
    count = ids.shape[1]
    if count + sampler['max_output_tokens'] > self.context_limit:
      raise contract_errors.InputRejected(
          f'Context requires {count} input tokens plus'
          f' {sampler["max_output_tokens"]} output tokens; limit is'
          f' {self.context_limit}; input was not truncated'
      )
    return encoded, serialized, serialization

  def execute(
      self, request: Mapping[str, Any], emit: Callable[..., None]
  ) -> dict[str, Any]:
    """Executes a reset, initialize, preflight, or generate operation."""
    started = time.monotonic()
    operation = request.get('operation')
    model_was_loaded = self.model is not None
    try:
      worker._cancelled(request['output'])
      if operation == 'reset':
        self.reset_chat()
        return dict(
            runtime_instance=self.identity,
            model_load_count=self.model_load_count,
            chat_closed=True,
        )
      if operation == 'initialize':
        self.initialize(request, emit)
        return dict(
            self.base or {},
            turn_sequence=self.turn,
            token_count=0,
            processed_token_count=None,
            elapsed_seconds_with_capture=time.monotonic() - started,
        )
      if operation == 'preflight':
        encoded, _, _ = self.preflight(request)
        return dict(
            context_token_count=int(encoded['input_ids'].shape[1]),
            context_limit=self.context_limit,
            max_output_tokens=worker._settings(request)['max_output_tokens'],
        )
      if operation != 'generate':
        raise ValueError('Unsupported PyTorch runtime operation')
      if (
          self.state != 'ready'
          or self.model is None
          or self.configuration(request) != self.config
      ):
        raise ValueError(
            'PyTorch session is unavailable or configuration changed.'
            ' Initialize again.'
        )
      if (
          'messages' in request
          and worker._history(request['messages']) != self.messages
      ):
        raise ValueError(
            'Saved conversation history differs from resident history.'
            ' Initialize again.'
        )
      if 'turn' in request and request['turn'] != self.turn + 1:
        raise ValueError(
            'PyTorch turn differs from resident history. Initialize again.'
        )
      prepared = self.preflight(request)
      self.state = 'running'
      result = self.generate(request, emit, prepared)
      self.state = 'ready'
      result['elapsed_seconds_with_capture'] = time.monotonic() - started
      return result
    except contract_errors.InputRejected:
      raise
    except InterruptedError:
      self.reset_chat()
      raise
    except BaseException:
      self.reset_chat()
      if operation != 'initialize' or not model_was_loaded:
        self.state = 'invalidated'
      raise

  def _reuse_prefix(
      self, ids: Any, mask: Any, extra: Mapping[str, Any]
  ) -> tuple[int, str]:
    if self.past is None or self.consumed_ids is None:
      return (
          0,
          'history_replay_after_initialize' if self.messages else 'empty_cache',
      )
    length = self.consumed_ids.shape[1]
    if extra:
      return 0, 'additional_tokenizer_inputs_require_replay'
    try:
      actual = self.past.get_seq_length()
    except (AttributeError, TypeError, ValueError):
      return 0, 'cache_length_unavailable'
    if actual != length or self.processed_token_count != length:
      return 0, 'cache_length_changed'
    if ids.shape[1] <= length:
      return 0, 'serialized_input_has_no_uncached_suffix'
    if not self.torch.equal(ids[:, :length].cpu(), self.consumed_ids):
      return 0, 'serialized_token_prefix_changed'
    if not self.torch.equal(mask[:, :length].cpu(), self.consumed_mask):
      return 0, 'attention_mask_prefix_changed'
    return length, 'exact_consumed_prefix'

  @staticmethod
  def _prefix(ids: Any, mask: Any, length: int) -> dict[str, Any]:
    return dict(
        processed_token_count=length,
        token_ids=ids[0, :length].detach().cpu().tolist(),
        attention_mask=mask[0, :length].detach().cpu().tolist(),
    )

  def _forward_proof(
      self,
      prepared: Mapping[str, Any],
      prefix: Mapping[str, Any],
      tokenizer: Mapping[str, Any],
      serialization: Mapping[str, Any],
  ) -> dict[str, Any]:
    tensors = {
        key: dict(
            shape=list(value.shape),
            dtype=str(value.dtype).removeprefix('torch.'),
            values=value.detach().cpu().tolist(),
        )
        for key, value in prepared.items()
        if self.torch.is_tensor(value)
    }
    proof = dict(
        scope='same_logical_context',
        tensors=tensors,
        cache_context=prefix,
        tokenizer=copy.deepcopy(tokenizer),
        serialization=copy.deepcopy(serialization),
    )
    return dict(
        input_identity=worker._digest(proof),
        input_proof=proof,
        comparison_basis='logical_context',
    )

  @staticmethod
  def _verify_consumed_cache(output: Any, expected_count: int) -> None:
    cache = getattr(output, 'past_key_values', None)
    if cache is None:
      raise ValueError('Model did not return a KV cache for eager generation')
    try:
      actual = cache.get_seq_length()
    except (AttributeError, TypeError, ValueError) as error:
      raise ValueError(
          'Cannot verify the actual consumed-token cache length'
      ) from error
    if actual != expected_count:
      raise ValueError(
          f'Actual KV cache length {actual} differs from consumed token count'
          f' {expected_count}'
      )

  def generate(
      self,
      request: Mapping[str, Any],
      emit: Callable[..., None],
      prepared_input: tuple[dict[str, Any], str, dict[str, Any]],
  ) -> dict[str, Any]:
    """Runs eager generation with CaptureRun hooks and exports debug artifacts."""
    _ensure_capture_package_on_path()
    import transformers
    from safetensors import torch as safetensors_torch
    import ai_edge_debugger_pytorch
    from model_debugger_runner import pytorch_capture

    torch, model, tokenizer = self.torch, self.model, self.tokenizer
    directory = pathlib.Path(request['output']).resolve()
    device = self.device
    sampler, generator = worker._settings(request), self.generator
    result = copy.deepcopy(self.base or {})
    result['effective']['sampler'] = sampler
    started = time.monotonic()
    encoded, serialized, serialization = prepared_input
    ids = encoded.get('input_ids')
    encoded = {key: value.to(device) for key, value in encoded.items()}
    ids = encoded.pop('input_ids')
    mask = encoded.pop('attention_mask', torch.ones_like(ids))
    prefix_length, reuse_reason = self._reuse_prefix(ids, mask, encoded)
    if not prefix_length:
      self.past = None
    past = self.past if prefix_length else None
    prefix = self._prefix(ids, mask, prefix_length)
    prompt_length = ids.shape[1]
    cache_position = torch.arange(prefix_length, ids.shape[1], device=device)
    # HF 5 expects sliced inputs; HF 4 also accepts this exact uncached suffix.
    prepared = model.prepare_inputs_for_generation(
        ids[:, prefix_length:],
        past_key_values=past,
        attention_mask=mask,
        cache_position=cache_position,
        use_cache=True,
        **encoded,
    )
    input_tensors = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in prepared.items()
        if torch.is_tensor(value)
    }
    proof = dict(
        tensors={
            key: dict(
                shape=list(value.shape),
                dtype=str(value.dtype).removeprefix('torch.'),
                values=value.tolist(),
            )
            for key, value in input_tensors.items()
        },
        tokenizer=dict(
            tokenizer_class=type(tokenizer).__name__,
            vocab_sha256=worker._digest(tokenizer.get_vocab()),
            special_tokens_sha256=worker._digest(tokenizer.special_tokens_map),
        ),
        serialization=serialization,
    )
    if hasattr(tokenizer, 'backend_tokenizer'):
      proof['tokenizer']['backend_sha256'] = worker._digest(
          tokenizer.backend_tokenizer.to_str()
      )
    first_proof = self._forward_proof(
        prepared, prefix, proof['tokenizer'], serialization
    )
    proof = first_proof['input_proof']
    input_identity = first_proof['input_identity']
    raw = directory / 'raw'
    raw.mkdir()
    safetensors_torch.save_file(input_tensors, raw / 'inputs.safetensors')
    result.update(
        input=request['prompt'],
        messages=copy.deepcopy(self.messages),
        serialized_input=serialized,
        input_identity=input_identity,
        input_proof=proof,
        input_tensor_path='raw/inputs.safetensors',
        tokens=[],
    )
    generated = []
    deltas = []

    class Stream(transformers.TextStreamer):

      def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if text:
          deltas.append(text)
          emit('delta', text=text)

    streamer = Stream(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    context = dict(
        phase='prefill', step=0, pos_offset=prefix_length, turn=self.turn + 1
    )
    worker._cancelled(directory)
    # The historical directory name stays; this scope covers every real forward.
    with ai_edge_debugger_pytorch.CaptureRun(
        model,
        granularity='sublayer',
        layers='0',
        outside=False,
        out_dir=raw / 'prefill',
    ) as capture:
      capture.run['forward_input_proofs'] = [dict(forward_id=0, **first_proof)]
      try:
        with capture.forward_context(**context):
          output = model(**prepared)
        self._verify_consumed_cache(output, ids.shape[1])
        source_forward_id = 0
        for step in range(sampler['max_output_tokens']):
          worker._cancelled(directory)
          token = worker._choose_token(torch, output.logits, sampler, generator)
          token_text = tokenizer.decode(
              [token],
              skip_special_tokens=False,
              clean_up_tokenization_spaces=False,
          )
          capture.observe_tokens(
              [token], forward_id=source_forward_id, texts=[token_text]
          )
          generated.append(token)
          result['tokens'].append(
              dict(
                  id=token,
                  text=token_text,
                  step=step,
                  source_forward_id=source_forward_id,
                  **(
                      {'kind': 'special'}
                      if token
                      in (getattr(tokenizer, 'all_special_ids', None) or [])
                      else {}
                  ),
              )
          )
          streamer.put(torch.tensor([token]))
          if token in eos or step + 1 == sampler['max_output_tokens']:
            break
          past = getattr(output, 'past_key_values', None)
          if past is None:
            raise ValueError(
                'Model did not return a KV cache for eager generation'
            )
          ids = torch.cat(
              (ids, torch.tensor([[token]], device=device, dtype=ids.dtype)),
              dim=1,
          )
          mask = torch.cat(
              (mask, torch.ones((1, 1), device=device, dtype=mask.dtype)), dim=1
          )
          if 'token_type_ids' in encoded:
            types = encoded['token_type_ids']
            encoded['token_type_ids'] = torch.cat((types, types[:, -1:]), dim=1)
          # Models prepare their own cache-aware positional inputs;
          # no forwards are replaced.
          position = ids.shape[1] - 1
          prepared = model.prepare_inputs_for_generation(
              ids[:, -1:],
              past_key_values=past,
              attention_mask=mask,
              cache_position=torch.tensor([position], device=device),
              use_cache=True,
              **encoded,
          )
          worker._cancelled(directory)
          next_proof = self._forward_proof(
              prepared,
              self._prefix(ids, mask, position),
              proof['tokenizer'],
              serialization,
          )
          capture.run['forward_input_proofs'].append(
              dict(forward_id=source_forward_id + 1, **next_proof)
          )
          with capture.forward_context(
              **dict(context, phase='decode', step=step, pos_offset=position)
          ):
            output = model(**prepared)
          self._verify_consumed_cache(output, ids.shape[1])
          source_forward_id += 1
        streamer.end()
        worker._cancelled(directory)
        stop_reason = 'eos' if generated[-1] in eos else 'max_output_tokens'
        capture.finish_generation(stop_reason)
      except BaseException as error:
        reason = (
            'cancelled'
            if isinstance(error, (InterruptedError, KeyboardInterrupt))
            else 'error'
        )
        try:
          capture.finish_generation(reason)
        except BaseException as finish_error:
          error.add_note(f'Capture finalization also failed: {finish_error}')
        raise
    snapshots = capture.kv_snapshots
    result.update(
        capture_environment=capture.run,
        raw_capture=dict(
            scope='whole_generation',
            directory='raw/prefill',
            forward_index='raw/prefill/forward_index.json',
            kv_index='raw/prefill/kv_index.json',
            tokens='raw/prefill/tokens.jsonl',
            generation='raw/prefill/generation.json',
            forward_count=source_forward_id + 1,
            kv_snapshot_count=len(snapshots),
            token_record_count=len(capture.token_records),
            kv_snapshots=[
                {
                    key: snapshot[key]
                    for key in (
                        'snapshot_id',
                        'moment',
                        'state',
                        'storage_complete',
                    )
                }
                for snapshot in snapshots
            ],
        ),
    )
    generation = capture.generation
    self.past = getattr(output, 'past_key_values', None)
    self.processed_token_count = generation['processed_token_count']
    self.consumed_ids = (
        ids.detach().cpu().clone()
        if self.past is not None and self.processed_token_count == ids.shape[1]
        else None
    )
    self.consumed_mask = (
        mask.detach().cpu().clone() if self.consumed_ids is not None else None
    )
    self.token_history = ids[0].detach().cpu().tolist() + [generated[-1]]
    self.messages.extend([
        dict(role='user', content=request['prompt']),
        dict(role='assistant', content=''.join(deltas)),
    ])
    self.turn += 1
    result.update(
        turn_sequence=self.turn,
        token_count_before=prefix_length,
        processed_token_count=self.processed_token_count,
        context_token_ids=list(self.token_history),
        pending_token_ids=list(generation['pending_token_ids']),
        cache_reuse=dict(
            reused=prefix_length > 0,
            reason=reuse_reason,
            reused_token_count=prefix_length,
            input_token_count=prompt_length - prefix_length,
            prompt_token_count=prompt_length,
        ),
        decode_capture=source_forward_id > 0,
        kv_capture=any(s['state'] == 'available' for s in snapshots),
    )
    result.update(
        output=''.join(deltas),
        token_count=len(generated),
        stop_reason=stop_reason,
        generation_status='completed',
        elapsed_seconds_with_capture=time.monotonic() - started,
    )
    # Full successful output and resident context are committed before Debug
    # export. Export failure must neither erase text nor rewind the model.
    worker._write_result(directory / 'generation-result.json', result)
    emit('generation_completed', result=result)
    try:
      index = pytorch_capture.export_capture(capture, directory, input_identity)
      result.update(
          topology=index['topology'],
          captured_tensors=len(index['tensors']),
          debug_data=dict(status='available', error=''),
      )
    except Exception as error:
      result.update(
          captured_tensors=None,
          debug_data=dict(status='unavailable', error=str(error)),
      )
    return result
