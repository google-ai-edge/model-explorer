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

"""HF eager worker protocol and generation helpers shared by resident runs."""

import hashlib
import json
import os
from pathlib import Path
import sys
from model_debugger_contracts.errors import InputRejected


def _digest(value):
  return hashlib.sha256(
      json.dumps(
          value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
      ).encode()
  ).hexdigest()


def _cancelled(directory):
  if (Path(directory).parent / 'cancel').exists():
    raise InterruptedError('Cancelled')


def _settings(request):
  generation = request.get('generation') or {}
  if not isinstance(generation, dict):
    raise ValueError('Invalid generation settings')
  allowed = {
      'temperature',
      'topK',
      'topP',
      'seed',
      'systemPrompt',
      'maxOutputTokens',
      'thinking',
      'thinkingBudget',
  }
  if set(generation) - allowed:
    raise ValueError(
        'Unsupported PyTorch generation settings: '
        + ', '.join(sorted(set(generation) - allowed))
    )
  if (
      generation.get('thinking') not in (None, '', 'default')
      or 'thinkingBudget' in generation
  ):
    raise ValueError('PyTorch thinking controls are not supported')
  config = dict(
      temperature=generation.get('temperature', 0.0),
      top_k=generation.get('topK', 1),
      top_p=generation.get('topP', 1.0),
      seed=generation.get('seed', 0),
      max_output_tokens=generation.get(
          'maxOutputTokens', request.get('max_output_tokens', 32)
      ),
  )
  import math

  for key, low, high, integer in [
      ('temperature', 0, 100, False),
      ('top_k', 1, 2147483647, True),
      ('top_p', 0, 1, False),
      ('seed', 0, 4294967295, True),
      ('max_output_tokens', 1, 256, True),
  ]:
    value = config[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
        or (integer and int(value) != value)
    ):
      raise ValueError(f'Invalid PyTorch {key}')
    if integer:
      config[key] = int(value)
  config['do_sample'] = config['temperature'] > 0 and config['top_k'] != 1
  return config


def _history(messages):
  result = []
  for message in messages or []:
    if not isinstance(message, dict) or message.get('role') not in (
        'system',
        'user',
        'assistant',
    ):
      raise ValueError('Only text system/user/assistant messages are supported')
    content = message.get('content')
    if isinstance(content, list):
      if any(
          not isinstance(part, dict)
          or part.get('type') != 'text'
          or not isinstance(part.get('text'), str)
          for part in content
      ):
        raise ValueError('Only text messages are supported')
      content = ''.join(part['text'] for part in content)
    if not isinstance(content, str):
      raise ValueError('Only text messages are supported')
    result.append(dict(role=message['role'], content=content))
  return result


def _messages(request):
  result = []
  system = (request.get('generation') or {}).get('systemPrompt', '')
  if not isinstance(system, str) or len(system) > 16000:
    raise ValueError('Invalid system prompt')
  if system:
    result.append(dict(role='system', content=system))
  result.extend(_history(request.get('messages')))
  prompt = request.get('prompt', '')
  if not isinstance(prompt, str) or not prompt.strip():
    raise InputRejected('Enter a non-empty prompt')
  result.append(dict(role='user', content=prompt))
  return result


def _tokenize(tokenizer, messages):
  if tokenizer.chat_template:
    template = tokenizer.get_chat_template()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        text, return_tensors='pt', add_special_tokens=False, truncation=False
    )
    serialization = dict(
        kind='tokenizer_chat_template', template_sha256=_digest(template)
    )
  else:
    text = (
        ''.join(
            message['role'] + ': ' + message['content'] + '\n'
            for message in messages
        )
        + 'assistant: '
    )
    encoded = tokenizer(
        text, return_tensors='pt', add_special_tokens=True, truncation=False
    )
    serialization = dict(
        kind='plain_role_lines_v1', format='{role}: {content}\\n...assistant: '
    )
  return dict(encoded), text, serialization


def _context_limit(model, tokenizer, run):
  config = (
      model.config.get_text_config()
      if hasattr(model.config, 'get_text_config')
      else model.config
  )
  limits = [
      getattr(config, key, None)
      for key in (
          'max_position_embeddings',
          'n_positions',
          'max_sequence_length',
      )
  ]
  limits.append(getattr(tokenizer, 'model_max_length', None))
  limits = [
      value for value in limits if isinstance(value, int) and 0 < value < 10**9
  ]
  explicit = run.get('contextLength')
  if explicit not in (None, ''):
    if (
        isinstance(explicit, bool)
        or str(explicit).isdigit() is False
        or int(explicit) < 1
    ):
      raise ValueError('Invalid PyTorch contextLength')
    limits.append(int(explicit))
  if not limits:
    raise ValueError(
        'Model context length is unknown; set contextLength explicitly'
    )
  return min(limits)


def _choose_token(torch, logits, sampler, generator):
  scores = logits[0, -1].detach().float().cpu()
  if not torch.isfinite(scores).all():
    raise ValueError('Model produced non-finite logits')
  if not sampler['do_sample']:
    return int(scores.argmax())
  scores = scores / sampler['temperature']
  k = min(sampler['top_k'], scores.numel())
  scores[scores < torch.topk(scores, k).values[-1]] = -float('inf')
  if sampler['top_p'] < 1:
    sorted_scores, indices = torch.sort(scores, descending=True)
    remove = torch.softmax(sorted_scores, dim=-1).cumsum(-1) > sampler['top_p']
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    scores[indices[remove]] = -float('inf')
  return int(
      torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=generator)
  )


def execute(request, emit):
  """Run one disposable request through the same resident implementation."""
  _cancelled(Path(request['output']).resolve())
  from .pytorch_session import PyTorchResidentRuntime

  runtime = PyTorchResidentRuntime()
  try:
    if request.get('operation') == 'generate':
      runtime.execute({**request, 'operation': 'initialize'}, emit)
    return runtime.execute(request, emit)
  finally:
    runtime.close()


def _write_result(path, result):
  temporary = path.with_suffix('.tmp')
  temporary.write_text(json.dumps(result, allow_nan=False))
  os.replace(temporary, path)


def serve_request(runtime, request):
  directory = Path(request['output']).resolve()
  directory.mkdir(exist_ok=False)
  events = directory / 'events.jsonl'

  def emit(kind, **data):
    with events.open('a') as file:
      file.write(json.dumps(dict(type=kind, **data)) + '\n')
      file.flush()

  try:
    result = runtime.execute(request, emit)
    _write_result(directory / 'result.json', result)
    emit('finished')
  except BaseException as error:
    status = (
        'rejected'
        if isinstance(error, InputRejected)
        else 'stopped'
        if isinstance(error, InterruptedError)
        else 'failed'
    )
    _write_result(
        directory / 'terminal.json',
        dict(generation_status=status, error=str(error)),
    )
    emit(
        'error',
        error=f'{type(error).__name__}: {error}',
        errorCode=InputRejected.code
        if isinstance(error, InputRejected)
        else status,
    )


def main():
  if sys.argv[1:] != ['--serve']:
    raise SystemExit(
        'Usage: python -m model_debugger_runner.pytorch_worker --serve'
    )
  from .pytorch_session import PyTorchResidentRuntime

  runtime = PyTorchResidentRuntime()
  try:
    for line in sys.stdin:
      command = json.loads(line)
      if command.get('operation') == 'close':
        break
      serve_request(runtime, json.loads(Path(command['request']).read_text()))
  finally:
    runtime.close()


if __name__ == '__main__':
  main()
