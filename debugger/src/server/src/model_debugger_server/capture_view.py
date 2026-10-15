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

"""The UI's view of a chat's capture: the response of GET /api/session.

Its shape is the UI's `Session` (`data/contracts/capture.ts`). A turn's
`step_start` and `prefill_tokens` are the reference side's: decode token `i`
at `step_start + i`, input token `j` at `step_start - prefill_tokens + j`.
The target's tokens may sit elsewhere, when the two histories differ in
length, so every token also carries its own side's `position`.
"""

import re
from typing import Any

from model_debugger_server import capture_index

_SIDES = ('ref', 'target')
# A chat-template marker such as `<|turn>`, `<turn|>` or `<|channel>`. A
# plain `<b>` is text: it is a marker only with the bar.
_MARKER = re.compile(r'<\|[^\s<>|]*>|<[^\s<>|]*\|>')
_OPENER = re.compile(r'<\|[^\s<>|]*>')
# Gemma's thinking channel: the reply's tokens from `<|channel>` to
# `<channel|>` are its reasoning, as the runtime's jsonl shows them.
_THINKING = ('<|channel>', '<channel|>')
# Where a token's text comes from when its own source names none: the
# runner's tokenization of the same id in a prompt of this side.
_TEXT_BASIS = 'runner prompt tokens'


def session_view(
    index: capture_index.CaptureIndex, record: dict[str, Any]
) -> dict[str, Any]:
  """Returns the UI's `Session` for a chat's capture.

  Args:
    index: The chat's capture.
    record: The chat's record (name, model, runs, generation, notice).
  """
  return {
      'model': record['model'],
      'runs': [_run(index.sides[side], record, side) for side in _SIDES],
      'turns': [_turn(turn) for turn in index.turns],
      'phases': [
          {'id': 'prefill', 'label': 'Prefill', 'captured': True},
          {'id': 'decode', 'label': 'Decode', 'captured': True},
      ],
      'batches': [],
      'layers': [],
      'anchors': [],
      'notice': record.get('notice', ''),
      'name': record['name'],
      'created_at': record['created_at'],
      'capture_id': record['id'],
      'generation': record.get('generation', {}),
      'conversation': [
          _conversation(index, turn, side)
          for turn in index.turns
          for side in _SIDES
      ],
  }


def _run(
    fixed: dict[str, Any], record: dict[str, Any], side: str
) -> dict[str, Any]:
  """Returns a `CapturedRun`: the record's run with what really ran."""
  run = next(run for run in record['runs'] if run['id'] == side)
  return {
      **run,
      'runtime': 'LiteRT-LM',
      'backend': fixed['load']['device'],
      # The runner reports no precision of its own.
      'precision': '',
      'artifact_name': fixed['model']['name'],
      'device': fixed['device_id'],
      'contextLength': str(fixed['load']['context_length']),
      'provenance': {
          'model_sha256': fixed['model']['sha256'],
          'runtime_version': fixed['runtime']['version'],
          'device_name': fixed['device_name'],
          'host': fixed['host'],
      },
  }


def _turn(turn: dict[str, Any]) -> dict[str, Any]:
  """Returns a `CapturedTurn` in the reference side's positions.

  The two sides of a turn may sit at different positions when their
  histories differ; the turn takes one origin, the reference side's, and
  each token its own side's position.
  """
  ref = turn['sides']['ref']
  return {
      'n': turn['n'],
      'prefill_tokens': len(ref['input']),
      'decode_tokens': len(ref['output']),
      'step_start': ref['output'][0]['position'],
      'step_end': ref['output'][-1]['position'],
  }


def _conversation(
    index: capture_index.CaptureIndex, turn: dict[str, Any], side: str
) -> dict[str, Any]:
  """Returns the `CapturedConversation` of one side of one turn."""
  entry = turn['sides'][side]
  completion = entry['completion']
  names = _names(index, side)
  stop_ids = {
      token[0]
      for token in index.sides[side]['stop_tokens']
      if isinstance(token, list) and len(token) == 1
  }
  start_token = index.sides[side]['start_token']
  special_ids = stop_ids | ({start_token} if start_token is not None else set())

  messages = [
      dict(message) for message in index.sides[side]['chat']['messages']
  ]
  for earlier in index.turns[: turn['n'] - 1]:
    messages.append({'role': 'user', 'content': earlier['prompt']})
    messages.append({
        'role': 'assistant',
        'content': earlier['sides'][side]['completion']['content'],
    })

  kinds = _kinds(entry['input'], special_ids)
  input_tokens = [
      _token(token, names, step=step, kind=kind)
      for step, (token, kind) in enumerate(zip(entry['input'], kinds))
  ]
  tokens = []
  thinking = False
  kinds = _kinds(entry['output'], special_ids)
  for step, (token, kind) in enumerate(zip(entry['output'], kinds)):
    thinking = thinking or token['text'] == _THINKING[0]
    view = _token(token, names, step=step, kind=kind)
    view.update(
        source_forward_id=token['forward'],
        released=token['released'],
        phase='thinking' if thinking else 'response',
    )
    if token.get('stop'):
      view['stop'] = True
    tokens.append(view)
    thinking = thinking and token['text'] != _THINKING[1]

  conversation = {
      'turn': turn['n'],
      'run': side,
      'input': turn['prompt'],
      'messages': messages,
      'input_tokens': input_tokens,
      'input_token_count': len(input_tokens),
      'output': completion['content'],
      'tokens': tokens,
      'stop_reason': entry['stop_reason'],
  }
  if completion['reasoning_content']:
    conversation['thinking'] = completion['reasoning_content']
  return conversation


def _names(index: capture_index.CaptureIndex, side: str) -> dict[int, str]:
  """Returns the text the runner gave each id in this side's prompts."""
  names = {}
  for turn in index.turns:
    for token in turn['sides'][side]['completion']['prompt_tokens']:
      names.setdefault(token['id'], token['text'])
  return names


def _token(
    token: dict[str, Any], names: dict[int, str], *, step: int, kind: str
) -> dict[str, Any]:
  """Returns a `CapturedToken`, its missing text named by the prompts.

  A token whose id the runner did not name (a sampled chat's stop token, or
  what the runtime fed of its own rendering of earlier turns) has no `id`.
  """
  view = {
      'kind': kind,
      'text': token['text'],
      'step': step,
      'position': token['position'],
  }
  if token['id'] is not None:
    view['id'] = token['id']
    if token['text'] is None and token['id'] in names:
      view.update(text=names[token['id']], text_basis=_TEXT_BASIS)
  return view


def _is_marker(text: str | None) -> bool:
  return text is not None and _MARKER.fullmatch(text) is not None


def _kinds(tokens: list[dict[str, Any]], special_ids: set[int]) -> list[str]:
  """Classifies tokens as the UI groups them into message rows.

  `special`: the start and stop tokens and chat-template markers;
  `template`: what the runtime fed of its own rendering of earlier turns,
  the word after an opening marker (a role, or `thought`), and the newline
  after that word or after a closing marker; `text`: everything else.
  """
  kinds = []
  for index, token in enumerate(tokens):
    text = token['text']
    before = tokens[index - 1]['text'] if index else None
    if token['id'] in special_ids or _is_marker(text):
      kinds.append('special')
    elif token.get('origin') == 'history':
      kinds.append('template')
    elif (
        before is not None
        and _OPENER.fullmatch(before)
        and text is not None
        and text.strip().isalpha()
    ):
      kinds.append('template')
    elif (
        text == '\n'
        and index
        and (
            kinds[-1] == 'special'
            or (kinds[-1] == 'template' and before != '\n')
        )
    ):
      kinds.append('template')
    else:
      kinds.append('text')
  return kinds
