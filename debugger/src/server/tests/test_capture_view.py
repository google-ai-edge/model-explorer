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

"""Tests of the UI's view of a chat's capture."""

import capture_fixtures as fixtures
from model_debugger_server import capture_index, capture_view
import pytest

_RECORD = {
    'id': 'chat-1',
    'name': 'Two plus two',
    'model': 'Tiny',
    'created_at': '2026-10-14T09:00:00Z',
    'notice': '',
    'runs': [
        {
            'id': 'ref',
            'runtime': 'LiteRT-LM',
            'artifact': 'model-abab',
            'source': 'upload',
            'device': 'local',
            'backend': 'CPU',
            'contextLength': '128',
            'cpuThreads': '4',
        },
        {
            'id': 'target',
            'runtime': 'LiteRT-LM',
            'artifact': 'model-abab',
            'source': 'upload',
            'device': 'local',
            'backend': 'CPU',
            'contextLength': '128',
        },
    ],
    'generation': {'ref': {'maxOutputTokens': 4}, 'target': {}},
}
_EXCHANGES = [
    ('What is it?', ['Four', '.'], True),
    ('And more?', ['a', 'b', 'c', 'd'], True),
    ('Thanks.', ['ok'], True),
]


@pytest.fixture(name='view')
def _view(tmp_path):
  index = fixtures.capture(
      tmp_path, _EXCHANGES, system='Be brief.', max_completion_tokens=4
  )
  return index, capture_view.session_view(index, _RECORD)


def _conversation(view, turn, run):
  (item,) = [
      c for c in view['conversation'] if c['turn'] == turn and c['run'] == run
  ]
  return item


def test_session_shape(view):
  """Session shape."""
  _, view = view

  assert view['model'] == 'Tiny'
  assert view['name'] == 'Two plus two'
  assert view['created_at'] == '2026-10-14T09:00:00Z'
  assert view['capture_id'] == 'chat-1'
  assert view['generation'] == _RECORD['generation']
  assert view['batches'] == view['layers'] == view['anchors'] == []
  assert view['phases'] == [
      {'id': 'prefill', 'label': 'Prefill', 'captured': True},
      {'id': 'decode', 'label': 'Decode', 'captured': True},
  ]
  assert [(c['turn'], c['run']) for c in view['conversation']] == [
      (n, run) for n in (1, 2, 3) for run in ('ref', 'target')
  ]


def test_runs_say_what_ran(view):
  """Runs say what ran."""
  _, view = view

  ref = view['runs'][0]
  assert [run['id'] for run in view['runs']] == ['ref', 'target']
  assert ref['cpuThreads'] == '4'
  assert ref['runtime'] == 'LiteRT-LM'
  assert ref['backend'] == 'CPU'
  assert ref['precision'] == ''
  assert ref['artifact'] == 'model-abab'
  assert ref['artifact_name'] == 'tiny.litertlm'
  assert ref['device'] == 'local'
  assert ref['contextLength'] == str(fixtures.CONTEXT_LENGTH)
  assert ref['provenance'] == {
      'model_sha256': 'ab' * 32,
      'runtime_version': 'a0f1e97361155c8b',
      'device_name': 'This computer',
      'host': {'platform': 'MACOS', 'chip': 'Apple M2 Max'},
  }


def test_turn_coordinates_are_context_positions(view):
  """Turn coordinates are context positions."""
  index, view = view

  for turn, entry in zip(view['turns'], index.turns):
    ref = entry['sides']['ref']
    assert turn['n'] == entry['n']
    assert turn['prefill_tokens'] == len(ref['input'])
    assert turn['decode_tokens'] == len(ref['output'])
    conversation = _conversation(view, turn['n'], 'ref')
    # The UI's formulas (conversation_panel.ts).
    for token, input in zip(conversation['input_tokens'], ref['input']):
      position = turn['step_start'] - turn['prefill_tokens'] + token['step']
      assert position == input['position']
    for token, output in zip(conversation['tokens'], ref['output']):
      assert turn['step_start'] + token['step'] == output['position']
    assert turn['step_end'] == ref['output'][-1]['position']


def test_decode_tokens_link_their_forward_pass(view):
  """Decode tokens link their forward pass."""
  index, view = view

  conversation = _conversation(view, 2, 'target')
  output = index.turn(2)['sides']['target']['output']

  assert [t['source_forward_id'] for t in conversation['tokens']] == [
      token['forward'] for token in output
  ]
  assert [t['step'] for t in conversation['tokens']] == list(range(4))
  assert {t['phase'] for t in conversation['tokens']} == {'response'}
  assert [t['released'] for t in conversation['tokens']] == [True] * 4
  assert conversation['stop_reason'] == 'max_output_tokens'
  assert conversation['output'] == 'abcd'


def test_a_stop_token_is_special_and_named_by_the_prompts(view):
  """A stop token is special and named by the prompts."""
  _, view = view

  conversation = _conversation(view, 1, 'ref')
  stop = conversation['tokens'][-1]

  assert stop == {
      'kind': 'special',
      'text': '<turn|>',
      'text_basis': 'runner prompt tokens',
      'id': fixtures.TURN_CLOSE,
      'step': 2,
      'position': conversation['tokens'][-2]['position'] + 1,
      'source_forward_id': conversation['tokens'][-2]['source_forward_id'] + 1,
      'released': False,
      'stop': True,
      'phase': 'response',
  }
  assert conversation['stop_reason'] == 'eos'
  assert conversation['output'] == 'Four.'
  assert 'thinking' not in conversation


def test_input_tokens_are_kinded_as_the_ui_groups_them(view):
  """Input tokens are kinded as the ui groups them."""
  _, view = view

  conversation = _conversation(view, 1, 'ref')
  kinds = [(t['kind'], t['text']) for t in conversation['input_tokens']]

  assert kinds == [
      ('special', None),
      ('special', '<|turn>'),
      ('template', 'system'),
      ('template', '\n'),
      ('text', 'Be'),
      ('text', ' brief.'),
      ('special', '<turn|>'),
      ('template', '\n'),
      ('special', '<|turn>'),
      ('template', 'user'),
      ('template', '\n'),
      ('text', 'What'),
      ('text', ' is'),
      ('text', ' it?'),
      ('special', '<turn|>'),
      ('template', '\n'),
      ('special', '<|turn>'),
      ('template', 'model'),
      ('template', '\n'),
  ]
  assert conversation['input_tokens'][0]['id'] == fixtures.BOS
  # Nothing names the start token: its text stays unknown.
  assert 'text_basis' not in conversation['input_tokens'][0]
  assert conversation['input_token_count'] == len(kinds)
  assert [t['step'] for t in conversation['input_tokens']] == list(
      range(len(kinds))
  )


def test_a_later_turn_starts_with_the_carried_token(view):
  """A later turn starts with the carried token."""
  _, view = view

  carried_stop = _conversation(view, 2, 'ref')['input_tokens'][0]
  carried_text = _conversation(view, 3, 'ref')['input_tokens'][0]

  assert carried_stop['kind'] == 'special'
  assert carried_stop['text'] == '<turn|>'
  assert carried_stop['id'] == fixtures.TURN_CLOSE
  assert carried_text == {
      'kind': 'text',
      'text': 'd',
      'id': fixtures.token('d')[0],
      'step': 0,
      'position': (
          view['turns'][2]['step_start']
          - len(_conversation(view, 3, 'ref')['input_tokens'])
      ),
  }


def test_messages_are_the_history_before_the_turn(view):
  """Messages are the history before the turn."""
  _, view = view

  assert _conversation(view, 1, 'ref')['messages'] == [
      {'role': 'system', 'content': 'Be brief.'}
  ]
  assert _conversation(view, 3, 'target')['messages'] == [
      {'role': 'system', 'content': 'Be brief.'},
      {'role': 'user', 'content': 'What is it?'},
      {'role': 'assistant', 'content': 'Four.'},
      {'role': 'user', 'content': 'And more?'},
      {'role': 'assistant', 'content': 'abcd'},
  ]
  assert _conversation(view, 3, 'target')['input'] == 'Thanks.'


def test_thinking_is_the_reasoning_content(tmp_path):
  """Thinking is the reasoning content."""
  index = fixtures.capture(tmp_path, [('Hi', ['a'], True)])
  index.turns[0]['sides']['ref']['completion']['reasoning_content'] = 'hm'

  view = capture_view.session_view(index, _RECORD)

  assert _conversation(view, 1, 'ref')['thinking'] == 'hm'
  assert 'thinking' not in _conversation(view, 1, 'target')


def test_thinking_tokens_are_in_the_thinking_phase(tmp_path):
  """Thinking tokens are in the thinking phase."""
  index = fixtures.capture(
      tmp_path,
      [('Hi', ['ok'], True, ['hmm']), ('More', ['b'], True)],
      max_completion_tokens=12,
  )

  view = capture_view.session_view(index, _RECORD)
  first = _conversation(view, 1, 'ref')
  second = _conversation(view, 2, 'ref')

  assert [(t['kind'], t['text'], t['phase']) for t in first['tokens']] == [
      ('special', '<|channel>', 'thinking'),
      ('template', 'thought', 'thinking'),
      ('template', '\n', 'thinking'),
      ('text', 'hmm', 'thinking'),
      ('special', '<channel|>', 'thinking'),
      ('text', 'ok', 'response'),
      ('special', '<turn|>', 'response'),
  ]
  assert first['thinking'] == 'hmm'
  assert first['output'] == 'ok'
  # The runtime's own rendering of turn 1 has no ids: the tokens have none.
  assert [
      (t['kind'], 'id' in t, t['text']) for t in second['input_tokens'][:4]
  ] == [('template', False, None)] * 4


def test_markers_are_bar_delimited():
  """Markers are bar delimited."""
  special_ids = set()
  tokens = [
      {'id': 9, 'text': '<b>'},
      {'id': 10, 'text': '<|channel>'},
      {'id': 11, 'text': 'thought'},
      {'id': 12, 'text': '\n'},
      {'id': 13, 'text': '\n'},
      {'id': 14, 'text': '<channel|>'},
  ]

  kinds = capture_view._kinds(tokens, special_ids)

  assert kinds == ['text', 'special', 'template', 'template', 'text', 'special']


def test_each_token_has_its_own_sides_position(tmp_path):
  """Sides of different lengths: the turn is the ref's, each token its own."""
  chats = {
      'ref': fixtures.TinyChat('', 8, seed=0),
      'target': fixtures.TinyChat('Be brief.', 8, seed=1),
  }
  index = capture_index.CaptureIndex.create(
      tmp_path, {side: fixtures.side_info(chat) for side, chat in chats.items()}
  )
  completions = {
      side: fixtures.write_turn(
          index.turn_directory(side, 1),
          chat.send('What is it?', ['Four.'], True),
      )
      for side, chat in chats.items()
  }
  index.add_turn(1, 'What is it?', completions)

  view = capture_view.session_view(index, _RECORD)

  sides = index.turn(1)['sides']
  turn = view['turns'][0]
  assert turn['step_start'] == sides['ref']['output'][0]['position']
  assert turn['prefill_tokens'] == len(sides['ref']['input'])
  for side in ('ref', 'target'):
    conversation = _conversation(view, 1, side)
    assert [t['position'] for t in conversation['input_tokens']] == [
        t['position'] for t in sides[side]['input']
    ]
    assert [t['position'] for t in conversation['tokens']] == [
        t['position'] for t in sides[side]['output']
    ]
  target = _conversation(view, 1, 'target')
  assert target['tokens'][0]['position'] > turn['step_start']
