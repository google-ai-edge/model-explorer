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

import contextlib
import json
import pathlib
import threading
from unittest import mock

from absl.testing import absltest
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index
from model_debugger.server.service import session_view
from model_debugger.server.testing import capture_fixtures as fixtures
from model_debugger.server.testing import test_main

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
}
# The Session's chat the views show.
_CHAT = {
    'n': 1,
    'generation': {'ref': {'maxOutputTokens': 4}, 'target': {}},
    'has_capture': True,
    'over': False,
    'notice': '',
}
_RECORD['chats'] = [_CHAT]
_EXCHANGES = [
    ('What is it?', ['Four', '.'], True),
    ('And more?', ['a', 'b', 'c', 'd'], True),
    ('Thanks.', ['ok'], True),
]


_AVAILABLE = {'available': True, 'reasons': []}


def _unavailable(*reasons):
  return {'available': False, 'reasons': list(reasons)}


def _why(table, code, side=None):
  """The reason of `code` in the words its module's table gives it."""
  return metrics.reason(code, table[code], side)


def _conversation(view, turn, run):
  (item,) = [
      c for c in view['conversation'] if c['turn'] == turn and c['run'] == run
  ]
  return item


class SessionViewTest(absltest.TestCase):
  """Tests of the UI's view of a chat's capture."""

  def setUp(self):
    """An empty directory; a captured three-exchange chat, and its view."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(
        pathlib.Path(self.create_tempdir().full_path),
        _EXCHANGES,
        system='Be brief.',
        max_completion_tokens=4,
    )
    self.captured = index, session_view.session_view(index, _RECORD, _CHAT)

  def test_session_shape(self):
    """Session shape."""
    _, view = self.captured

    self.assertEqual(view['model'], 'Tiny')
    self.assertEqual(view['name'], 'Two plus two')
    self.assertEqual(view['created_at'], '2026-10-14T09:00:00Z')
    self.assertEqual(view['capture_id'], 'chat-1')
    self.assertEqual(view['generation'], _CHAT['generation'])
    self.assertEqual(view['chat'], 1)
    self.assertEqual(view['batches'], [])
    expected = []
    for n in (1, 2, 3):
      for run in ('ref', 'target'):
        expected.append((n, run))
    self.assertEqual(
        [(c['turn'], c['run']) for c in view['conversation']],
        expected,
    )

  def test_a_rejected_turn_says_why_and_has_no_tokens(self):
    """It follows the turns read, with the text to show."""
    index, _ = self.captured
    cut = min((index.root / 'target' / 'turn-2').glob('*.safetensors'))
    size = cut.stat().st_size
    cut.write_bytes(cut.read_bytes()[:-1])

    view = session_view.session_view(
        capture_index.CaptureIndex.load(index.root), _RECORD, _CHAT
    )

    self.assertEqual([turn['n'] for turn in view['turns']], [1, 2, 3])
    self.assertNotIn('rejected', view['turns'][0])
    self.assertEqual(
        view['turns'][1],
        {
            'n': 2,
            'prompt': 'And more?',
            'step_start': None,
            'rejected': {
                'code': session_view.TURN_UNREADABLE,
                'side': 'target',
                'text': (
                    f'Target: Turn 2 could not be read: {cut.name} '
                    f'holds {size - 1} bytes; the capture recorded {size}.'
                ),
            },
            'stages': view['turns'][1]['stages'],
        },
    )
    # Its analyses are unavailable, for the rejection; it has no
    # positions and no KV snapshot.
    rejected = view['turns'][1]['rejected']
    for stage, moment in session_view.STAGE_MOMENTS.items():
      self.assertEqual(
          view['turns'][1]['stages'][stage],
          {
              'positions': {'ref': None, 'target': None},
              'kv': {'moment': moment, 'context_id': None},
              'analyses': dict.fromkeys(
                  session_view.ANALYSES,
                  _unavailable(
                      metrics.reason(rejected['code'], rejected['text'])
                  ),
              ),
          },
      )
    self.assertEqual(
        view['turns'][2]['rejected'],
        {
            'code': session_view.TURN_UNREADABLE,
            'side': None,
            'text': (
                'Turn 3 could not be read: It follows turn 2, which '
                'was rejected.'
            ),
        },
    )
    self.assertEqual(
        [(c['turn'], c['run']) for c in view['conversation']],
        [(1, 'ref'), (1, 'target')],
    )

  def test_runs_say_what_ran(self):
    """Runs say what ran."""
    _, view = self.captured

    ref = view['runs'][0]
    self.assertEqual([run['id'] for run in view['runs']], ['ref', 'target'])
    self.assertEqual(ref['cpuThreads'], '4')
    self.assertEqual(ref['runtime'], 'LiteRT-LM')
    self.assertEqual(ref['backend'], 'CPU')
    self.assertEqual(ref['artifact'], 'model-abab')
    self.assertEqual(ref['device'], 'local')
    self.assertEqual(ref['contextLength'], str(fixtures.CONTEXT_LENGTH))
    self.assertEqual(
        ref['provenance'],
        {
            'runtime_version': 'a0f1e97361155c8b',
            'device_name': 'This computer',
            'host': {'platform': 'MACOS', 'chip': 'Apple M2 Max'},
        },
    )

  def test_turn_coordinates_are_context_positions(self):
    """Turn coordinates are context positions."""
    index, view = self.captured

    for turn, entry in zip(view['turns'], index.turns):
      ref = entry['sides']['ref']
      self.assertEqual(turn['n'], entry['n'])
      conversation = _conversation(view, turn['n'], 'ref')
      for token, input_token in zip(conversation['input_tokens'], ref['input']):
        self.assertEqual(token['position'], input_token['position'])
      for token, output in zip(conversation['tokens'], ref['output']):
        self.assertEqual(turn['step_start'] + token['step'], output['position'])

  def test_decode_tokens_link_their_forward_pass(self):
    """Decode tokens link their forward pass."""
    index, view = self.captured

    conversation = _conversation(view, 2, 'target')
    output = index.turn(2)['sides']['target']['output']

    self.assertEqual(
        [t['source_forward_id'] for t in conversation['tokens']],
        [token['forward'] for token in output],
    )
    self.assertEqual(
        [t['step'] for t in conversation['tokens']], list(range(4))
    )
    self.assertEqual({t['phase'] for t in conversation['tokens']}, {'response'})
    self.assertEqual(
        [t['released'] for t in conversation['tokens']], [True] * 4
    )
    self.assertEqual(conversation['stop_reason'], 'max_output_tokens')
    self.assertEqual(
        conversation['stop_text'], 'Output limit reached (4 tokens)'
    )
    self.assertEqual(conversation['output'], 'abcd')

  def test_a_stop_token_is_special_and_named_by_the_prompts(self):
    """A stop token is special and named by the prompts."""
    _, view = self.captured

    conversation = _conversation(view, 1, 'ref')
    stop = conversation['tokens'][-1]
    position = conversation['tokens'][-2]['position'] + 1

    self.assertEqual(
        stop,
        {
            'kind': 'special',
            'text': '<turn|>',
            'text_basis': 'runner prompt tokens',
            'id': fixtures.TURN_CLOSE,
            'step': 2,
            'position': position,
            # A decode step labelled s reads s - 1 and produces s.
            'produced_text': (
                f'Decode step {position} · reads "." · produces "<turn|>"'
            ),
            'written_text': 'No step of this turn reads "<turn|>"',
            'source_forward_id': (
                conversation['tokens'][-2]['source_forward_id'] + 1
            ),
            'released': False,
            'stop': True,
            'phase': 'response',
        },
    )
    self.assertEqual(conversation['stop_reason'], 'eos')
    self.assertEqual(conversation['stop_text'], 'Stop token')
    self.assertEqual(conversation['output'], 'Four.')
    self.assertNotIn('thinking', conversation)

  def test_an_unknown_stop_token_is_the_one_the_next_turn_carried(self):
    """Another sampler's stop: the token the next turn fed there."""
    index = fixtures.capture(
        self.tmp_path,
        [('Hi', ['a', '.'], True), ('More', ['b'], True)],
        top_k=40,
    )
    assert index.turns[0]['sides']['ref']['output'][-1]['id'] is None

    view = session_view.session_view(index, _RECORD, _CHAT)

    stop = _conversation(view, 1, 'ref')['tokens'][-1]
    self.assertTrue(stop['stop'])
    self.assertEqual(stop['id'], fixtures.TURN_CLOSE)
    self.assertEqual(stop['text'], '<turn|>')

  def test_a_reply_that_filled_the_context_says_so(self):
    """A reply cut by the context, not its limit, says the context."""
    # 115 words leave room for 4 generated tokens in the tiny context.
    index = fixtures.capture(
        self.tmp_path, [(' '.join(['w'] * 115), ['a'] * 6, True)]
    )

    view = session_view.session_view(index, _RECORD, _CHAT)

    for side in capture_index.SIDES:
      conversation = _conversation(view, 1, side)
      self.assertEqual(conversation['stop_reason'], 'context_full')
      self.assertEqual(
          conversation['stop_text'],
          f'Context full ({fixtures.CONTEXT_LENGTH} tokens)',
      )

  def test_a_step_that_cannot_be_confirmed_says_so_and_why(self):
    """Tied top logits: that token carries the text and the reason."""
    index, _ = self.captured
    directory = index.root / 'target' / 'turn-1'
    four = fixtures.token('Four')[0]

    def tie(logits):
      logits[3] = logits[four]

    step = fixtures.edit_logits(directory, 0, tie)
    index = capture_index.CaptureIndex.load(index.root)
    reason = index.turn(1)['sides']['target']['output'][0]['cannot_confirm']

    view = session_view.session_view(index, _RECORD, _CHAT)

    tokens = _conversation(view, 1, 'target')['tokens']
    self.assertEqual(
        tokens[0]['cannot_confirm'],
        {
            'text': 'Cannot confirm greedy decoding',
            'reason': metrics.reason(
                token_analysis.GREEDY_UNCONFIRMED, reason, 'target'
            ),
        },
    )
    self.assertTrue(reason.startswith(f'Decode step {step} has 2 logits tied'))
    self.assertEqual(tokens[0]['text'], 'Four')
    self.assertFalse(any('cannot_confirm' in t for t in tokens[1:]))
    self.assertFalse(
        any(
            'cannot_confirm' in t
            for t in _conversation(view, 1, 'ref')['tokens']
        )
    )
    self.assertLen(view['turns'], 3)
    self.assertNotIn('rejected', view['turns'][0])

  def test_input_tokens_are_kinded_as_the_ui_groups_them(self):
    """Input tokens are kinded as the ui groups them."""
    _, view = self.captured

    conversation = _conversation(view, 1, 'ref')
    kinds = [(t['kind'], t['text']) for t in conversation['input_tokens']]

    self.assertEqual(
        kinds,
        [
            ('special', '<bos>'),
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
        ],
    )
    self.assertEqual(conversation['input_tokens'][0]['id'], fixtures.BOS)
    # The runner names the start token it fed.
    self.assertNotIn('text_basis', conversation['input_tokens'][0])
    self.assertLen(kinds, conversation['input_token_count'])
    self.assertEqual(
        [t['step'] for t in conversation['input_tokens']],
        list(range(len(kinds))),
    )

  def test_a_later_turn_starts_with_the_carried_token(self):
    """A later turn starts with the carried token."""
    _, view = self.captured

    carried_stop = _conversation(view, 2, 'ref')['input_tokens'][0]
    carried_text = _conversation(view, 3, 'ref')['input_tokens'][0]

    self.assertEqual(carried_stop['kind'], 'special')
    self.assertEqual(carried_stop['text'], '<turn|>')
    self.assertEqual(carried_stop['id'], fixtures.TURN_CLOSE)
    # The turn before produced it: no step of this one did.
    self.assertEqual(
        carried_text.pop('produced_text'),
        'No step of this turn produces "d"',
    )
    self.assertRegex(
        carried_text.pop('written_text'),
        r'^Written at prefill step \d+, row 0, when the model reads "d"$',
    )
    self.assertEqual(
        carried_text,
        {
            'kind': 'text',
            'text': 'd',
            'id': fixtures.token('d')[0],
            'step': 0,
            'position': (
                view['turns'][2]['step_start']
                - len(_conversation(view, 3, 'ref')['input_tokens'])
            ),
        },
    )

  def test_each_token_says_which_step_it_is(self):
    """The step that produced it, and the one that wrote its K and V."""
    index, view = self.captured
    forwards = index.turn(1)['sides']['ref']['forwards']
    conversation = _conversation(view, 1, 'ref')
    inputs, tokens = conversation['input_tokens'], conversation['tokens']
    chunk = forwards[0]
    self.assertEqual(chunk['phase'], 'prefill')
    self.assertEqual(
        inputs[0]['produced_text'], 'No step of this turn produces "<bos>"'
    )
    self.assertEqual(
        inputs[0]['written_text'],
        f'Written at prefill step {chunk["step"]}, row 0, when the model '
        'reads "<bos>"',
    )
    self.assertEqual(
        inputs[1]['produced_text'],
        f'Prefill step {chunk["step"]}, row 0 · reads "<bos>" · next '
        f'token {json.dumps(inputs[1]["text"])}',
    )
    # The first decode step reads the input's last token and produces
    # the first token generated: the step Final Logits shows there.
    first = forwards[tokens[0]['source_forward_id']]
    self.assertEqual(first['input_position'], inputs[-1]['position'])
    last_input = json.dumps(inputs[-1]['text'])
    self.assertEqual(
        inputs[-1]['written_text'],
        f'Written at decode step {first["step"]}, when the model reads '
        f'{last_input}',
    )
    self.assertEqual(
        tokens[0]['produced_text'],
        f'Decode step {first["step"]} · reads {last_input} · produces '
        f'{json.dumps(tokens[0]["text"])}',
    )
    for token in tokens:
      step = forwards[token['source_forward_id']]['step']
      self.assertTrue(
          token['produced_text'].startswith(f'Decode step {step} ·'),
          token['produced_text'],
      )
    for token, after in zip(tokens, tokens[1:]):
      step = forwards[after['source_forward_id']]['step']
      self.assertTrue(
          token['written_text'].startswith(f'Written at decode step {step}, '),
          token['written_text'],
      )

  def test_only_turn_one_carries_the_opening_messages(self):
    """Later turns' history is the earlier turns' prompts and replies."""
    _, view = self.captured

    for run in capture_index.SIDES:
      self.assertEqual(
          _conversation(view, 1, run)['messages'],
          [{'role': 'system', 'content': 'Be brief.'}],
      )
    self.assertNotIn('messages', _conversation(view, 2, 'ref'))
    self.assertNotIn('messages', _conversation(view, 3, 'target'))
    self.assertEqual(
        [
            (c['input'], c['output'])
            for c in view['conversation']
            if c['run'] == 'target'
        ],
        [
            ('What is it?', 'Four.'),
            ('And more?', 'abcd'),
            ('Thanks.', 'ok'),
        ],
    )

  def test_a_tool_response_an_older_server_recorded_is_its_input(self):
    """A tool's response of an older server's chat shows as its text."""
    index, _ = self.captured
    index.turns[1]['sides']['target']['message'] = {
        'role': 'tool',
        'content': '{"result":"B"}',
        'name': 'run_js',
    }

    view = session_view.session_view(index, _RECORD, _CHAT)

    self.assertEqual(
        _conversation(view, 2, 'target')['input'], '{"result":"B"}'
    )

  def test_thinking_is_the_reasoning_content(self):
    """Thinking is the reasoning content."""
    index = fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])
    index.turns[0]['sides']['ref']['completion']['reasoning_content'] = 'hm'

    view = session_view.session_view(index, _RECORD, _CHAT)

    self.assertEqual(_conversation(view, 1, 'ref')['thinking'], 'hm')
    self.assertNotIn('thinking', _conversation(view, 1, 'target'))

  def test_thinking_tokens_are_in_the_thinking_phase(self):
    """Thinking tokens are in the thinking phase."""
    index = fixtures.capture(
        self.tmp_path,
        [('Hi', ['ok'], True, ['hmm']), ('More', ['b'], True)],
        max_completion_tokens=12,
    )

    view = session_view.session_view(index, _RECORD, _CHAT)
    first = _conversation(view, 1, 'ref')
    second = _conversation(view, 2, 'ref')

    self.assertEqual(
        [(t['kind'], t['text'], t['phase']) for t in first['tokens']],
        [
            ('special', '<|channel>', 'thinking'),
            ('template', 'thought', 'thinking'),
            ('template', '\n', 'thinking'),
            ('text', 'hmm', 'thinking'),
            ('special', '<channel|>', 'thinking'),
            ('text', 'ok', 'response'),
            ('special', '<turn|>', 'response'),
        ],
    )
    self.assertEqual(first['thinking'], 'hmm')
    self.assertEqual(first['output'], 'ok')
    # Turn 1 rewrote its reply without the thoughts and held back the
    # newline after it, which turn 2 feeds first, where the rewrite
    # ended.
    self.assertEqual(
        [(t['id'], t['text']) for t in second['input_tokens'][:4]],
        [
            (fixtures.NEWLINE, '\n'),
            (fixtures.TURN_OPEN, '<|turn>'),
            (fixtures.ROLES['user'], 'user'),
            (fixtures.NEWLINE, '\n'),
        ],
    )
    # The newline after the rewrite's `<turn|>` is the template's.
    self.assertEqual(second['input_tokens'][0]['kind'], 'template')
    rewrite = index.turn(1)['sides']['ref']['rewrite']
    self.assertEqual(
        second['input_tokens'][0]['position'], rewrite[-1]['position'] + 1
    )
    self.assertEqual(
        view['turns'][1]['step_start'] - len(second['input_tokens']),
        second['input_tokens'][0]['position'],
    )

  def test_markers_are_bar_delimited(self):
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

    kinds = session_view._kinds(tokens, special_ids)

    self.assertEqual(
        kinds,
        ['text', 'special', 'template', 'template', 'text', 'special'],
    )

  def test_each_token_has_its_own_sides_position(self):
    """Sides of different lengths: the turn is the ref's.

    Each token has its own side's position.
    """
    chats = {
        'ref': fixtures.TinyChat('', 8, seed=0),
        'target': fixtures.TinyChat('Be brief.', 8, seed=1),
    }
    index = capture_index.CaptureIndex.create(
        self.tmp_path,
        {side: fixtures.side_info(chat) for side, chat in chats.items()},
    )
    completions = {
        side: fixtures.write_turn(
            index.turn_directory(side, 1),
            chat.send('What is it?', ['Four.'], True),
        )
        for side, chat in chats.items()
    }
    index.add_turn(1, 'What is it?', completions)

    view = session_view.session_view(index, _RECORD, _CHAT)

    sides = index.turn(1)['sides']
    turn = view['turns'][0]
    self.assertEqual(turn['step_start'], sides['ref']['output'][0]['position'])
    for side in ('ref', 'target'):
      conversation = _conversation(view, 1, side)
      self.assertEqual(
          [t['position'] for t in conversation['input_tokens']],
          [t['position'] for t in sides[side]['input']],
      )
      self.assertEqual(
          [t['position'] for t in conversation['tokens']],
          [t['position'] for t in sides[side]['output']],
      )
    target = _conversation(view, 1, 'target')
    self.assertGreater(target['tokens'][0]['position'], turn['step_start'])
    # Each stage's positions are each side's own.
    for side in ('ref', 'target'):
      for stage, key in (('prefill', 'input'), ('decode', 'output')):
        tokens = sides[side][key]
        self.assertEqual(
            turn['stages'][stage]['positions'][side],
            {
                'start': tokens[0]['position'],
                'end': tokens[-1]['position'] + 1,
            },
            (side, stage),
        )
    positions = turn['stages']['decode']['positions']
    self.assertGreater(positions['target']['start'], positions['ref']['start'])

  def test_each_stage_says_what_can_be_added_and_where_it_is(self):
    """A greedy chat: what each stage of each turn offers, and why not."""
    index, view = self.captured
    for turn in view['turns']:
      n, stages = turn['n'], turn['stages']
      sides = index.turn(n)['sides']
      self.assertEqual(set(stages), {'prefill', 'decode'})
      for stage, key in (('prefill', 'input'), ('decode', 'output')):
        moment = session_view.STAGE_MOMENTS[stage]
        self.assertEqual(
            stages[stage]['kv'],
            {
                'moment': moment,
                'context_id': kv_analysis.context_id(n, moment),
            },
        )
        for side in ('ref', 'target'):
          tokens = sides[side][key]
          self.assertEqual(
              stages[stage]['positions'][side],
              {
                  'start': tokens[0]['position'],
                  'end': tokens[-1]['position'] + 1,
              },
          )
      # The stage's columns start where the ref side's stage does.
      self.assertEqual(
          stages['prefill']['positions']['ref']['start'],
          turn['comparison']['prefill']['positions']['start'],
      )
      self.assertEqual(
          stages['decode']['positions']['ref']['start'],
          turn['step_start'],
      )
      # A Prefill names no generated token: the token analyses serve
      # only a Decode.
      self.assertEqual(
          stages['prefill']['analyses'],
          {
              'final_logits': _unavailable(
                  _why(token_analysis.REASONS, token_analysis.NOT_DECODE)
              ),
              'sampling': _unavailable(
                  _why(token_analysis.REASONS, token_analysis.NOT_DECODE)
              ),
              'kv': _AVAILABLE,
              'intermediate_tensors': _unavailable(
                  _why(
                      tensor_analysis.REASONS,
                      tensor_analysis.NOT_OBSERVED,
                  )
              ),
          },
      )
      self.assertEqual(
          stages['decode']['analyses'],
          {
              'final_logits': _AVAILABLE,
              'sampling': _AVAILABLE,
              'kv': _AVAILABLE,
              'intermediate_tensors': _unavailable(
                  _why(
                      tensor_analysis.REASONS,
                      tensor_analysis.NOT_OBSERVED,
                  )
              ),
          },
      )

  def test_observed_tensors_are_offered_where_the_capture_holds_them(self):
    """Both stages, their batches; one graph's points only in its stages."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], observe=True)
    sides = {
        side: tensor_analysis.Side('ab' * 32, fixtures.TAP_POINTS)
        for side in capture_index.SIDES
    }
    views = session_view.SessionViews(
        observed=lambda record_id: tensor_analysis.Observed(sides)
    )
    view = views.session('chat-1', 1, index, _RECORD)
    stages = view['turns'][0]['stages']
    for stage in ('prefill', 'decode'):
      self.assertEqual(
          stages[stage]['analyses']['intermediate_tensors'], _AVAILABLE
      )
    self.assertEqual(
        [(b['turn'], b['phase'], b['signature']) for b in view['batches']],
        [
            (1, 'prefill', fixtures.SIGNATURE),
            (1, 'decode', 'decode'),
        ],
    )
    prefill_only = {
        side: tensor_analysis.Side(
            'ab' * 32,
            tuple(p for p in fixtures.TAP_POINTS if p.signature != 'decode'),
        )
        for side in capture_index.SIDES
    }
    view = session_view.SessionViews(
        observed=lambda record_id: tensor_analysis.Observed(prefill_only)
    ).session('chat-1', 1, index, _RECORD)
    (why,) = view['turns'][0]['stages']['decode']['analyses'][
        'intermediate_tensors'
    ]['reasons']
    self.assertEqual(why['code'], tensor_analysis.NOT_IN_STAGE)

  def test_a_stages_tensors_are_compared_once_per_view(self):
    """However many ask, and at either alignment, each once."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], observe=True)
    sides = {
        side: tensor_analysis.Side('ab' * 32, fixtures.TAP_POINTS)
        for side in capture_index.SIDES
    }
    view = session_view.compared(
        index, observed=tensor_analysis.Observed(sides)
    )
    real = tensor_analysis.stages
    with mock.patch.object(
        tensor_analysis, 'stages', side_effect=real
    ) as stages:
      first = view.tensors(1, 'decode', 'content')
      self.assertIs(view.tensors(1, 'decode', 'content'), first)
      # The Decode's other alignment was compared in the same pass.
      steps = view.tensors(1, 'decode', 'steps')
      positions = view.tensors(1, 'decode', None)
    self.assertEqual(stages.call_count, 2)
    self.assertEqual(
        stages.call_args_list[0].args[4],
        [
            view.columns(1, 'decode', 'content'),
            view.columns(1, 'decode', 'steps'),
        ],
    )
    self.assertIsNot(steps, first)
    self.assertEqual(
        (first.start, first.end),
        (
            view.columns(1, 'decode', 'content').start,
            view.columns(1, 'decode', 'content').end(),
        ),
    )
    self.assertEqual(
        positions.start,
        index.turn(1)['sides']['ref']['output'][0]['position'],
    )

  def test_kept_tensors_count_toward_the_size_and_go_oldest_first(self):
    """A stage kept grows its view; beyond what is kept, older go."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], observe=True)
    sides = {
        side: tensor_analysis.Side('ab' * 32, fixtures.TAP_POINTS)
        for side in capture_index.SIDES
    }
    observed = tensor_analysis.Observed(sides)
    alone = session_view.compared(index, observed=observed)
    bare = alone.size
    decode = alone.tensors(1, 'decode', None)
    self.assertEqual(
        alone.size,
        bare + session_view._CELL_BYTES * len(decode.cells),
    )

    views = session_view.SessionViews(
        kept_bytes=alone.size + 1, observed=lambda record_id: observed
    )
    view = views.compared('chat-1', 1, index)
    first = view.tensors(1, 'decode', None)
    self.assertIs(view.tensors(1, 'decode', None), first)
    # The prefill stage outgrows what is kept: the decode stage goes.
    view.tensors(1, 'prefill', None)
    self.assertIsNot(view.tensors(1, 'decode', None), first)
    self.assertIs(views.compared('chat-1', 1, index), view)

  def test_an_unreadable_model_is_asked_again(self):
    """A model briefly missing does not leave its view without tensors."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], observe=True)
    sides = {
        side: tensor_analysis.Side('ab' * 32, fixtures.TAP_POINTS)
        for side in capture_index.SIDES
    }
    answers = [
        tensor_analysis.nothing(
            tensor_analysis.MODEL_UNREADABLE, 'The model is gone.'
        )
    ]
    asked = []

    def observed(record_id):
      asked.append(record_id)
      return answers[-1]

    views = session_view.SessionViews(observed=observed)
    gone = views.session('chat-1', 1, index, _RECORD)
    (why,) = gone['turns'][0]['stages']['decode']['analyses'][
        'intermediate_tensors'
    ]['reasons']
    self.assertEqual(why['code'], tensor_analysis.MODEL_UNREADABLE)
    self.assertEqual(gone['batches'], [])
    # Still gone: the same view, asked again.
    view = views.compared('chat-1', 1, index)
    self.assertIs(views.compared('chat-1', 1, index), view)
    self.assertLen(asked, 3)
    answers.append(tensor_analysis.Observed(sides))
    back = views.session('chat-1', 1, index, _RECORD)
    self.assertEqual(
        back['turns'][0]['stages']['decode']['analyses'][
            'intermediate_tensors'
        ],
        _AVAILABLE,
    )
    self.assertNotEmpty(back['batches'])
    again = views.compared('chat-1', 1, index)
    self.assertIs(views.compared('chat-1', 1, index), again)
    self.assertIsNot(again, view)

  def test_prefill_offers_no_final_logits_even_with_post_logits(self):
    """A prefill chunk's logits produced no token the routes could name."""
    index, _ = self.captured
    forward = index.turn(1)['sides']['target']['forwards'][0]
    self.assertEqual(forward['phase'], 'prefill')
    with mock.patch.dict(
        forward['tensors'], {'post_logits': 'prefill_post_logits.st'}
    ):
      view = session_view.session_view(index, _RECORD, _CHAT)
    analyses = view['turns'][0]['stages']['prefill']['analyses']
    self.assertEqual(
        analyses['final_logits'],
        _unavailable(token_analysis.stage_reason('prefill')),
    )

  def test_decode_without_post_logits_offers_no_final_logits(self):
    """No decode step of either side captured its logits."""
    index, _ = self.captured
    forwards = []
    for side in capture_index.SIDES:
      for forward in index.turn(1)['sides'][side]['forwards']:
        if forward['phase'] == 'decode':
          forwards.append(forward)
    tensors = [
        {k: v for k, v in forward['tensors'].items() if k != 'post_logits'}
        for forward in forwards
    ]
    with contextlib.ExitStack() as stack:
      for forward, kept in zip(forwards, tensors):
        stack.enter_context(
            mock.patch.dict(forward['tensors'], kept, clear=True)
        )
      view = session_view.session_view(index, _RECORD, _CHAT)
    analyses = view['turns'][0]['stages']['decode']['analyses']
    # Sampling is the softmax of those logits: none either.
    for analysis in ('final_logits', 'sampling'):
      self.assertEqual(
          analyses[analysis],
          _unavailable(
              metrics.reason(
                  token_analysis.LOGITS_NOT_CAPTURED,
                  'Final logits not captured',
              )
          ),
      )

  def test_kv_is_unavailable_by_the_rule_kv_head_reads_by(self):
    """Snapshots of no readable layout: unavailable, and why, per side."""
    index, _ = self.captured
    stored = []
    for side in capture_index.SIDES:
      for snapshot in index.turn(1)['sides'][side]['kv']:
        if snapshot['moment'] == 'terminal':
          for layer in snapshot['layers']:
            for kind in kv_analysis.KINDS:
              if layer.get(kind) is not None:
                stored.append(layer[kind])
    self.assertTrue(stored)
    with contextlib.ExitStack() as stack:
      for entry in stored:
        stack.enter_context(mock.patch.dict(entry, {'layout': None}))
      view = session_view.session_view(index, _RECORD, _CHAT)
    stages = view['turns'][0]['stages']
    self.assertEqual(
        stages['decode']['analyses']['kv'],
        _unavailable(
            _why(kv_analysis.REASONS, 'layout_unavailable'),
        ),
    )
    self.assertEqual(stages['prefill']['analyses']['kv'], _AVAILABLE)

  def test_sampling_is_offered_whatever_the_sampler(self):
    """The softmax of a step's logits: any sampler, any step has one."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['ok'], False)], max_completion_tokens=1
    )
    options = index.sides['ref']['chat']['options']

    def sampling(view):
      return view['turns'][0]['stages']['decode']['analyses']['sampling']

    view = session_view.session_view(index, _RECORD, _CHAT)
    self.assertEqual(sampling(view), _AVAILABLE)
    speculative = {
        'top_k': 40,
        'runtime_options': {'speculative_decoding': True},
    }
    with (
        mock.patch.dict(options, speculative),
        mock.patch.dict(index.sides['target']['chat']['options'], speculative),
    ):
      view = session_view.session_view(index, _RECORD, _CHAT)
    self.assertEqual(sampling(view), _AVAILABLE)

  def test_kv_needs_a_snapshot_at_the_stages_moment(self):
    """A stage whose snapshot holds no position offers no KV cache."""
    index, _ = self.captured
    sides = index.turn(1)['sides']
    kept = {
        side: [
            snapshot
            for snapshot in sides[side]['kv']
            if snapshot['moment'] != 'terminal'
        ]
        for side in ('ref', 'target')
    }
    with (
        mock.patch.dict(sides['ref'], {'kv': kept['ref']}),
        mock.patch.dict(sides['target'], {'kv': kept['target']}),
    ):
      view = session_view.session_view(index, _RECORD, _CHAT)
    stages = view['turns'][0]['stages']
    self.assertEqual(stages['prefill']['analyses']['kv'], _AVAILABLE)
    self.assertEqual(
        stages['decode']['analyses']['kv'],
        _unavailable(_why(session_view.REASONS, session_view.KV_NOT_CAPTURED)),
    )
    self.assertEqual(
        stages['decode']['kv'], {'moment': 'terminal', 'context_id': None}
    )

  def test_each_turn_carries_its_comparison(self):
    """Identical sides: the same history, input and generation."""
    _, view = self.captured

    first, second = (view['turns'][k]['comparison'] for k in (0, 1))
    conversation = _conversation(view, 1, 'ref')

    self.assertEqual(first['history'], {'prefill': None, 'decode': 'same'})
    self.assertEqual(second['history'], {'prefill': 'same', 'decode': 'same'})
    self.assertEqual(
        first['prefill']['conclusion'], {'kind': 'same', 'row': None}
    )
    self.assertEqual(
        [row['role'] for row in first['prefill']['messages']],
        ['system', 'user'],
    )
    self.assertEqual(
        first['prefill']['messages'], conversation['input_messages']
    )
    self.assertEqual(
        len(first['prefill']['rows']), len(conversation['input_tokens'])
    )
    self.assertEqual(set(first['alignments']), {'content', 'steps'})
    for compared in first['alignments'].values():
      self.assertEqual(
          compared['rows'],
          [
              {'ref': k, 'target': k, 'same': True}
              for k in range(len(conversation['tokens']))
          ],
      )
      self.assertEqual(
          compared['conclusion'],
          {'kind': 'same', 'step': None, 'prefix': 3},
      )
      self.assertIsNone(compared['first_mismatch'])
      self.assertIsNone(compared['first_divergence'])
      self.assertEqual(
          compared['stages'],
          {
              stage: {
                  'match_rate': 1.0,
                  'cumulative_match_rate': 1.0,
                  'first_mismatch': False,
              }
              for stage in ('prefill', 'decode')
          },
      )

  def test_a_turn_that_diverges_says_where(self):
    """Different replies: the first differing step, by both alignments."""
    chats = {
        side: fixtures.TinyChat('', 8, seed=seed)
        for seed, side in enumerate(capture_index.SIDES)
    }
    index = capture_index.CaptureIndex.create(
        self.tmp_path,
        {side: fixtures.side_info(chat) for side, chat in chats.items()},
    )
    replies = {'ref': ['a', 'b'], 'target': ['a', ' x', 'b']}
    completions = {
        side: fixtures.write_turn(
            index.turn_directory(side, 1),
            chat.send('Hi', replies[side], True),
        )
        for side, chat in chats.items()
    }
    index.add_turn(1, 'Hi', completions)

    view = session_view.session_view(index, _RECORD, _CHAT)

    compared = view['turns'][0]['comparison']['alignments']
    self.assertEqual(
        [
            (row['ref'], row['target'], row['same'])
            for row in compared['content']['rows']
        ],
        [(0, 0, True), (None, 1, False), (1, 2, True), (2, 3, True)],
    )
    self.assertEqual(
        compared['content']['first_divergence'],
        {'row': 1, 'ref': 1, 'target': 1, 'filled': 'ref', 'partner': 2},
    )
    for kind in ('content', 'steps'):
      self.assertEqual(
          compared[kind]['conclusion'],
          {'kind': 'diverged', 'step': 1, 'prefix': 1},
      )
      self.assertEqual(
          compared[kind]['stages']['decode']['first_mismatch'], True
      )
    self.assertEqual(
        view['turns'][0]['comparison']['history'],
        {'prefill': None, 'decode': 'same'},
    )
    # Every analysis has every column after the divergence too: those
    # whose sides read different tokens still compare, and say so, by one
    # rule. The first such column is the one after the divergence, or one
    # whose two sides sit at different positions.
    for kind in ('content', 'steps'):
      positions = compared[kind]['positions']
      columns = positions['columns']
      start = positions['start']
      end = start + len(positions['pairs'])
      for analysis in ('final_logits', 'sampling'):
        self.assertEqual(
            (
                columns[analysis]['start'],
                columns[analysis]['end'],
                columns[analysis]['omitted'],
            ),
            (start, end, []),
        )
      self.assertIsNone(columns['sampling']['context'])
      for analysis, values in (
          ('final_logits', 'logits'),
          ('kv', 'K and V'),
          ('intermediate_tensors', 'tensors'),
      ):
        self.assertEqual(
            columns[analysis]['context'],
            {
                'start': start + 2,
                'reason': metrics.reason(
                    token_analysis.CONTEXT_DIFFERENT,
                    f'After the divergence at position {start + 1} '
                    'the two sides read different tokens; the '
                    f'{values} still compare.',
                ),
            },
            (kind, analysis),
        )

  def test_each_analysis_has_the_columns_it_can_show(self):
    """No column no forward pass produced, nor one no KV holds."""
    index, view = self.captured
    for turn in view['turns']:
      n = turn['n']
      comparison = turn['comparison']
      sides = index.turn(n)['sides']
      processed = {}
      for moment in ('prefill_post', 'terminal'):
        counts = []
        for side in capture_index.SIDES:
          for snapshot in sides[side]['kv']:
            if snapshot['moment'] == moment:
              counts.append(snapshot['processed'])
        processed[moment] = max(counts)
      prefill = comparison['prefill']['positions']
      kv = prefill['columns']['kv']
      # After Prefill, the input's last token is not in the cache yet.
      self.assertEqual(
          (kv['start'], kv['end']),
          (prefill['start'], processed['prefill_post']),
      )
      self.assertEqual(
          [r['code'] for r in kv['omitted']],
          [session_view.KV_NOT_WRITTEN],
      )
      decode = comparison['alignments']['steps']['positions']
      kv = decode['columns']['kv']
      # At the end, the reply's last token is never fed.
      self.assertEqual(kv['end'], processed['terminal'])
      self.assertEqual(
          decode['columns']['final_logits']['end'],
          decode['start'] + len(decode['pairs']),
      )

  def test_each_stage_names_its_rows_positions(self):
    """Columns from the ref side's first position; each side its own."""
    _, view = self.captured
    for turn in view['turns']:
      n = turn['n']
      comparison = turn['comparison']
      ref, target = (_conversation(view, n, side) for side in ('ref', 'target'))
      stages = [
          (comparison['prefill'], 'input_tokens'),
          *(
              (compared, 'tokens')
              for compared in comparison['alignments'].values()
          ),
      ]
      for compared, key in stages:
        columns = compared['positions']
        self.assertEqual(columns['start'], ref[key][0]['position'])
        self.assertEqual(
            columns['pairs'],
            [
                [
                    ref[key][row['ref']]['position'],
                    target[key][row['target']]['position'],
                ]
                for row in compared['rows']
            ],
        )

  def test_views_are_kept_until_a_turn_is_added(self):
    """The same comparison answers until the capture grows."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    views = session_view.SessionViews()

    first = views.compared('chat-1', 1, index)
    self.assertIs(views.compared('chat-1', 1, index), first)
    self.assertEqual(
        views.session('chat-1', 1, index, _RECORD),
        session_view.session_view(index, _RECORD, _CHAT),
    )

    index.add_turn(2, _EXCHANGES[1][0], self._second_turn(index))

    again = views.compared('chat-1', 1, index)
    self.assertIsNot(again, first)
    self.assertEqual(sorted(again.comparisons), [1, 2])

  def test_only_the_last_chats_are_kept(self):
    """Views beyond the limit are made again."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    views = session_view.SessionViews(kept=1)

    first = views.compared('chat-1', 1, index)
    views.compared('chat-2', 1, index)
    self.assertIsNot(views.compared('chat-1', 1, index), first)

  def test_views_are_kept_up_to_a_size(self):
    """Views beyond the bytes allowed are made again, but the last one."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    size = session_view.compared(index).size
    self.assertGreater(size, 0)

    views = session_view.SessionViews(kept_bytes=2 * size)
    first = views.compared('chat-1', 1, index)
    second = views.compared('chat-2', 1, index)
    self.assertIs(views.compared('chat-1', 1, index), first)
    views.compared('chat-3', 1, index)
    self.assertIs(views.compared('chat-1', 1, index), first)
    self.assertIsNot(views.compared('chat-2', 1, index), second)

    alone = session_view.SessionViews(kept_bytes=size - 1)
    last = alone.compared('chat-1', 1, index)
    self.assertIs(alone.compared('chat-1', 1, index), last)

  def test_a_chats_pairs_are_kept_across_its_views(self):
    """A turn added makes a new view; the chat's pairs stay."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    views = session_view.SessionViews()
    first = views.compared('chat-1', 1, index)
    first.pairs.values(first.index, 1, [(0, 0), (1, 1)])
    kept = first.pairs.size()

    index.add_turn(2, _EXCHANGES[1][0], self._second_turn(index))
    again = views.compared('chat-1', 1, index)

    self.assertIsNot(again, first)
    self.assertIs(again.pairs, first.pairs)
    self.assertEqual(again.pairs.size(), kept)
    # Another capture of the chat is another chat: nothing is kept.
    other = fixtures.capture(
        pathlib.Path(self.create_tempdir().full_path),
        _EXCHANGES[:1],
        max_completion_tokens=4,
    )
    self.assertIsNot(views.compared('chat-1', 1, other).pairs, first.pairs)

  def test_kept_pairs_count_toward_the_size(self):
    """A chat whose pairs outgrow what is kept is dropped, pairs and all."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    size = session_view.compared(index).size
    views = session_view.SessionViews(kept_bytes=3 * size)
    first = views.compared('chat-1', 1, index)
    second = views.compared('chat-2', 1, index)

    with mock.patch.object(first.pairs, 'size', return_value=2 * size):
      self.assertIs(views.compared('chat-2', 1, index), second)

    self.assertIsNot(views.compared('chat-1', 1, index), first)

  def test_a_view_is_made_once_however_many_ask(self):
    """Who asks while the view is made waits for it, errors and all."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    views = session_view.SessionViews()
    release, started = threading.Event(), threading.Event()
    made = []
    real = session_view.compared

    def slow(index, pairs=None, observed=None):
      made.append(index)
      started.set()
      release.wait()
      return real(index, pairs, observed)

    answers = []
    with mock.patch.object(session_view, 'compared', side_effect=slow):
      threads = [
          threading.Thread(
              target=lambda: answers.append(views.compared('chat-1', 1, index))
          )
          for _ in range(3)
      ]
      threads[0].start()
      started.wait()
      for thread in threads[1:]:
        thread.start()
      threading.Event().wait(0.2)  # They ask while it is made.
      release.set()
      for thread in threads:
        thread.join()

    self.assertLen(made, 1)
    self.assertLen(answers, 3)
    self.assertTrue(all(answer is answers[0] for answer in answers))

    index.add_turn(2, _EXCHANGES[1][0], self._second_turn(index))
    release.clear()
    failures = []

    def failing(index, pairs=None, observed=None):
      del index, pairs, observed  # Unused.
      started.set()
      release.wait()
      raise OSError('unreadable')

    def ask():
      try:
        views.compared('chat-1', 1, index)
      except OSError as error:
        failures.append(error)

    started.clear()
    with mock.patch.object(session_view, 'compared', side_effect=failing):
      threads = [threading.Thread(target=ask) for _ in range(2)]
      threads[0].start()
      started.wait()
      threads[1].start()
      threading.Event().wait(0.2)
      release.set()
      for thread in threads:
        thread.join()
    self.assertLen(failures, 2)
    self.assertEqual(
        sorted(views.compared('chat-1', 1, index).comparisons), [1, 2]
    )

  def _second_turn(self, index):
    """Writes turn 2 of a capture of `_EXCHANGES[:1]`; its completions."""
    chats = fixtures.TinyChat('', 4, 0), fixtures.TinyChat('', 4, 1)
    for chat in chats:
      chat.send(*_EXCHANGES[0])
    return {
        side: fixtures.write_turn(
            index.turn_directory(side, 2), chat.send(*_EXCHANGES[1])
        )
        for side, chat in zip(capture_index.SIDES, chats)
    }

  def test_a_view_made_while_a_turn_failed_to_write_is_not_kept(self):
    """add_turn drops the turn again; the turn added next is another."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    views = session_view.SessionViews()
    completions = self._second_turn(index)
    during = []

    def fail_to_write():
      during.append(views.compared('chat-1', 1, index))
      raise OSError('disk full')

    with mock.patch.object(index, '_write', side_effect=fail_to_write):
      with self.assertRaises(OSError):
        index.add_turn(2, _EXCHANGES[1][0], completions)
    self.assertLen(during, 1)
    stale = during[0]
    self.assertEqual(sorted(stale.comparisons), [1, 2])
    self.assertLen(index.turns, 1)

    index.add_turn(2, _EXCHANGES[1][0], completions)
    added = views.compared('chat-1', 1, index)
    self.assertIsNot(added, stale)
    self.assertIs(added.index.turns[-1], index.turns[-1])

  def test_a_session_is_of_its_views_turns(self):
    """A turn added, or rejected, after the view does not show in it."""
    root = pathlib.Path(self.create_tempdir().full_path)
    index = fixtures.capture(root, _EXCHANGES[:1], max_completion_tokens=4)
    views = session_view.SessionViews()
    made = views.compared
    grow = []

    def compared_then_grow(record_id, n, index):
      view = made(record_id, n, index)
      grow.pop()()
      return view

    views.compared = compared_then_grow
    completions = self._second_turn(index)

    grow.append(lambda: index.add_turn(2, _EXCHANGES[1][0], completions))
    session = views.session('chat-1', 1, index, _RECORD)
    self.assertEqual([turn['n'] for turn in session['turns']], [1])
    self.assertLen(index.turns, 2)

    def reject():
      completions = self._third_turn(index)
      completions['target']['usage']['completion_tokens'] += 1
      with self.assertRaises(capture_index.Rejected):
        index.add_turn(3, _EXCHANGES[2][0], completions)

    grow.append(reject)
    session = views.session('chat-1', 1, index, _RECORD)
    self.assertEqual([turn['n'] for turn in session['turns']], [1, 2])
    self.assertLen(index.rejected, 1)
    views.compared = made
    session = views.session('chat-1', 1, index, _RECORD)
    self.assertEqual(
        [(turn['n'], 'rejected' in turn) for turn in session['turns']],
        [(1, False), (2, False), (3, True)],
    )

  def _third_turn(self, index):
    """Writes turn 3 after `_second_turn`; its completions."""
    chats = fixtures.TinyChat('', 4, 0), fixtures.TinyChat('', 4, 1)
    for chat in chats:
      chat.send(*_EXCHANGES[0])
      chat.send(*_EXCHANGES[1])
    return {
        side: fixtures.write_turn(
            index.turn_directory(side, 3), chat.send(*_EXCHANGES[2])
        )
        for side, chat in zip(capture_index.SIDES, chats)
    }


if __name__ == '__main__':
  test_main.main()
