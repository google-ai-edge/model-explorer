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

"""Tests of indexing a chat's capture."""

import json
import os
import pathlib
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from google.protobuf import json_format
from model_debugger.protocol import runner_pb2
from model_debugger.server.capture import capture_index
from model_debugger.server.testing import capture_fixtures as fixtures
from model_debugger.server.testing import real_capture
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors
import numpy
from safetensors.numpy import load_file
from safetensors.numpy import save_file

# A prompt that takes several prefill chunks.
_LONG = ' '.join(f'w{i}' for i in range(20))


# Contradictions between files and completion are runner bugs: the turn is
# rejected.


# A prompt of 115 words: turn 1 then leaves room for 4 generated tokens in
# the tiny context.
_FILLING = ' '.join(['w'] * 115)


def _broken(
    tmp_path: pathlib.Path,
    change: Any,
    reply: tuple[str, ...] = ('a', 'b'),
    stop: bool = True,
) -> str:
  """Adds one turn whose files or completion `change` edits; returns why."""
  chat = fixtures.TinyChat()
  index = capture_index.CaptureIndex.create(
      tmp_path, {'ref': fixtures.side_info(chat)}
  )
  turn = chat.send('Hi there', list(reply), stop)
  completion = fixtures.write_turn(index.turn_directory('ref', 1), turn)
  change(index.turn_directory('ref', 1), completion)
  try:
    index.add_turn(1, 'Hi there', {'ref': completion})
  except capture_index.Rejected as error:
    message = error.reason
  else:
    raise AssertionError('the broken turn was added')
  assert not index.turns
  assert index.rejected[0]['rejected'] == {
      'side': 'ref',
      'reason': message,
      'on_read': False,
  }
  return message


def _cut(path: pathlib.Path) -> None:
  """Cuts a file short by one byte, as an interrupted copy would."""
  path.write_bytes(path.read_bytes()[:-1])


def _set_nan_logit(logits: Any) -> None:
  logits[3] = numpy.nan


def _decode_files(directory, step):
  return list(directory.glob(f'decode_*_step_{step}.safetensors'))


def _greedy_turn(tmp_path, nth, change, reply=('a', 'b'), top_k=1):
  """Adds a turn whose `nth` decode step's logits `change` edits.

  Args:
    tmp_path: Directory path for the session.
    nth: Decode step index to edit.
    change: Function to modify logits.
    reply: Sequence of reply tokens.
    top_k: Top-k value for sampling.

  Returns:
    The index, and the step changed.
  """
  chat = fixtures.TinyChat()
  fixed = fixtures.side_info(chat)
  fixed['chat']['options']['top_k'] = top_k
  index = capture_index.CaptureIndex.create(tmp_path, {'ref': fixed})
  turn = chat.send('Hi there', list(reply), True)
  directory = index.turn_directory('ref', 1)
  completion = fixtures.write_turn(directory, turn)
  step = fixtures.edit_logits(directory, nth, change)
  index.add_turn(1, 'Hi there', {'ref': completion})
  return index, step


# REAL_SESSIONS names a directory of real sessions: sessionA (four turns,
# one prefilled in chunks) and sessionD (a turn that thinks), each turn in
# turn<n>/ with the runner's files and its Completion in _completion.json.
_REAL_SESSIONS = os.environ.get('REAL_SESSIONS', '')


_REAL_SESSION = pathlib.Path(_REAL_SESSIONS, 'sessionA')


_GEMMA = {
    'start_token': 2,
    'stop_tokens': [[1], [50], [106]],
    'load': {'device': 'CPU', 'context_length': 4096},
    'chat': {'messages': [], 'options': {'top_k': 1}},
}


def _real_completion(path: pathlib.Path) -> dict[str, Any]:
  """Reads a real Completion of a runner before fed_tokens, as stored."""
  data = json.loads(path.read_text())
  data = data.get('completion', data)
  prompt = data['modelDebugger'].pop('promptTokens')
  completion = capture_index.completion_dict(
      json_format.ParseDict(data, runner_pb2.Completion())
  )
  del completion['fed_tokens']
  completion['prompt_tokens'] = [
      {'id': token['id'], 'text': token['text']} for token in prompt
  ]
  return completion


def _link_turn(index, n, source):
  for side in index.sides:
    directory = index.turn_directory(side, n)
    for path in source.iterdir():
      if not path.name.startswith('_'):
        os.symlink(path, directory / path.name)


_REAL_THINKING = pathlib.Path(_REAL_SESSIONS, 'sessionD')


class CaptureIndexTest(absltest.TestCase):
  """Tests of indexing a chat's capture."""

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def test_side_entry_keeps_the_fixed_facts_of_a_side(self):
    """A side's fixed facts, as index.json holds them.

    Its model, runtime, device, Load, chat, tokens and the model's own
    sampling.
    """
    info = runner_pb2.RunnerInfo(
        host=runner_pb2.HostInfo(architecture='arm64', cpu_count=12),
        runtime=runner_pb2.RuntimeInfo(
            name=runner_pb2.RuntimeInfo.LITERT_LM, version='a0f1e973'
        ),
    )
    load = runner_pb2.LoadRequest(
        sha256='0' * 64, device=runner_pb2.CPU, context_length=128
    )
    load.runtime_options.update({'cpuThreads': 4})
    chat = runner_pb2.CreateChatRequest(
        messages=[
            runner_pb2.ChatMessage(
                role=runner_pb2.ChatMessage.SYSTEM, content='Hi'
            )
        ],
        options=runner_pb2.ChatOptions(max_completion_tokens=32, top_k=1),
    )
    loaded = runner_pb2.LoadResponse(
        start_token=2,
        stop_tokens=[
            runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[1])),
            runner_pb2.StopToken(text='<end>'),
        ],
        sampler_defaults=runner_pb2.SamplerDefaults(
            type=runner_pb2.SamplerDefaults.TOP_P, top_k=48, top_p=0.5
        ),
    )
    entry = capture_index.side_entry(
        model_sha256='0' * 64,
        model_name='gemma.litertlm',
        model_size=1000,
        device_id='local',
        device_name='This computer',
        info=info,
        load=load,
        chat=chat,
        loaded=loaded,
    )
    self.assertEqual(
        entry,
        {
            'model': {
                'sha256': '0' * 64,
                'name': 'gemma.litertlm',
                'size': 1000,
            },
            'runtime': {'name': 'LiteRT-LM', 'version': 'a0f1e973'},
            'host': {'architecture': 'arm64', 'cpu_count': 12},
            'device_id': 'local',
            'device_name': 'This computer',
            'load': {
                'device': 'CPU',
                'context_length': 128,
                'runtime_options': {'cpuThreads': 4.0},
            },
            'chat': {
                'messages': [{'role': 'system', 'content': 'Hi'}],
                'options': {'max_completion_tokens': 32, 'top_k': 1},
            },
            'start_token': 2,
            'stop_tokens': [[1], {'text': '<end>'}],
            'sampler_defaults': {
                'type': 'TOP_P',
                'top_k': 48,
                'top_p': 0.5,
            },
        },
    )
    # A runner that names no start token says so.
    loaded.ClearField('start_token')
    self.assertIsNone(
        capture_index.side_entry(
            model_sha256='0' * 64,
            model_name='gemma.litertlm',
            model_size=1000,
            device_id='local',
            device_name='This computer',
            info=info,
            load=load,
            chat=chat,
            loaded=loaded,
        )['start_token']
    )

  def test_turn_one_feeds_the_start_token_and_the_prompt(self):
    """Turn one feeds the start token and the prompt."""
    index = fixtures.capture(
        self.tmp_path,
        [('What is it?', ['Four', '.'], True)],
        system='Be brief.',
    )

    side = index.turn(1)['sides']['ref']
    fed = side['completion']['fed_tokens']

    self.assertEqual(
        side['input'][0],
        {
            'id': fixtures.BOS,
            'text': '<bos>',
            'origin': 'start',
            'position': 0,
        },
    )
    # The input is what the prefill fed, then what the first decode step
    # fed: the prompt's last token.
    self.assertEqual(
        [(t['id'], t['text'], t['position']) for t in side['input']],
        [
            (t['id'], t['text'], t['position'])
            for t in fed[: len(side['input'])]
        ],
    )
    self.assertEqual(
        [t['phase'] for t in fed[: len(side['input'])]],
        ['prefill'] * (len(side['input']) - 1) + ['decode'],
    )
    self.assertEqual(side['input'][-1]['text'], '\n')
    self.assertEqual(
        {token['origin'] for token in side['input'][1:]}, {'prompt'}
    )
    self.assertEqual(
        [token['position'] for token in side['input']],
        list(range(len(side['input']))),
    )
    self.assertEqual(side['rewrite'], [])
    usage = side['completion']['usage']
    self.assertLen(side['input'], usage['prompt_tokens'])
    self.assertEqual(
        usage['total_tokens'],
        usage['prompt_tokens'] + usage['completion_tokens'],
    )

  def test_a_stop_token_is_generated_but_not_released(self):
    """A stop token is generated but not released."""
    index = fixtures.capture(self.tmp_path, [('Hi', ['Four', '.'], True)])

    side = index.turn(1)['sides']['ref']
    first = len(side['input'])
    chunks = sum(
        forward['phase'] == 'prefill' for forward in side['forwards']
    )

    self.assertEqual(
        side['output'],
        [
            {
                'id': fixtures.token('Four')[0],
                'text': 'Four',
                'position': first,
                'forward': chunks,
                'released': True,
            },
            {
                'id': fixtures.token('.')[0],
                'text': '.',
                'position': first + 1,
                'forward': chunks + 1,
                'released': True,
            },
            {
                'id': fixtures.TURN_CLOSE,
                'text': None,
                'position': first + 2,
                'forward': chunks + 2,
                'released': False,
                'stop': True,
            },
        ],
    )
    self.assertEqual(side['stop_reason'], 'eos')
    self.assertEqual(side['completion']['finish_reason'], 'stop')

  def test_a_reply_cut_at_the_limit_has_no_stop(self):
    """A reply cut at the limit has no stop."""
    index = fixtures.capture(
        self.tmp_path,
        [('Hi', ['a', 'b', 'c'], True)],
        max_completion_tokens=2,
    )

    side = index.turn(1)['sides']['ref']

    self.assertEqual(
        [token['released'] for token in side['output']], [True, True]
    )
    self.assertEqual(side['stop_reason'], 'max_output_tokens')
    self.assertEqual(side['completion']['finish_reason'], 'length')

  def test_a_stop_at_the_limit_is_an_end_of_sequence(self):
    """A stop at the limit is an end of sequence."""
    # The runner reports LENGTH when the stop token is the limit's last
    # token.
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a'], True)], max_completion_tokens=2
    )

    side = index.turn(1)['sides']['ref']

    self.assertEqual(side['completion']['finish_reason'], 'length')
    self.assertIs(side['output'][-1]['stop'], True)
    self.assertEqual(side['stop_reason'], 'eos')

  def test_a_reply_cut_by_the_context_has_filled_it(self):
    """A reply cut by the context has filled it; its limit was not hit."""
    # Turn 1's 115 words and 9 other input tokens leave room for 4.
    index = fixtures.capture(
        self.tmp_path, [(_FILLING, ['a', 'b', 'c', 'd', 'e'], True)]
    )

    side = index.turn(1)['sides']['ref']

    self.assertLen(side['output'], 4)
    self.assertNotIn('stop', side['output'][-1])
    self.assertEqual(
        side['completion']['usage']['total_tokens'],
        fixtures.CONTEXT_LENGTH,
    )
    self.assertEqual(side['completion']['finish_reason'], 'length')
    self.assertEqual(side['stop_reason'], 'context_full')
    self.assertEqual(
        capture_index.ending(index.sides['ref'], side),
        f'Context full ({fixtures.CONTEXT_LENGTH} tokens)',
    )

  def test_a_reply_at_both_limits_has_filled_the_context(self):
    """A reply that reaches its own limit as it fills the context."""
    # Turn 1's 115 words and 9 other input tokens leave room for 4, the
    # reply's own limit too.
    index = fixtures.capture(
        self.tmp_path,
        [(_FILLING, ['a', 'b', 'c', 'd', 'e'], True)],
        max_completion_tokens=4,
    )

    side = index.turn(1)['sides']['ref']

    usage = side['completion']['usage']
    self.assertEqual(usage['completion_tokens'], 4)
    self.assertEqual(usage['total_tokens'], fixtures.CONTEXT_LENGTH)
    self.assertEqual(side['completion']['finish_reason'], 'length')
    self.assertEqual(side['stop_reason'], 'context_full')

  def test_a_stop_that_fills_the_context_is_a_full_context(self):
    """A stop token that fills the context still ends the chat's room."""
    index = fixtures.capture(self.tmp_path, [(_FILLING, ['a', 'b', 'c'], True)])

    side = index.turn(1)['sides']['ref']

    self.assertIs(side['output'][-1]['stop'], True)
    self.assertEqual(side['completion']['finish_reason'], 'length')
    self.assertEqual(side['stop_reason'], 'context_full')

  def test_the_ending_of_each_reason(self):
    """The text of each reason names its limit."""
    index = fixtures.capture(
        self.tmp_path,
        [('Hi', ['a', 'b', 'c'], True)],
        max_completion_tokens=2,
    )
    fixed = index.sides['ref']

    self.assertEqual(
        [
            capture_index.ending(fixed, {'stop_reason': reason})
            for reason in ('eos', 'max_output_tokens', 'context_full')
        ],
        [
            'Stop token',
            'Output limit reached (2 tokens)',
            f'Context full ({fixtures.CONTEXT_LENGTH} tokens)',
        ],
    )

  def test_a_reason_is_derived_and_not_stored(self):
    """index.json keeps the finish reason; why the reply ended is read."""
    fixtures.capture(
        self.tmp_path, [(_FILLING, ['a', 'b', 'c', 'd', 'e'], True)]
    )
    stored = json.loads((self.tmp_path / 'index.json').read_text())

    (entry,) = {
        side['completion']['finish_reason']
        for side in stored['turns'][0]['sides'].values()
    }
    self.assertEqual(entry, 'length')
    self.assertNotIn('stop_reason', stored['turns'][0]['sides']['ref'])
    loaded = capture_index.CaptureIndex.load(self.tmp_path)
    self.assertEqual(
        loaded.turn(1)['sides']['ref']['stop_reason'], 'context_full'
    )

  def test_a_later_turn_first_feeds_the_previous_last_token(self):
    """A later turn first feeds the previous last token."""
    index = fixtures.capture(
        self.tmp_path,
        [
            ('Hi', ['a'], True),
            ('More', ['b', 'c'], False),
            ('Again', ['d'], True),
        ],
        max_completion_tokens=4,
    )

    for n in (2, 3):
      side = index.turn(n)['sides']['ref']
      before = index.turn(n - 1)['sides']['ref']
      previous = before['output'][-1]
      self.assertEqual(
          (side['input'][0]['id'], side['input'][0]['origin']),
          (previous['id'], 'carried'),
      )
      self.assertEqual(
          side['input'][0]['position'],
          before['completion']['context_token_count'] - 1,
      )
      self.assertEqual(previous['position'], side['input'][0]['position'])
      self.assertEqual(side['input'][1]['origin'], 'prompt')
    # After a stop the carried token is the stop, named by the runner;
    # after a cut, a released one.
    self.assertEqual(
        index.turn(2)['sides']['ref']['input'][0]['id'], fixtures.TURN_CLOSE
    )
    self.assertEqual(
        index.turn(2)['sides']['ref']['input'][0]['text'], '<turn|>'
    )
    self.assertEqual(index.turn(3)['sides']['ref']['input'][0]['text'], 'c')

  def test_a_turn_that_thinks_rewrites_its_reply_without_the_thoughts(self):
    """A reply that thought is fed again without the thoughts, at once."""
    index = fixtures.capture(
        self.tmp_path,
        [
            ('Hi', ['ok'], True, ['hmm', ' yes']),
            ('More', ['b'], True),
        ],
        max_completion_tokens=12,
    )

    first = index.turn(1)['sides']['ref']
    second = index.turn(2)['sides']['ref']

    self.assertEqual(first['completion']['reasoning_content'], 'hmm yes')
    self.assertEqual(first['completion']['content'], 'ok')
    self.assertEqual(
        [t['text'] for t in first['output'][:3]],
        ['<|channel>', 'thought', '\n'],
    )
    # From the first decode step's position, which held the prompt's last
    # token: that token again, `ok<turn|>`, and `\n` held back.
    start = first['input'][-1]['position']
    self.assertEqual(
        [(t['text'], t['position']) for t in first['rewrite']],
        [('\n', start), ('ok', start + 1), ('<turn|>', start + 2)],
    )
    self.assertNotIn(None, [t['id'] for t in first['rewrite']])
    # The rewrite's chunk is the last forward pass, after the decode steps.
    rewrite = first['forwards'][-1]
    self.assertEqual(
        (rewrite['phase'], rewrite['input_positions']),
        ('prefill', [start, start + 2]),
    )
    self.assertEqual({t['forward'] for t in first['rewrite']}, {rewrite['id']})
    self.assertEqual(
        [f['phase'] for f in first['forwards']],
        ['prefill'] * (len(first['forwards']) - len(first['output']) - 1)
        + ['decode'] * len(first['output'])
        + ['prefill'],
    )
    # The KV cache after the prefill is the input's, not the rewrite's.
    moments = {s['moment']: s for s in first['kv']}
    self.assertEqual(moments['prefill_post']['processed'], start)
    self.assertEqual(
        moments['terminal']['processed'], first['output'][-1]['position']
    )
    # The counts: the context the reply came from, the reply, and what the
    # context holds after the rewrite, the held token included.
    usage = first['completion']['usage']
    self.assertEqual(usage['prompt_tokens'], start + 1)
    self.assertLen(first['output'], usage['completion_tokens'])
    self.assertEqual(usage['total_tokens'], first['output'][-1]['position'] + 1)
    self.assertEqual(first['completion']['context_token_count'], start + 4)
    # Turn 2 starts with the token turn 1 held back, after the rewrite.
    self.assertEqual(
        second['input'][0],
        {
            'id': fixtures.NEWLINE,
            'text': '\n',
            'origin': 'carried',
            'position': start + 3,
        },
    )
    self.assertEqual({t['origin'] for t in second['input'][1:]}, {'prompt'})
    self.assertEqual(second['kv'][0]['processed'], start + 3)
    self.assertEqual(second['rewrite'], [])

  def test_feeds_are_the_inputs_and_the_decode_steps_not_a_rewrite(self):
    """Each position's forward pass and row; the last token has none."""
    index = fixtures.capture(
        self.tmp_path,
        [('Hi', ['ok'], True, ['hmm', ' yes'])],
        max_completion_tokens=12,
    )
    side = index.turn(1)['sides']['ref']
    feeds = capture_index.feeds(side)
    rewritten = {token['forward'] for token in side['rewrite']}
    self.assertNotEmpty(rewritten)
    self.assertNotIn(
        True, [feed.forward['id'] in rewritten for feed in feeds.values()]
    )
    last_input = side['input'][-1]['position']
    for token in side['input']:
      feed = feeds[token['position']]
      self.assertEqual(feed.position, token['position'])
      self.assertEqual(
          feed.forward['input_positions'][0] + feed.row, feed.position
      )
      self.assertEqual(
          feed.forward['phase'],
          'decode' if token['position'] == last_input else 'prefill',
      )
    # Each generated token is read by the step that produces the next.
    output = side['output']
    for token, after in zip(output, output[1:]):
      feed = feeds[token['position']]
      self.assertEqual(feed.forward['id'], after['forward'])
      self.assertEqual(feed.row, 0)
    self.assertNotIn(output[-1]['position'], feeds)

  def test_prefill_chunks_come_first_in_position_order(self):
    """Prefill chunks come first in position order."""
    index = fixtures.capture(self.tmp_path, [(_LONG, ['a'], True)])

    side = index.turn(1)['sides']['ref']
    prefills = [f for f in side['forwards'] if f['phase'] == 'prefill']
    decodes = [f for f in side['forwards'] if f['phase'] == 'decode']

    self.assertEqual(
        [f['id'] for f in side['forwards']],
        list(range(len(side['forwards']))),
    )
    self.assertEqual(side['forwards'][: len(prefills)], prefills)
    # Turn 1's first chunk holds back its last id: one row fewer.
    self.assertEqual(
        prefills[0]['input_positions'], [0, fixtures.PREFILL_SIZE - 2]
    )
    for before, after in zip(prefills, prefills[1:]):
      self.assertEqual(
          after['input_positions'][0], before['input_positions'][1] + 1
      )
    # A chunk is labelled with its last position + 2.
    self.assertEqual(
        [f['step'] for f in prefills],
        [f['input_positions'][1] + 2 for f in prefills],
    )
    self.assertEqual(prefills[-1]['step'], decodes[0]['step'])
    self.assertEqual(decodes[0]['input_position'], len(side['input']) - 1)
    self.assertEqual(
        decodes[0]['input_positions'], [decodes[0]['input_position']]
    )
    expected_tensors = {
        'pre_input_pos',
        'pre_mask',
        'pre_embeddings',
        'pre_per_layer_embeddings',
        'pre_param_tensor',
    }
    for m in ('pre', 'post'):
      for k in 'kv':
        for layer in (0, 1):
          expected_tensors.add(f'{m}_kv_cache_{k}_{layer}')
    self.assertEqual(set(prefills[0]['tensors']), expected_tensors)
    self.assertEqual(
        decodes[0]['tensors']['post_logits'],
        f'decode_post_logits_step_{decodes[0]["step"]}.safetensors',
    )

  def test_each_forward_lists_its_files_by_key_and_its_rows(self):
    """Observed tensors by their keys; a chunk's rows, its padding too."""
    index = fixtures.capture(
        self.tmp_path, [(_LONG, ['a', 'b'], True)], observe=True
    )

    side = index.turn(1)['sides']['ref']
    prefill, decode = side['forwards'][0], side['forwards'][-1]
    self.assertIn('post_tap_sg1_op3_out0', prefill['tensors'])
    self.assertIn('post_tap_sg0_op5_out0', decode['tensors'])
    # An output the signature has already is under its own key.
    self.assertIn('post_activations', decode['tensors'])
    self.assertEqual(prefill['rows'], fixtures.PREFILL_SIZE)
    self.assertEqual(decode['rows'], 1)
    self.assertNotIn('observed', prefill)

  def test_observed_files_of_one_signature_name_one_subgraph(self):
    """A file naming another subgraph of its signature rejects the turn."""

    def change(directory, completion):
      del completion  # Unused.
      path = next(directory.glob('decode_post_tap_sg0_op3_out0_step_*'))
      step = path.name.rsplit('_', 1)[1].split('.')[0]
      path.unlink()

      save_file(
          {'post_tap_sg4_op3_out0': numpy.zeros((1, 1, 6), numpy.float32)},
          str(
              directory
              / f'decode_post_tap_sg4_op3_out0_step_{step}.safetensors'
          ),
          metadata={'signature': 'decode', 'step': step},
      )

    chat = fixtures.TinyChat(observe=True)
    index = capture_index.CaptureIndex.create(
        self.tmp_path, {'ref': fixtures.side_info(chat)}
    )
    turn = chat.send('Hi there', ['a', 'b'], True)
    completion = fixtures.write_turn(index.turn_directory('ref', 1), turn)
    change(index.turn_directory('ref', 1), completion)
    with self.assertRaisesRegex(capture_index.Rejected, 'subgraph'):
      index.add_turn(1, 'Hi there', {'ref': completion})

  def test_kv_snapshots_before_and_after_prefill_and_at_the_end(self):
    """KV snapshots before and after prefill and at the end."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a'], True), (_LONG, ['b', 'c'], True)]
    )

    side = index.turn(2)['sides']['ref']
    prefills = [f for f in side['forwards'] if f['phase'] == 'prefill']
    decodes = [f for f in side['forwards'] if f['phase'] == 'decode']
    kv = side['kv']

    self.assertEqual(
        [(s['moment'], s['step'], s['processed']) for s in kv],
        [
            (
                'prefill_pre',
                prefills[0]['step'],
                side['input'][0]['position'],
            ),
            (
                'prefill_post',
                prefills[-1]['step'],
                prefills[-1]['input_positions'][1] + 1,
            ),
            (
                'terminal',
                decodes[-1]['step'],
                decodes[-1]['input_position'] + 1,
            ),
        ],
    )
    layer = kv[2]['layers'][1]
    self.assertEqual(layer['layer'], 1)
    self.assertEqual(
        layer['key'],
        {
            'file': (
                f'decode_post_kv_cache_k_1_step_{decodes[-1]["step"]}.safetensors'
            ),
            'dtype': 'I8',
            'shape': [1, 1, fixtures.CONTEXT_LENGTH, fixtures.HEAD_DIM],
            'layout': ['batch', 'kv_head', 'sequence', 'head_dim'],
        },
    )
    self.assertEqual(
        layer['value']['shape'],
        [
            1,
            1,
            fixtures.HEAD_DIM,
            fixtures.CONTEXT_LENGTH,
        ],
    )
    self.assertEqual(
        layer['value']['layout'],
        [
            'batch',
            'kv_head',
            'head_dim',
            'sequence',
        ],
    )

  def test_kv_that_does_not_span_the_context_has_no_layout(self):
    """KV that does not span the context has no layout."""
    chat = fixtures.TinyChat()
    index = capture_index.CaptureIndex.create(
        self.tmp_path, {'ref': fixtures.side_info(chat)}
    )
    index.sides['ref']['load']['context_length'] = 4096
    completion = fixtures.write_turn(
        index.turn_directory('ref', 1), chat.send('Hi', ['a'])
    )

    index.add_turn(1, 'Hi', {'ref': completion})

    layer = index.turn(1)['sides']['ref']['kv'][0]['layers'][0]
    self.assertIsNone(layer['key']['layout'])
    self.assertEqual(
        layer['key']['shape'],
        [
            1,
            1,
            fixtures.CONTEXT_LENGTH,
            fixtures.HEAD_DIM,
        ],
    )

  def test_pytorch_kv_takes_its_layout_from_its_metadata(self):
    """A PyTorch KV file's `layout` is read; its sequence is processed."""
    index = fixtures.capture(
        self.tmp_path,
        [('Hi', ['a'], True), (_LONG, ['b', 'c'], True)],
        pytorch=True,
    )

    first = index.turn(1)['sides']['ref']['kv']
    kv = index.turn(2)['sides']['ref']['kv']

    # Before the first prefill the cache is empty, and not written.
    self.assertEqual([s['moment'] for s in first], ['prefill_post', 'terminal'])
    self.assertEqual(
        [s['moment'] for s in kv],
        ['prefill_pre', 'prefill_post', 'terminal'],
    )
    for snapshot in kv:
      for kind in ('key', 'value'):
        stored = snapshot['layers'][1][kind]
        self.assertEqual(stored['dtype'], 'BF16')
        self.assertEqual(
            stored['shape'],
            [1, 1, snapshot['processed'], fixtures.HEAD_DIM],
        )
        self.assertEqual(
            stored['layout'],
            ['batch', 'kv_head', 'sequence', 'head_dim'],
        )
        self.assertNotIn('positions', stored)

  def test_kv_positions_say_which_positions_a_layer_keeps(self):
    """A KV file's `kv_positions` is kept; a malformed one rejects."""
    chat = fixtures.TinyChat(pytorch=True)
    index = capture_index.CaptureIndex.create(
        self.tmp_path, {'ref': fixtures.side_info(chat)}
    )
    directory = index.turn_directory('ref', 1)
    completion = fixtures.write_turn(directory, chat.send('Hi', ['a']))
    for path in directory.glob('*_kv_cache_*_0_*'):
      fixtures.set_metadata(path, kv_positions='2-5')

    index.add_turn(1, 'Hi', {'ref': completion})

    layers = index.turn(1)['sides']['ref']['kv'][0]['layers']
    self.assertEqual(layers[0]['key']['positions'], [2, 5])
    self.assertNotIn('positions', layers[1]['key'])

  def test_kv_metadata_not_of_its_form_rejects_the_turn(self):
    """KV metadata that is not of its form rejects the turn."""
    for n, metadata in enumerate((
        {'kv_positions': '5-2'},
        {'kv_positions': 'all'},
        {'layout': 'k:bhs,v:bhsd'},
    )):
      with self.subTest(metadata):
        chat = fixtures.TinyChat(pytorch=True)
        index = capture_index.CaptureIndex.create(
            self.tmp_path / str(n), {'ref': fixtures.side_info(chat)}
        )
        directory = index.turn_directory('ref', 1)
        completion = fixtures.write_turn(directory, chat.send('Hi', ['a']))
        for path in directory.glob('*_kv_cache_k_0_*'):
          fixtures.set_metadata(path, **metadata)

        with self.assertRaises(capture_index.Rejected):
          index.add_turn(1, 'Hi', {'ref': completion})

  def test_pytorch_bf16_logits_confirm_greedy_tokens(self):
    """BF16 logits are read for the tokens greedy decoding generated."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a', 'b'], True)], pytorch=True
    )

    side = index.turn(1)['sides']['ref']

    self.assertEqual([t['text'] for t in side['output'][:2]], ['a', 'b'])
    self.assertTrue(side['output'][-1]['stop'])
    self.assertEqual(index.sides['ref']['runtime']['name'], 'PyTorch')

  def test_files_are_listed_with_size(self):
    """Files are listed with their size."""
    index = fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])

    side = index.turn(1)['sides']['target']
    directory = self.tmp_path / 'target' / 'turn-1'

    self.assertEqual(side['dir'], 'target/turn-1')
    self.assertEqual(
        [f['path'] for f in side['files']],
        sorted(p.name for p in directory.iterdir()),
    )
    for file in side['files']:
      size = (directory / file['path']).stat().st_size
      self.assertEqual(file, {'path': file['path'], 'size': size})
    self.assertEqual(
        index.path('target', 1, 'generated_tokens.jsonl'),
        directory / 'generated_tokens.jsonl',
    )

  def test_load_reads_what_add_turn_wrote(self):
    """Load reads what add turn wrote."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a'], True), ('Yo', ['b'], True)]
    )

    loaded = capture_index.CaptureIndex.load(self.tmp_path)

    self.assertEqual(loaded.sides, index.sides)
    self.assertEqual(loaded.turns, index.turns)
    self.assertEqual(
        json.loads((self.tmp_path / 'index.json').read_text())[
            'format_version'
        ],
        1,
    )

  def test_index_json_holds_only_the_facts(self):
    """index.json keeps each turn's prompt, completions and files only."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a'], True), ('Yo', ['b'], True)]
    )

    stored = json.loads((self.tmp_path / 'index.json').read_text())

    self.assertEqual(stored['sides'], index.sides)
    for turn, kept in zip(index.turns, stored['turns'], strict=True):
      self.assertEqual(
          kept,
          {
              'n': turn['n'],
              'prompt': turn['prompt'],
              'sides': {
                  side: {
                      'completion': entry['completion'],
                      'files': entry['files'],
                  }
                  for side, entry in turn['sides'].items()
              },
          },
      )

  def test_load_derives_again_what_an_older_index_stored(self):
    """Fields an older server derived and stored are derived again."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a'], True), ('Yo', ['b'], True)]
    )
    path = self.tmp_path / 'index.json'
    stored = json.loads(path.read_text())
    for turn in stored['turns']:
      for entry in turn['sides'].values():
        entry.update(
            dir='elsewhere',
            stop_reason='unknown',
            input=[],
            output=[],
            forwards=[],
            kv=[],
        )
    path.write_text(json.dumps(stored))

    loaded = capture_index.CaptureIndex.load(self.tmp_path)

    self.assertEqual(loaded.turns, index.turns)

  def test_load_refuses_turns_out_of_order(self):
    """A stored turn that does not follow the one before is refused."""
    fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])
    path = self.tmp_path / 'index.json'
    stored = json.loads(path.read_text())
    stored['turns'][0]['n'] = 2
    path.write_text(json.dumps(stored))

    with self.assertRaisesRegex(ValueError, 'turn 2 after turn 0'):
      capture_index.CaptureIndex.load(self.tmp_path)

  def test_load_without_a_capture_is_not_found(self):
    """Load without a capture is not found."""
    with self.assertRaises(errors.NotFound):
      capture_index.CaptureIndex.load(self.tmp_path)

  def test_turn_out_of_range_is_not_found(self):
    """Turn out of range is not found."""
    index = fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])

    for n in (0, 2):
      with self.assertRaises(errors.NotFound):
        index.turn(n)

  def test_a_snapshot_does_not_grow_with_the_capture(self):
    """Turns added after it are the capture's, not the snapshot's."""
    index = fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])
    snapshot = index.snapshot()
    index.turns.append({'n': 2})
    index.rejected.append({'n': 3})

    self.assertEqual([turn['n'] for turn in snapshot.turns], [1])
    self.assertIs(snapshot.turns[0], index.turns[0])
    self.assertEqual(snapshot.rejected, [])
    self.assertIs(snapshot.sides, index.sides)

  def test_add_turn_must_follow_the_last(self):
    """Add turn must follow the last."""
    index = fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])

    with self.assertRaisesRegex(ValueError, 'Turn 3 cannot follow turn 1'):
      index.add_turn(3, 'x', {})

  def test_a_rejected_turn_is_recorded_and_its_files_kept(self):
    """The rejection and its reason are facts; nothing is derived."""
    chat = fixtures.TinyChat()
    index = capture_index.CaptureIndex.create(
        self.tmp_path, {'ref': fixtures.side_info(chat)}
    )
    directory = index.turn_directory('ref', 1)
    completion = fixtures.write_turn(
        directory, chat.send('Hi there', ['a', 'b'], True)
    )
    completion['usage']['completion_tokens'] += 1
    names = sorted(path.name for path in directory.iterdir())

    with self.assertRaises(capture_index.Rejected) as raised:
      index.add_turn(1, 'Hi there', {'ref': completion})

    reason = 'The runner counts 4 completion tokens; turn 1 has 3 decode steps.'
    self.assertEqual(
        str(raised.exception),
        f'The capture index rejected turn 1: {reason}',
    )
    self.assertEqual((raised.exception.side, raised.exception.n), ('ref', 1))
    self.assertEqual(sorted(path.name for path in directory.iterdir()), names)
    (stored,) = json.loads((self.tmp_path / 'index.json').read_text())['turns']
    self.assertEqual(stored['rejected'], {'side': 'ref', 'reason': reason})
    self.assertEqual(stored['sides']['ref']['completion'], completion)
    self.assertEqual(
        [file['path'] for file in stored['sides']['ref']['files']], names
    )
    self.assertEqual(index.turns, [])
    self.assertEqual(index.next_turn(), 2)
    text = f'Reference: The capture index rejected turn 1: {reason}'
    self.assertEqual(capture_index.rejection(index.rejected[0]), text)
    with self.assertRaisesRegex(errors.Conflict, f'^{text}$'):
      index.turn(1)
    with self.assertRaisesRegex(ValueError, 'which was rejected'):
      index.add_turn(2, 'More', {'ref': completion})
    # Read back, it stays rejected, as recorded: it is not derived again.
    loaded = capture_index.CaptureIndex.load(self.tmp_path)
    self.assertEqual(loaded.turns, [])
    self.assertEqual(loaded.rejected, index.rejected)

  def test_load_rejects_a_turn_cut_short_and_reads_the_others(self):
    """The turns before it are read; it and those after it are marked."""
    index = fixtures.capture(
        self.tmp_path,
        [('Hi', ['a'], True), ('Yo', ['b'], True), ('So', ['c'], True)],
    )
    stored = (self.tmp_path / 'index.json').read_bytes()
    cut = min((self.tmp_path / 'target' / 'turn-2').glob('*.safetensors'))
    size = cut.stat().st_size
    _cut(cut)

    loaded = capture_index.CaptureIndex.load(self.tmp_path)

    self.assertEqual(loaded.turns, index.turns[:1])
    self.assertEqual([turn['n'] for turn in loaded.rejected], [2, 3])
    self.assertEqual(
        loaded.rejected[0]['rejected'],
        {
            'side': 'target',
            'reason': (
                f'{cut.name} holds {size - 1} bytes; the capture '
                f'recorded {size}.'
            ),
            'on_read': True,
        },
    )
    self.assertEqual(
        loaded.rejected[1]['rejected'],
        {
            'side': None,
            'reason': 'It follows turn 2, which was rejected.',
            'on_read': True,
        },
    )
    self.assertEqual(
        capture_index.rejection(loaded.rejected[1]),
        'Turn 3 could not be read: It follows turn 2, which was rejected.',
    )
    with self.assertRaisesRegex(
        errors.Conflict, f'^Target: Turn 2 could not be read: {cut.name}'
    ):
      loaded.turn(2)
    self.assertEqual(
        loaded.rejected[0]['sides']['ref'],
        {
            key: index.turns[1]['sides']['ref'][key]
            for key in ('completion', 'files', 'dir')
        },
    )
    # A rejection on read is derived: index.json is as it was.
    self.assertEqual((self.tmp_path / 'index.json').read_bytes(), stored)

  def test_load_rejects_a_turn_whose_file_is_missing(self):
    """A missing file names itself."""
    fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])
    (self.tmp_path / 'ref' / 'turn-1' / 'generated_tokens.jsonl').unlink()

    loaded = capture_index.CaptureIndex.load(self.tmp_path)

    self.assertEqual(loaded.turns, [])
    self.assertEqual(
        loaded.rejected[0]['rejected'],
        {
            'side': 'ref',
            'reason': 'generated_tokens.jsonl is missing.',
            'on_read': True,
        },
    )

  def test_load_rejects_a_turn_whose_tensors_are_malformed(self):
    """A file of an unexpected shape marks its turn, whatever it raises."""
    fixtures.capture(self.tmp_path, [('Hi', ['a'], True)])

    for error in (KeyError('dtype'), TypeError('x'), IndexError('y')):
      with mock.patch.object(
          capture_index, '_read_side_turn', side_effect=error
      ):
        loaded = capture_index.CaptureIndex.load(self.tmp_path)

      self.assertEqual(loaded.turns, [])
      self.assertEqual(loaded.rejected[0]['rejected']['side'], 'ref')

  def test_load_rejects_a_turn_a_check_now_refuses(self):
    """An old capture a stricter derivation refuses is marked, not fatal."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a'], True), ('Yo', ['b'], True)]
    )
    path = self.tmp_path / 'index.json'
    stored = json.loads(path.read_text())
    stored['turns'][1]['sides']['ref']['completion']['usage'][
        'completion_tokens'
    ] += 1
    path.write_text(json.dumps(stored))

    loaded = capture_index.CaptureIndex.load(self.tmp_path)

    self.assertEqual(loaded.turns, index.turns[:1])
    self.assertEqual(loaded.rejected[0]['rejected']['side'], 'ref')
    self.assertIn('completion tokens', loaded.rejected[0]['rejected']['reason'])

  def test_completion_dict(self):
    """Completion dict."""
    completion = runner_pb2.Completion(
        content='2',
        reasoning_content='think',
        finish_reason=runner_pb2.FINISH_REASON_LENGTH,
        usage=runner_pb2.Usage(
            prompt_tokens=2, completion_tokens=1, total_tokens=3
        ),
        model_debugger=runner_pb2.ModelDebugger(
            turn=1,
            context_token_count=3,
            fed_tokens=[
                runner_pb2.FedToken(
                    position=0,
                    phase=runner_pb2.FedToken.PREFILL,
                    id=2,
                    text='<bos>',
                ),
                runner_pb2.FedToken(
                    position=1, phase=runner_pb2.FedToken.DECODE
                ),
            ],
        ),
    )

    self.assertEqual(
        capture_index.completion_dict(completion),
        {
            'content': '2',
            'reasoning_content': 'think',
            'finish_reason': 'length',
            'usage': {
                'prompt_tokens': 2,
                'completion_tokens': 1,
                'total_tokens': 3,
            },
            'turn': 1,
            'context_token_count': 3,
            'fed_tokens': [
                {
                    'position': 0,
                    'phase': 'prefill',
                    'id': 2,
                    'text': '<bos>',
                },
                # The runner could not tell which token it was.
                {
                    'position': 1,
                    'phase': 'decode',
                    'id': None,
                    'text': None,
                },
            ],
        },
    )

  def test_completion_dict_keeps_a_fed_token_of_no_phase(self):
    """Stored as reported, for add_turn to reject with its files kept."""
    completion = runner_pb2.Completion(
        finish_reason=runner_pb2.FINISH_REASON_STOP,
        model_debugger=runner_pb2.ModelDebugger(
            turn=1, fed_tokens=[runner_pb2.FedToken(position=4, id=2)]
        ),
    )

    self.assertEqual(
        capture_index.completion_dict(completion)['fed_tokens'],
        [{'position': 4, 'phase': None, 'id': 2, 'text': ''}],
    )

  def test_completion_dict_refuses_a_completion_without_debugger(self):
    """Completion dict refuses a completion without debugger."""
    with self.assertRaisesRegex(ValueError, 'model_debugger'):
      capture_index.completion_dict(
          runner_pb2.Completion(finish_reason=runner_pb2.FINISH_REASON_STOP)
      )

  def test_a_greedy_chat_generates_what_it_released(self):
    """A greedy step whose argmax is not the released token is kept.

    The index does not check the runtime's sampling: the token is the
    one released, and the step is not marked.
    """
    a = fixtures.token('a')[0]

    def overtake(logits):
      logits[3] = logits[a] + 1

    index, _ = _greedy_turn(self.tmp_path, 0, overtake)

    output = index.turn(1)['sides']['ref']['output']
    self.assertEqual((output[0]['id'], output[0]['text']), (a, 'a'))
    self.assertNotIn('cannot_confirm', output[0])
    self.assertEqual(index.rejected, [])

  def test_a_tie_at_the_top_cannot_be_confirmed(self):
    """Tied top logits: the step is unconfirmed, the turn is kept.

    The released token is not the first of the tied ids.
    """
    a = fixtures.token('a')[0]
    other = fixtures.OTHER
    tied = {}

    def tie(logits):
      fixtures.greedy_edits()['tie'](logits)
      tied['value'] = float(logits[a])

    index, step = _greedy_turn(self.tmp_path, 0, tie)

    output = index.turn(1)['sides']['ref']['output']
    low, high = sorted((a, other))
    self.assertEqual(
        output[0]['cannot_confirm'],
        f'Decode step {step} has 2 logits tied at the top, '
        f'{tied["value"]!r}: {low} or {high}.',
    )
    self.assertEqual((output[0]['id'], output[0]['text']), (a, 'a'))
    self.assertNotIn('cannot_confirm', output[1])
    self.assertNotIn('cannot_confirm', output[2])
    self.assertEqual(index.rejected, [])
    # Derived, not stored: index.json holds none of it, a read derives it.
    self.assertNotIn(
        'cannot_confirm', (self.tmp_path / 'index.json').read_text()
    )
    loaded = capture_index.CaptureIndex.load(self.tmp_path)
    self.assertEqual(loaded.turns, index.turns)

  def test_a_tie_that_leaves_out_the_released_token_is_kept(self):
    """No tied id is the released token: unconfirmed, the turn kept."""
    a = fixtures.token('a')[0]
    others = [id for id in range(12, 48) if id != a][:2]

    def tie(logits):
      logits[others] = logits[a] + 1

    index, step = _greedy_turn(self.tmp_path, 0, tie)

    token, *_ = index.turn(1)['sides']['ref']['output']
    self.assertEqual(token['id'], a)
    self.assertIn(
        f'Decode step {step} has 2 logits tied', token['cannot_confirm']
    )
    self.assertEqual(index.rejected, [])

  def test_a_nan_logit_cannot_be_confirmed(self):
    """A NaN among the logits: the step is unconfirmed, not rejected."""
    index, step = _greedy_turn(self.tmp_path, 1, _set_nan_logit)

    output = index.turn(1)['sides']['ref']['output']
    self.assertEqual(
        output[1]['cannot_confirm'],
        f'Decode step {step} has 1 of {fixtures.VOCABULARY} logits that '
        'are NaN.',
    )
    self.assertEqual(output[1]['text'], 'b')
    self.assertNotIn('cannot_confirm', output[0])

  def test_one_plus_inf_is_greedys_pick(self):
    """One +inf is the largest logit: the step is confirmed."""
    a = fixtures.token('a')[0]

    index, _ = _greedy_turn(
        self.tmp_path, 0, fixtures.greedy_edits()['plus_inf']
    )

    token, *_ = index.turn(1)['sides']['ref']['output']
    self.assertEqual(token['id'], a)
    self.assertNotIn('cannot_confirm', token)
    self.assertEqual(index.rejected, [])

  def test_two_plus_inf_tie_at_the_top(self):
    """Two +inf tie: the step is unconfirmed, between those two ids."""
    a = fixtures.token('a')[0]

    index, step = _greedy_turn(
        self.tmp_path, 0, fixtures.greedy_edits()['two_plus_inf']
    )

    output = index.turn(1)['sides']['ref']['output']
    low, high = sorted((a, fixtures.OTHER))
    self.assertEqual(
        output[0]['cannot_confirm'],
        f'Decode step {step} has 2 logits tied at the top, inf: '
        f'{low} or {high}.',
    )
    self.assertEqual(output[0]['id'], a)

  def test_minus_inf_is_only_a_low_logit(self):
    """A vocabulary masked with -inf is read as any other: confirmed."""
    a = fixtures.token('a')[0]

    index, _ = _greedy_turn(self.tmp_path, 0, fixtures.greedy_edits()['masked'])

    output = index.turn(1)['sides']['ref']['output']
    self.assertEqual(output[0]['id'], a)
    self.assertFalse(any('cannot_confirm' in token for token in output))

  def test_a_row_of_minus_inf_cannot_be_confirmed(self):
    """Every logit -inf names no pick: unconfirmed, not a mismatch."""
    index, step = _greedy_turn(
        self.tmp_path, 0, fixtures.greedy_edits()['all_minus_inf']
    )

    output = index.turn(1)['sides']['ref']['output']
    self.assertEqual(
        output[0]['cannot_confirm'],
        f'Decode step {step} has all {fixtures.VOCABULARY} logits at -inf.',
    )
    self.assertEqual(output[0]['id'], fixtures.token('a')[0])
    self.assertEqual(index.rejected, [])

  def test_an_unconfirmed_stop_step_names_the_stop_token_tied(self):
    """The unreleased last step ties a stop token with another id."""

    def tie(logits):
      logits[3] = logits[fixtures.TURN_CLOSE]

    index, step = _greedy_turn(self.tmp_path, 2, tie)

    stop = index.turn(1)['sides']['ref']['output'][2]
    self.assertEqual(stop['id'], fixtures.TURN_CLOSE)
    self.assertTrue(stop['stop'] and not stop['released'])
    self.assertIn(
        f'Decode step {step} has 2 logits tied', stop['cannot_confirm']
    )

  def test_a_stop_step_whose_logits_are_nan_is_a_stop_of_unknown_id(self):
    """Every logit NaN at the last, unreleased step: a stop, id unknown."""

    def nan(logits):
      logits[:] = numpy.nan

    index, _ = _greedy_turn(self.tmp_path, 2, nan)

    side = index.turn(1)['sides']['ref']
    self.assertIsNone(side['output'][2]['id'])
    self.assertTrue(side['output'][2]['stop'])
    self.assertIn('48 of 48 logits', side['output'][2]['cannot_confirm'])
    self.assertEqual(side['stop_reason'], 'eos')

  def test_a_sampled_chat_has_nothing_to_confirm(self):
    """A tie in a sampled chat is no greedy step to confirm."""
    a = fixtures.token('a')[0]

    def tie(logits):
      logits[3] = logits[a]

    index, _ = _greedy_turn(self.tmp_path, 0, tie, top_k=40)

    output = index.turn(1)['sides']['ref']['output']
    self.assertFalse(any('cannot_confirm' in token for token in output))

  def test_a_sampled_chat_generates_what_it_released(self):
    """A sampled chat need not generate the argmax; its stop is unknown."""
    chat = fixtures.TinyChat()
    fixed = fixtures.side_info(chat)
    fixed['chat']['options']['top_k'] = 40
    index = capture_index.CaptureIndex.create(self.tmp_path, {'ref': fixed})
    turn = chat.send('Hi there', ['a', 'b'], True)
    completion = fixtures.write_turn(index.turn_directory('ref', 1), turn)
    path = index.turn_directory('ref', 1) / 'generated_tokens.jsonl'
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({'token_ids': [[7]], 'texts': ['x']})
    path.write_text('\n'.join(lines) + '\n')
    # The second decode step feeds what the first generated.
    decodes = [t for t in completion['fed_tokens'] if t['phase'] == 'decode']
    decodes[1].update(id=7, text='x')

    index.add_turn(1, 'Hi there', {'ref': completion})

    output = index.turn(1)['sides']['ref']['output']
    self.assertEqual((output[0]['id'], output[0]['text']), (7, 'x'))
    self.assertIsNone(output[-1]['id'])
    self.assertTrue(output[-1]['stop'] and not output[-1]['released'])

  def test_a_missing_decode_step_is_a_gap(self):
    """A missing decode step is a gap."""

    def change(directory, completion):
      del completion  # Unused.
      steps = sorted(
          int(p.name.rsplit('_', 1)[1].split('.')[0])
          for p in directory.glob('decode_post_logits_step_*')
      )
      for path in _decode_files(directory, steps[1]):
        path.unlink()

    self.assertIn(
        'comes next', _broken(self.tmp_path, change, reply=('a', 'b', 'c'))
    )

  def test_an_unreleased_step_whose_argmax_is_no_stop_token(self):
    """The unreleased last step is a stop token, of unknown id.

    The runtime releases every token but a stop token; the index does
    not check its sampling against the step's argmax.
    """
    chat = fixtures.TinyChat()
    index = capture_index.CaptureIndex.create(
        self.tmp_path, {'ref': fixtures.side_info(chat)}
    )
    turn = chat.send('Hi there', ['a', 'b'], False)
    directory = index.turn_directory('ref', 1)
    completion = fixtures.write_turn(directory, turn)
    path = directory / 'generated_tokens.jsonl'
    path.write_text(''.join(path.read_text().splitlines(True)[:-1]))
    completion['content'] = 'a'

    index.add_turn(1, 'Hi there', {'ref': completion})

    last = index.turn(1)['sides']['ref']['output'][-1]
    self.assertIsNone(last['id'])
    self.assertTrue(last['stop'] and not last['released'])

  def test_more_decode_steps_than_tokens(self):
    """More decode steps than tokens."""

    def change(directory, completion):
      del completion  # Unused.
      (directory / 'generated_tokens.jsonl').write_text('')

    self.assertIn(
        '3 decode steps for 0 released tokens',
        _broken(self.tmp_path, change),
    )

  def test_a_wrong_context_count(self):
    """A wrong context count."""

    def change(directory, completion):
      del directory  # Unused.
      completion['context_token_count'] += 1

    self.assertIn('context tokens', _broken(self.tmp_path, change))

  def test_a_wrong_total_count(self):
    """A total that is not the context count."""

    def change(directory, completion):
      del directory  # Unused.
      completion['usage']['total_tokens'] += 1

    self.assertIn('total tokens', _broken(self.tmp_path, change))

  def test_a_length_that_reached_neither_limit(self):
    """LENGTH with room in both the reply and the context."""

    def change(directory, completion):
      del directory  # Unused.
      completion['finish_reason'] = 'length'

    self.assertIn(
        'reply has 2 of 8 tokens, and the context',
        _broken(self.tmp_path, change, stop=False),
    )

  def test_a_length_of_a_reply_without_a_limit(self):
    """LENGTH of a reply without a limit of its own, with room left."""
    chat = fixtures.TinyChat()
    fixed = fixtures.side_info(chat)
    del fixed['chat']['options']['max_completion_tokens']
    index = capture_index.CaptureIndex.create(self.tmp_path, {'ref': fixed})
    turn = chat.send('Hi there', ['a', 'b'], False)
    completion = fixtures.write_turn(index.turn_directory('ref', 1), turn)
    completion['finish_reason'] = 'length'

    with self.assertRaises(capture_index.Rejected) as raised:
      index.add_turn(1, 'Hi there', {'ref': completion})

    self.assertIn(
        'FINISH_REASON_LENGTH for turn 1, but its reply has 2 tokens and '
        'no limit, and the context',
        raised.exception.reason,
    )
    self.assertNotIn('None', raised.exception.reason)

  def test_a_wrong_completion_count(self):
    """A wrong completion count."""

    def change(directory, completion):
      del directory  # Unused.
      completion['usage']['completion_tokens'] = 2

    self.assertIn('completion tokens', _broken(self.tmp_path, change))

  def test_a_completion_from_before_fed_tokens_is_rejected(self):
    """A runner from before fed_tokens listed `prompt_tokens` instead."""

    def change(directory, completion):
      del directory  # Unused.
      del completion['fed_tokens']
      completion['prompt_tokens'] = [{'id': 1, 'text': 'Hi'}]

    self.assertEqual(
        _broken(self.tmp_path, change),
        'Turn 1 was captured by a runner from before fed_tokens; such '
        'captures are no longer read. Capture the chat again.',
    )

  def test_load_rejects_a_turn_recorded_before_fed_tokens(self):
    """An older capture opens; its turns without fed_tokens are rejected."""
    index = fixtures.capture(
        self.tmp_path, [('Hi', ['a'], True), ('Yo', ['b'], True)]
    )
    path = self.tmp_path / 'index.json'
    stored = json.loads(path.read_text())
    completion = stored['turns'][1]['sides']['ref']['completion']
    del completion['fed_tokens']
    completion['prompt_tokens'] = [{'id': 1, 'text': 'Yo'}]
    path.write_text(json.dumps(stored))

    loaded = capture_index.CaptureIndex.load(self.tmp_path)

    self.assertEqual(loaded.turns, index.turns[:1])
    self.assertEqual(
        loaded.rejected[0]['rejected'],
        {
            'side': 'ref',
            'reason': (
                'Turn 2 was captured by a runner from before '
                'fed_tokens; such captures are no longer read. Capture the '
                'chat again.'
            ),
            'on_read': True,
        },
    )

  def test_no_fed_tokens_asks_for_a_rebuilt_runner(self):
    """A runner built before fed_tokens, its turn recorded as rejected."""

    def change(directory, completion):
      del directory  # Unused.
      completion['fed_tokens'] = []

    self.assertIn('rebuild it', _broken(self.tmp_path, change))

  def test_a_fed_token_of_no_phase_is_rejected(self):
    """A runner that breaks the protocol: its turn and files are kept."""

    def change(directory, completion):
      del directory  # Unused.
      completion['fed_tokens'][2]['phase'] = None

    self.assertEqual(
        _broken(self.tmp_path, change),
        'The runner reports a fed token at position 2 of no phase.',
    )

  def test_fed_tokens_the_files_do_not_feed(self):
    """Fed tokens the files do not feed."""

    def missing(directory, completion):
      del directory  # Unused.
      completion['fed_tokens'].pop()

    def moved(directory, completion):
      del directory  # Unused.
      completion['fed_tokens'][3]['position'] += 1

    def decoded(directory, completion):
      del directory  # Unused.
      completion['fed_tokens'][0]['phase'] = 'decode'

    self.assertIn('the files feed', _broken(self.tmp_path / 'missing', missing))
    self.assertIn(
        'fed token 3 is a prefill at position 4; the files feed a '
        'prefill at position 3',
        _broken(self.tmp_path / 'moved', moved),
    )
    self.assertIn(
        'fed token 0 is a decode',
        _broken(self.tmp_path / 'decoded', decoded),
    )

  def test_a_first_token_that_is_not_the_start_token(self):
    """Turn 1 first feeds the model's start token."""

    def change(directory, completion):
      del directory  # Unused.
      completion['fed_tokens'][0]['id'] = fixtures.EOS

    self.assertIn(
        f'first feeds {fixtures.EOS}, not the start token',
        _broken(self.tmp_path, change),
    )

  def test_a_decode_step_that_fed_another_token(self):
    """A decode step feeds what the step before it generated."""

    def change(directory, completion):
      del directory  # Unused.
      completion['fed_tokens'][-1]['id'] = fixtures.EOS

    self.assertIn(
        f'fed {fixtures.EOS}; the step before generated',
        _broken(self.tmp_path, change),
    )

  def test_a_wrong_prompt_count(self):
    """The prompt counts the positions up to the first decode step's."""

    def change(directory, completion):
      del directory  # Unused.
      completion['usage']['prompt_tokens'] += 1
      completion['usage']['total_tokens'] += 1

    self.assertIn('prompt tokens', _broken(self.tmp_path, change))

  def test_a_later_turn_that_starts_elsewhere(self):
    """A later turn starts at the token held back for it."""
    chat = fixtures.TinyChat()
    index = capture_index.CaptureIndex.create(
        self.tmp_path, {'ref': fixtures.side_info(chat)}
    )
    for n in (1, 2):
      completion = fixtures.write_turn(
          index.turn_directory('ref', n), chat.send('Hi', ['a'])
      )
      if n == 1:
        index.add_turn(1, 'Hi', {'ref': completion})
        # As if the runner had counted one position less.
        index.turns[0]['sides']['ref']['completion']['context_token_count'] -= 1

    with self.assertRaises(capture_index.Rejected) as raised:
      index.add_turn(2, 'Hi', {'ref': completion})

    self.assertIn('held the token at position', raised.exception.reason)

  def test_a_turn_that_rolls_back_at_its_start(self):
    """A turn that starts before the token held: the runtime rolled back."""
    chat = fixtures.TinyChat()
    index = capture_index.CaptureIndex.create(
        self.tmp_path, {'ref': fixtures.side_info(chat)}
    )
    for n in (1, 2):
      completion = fixtures.write_turn(
          index.turn_directory('ref', n), chat.send('Hi', ['a', 'b'])
      )
      if n == 1:
        index.add_turn(1, 'Hi', {'ref': completion})
        index.turns[0]['sides']['ref']['completion']['context_token_count'] += 2

    index.add_turn(2, 'Hi', {'ref': completion})

    self.assertEqual(
        {t['origin'] for t in index.turn(2)['sides']['ref']['input']},
        {'history'},
    )

  def test_a_turn_the_runner_numbers_otherwise(self):
    """A turn the runner numbers otherwise."""

    def change(directory, completion):
      del directory  # Unused.
      completion['turn'] = 2

    self.assertIn('calls turn 1 turn 2', _broken(self.tmp_path, change))

  def test_an_unexpected_file(self):
    """An unexpected file."""

    def change(directory, completion):
      del completion  # Unused.
      (directory / 'notes.txt').write_text('x')

    self.assertIn('unexpected file, notes.txt', _broken(self.tmp_path, change))

  def test_a_mask_that_disagrees_with_the_positions(self):
    """A mask that disagrees with the positions."""

    def change(directory, completion):
      del completion  # Unused.
      path = sorted(directory.glob('prefill_*_pre_mask_step_*.safetensors'))[0]
      mask = load_file(path)['pre_mask']
      mask[0, 0, -1, 0] = True
      save_file(
          {'pre_mask': mask},
          str(path),
          metadata={
              'signature': fixtures.SIGNATURE,
              'step': path.name.rsplit('_', 1)[1].split('.')[0],
          },
      )

    self.assertIn('mask attends from', _broken(self.tmp_path, change))

  def test_a_file_without_step_metadata(self):
    """A file without step metadata."""

    def change(directory, completion):
      del completion  # Unused.
      path = next(directory.glob('decode_pre_mask_step_*.safetensors'))
      save_file(load_file(path), str(path))

    self.assertIn('names no signature and step', _broken(self.tmp_path, change))

  def _real_turn(self, backend):
    """Indexes the real turn on `backend`; returns its side's entry."""
    real = real_capture.turn(backend)
    index = capture_index.CaptureIndex.create(self.tmp_path, {'ref': real.side})
    _link_turn(index, 1, real.directory)
    index.add_turn(1, real_capture.PROMPT, {'ref': real.completion})
    side = index.turn(1)['sides']['ref']
    # Gemma's turn 1: the start token and 16 prompt tokens, prefilled in
    # one chunk; 8 decode steps, the first of which feeds the last input.
    self.assertLen(side['input'], 17)
    self.assertEqual(
        [f['step'] for f in side['forwards']], [17, *range(17, 25)]
    )
    self.assertEqual(side['forwards'][0]['input_positions'], [0, 15])
    self.assertEqual(
        ''.join(t['text'] for t in side['output']),
        real.completion['content'],
    )
    self.assertEqual(
        [t['position'] for t in side['output']], list(range(17, 25))
    )
    self.assertEqual(side['stop_reason'], 'max_output_tokens')
    # The CPU's KV before the prefill too; the GPU's is not captured.
    self.assertEqual(
        [
            (s['moment'], s['step'], s['processed'])
            for s in side['kv']
            if s['moment'] != 'prefill_pre'
        ],
        [('prefill_post', 17, 16), ('terminal', 24, 24)],
    )
    layers = side['kv'][-1]['layers']
    self.assertEqual([layer['layer'] for layer in layers], list(range(15)))
    self.assertEqual(
        layers[4]['value']['layout'],
        [
            'batch',
            'kv_head',
            'head_dim',
            'sequence',
        ],
    )
    return side

  @absltest.skipUnless(real_capture.AVAILABLE, real_capture.UNAVAILABLE)
  def test_a_real_turn_on_the_cpu(self):
    """A real turn on the CPU: every step's argmax is unique."""
    side = self._real_turn('CPU')

    self.assertEqual(
        [t['text'] for t in side['output']],
        ['2', ' +', ' ', '2', ' is', ' **', '4', '**.'],
    )
    self.assertFalse(any('cannot_confirm' in token for token in side['output']))
    self.assertEqual(
        [(s['moment'], s['step'], s['processed']) for s in side['kv']][0],
        ('prefill_pre', 17, 0),
    )
    self.assertEqual(
        side['kv'][-1]['layers'][4]['key']['shape'], [1, 1, 4096, 512]
    )

  @absltest.skipUnless(real_capture.AVAILABLE, real_capture.UNAVAILABLE)
  def test_a_real_turn_on_the_gpu(self):
    """A real turn on the GPU is indexed, its F16 ties unconfirmed."""
    side = self._real_turn('GPU')

    unconfirmed = [
        token['cannot_confirm']
        for token in side['output']
        if 'cannot_confirm' in token
    ]
    logging.info(
        'GPU turn: %d of %d steps cannot be confirmed: %s',
        len(unconfirmed),
        len(side['output']),
        unconfirmed,
    )
    self.assertTrue(all(token['released'] for token in side['output']))

  @absltest.skipUnless(
      _REAL_SESSIONS and _REAL_SESSION.is_dir(), 'no REAL_SESSIONS'
  )
  def test_a_real_session_of_four_turns(self):
    """A real session of four turns."""
    index = capture_index.CaptureIndex.create(self.tmp_path, {'ref': _GEMMA})
    for n in range(1, 5):
      source = _REAL_SESSION / f'turn{n}'
      _link_turn(index, n, source)
      index.add_turn(
          n, '', {'ref': _real_completion(source / '_completion.json')}
      )

    sides = [turn['sides']['ref'] for turn in index.turns]
    self.assertEqual(
        [s['stop_reason'] for s in sides],
        [
            'eos',
            'eos',
            'max_output_tokens',
            'eos',
        ],
    )
    # Turn 2's 362 inputs went through three chunks of 128, 128 and 105.
    chunks = [f for f in sides[1]['forwards'] if f['phase'] == 'prefill']
    self.assertEqual(
        [(f['step'], f['input_positions']) for f in chunks],
        [
            (155, [26, 153]),
            (283, [154, 281]),
            (388, [282, 386]),
        ],
    )
    # Each later turn first feeds the previous turn's last token: the stop
    # `<turn|>` after a stop, the released ` earth` after a cut.
    self.assertEqual([s['input'][0]['id'] for s in sides[1:]], [106, 106, 7764])
    self.assertEqual(
        [s['input'][0]['position'] for s in sides[1:]], [26, 390, 418]
    )

  @absltest.skipUnless(
      _REAL_SESSIONS and _REAL_THINKING.is_dir(), 'no REAL_SESSIONS'
  )
  def test_a_real_session_that_thinks(self):
    """A real session that thinks: the next turn rolls the context back."""
    index = capture_index.CaptureIndex.create(self.tmp_path, {'ref': _GEMMA})
    for n in (1, 2):
      source = _REAL_THINKING / f'turn{n}'
      _link_turn(index, n, source)
      index.add_turn(
          n, '', {'ref': _real_completion(source / '_completion.json')}
      )

    first, second = (turn['sides']['ref'] for turn in index.turns)
    self.assertEqual(
        [t['text'] for t in first['output'][:3]],
        [
            '<|channel>',
            'thought',
            '\n',
        ],
    )
    self.assertEqual(first['output'][-1]['position'], 63)
    # Turn 2 starts back at position 23, turn 1's last input token, and
    # feeds three tokens of the runtime's own before its 13 prompt tokens.
    self.assertEqual(
        [(t['origin'], t['position']) for t in second['input'][:4]],
        [
            ('history', 23),
            ('history', 24),
            ('history', 25),
            ('prompt', 26),
        ],
    )
    self.assertLen(second['input'], 16)
    chunks = [f for f in second['forwards'] if f['phase'] == 'prefill']
    self.assertEqual(
        [f['input_positions'] for f in chunks], [[23, 24], [25, 37]]
    )


if __name__ == '__main__':
  test_main.main()
