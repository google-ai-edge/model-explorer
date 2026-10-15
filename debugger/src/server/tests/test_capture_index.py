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

import hashlib
import json
import os
import pathlib

import capture_fixtures as fixtures
from google.protobuf import json_format
from model_debugger.v1 import runner_pb2
from model_debugger_server import capture_index, errors
import pytest
from safetensors import numpy as safetensors_numpy

# A prompt that takes several prefill chunks.
_LONG = ' '.join(f'w{i}' for i in range(20))


def test_turn_one_feeds_the_start_token_and_the_prompt(tmp_path):
  """Turn one feeds the start token and the prompt."""
  index = fixtures.capture(
      tmp_path, [('What is it?', ['Four', '.'], True)], system='Be brief.'
  )

  side = index.turn(1)['sides']['ref']
  prompt = side['completion']['prompt_tokens']

  assert side['input'][0] == {
      'id': fixtures.BOS,
      'text': None,
      'origin': 'start',
      'position': 0,
  }
  assert [token['id'] for token in side['input'][1:]] == [
      token['id'] for token in prompt
  ]
  assert {token['origin'] for token in side['input'][1:]} == {'prompt'}
  assert [token['position'] for token in side['input']] == list(
      range(len(prompt) + 1)
  )


def test_a_stop_token_is_generated_but_not_released(tmp_path):
  """A stop token is generated but not released."""
  index = fixtures.capture(tmp_path, [('Hi', ['Four', '.'], True)])

  side = index.turn(1)['sides']['ref']
  first = len(side['input'])
  chunks = sum(f['phase'] == 'prefill' for f in side['forwards'])

  assert side['output'] == [
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
  ]
  assert side['stop_reason'] == 'eos'
  assert side['completion']['finish_reason'] == 'stop'


def test_a_reply_cut_at_the_limit_has_no_stop(tmp_path):
  """A reply cut at the limit has no stop."""
  index = fixtures.capture(
      tmp_path, [('Hi', ['a', 'b', 'c'], True)], max_completion_tokens=2
  )

  side = index.turn(1)['sides']['ref']

  assert [token['released'] for token in side['output']] == [True, True]
  assert side['stop_reason'] == 'max_output_tokens'
  assert side['completion']['finish_reason'] == 'length'


def test_a_stop_at_the_limit_is_an_end_of_sequence(tmp_path):
  """A stop at the limit is an end of sequence."""
  # The runner reports LENGTH when the stop token is the limit's last token.
  index = fixtures.capture(
      tmp_path, [('Hi', ['a'], True)], max_completion_tokens=2
  )

  side = index.turn(1)['sides']['ref']

  assert side['completion']['finish_reason'] == 'length'
  assert side['output'][-1]['stop'] is True
  assert side['stop_reason'] == 'eos'


def test_a_later_turn_first_feeds_the_previous_last_token(tmp_path):
  """A later turn first feeds the previous last token."""
  index = fixtures.capture(
      tmp_path,
      [
          ('Hi', ['a'], True),
          ('More', ['b', 'c'], False),
          ('Again', ['d'], True),
      ],
      max_completion_tokens=4,
  )

  for n in (2, 3):
    side = index.turn(n)['sides']['ref']
    previous = index.turn(n - 1)['sides']['ref']['output'][-1]
    assert side['input'][0] == {
        'id': previous['id'],
        'text': previous['text'],
        'origin': 'carried',
        'position': previous['position'],
    }
    assert side['input'][1]['origin'] == 'prompt'
  # After a stop the carried token is the stop; after a cut, a released one.
  assert index.turn(2)['sides']['ref']['input'][0]['id'] == fixtures.TURN_CLOSE
  assert index.turn(3)['sides']['ref']['input'][0]['text'] == 'c'


def test_a_turn_after_thinking_feeds_what_the_runtime_rendered(tmp_path):
  """A turn after thinking feeds what the runtime rendered."""
  index = fixtures.capture(
      tmp_path,
      [('Hi', ['ok'], True, ['hmm', ' yes']), ('More', ['b'], True)],
      max_completion_tokens=12,
  )

  first = index.turn(1)['sides']['ref']
  second = index.turn(2)['sides']['ref']
  prompt = second['completion']['prompt_tokens']

  assert first['completion']['reasoning_content'] == 'hmm yes'
  assert first['completion']['content'] == 'ok'
  assert [t['text'] for t in first['output'][:3]] == [
      '<|channel>',
      'thought',
      '\n',
  ]
  # The runtime rolls back to turn 1's last input token and feeds it, then
  # `ok<turn|>\n` without the thoughts; the runner names none of them.
  start = first['input'][-1]['position']
  assert second['input'][:4] == [
      {'id': None, 'text': None, 'origin': 'history', 'position': start + i}
      for i in range(4)
  ]
  assert [t['id'] for t in second['input'][4:]] == [t['id'] for t in prompt]
  assert second['kv'][0]['processed'] == start


def test_prefill_chunks_come_first_in_position_order(tmp_path):
  """Prefill chunks come first in position order."""
  index = fixtures.capture(tmp_path, [(_LONG, ['a'], True)])

  side = index.turn(1)['sides']['ref']
  prefills = [f for f in side['forwards'] if f['phase'] == 'prefill']
  decodes = [f for f in side['forwards'] if f['phase'] == 'decode']

  assert [f['id'] for f in side['forwards']] == list(
      range(len(side['forwards']))
  )
  assert side['forwards'][: len(prefills)] == prefills
  # Turn 1's first chunk holds back its last id: one row fewer.
  assert prefills[0]['input_positions'] == [0, fixtures.PREFILL_SIZE - 2]
  for before, after in zip(prefills, prefills[1:]):
    assert after['input_positions'][0] == before['input_positions'][1] + 1
  # A chunk is labelled with its last position + 2.
  assert [f['step'] for f in prefills] == [
      f['input_positions'][1] + 2 for f in prefills
  ]
  assert prefills[-1]['step'] == decodes[0]['step']
  assert decodes[0]['input_position'] == len(side['input']) - 1
  assert decodes[0]['input_positions'] == [decodes[0]['input_position']]
  assert set(prefills[0]['tensors']) == {
      'pre_input_pos',
      'pre_mask',
      'pre_embeddings',
      'pre_per_layer_embeddings',
      'pre_param_tensor',
      *(
          f'{m}_kv_cache_{k}_{layer}'
          for m in ('pre', 'post')
          for k in 'kv'
          for layer in (0, 1)
      ),
  }
  assert decodes[0]['tensors']['post_logits'] == (
      f'decode_post_logits_step_{decodes[0]["step"]}.safetensors'
  )


def test_kv_snapshots_before_and_after_prefill_and_at_the_end(tmp_path):
  """KV snapshots before and after prefill and at the end."""
  index = fixtures.capture(
      tmp_path, [('Hi', ['a'], True), (_LONG, ['b', 'c'], True)]
  )

  side = index.turn(2)['sides']['ref']
  prefills = [f for f in side['forwards'] if f['phase'] == 'prefill']
  decodes = [f for f in side['forwards'] if f['phase'] == 'decode']
  kv = side['kv']

  assert [(s['moment'], s['step'], s['processed']) for s in kv] == [
      ('prefill_pre', prefills[0]['step'], side['input'][0]['position']),
      (
          'prefill_post',
          prefills[-1]['step'],
          prefills[-1]['input_positions'][1] + 1,
      ),
      ('terminal', decodes[-1]['step'], decodes[-1]['input_position'] + 1),
  ]
  layer = kv[2]['layers'][1]
  assert layer['layer'] == 1
  assert layer['key'] == {
      'file': (
          f'decode_post_kv_cache_k_1_step_{decodes[-1]["step"]}.safetensors'
      ),
      'dtype': 'I8',
      'shape': [1, 1, fixtures.CONTEXT_LENGTH, fixtures.HEAD_DIM],
      'layout': ['batch', 'kv_head', 'sequence', 'head_dim'],
  }
  assert layer['value']['shape'] == [
      1,
      1,
      fixtures.HEAD_DIM,
      fixtures.CONTEXT_LENGTH,
  ]
  assert layer['value']['layout'] == [
      'batch',
      'kv_head',
      'head_dim',
      'sequence',
  ]


def test_kv_that_does_not_span_the_context_has_no_layout(tmp_path):
  """KV that does not span the context has no layout."""
  chat = fixtures.TinyChat()
  index = capture_index.CaptureIndex.create(
      tmp_path, {'ref': fixtures.side_info(chat)}
  )
  index.sides['ref']['load']['context_length'] = 4096
  completion = fixtures.write_turn(
      index.turn_directory('ref', 1), chat.send('Hi', ['a'])
  )

  index.add_turn(1, 'Hi', {'ref': completion})

  layer = index.turn(1)['sides']['ref']['kv'][0]['layers'][0]
  assert layer['key']['layout'] is None
  assert layer['key']['shape'] == [
      1,
      1,
      fixtures.CONTEXT_LENGTH,
      fixtures.HEAD_DIM,
  ]


def test_files_are_listed_with_size_and_digest(tmp_path):
  """Files are listed with size and digest."""
  index = fixtures.capture(tmp_path, [('Hi', ['a'], True)])

  side = index.turn(1)['sides']['target']
  directory = tmp_path / 'target' / 'turn-1'

  assert side['dir'] == 'target/turn-1'
  assert [f['path'] for f in side['files']] == sorted(
      p.name for p in directory.iterdir()
  )
  for file in side['files']:
    data = (directory / file['path']).read_bytes()
    assert file['size'] == len(data)
    assert file['sha256'] == hashlib.sha256(data).hexdigest()
  assert index.path('target', 1, 'generated_tokens.jsonl') == (
      directory / 'generated_tokens.jsonl'
  )


def test_load_reads_what_add_turn_wrote(tmp_path):
  """Load reads what add turn wrote."""
  index = fixtures.capture(tmp_path, [('Hi', ['a'], True), ('Yo', ['b'], True)])

  loaded = capture_index.CaptureIndex.load(tmp_path)

  assert loaded.sides == index.sides
  assert loaded.turns == index.turns
  assert (
      json.loads((tmp_path / 'index.json').read_text())['format_version'] == 1
  )


def test_load_without_a_capture_is_not_found(tmp_path):
  """Load without a capture is not found."""
  with pytest.raises(errors.NotFound):
    capture_index.CaptureIndex.load(tmp_path)


def test_turn_out_of_range_is_not_found(tmp_path):
  """Turn out of range is not found."""
  index = fixtures.capture(tmp_path, [('Hi', ['a'], True)])

  for n in (0, 2):
    with pytest.raises(errors.NotFound):
      index.turn(n)


def test_add_turn_must_follow_the_last(tmp_path):
  """Add turn must follow the last."""
  index = fixtures.capture(tmp_path, [('Hi', ['a'], True)])

  with pytest.raises(ValueError, match='Turn 3 cannot follow turn 1'):
    index.add_turn(3, 'x', {})


def test_completion_dict():
  """Completion dict."""
  completion = runner_pb2.Completion(
      content='2',
      reasoning_content='think',
      finish_reason=runner_pb2.FINISH_REASON_LENGTH,
      usage=runner_pb2.Usage(
          prompt_tokens=17, completion_tokens=1, total_tokens=18
      ),
      model_debugger=runner_pb2.ModelDebugger(
          turn=1,
          prompt_tokens=[runner_pb2.PromptToken(id=105, text='<|turn>')],
          context_token_count=18,
      ),
  )

  assert capture_index.completion_dict(completion) == {
      'content': '2',
      'reasoning_content': 'think',
      'finish_reason': 'length',
      'usage': {
          'prompt_tokens': 17,
          'completion_tokens': 1,
          'total_tokens': 18,
      },
      'turn': 1,
      'prompt_tokens': [{'id': 105, 'text': '<|turn>'}],
      'context_token_count': 18,
  }


def test_completion_dict_refuses_a_completion_without_debugger():
  """Completion dict refuses a completion without debugger."""
  with pytest.raises(ValueError, match='model_debugger'):
    capture_index.completion_dict(
        runner_pb2.Completion(finish_reason=runner_pb2.FINISH_REASON_STOP)
    )


# Contradictions between files and completion are runner bugs: raised.


def _broken(tmp_path, change, reply=('a', 'b'), stop=True):
  """Adds one turn whose files or completion `change` edits; returns error."""
  chat = fixtures.TinyChat()
  index = capture_index.CaptureIndex.create(
      tmp_path, {'ref': fixtures.side_info(chat)}
  )
  turn = chat.send('Hi there', list(reply), stop)
  completion = fixtures.write_turn(index.turn_directory('ref', 1), turn)
  change(index.turn_directory('ref', 1), completion)
  with pytest.raises(ValueError) as error:
    index.add_turn(1, 'Hi there', {'ref': completion})
  assert index.turns == []
  return str(error.value)


def _decode_files(directory, step):
  return list(directory.glob(f'decode_*_step_{step}.safetensors'))


def test_an_argmax_that_is_not_the_released_token(tmp_path):
  """An argmax that is not the released token."""

  def change(directory, completion):
    path = directory / 'generated_tokens.jsonl'
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({'token_ids': [[7]], 'texts': ['x']})
    path.write_text('\n'.join(lines) + '\n')

  assert 'released token is 7' in _broken(tmp_path, change)


def test_a_sampled_chat_generates_what_it_released(tmp_path):
  """A chat that samples need not generate the argmax; its stop is unknown."""
  chat = fixtures.TinyChat()
  fixed = fixtures.side_info(chat)
  fixed['chat']['options']['top_k'] = 40
  index = capture_index.CaptureIndex.create(tmp_path, {'ref': fixed})
  turn = chat.send('Hi there', ['a', 'b'], True)
  completion = fixtures.write_turn(index.turn_directory('ref', 1), turn)
  path = index.turn_directory('ref', 1) / 'generated_tokens.jsonl'
  lines = path.read_text().splitlines()
  lines[0] = json.dumps({'token_ids': [[7]], 'texts': ['x']})
  path.write_text('\n'.join(lines) + '\n')

  index.add_turn(1, 'Hi there', {'ref': completion})

  output = index.turn(1)['sides']['ref']['output']
  assert (output[0]['id'], output[0]['text']) == (7, 'x')
  assert output[-1]['id'] is None
  assert output[-1]['stop'] and not output[-1]['released']


def test_a_missing_decode_step_is_a_gap(tmp_path):
  """A missing decode step is a gap."""

  def change(directory, completion):
    steps = sorted(
        int(p.name.rsplit('_', 1)[1].split('.')[0])
        for p in directory.glob('decode_post_logits_step_*')
    )
    for path in _decode_files(directory, steps[1]):
      path.unlink()

  assert 'comes next' in _broken(tmp_path, change, reply=('a', 'b', 'c'))


def test_an_unreleased_token_that_is_not_a_stop(tmp_path):
  """An unreleased token that is not a stop."""

  def change(directory, completion):
    path = directory / 'generated_tokens.jsonl'
    path.write_text(''.join(path.read_text().splitlines(True)[:-1]))
    completion['content'] = 'a'

  message = _broken(tmp_path, change, stop=False)
  assert 'neither released nor a stop token' in message


def test_more_decode_steps_than_tokens(tmp_path):
  """More decode steps than tokens."""

  def change(directory, completion):
    (directory / 'generated_tokens.jsonl').write_text('')

  assert '3 decode steps for 0 released tokens' in _broken(tmp_path, change)


def test_a_wrong_context_count(tmp_path):
  """A wrong context count."""

  def change(directory, completion):
    completion['context_token_count'] += 1

  assert 'context tokens' in _broken(tmp_path, change)


def test_a_wrong_completion_count(tmp_path):
  """A wrong completion count."""

  def change(directory, completion):
    completion['usage']['completion_tokens'] = 2

  assert 'completion tokens' in _broken(tmp_path, change)


def test_prompt_tokens_that_do_not_fill_the_prefill(tmp_path):
  """Prompt tokens that do not fill the prefill."""

  def change(directory, completion):
    completion['prompt_tokens'].pop()

  assert 'Turn 1 fed positions 0..' in _broken(tmp_path, change)


def test_a_turn_the_runner_numbers_otherwise(tmp_path):
  """A turn the runner numbers otherwise."""

  def change(directory, completion):
    completion['turn'] = 2

  assert 'calls turn 1 turn 2' in _broken(tmp_path, change)


def test_an_unexpected_file(tmp_path):
  """An unexpected file."""

  def change(directory, completion):
    (directory / 'notes.txt').write_text('x')

  assert 'unexpected file, notes.txt' in _broken(tmp_path, change)


def test_a_mask_that_disagrees_with_the_positions(tmp_path):
  """A mask that disagrees with the positions."""

  def change(directory, completion):
    path = sorted(directory.glob('prefill_*_pre_mask_step_*.safetensors'))[0]
    mask = safetensors_numpy.load_file(path)['pre_mask']
    mask[0, 0, -1, 0] = True
    safetensors_numpy.save_file(
        {'pre_mask': mask},
        str(path),
        metadata={
            'signature': fixtures.SIGNATURE,
            'step': path.name.rsplit('_', 1)[1].split('.')[0],
        },
    )

  assert 'mask attends from' in _broken(tmp_path, change)


def test_a_file_without_step_metadata(tmp_path):
  """A file without step metadata."""

  def change(directory, completion):
    path = next(directory.glob('decode_pre_mask_step_*.safetensors'))
    safetensors_numpy.save_file(safetensors_numpy.load_file(path), str(path))

  assert 'names no signature and step' in _broken(tmp_path, change)


# The real runner's files, when this machine has them.

_REAL_TURN = pathlib.Path('/tmp/model_debugger_test_turn1')
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


def _real_completion(path: pathlib.Path) -> dict:
  data = json.loads(path.read_text())
  data = data.get('completion', data)
  return capture_index.completion_dict(
      json_format.ParseDict(data, runner_pb2.Completion())
  )


def _link_turn(index, n, source):
  for side in index.sides:
    directory = index.turn_directory(side, n)
    for path in source.iterdir():
      if not path.name.startswith('_'):
        os.symlink(path, directory / path.name)


@pytest.mark.skipif(not _REAL_TURN.is_dir(), reason='no real turn here')
def test_a_real_turn(tmp_path):
  """A real turn."""
  index = capture_index.CaptureIndex.create(tmp_path, {'ref': _GEMMA})
  _link_turn(index, 1, _REAL_TURN)

  index.add_turn(
      1,
      'What is 2 + 2?',
      {'ref': _real_completion(_REAL_TURN / '_completion.json')},
  )

  side = index.turn(1)['sides']['ref']
  assert len(side['files']) == 152
  assert len(side['input']) == 17
  assert [f['step'] for f in side['forwards']] == [17, *range(17, 25)]
  assert side['forwards'][0]['input_positions'] == [0, 15]
  assert [t['text'] for t in side['output']] == [
      '2',
      ' +',
      ' ',
      '2',
      ' is',
      ' **',
      '4',
      '**.',
  ]
  assert [t['position'] for t in side['output']] == list(range(17, 25))
  assert side['stop_reason'] == 'max_output_tokens'
  assert [(s['moment'], s['step'], s['processed']) for s in side['kv']] == [
      ('prefill_pre', 17, 0),
      ('prefill_post', 17, 16),
      ('terminal', 24, 24),
  ]
  layers = side['kv'][2]['layers']
  assert [layer['layer'] for layer in layers] == list(range(15))
  assert layers[4]['key']['shape'] == [1, 1, 4096, 512]
  assert layers[4]['value']['layout'] == [
      'batch',
      'kv_head',
      'head_dim',
      'sequence',
  ]


@pytest.mark.skipif(
    not (_REAL_SESSIONS and _REAL_SESSION.is_dir()), reason='no REAL_SESSIONS'
)
def test_a_real_session_of_four_turns(tmp_path):
  """A real session of four turns."""
  index = capture_index.CaptureIndex.create(tmp_path, {'ref': _GEMMA})
  for n in range(1, 5):
    source = _REAL_SESSION / f'turn{n}'
    _link_turn(index, n, source)
    index.add_turn(
        n, '', {'ref': _real_completion(source / '_completion.json')}
    )

  sides = [turn['sides']['ref'] for turn in index.turns]
  assert [s['stop_reason'] for s in sides] == [
      'eos',
      'eos',
      'max_output_tokens',
      'eos',
  ]
  # Turn 2's 362 inputs went through three chunks of 128, 128 and 105.
  chunks = [f for f in sides[1]['forwards'] if f['phase'] == 'prefill']
  assert [(f['step'], f['input_positions']) for f in chunks] == [
      (155, [26, 153]),
      (283, [154, 281]),
      (388, [282, 386]),
  ]
  # Each later turn first feeds the previous turn's last token: the stop
  # `<turn|>` after a stop, the released ` earth` after a cut.
  assert [s['input'][0]['id'] for s in sides[1:]] == [106, 106, 7764]
  assert [s['input'][0]['position'] for s in sides[1:]] == [26, 390, 418]


_REAL_THINKING = pathlib.Path(_REAL_SESSIONS, 'sessionD')


@pytest.mark.skipif(
    not (_REAL_SESSIONS and _REAL_THINKING.is_dir()), reason='no REAL_SESSIONS'
)
def test_a_real_session_that_thinks(tmp_path):
  """A real session that thinks: the next turn rolls the context back."""
  index = capture_index.CaptureIndex.create(tmp_path, {'ref': _GEMMA})
  for n in (1, 2):
    source = _REAL_THINKING / f'turn{n}'
    _link_turn(index, n, source)
    index.add_turn(
        n, '', {'ref': _real_completion(source / '_completion.json')}
    )

  first, second = (turn['sides']['ref'] for turn in index.turns)
  assert [t['text'] for t in first['output'][:3]] == [
      '<|channel>',
      'thought',
      '\n',
  ]
  assert first['output'][-1]['position'] == 63
  # Turn 2 starts back at position 23, turn 1's last input token, and feeds
  # three tokens of the runtime's own before its 13 prompt tokens.
  assert [(t['origin'], t['position']) for t in second['input'][:4]] == [
      ('history', 23),
      ('history', 24),
      ('history', 25),
      ('prompt', 26),
  ]
  assert len(second['input']) == 16
  chunks = [f for f in second['forwards'] if f['phase'] == 'prefill']
  assert [f['input_positions'] for f in chunks] == [[23, 24], [25, 37]]


def test_released_token_with_unicode_line_separator_is_indexed(tmp_path):
  """A released token with U+2028 or U+2029 in raw UTF-8 JSONL is indexed."""
  index = fixtures.capture(
      tmp_path, [('Hi', ['line1\u2028line2\u2029end'], True)]
  )

  side = index.turn(1)['sides']['ref']
  assert side['output'][0]['text'] == 'line1\u2028line2\u2029end'


def test_non_thinking_turn_with_five_tokens_has_empty_reasoning(tmp_path):
  """A non-thinking turn with five+ released tokens has empty reasoning."""
  index = fixtures.capture(
      tmp_path,
      [('Count to five', ['one', 'two', 'three', 'four', 'five'], True)],
  )

  side = index.turn(1)['sides']['ref']
  assert side['completion']['reasoning_content'] == ''
  assert side['completion']['content'] == 'onetwothreefourfive'
