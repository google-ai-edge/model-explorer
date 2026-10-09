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

"""The two request primitives every Runner exchange goes through.

Peers are synthetic.
"""

import unittest
from unittest.mock import patch

from model_explorer_debugger.runtime import turn_outcome
from model_explorer_debugger.runtime.runner_channel import call, request


class Peer:
  """Replies are functions of the first sent message, or exceptions to raise.

  Exceptions are raised from receive.
  """

  def __init__(self, *replies):
    self.replies = list(replies)
    self.sent = []

  def send(self, message):
    self.sent.append(message)

  def receive(self, timeout):
    if not self.replies:
      raise TimeoutError('Waiting for Runner response')
    reply = self.replies.pop(0)
    if isinstance(reply, BaseException):
      raise reply
    return reply(self.sent[0])


def echo(kind, **fields):
  return lambda sent: dict(type=kind, requestId=sent['requestId'], **fields)


class CallTests(unittest.TestCase):

  def test_reply_is_matched_by_identity_and_type(self):
    peer = Peer(echo('model_info', available=True))
    self.assertTrue(call(peer, 'model_info', modelSHA256='a' * 64)['available'])
    self.assertEqual(peer.sent[0]['modelSHA256'], 'a' * 64)

  def test_reply_may_have_a_different_type_than_the_request(self):
    self.assertEqual(
        call(Peer(echo('closed')), 'close', reply='closed')['type'], 'closed'
    )

  def test_foreign_reply_is_never_accepted(self):
    peer = Peer(
        lambda sent: dict(type='model_info', requestId='another-request')
    )
    with self.assertRaisesRegex(ValueError, 'identity mismatch'):
      call(peer, 'model_info')

  def test_runner_error_text_is_reported(self):
    with self.assertRaisesRegex(ValueError, 'Model transfer is incomplete'):
      call(
          Peer(echo('error', error='Model transfer is incomplete')),
          'model_commit',
      )

  def test_unexpected_type_uses_the_callers_failure_text(self):
    with self.assertRaisesRegex(ValueError, 'did not acknowledge'):
      call(
          Peer(echo('progress')),
          'reset',
          failure='Runner did not acknowledge Session reset',
      )


class RequestTests(unittest.TestCase):

  def message(self):
    return {'type': 'generate', 'requestId': 'request-1'}

  def test_events_are_yielded_until_the_caller_stops(self):
    peer = Peer(echo('delta', text='a'), TimeoutError(), echo('completed'))
    kinds = []
    for event in request(peer, self.message()):
      kinds.append(event['type'])
      if event['type'] == 'completed':
        break
    self.assertEqual(kinds, ['delta', 'completed'])
    self.assertEqual([m['type'] for m in peer.sent], ['generate'])

  def test_cancel_is_sent_once_and_the_terminal_event_is_still_awaited(self):
    peer = Peer(
        echo('delta', text='a'),
        echo('delta', text='b'),
        echo('error', generationStatus='stopped'),
    )
    events = list(
        zip(
            range(3),
            request(
                peer, self.message(), cancelled=lambda: len(peer.sent) >= 1
            ),
        )
    )
    self.assertEqual(events[-1][1]['generationStatus'], 'stopped')
    self.assertEqual([m['type'] for m in peer.sent], ['generate', 'cancel'])

  def test_foreign_event_ends_the_request(self):
    peer = Peer(
        lambda sent: dict(
            type='delta', requestId='another-request', text='unsafe'
        )
    )
    with self.assertRaisesRegex(ValueError, 'identity mismatch'):
      next(request(peer, self.message()))

  def test_unanswered_cancel_and_overlong_task_fail_instead_of_waiting_forever(
      self,
  ):
    clock = iter(range(0, 10_000, 20))
    with patch(
        'model_explorer_debugger.runtime.runner_channel.time.monotonic',
        lambda: next(clock),
    ):
      with self.assertRaisesRegex(
          ConnectionError, 'did not finish cancellation'
      ):
        next(request(Peer(), self.message(), cancelled=lambda: True))
      with self.assertRaisesRegex(TimeoutError, 'exceeded 10 minutes'):
        next(request(Peer(), self.message()))


class OutcomeTests(unittest.TestCase):

  def test_only_completed_text_is_reported(self):
    self.assertEqual(
        turn_outcome.text('completed', 'answer'),
        dict(
            output='answer',
            output_confirmed=True,
            generation_status='completed',
        ),
    )
    self.assertEqual(
        turn_outcome.text('stopped', 'partial'),
        dict(output='', output_confirmed=False, generation_status='stopped'),
    )

  def test_failed_dump_leaves_text_fields_alone(self):
    result = dict(turn_outcome.text('completed', 'answer'))
    result.update(turn_outcome.dump_failed(ConnectionError('cable'), lost=True))
    self.assertTrue(result['output_confirmed'])
    self.assertEqual(
        result['debug_data'], {'status': 'unavailable', 'error': 'cable'}
    )
    self.assertTrue(result['connection_lost'])

  def test_lost_link_before_text_is_unconfirmed_and_never_a_dump(self):
    result = turn_outcome.unconfirmed(TimeoutError('silent'), input='hello')
    self.assertEqual(
        (result['input'], result['output'], result['generation_status']),
        ('hello', '', 'failed'),
    )
    self.assertFalse(result['output_confirmed'] or result['dump_complete'])
    self.assertTrue(result['connection_lost'])


if __name__ == '__main__':
  unittest.main()
