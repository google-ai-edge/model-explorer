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

"""Tests of the fake runner's rules for Load and CreateChat.

The other tests lean on these rules being the real runner's
(runner_service.cc, options_schema.cc), with its codes and texts.
"""

import json
import threading
import time
from typing import Any

from absl.testing import parameterized
import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.protocol import runner_pb2_grpc
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main
# pyformat: disable
from google.protobuf import struct_pb2
# pyformat: enable

_SHA256 = 'a' * 64
_LOAD = runner_pb2.LoadRequest(
    sha256=_SHA256, context_length=64, device=runner_pb2.CPU
)
_CHAT = runner_pb2.CreateChatRequest(
    options=runner_pb2.ChatOptions(max_completion_tokens=8)
)
_TIMEOUT = 10
_LOAD_FAILED = 'A Load failed; a runner loads one model, once.'


class _Runner(fake_runner.FakeRunner):
  """A fake runner that shows its last Load's and CreateChat's progress.

  Attributes:
      contexts: The context of the last call of each, by its name.
      ended: Set once the last call of each has returned or failed.
  """

  def __init__(self) -> None:
    """A runner with the model in its store."""
    super().__init__(fake_runner.no_reply, store={_SHA256})
    self.contexts: dict[str, grpc.ServicerContext] = {}
    self.ended: dict[str, threading.Event] = {}

  def Load(self, request, context):  # noqa: D102
    return self._watch('Load', super().Load, request, context)

  def CreateChat(self, request, context):  # noqa: D102
    return self._watch('CreateChat', super().CreateChat, request, context)

  def _watch(self, name, method, request, context):
    """Calls `method`, noting its context and when it ends."""
    self.contexts[name] = context
    self.ended[name] = ended = threading.Event()
    try:
      return method(request, context)
    finally:
      ended.set()


def _fail(name: str, code: grpc.StatusCode, message: str):
  """A `before` that fails the RPC `name` with a status."""

  def before(method: str) -> None:
    if method == name:
      raise fake_runner.AbortError(code, message)

  return before


def _stall(name: str, release: threading.Event, reached: threading.Event):
  """A `before` that holds the RPC `name` until `release` is set."""

  def before(method: str) -> None:
    if method == name:
      reached.set()
      release.wait(_TIMEOUT)

  return before


def _options(**values) -> struct_pb2.Struct:
  """Runtime options of these values."""
  options = struct_pb2.Struct()
  options.update(values)
  return options


def _chat(**options) -> runner_pb2.CreateChatRequest:
  """A CreateChat with these ChatOptions; max_completion_tokens 8 if unset."""
  options.setdefault('max_completion_tokens', 8)
  return runner_pb2.CreateChatRequest(options=runner_pb2.ChatOptions(**options))


def _schema_info(schema: dict[str, Any] | str) -> runner_pb2.RunnerInfo:
  """The default info, with this load options schema."""
  info = runner_pb2.RunnerInfo()
  info.CopyFrom(fake_runner.DEFAULT_INFO)
  info.runtime.load_options_schema = (
      schema if isinstance(schema, str) else json.dumps(schema)
  )
  return info


class FakeRunnerTest(parameterized.TestCase):
  """The fake runner's Load and CreateChat, called over gRPC."""

  def setUp(self):
    """A fake runner holding the model, and a stub connected to it."""
    super().setUp()
    self.runner = _Runner().start()
    self.addCleanup(self.runner.stop)
    # pylint: disable=line-too-long
    channel = grpc.insecure_channel(self.runner.address)
    # pylint: enable=line-too-long
    self.addCleanup(channel.close)
    self.stub = runner_pb2_grpc.RunnerStub(channel)

  def assert_refused(self, call, request, code, text):
    """Asserts the RPC `call` fails with `code` and `text`."""
    with self.assertRaises(grpc.RpcError) as failed:
      call(request, timeout=_TIMEOUT)
    self.assertEqual(
        (failed.exception.code(), failed.exception.details()), (code, text)
    )

  def cancel(self, name, call, request):
    """Cancels the RPC `name` in its runtime, and waits for its end."""
    release, reached = threading.Event(), threading.Event()
    self.runner.before = _stall(name, release, reached)
    future = call.future(request)
    self.assertTrue(reached.wait(_TIMEOUT))
    context, ended = self.runner.contexts[name], self.runner.ended[name]
    future.cancel()
    # The runtime goes on until the runner has seen the cancel.
    deadline = time.monotonic() + _TIMEOUT
    while context.is_active() and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertFalse(context.is_active())
    self.runner.before = None
    release.set()
    self.assertTrue(ended.wait(_TIMEOUT))

  # ---- Load

  def test_a_load_not_in_the_store_is_not_final(self):
    """NOT_FOUND asks for an upload; the Load after it loads."""
    self.runner.store.clear()
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.NOT_FOUND,
        f'No model with sha256 {_SHA256} in the model store.',
    )
    self.runner.store.add(_SHA256)
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    self.assertEqual(self.runner.load_request, _LOAD)

  def test_a_refused_load_is_final(self):
    """A Load refused for its request leaves every later one refused."""
    request = runner_pb2.LoadRequest()
    request.CopyFrom(_LOAD)
    request.context_length = 0
    self.assert_refused(
        self.stub.Load,
        request,
        grpc.StatusCode.INVALID_ARGUMENT,
        'LoadRequest.context_length is 0; it must be positive.',
    )
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        _LOAD_FAILED,
    )
    self.assertIsNone(self.runner.load_request)

  def test_a_load_the_runtime_fails_is_final(self):
    """An AbortError from `before` is the runtime's failure: final."""
    self.runner.before = _fail(
        'Load', grpc.StatusCode.INTERNAL, 'Out of memory.'
    )
    self.assert_refused(
        self.stub.Load, _LOAD, grpc.StatusCode.INTERNAL, 'Out of memory.'
    )
    self.runner.before = None
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        _LOAD_FAILED,
    )

  def test_a_runtime_not_found_is_internal_and_final(self):
    """The model is in the store: what the runtime misses is INTERNAL."""
    self.runner.before = _fail(
        'Load', grpc.StatusCode.NOT_FOUND, 'No tokenizer.'
    )
    self.assert_refused(
        self.stub.Load, _LOAD, grpc.StatusCode.INTERNAL, 'No tokenizer.'
    )
    self.runner.before = None
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        _LOAD_FAILED,
    )

  def test_the_runtime_runs_after_the_state_refusals(self):
    """An injected failure is not reached where the state refuses."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    self.runner.before = _fail(
        'Load', grpc.StatusCode.INTERNAL, 'Out of memory.'
    )
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        'A model is loaded; a runner loads one.',
    )
    self.assertEqual(self.runner.load_request, _LOAD)

  def test_the_runtime_runs_after_a_failed_load_is_refused(self):
    """After a failed Load, the refusal, not another runtime failure."""
    self.runner.before = _fail(
        'Load', grpc.StatusCode.INTERNAL, 'Out of memory.'
    )
    self.assert_refused(
        self.stub.Load, _LOAD, grpc.StatusCode.INTERNAL, 'Out of memory.'
    )
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        _LOAD_FAILED,
    )

  def test_a_cancelled_load_is_discarded_and_final(self):
    """A Load cancelled in the runtime loads nothing; later ones fail."""
    self.cancel('Load', self.stub.Load, _LOAD)
    self.assertIsNone(self.runner.load_request)
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        _LOAD_FAILED,
    )

  def test_a_load_or_create_chat_meanwhile_is_refused(self):
    """While a Load is in the runtime, another Load or CreateChat waits."""
    release, reached = threading.Event(), threading.Event()
    self.runner.before = _stall('Load', release, reached)
    future = self.stub.Load.future(_LOAD, timeout=_TIMEOUT)
    try:
      self.assertTrue(reached.wait(_TIMEOUT))
      for call, request in (
          (self.stub.Load, _LOAD),
          (self.stub.CreateChat, _CHAT),
      ):
        self.assert_refused(
            call,
            request,
            grpc.StatusCode.FAILED_PRECONDITION,
            'A Load or CreateChat is in progress.',
        )
    finally:
      release.set()
    future.result(timeout=_TIMEOUT)
    self.assertEqual(self.runner.load_request, _LOAD)

  # ---- CreateChat

  def test_create_chat_before_load_is_refused_not_failed(self):
    """'No model is loaded.' changes nothing, an injected failure aside."""
    self.runner.before = _fail(
        'CreateChat', grpc.StatusCode.INTERNAL, 'The runtime failed.'
    )
    self.assert_refused(
        self.stub.CreateChat,
        _CHAT,
        grpc.StatusCode.FAILED_PRECONDITION,
        'No model is loaded.',
    )
    self.runner.before = None
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    self.stub.CreateChat(_CHAT, timeout=_TIMEOUT)
    self.assertEqual(self.runner.chat_request, _CHAT)

  def test_a_create_chat_the_runtime_fails_leaves_no_chat(self):
    """An AbortError from `before` on CreateChat makes no chat; another may."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    self.runner.before = _fail(
        'CreateChat', grpc.StatusCode.INTERNAL, 'The runtime failed.'
    )
    self.assert_refused(
        self.stub.CreateChat,
        _CHAT,
        grpc.StatusCode.INTERNAL,
        'The runtime failed.',
    )
    self.assertIsNone(self.runner.chat_request)
    self.runner.before = None
    self.stub.CreateChat(_CHAT, timeout=_TIMEOUT)
    self.assertEqual(self.runner.chat_request, _CHAT)

  def test_a_system_message_not_first_is_the_runtimes_failure(self):
    """LiteRT-LM refuses it: no chat, until another CreateChat."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    request = runner_pb2.CreateChatRequest()
    request.CopyFrom(_CHAT)
    request.messages.add(role=runner_pb2.ChatMessage.USER, content='Hi')
    request.messages.add(role=runner_pb2.ChatMessage.SYSTEM, content='No')
    self.assert_refused(
        self.stub.CreateChat,
        request,
        grpc.StatusCode.INVALID_ARGUMENT,
        'A system message may only be the first message.',
    )
    self.assertIsNone(self.runner.chat_request)
    self.stub.CreateChat(_CHAT, timeout=_TIMEOUT)
    self.assertEqual(self.runner.chat_request, _CHAT)

  def test_a_system_message_not_first_is_refused_before_the_runtime(self):
    """LiteRT-LM checks the messages first: the runtime is never reached."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    reached = []

    def before(method):
      reached.append(method)
      raise fake_runner.AbortError(
          grpc.StatusCode.INTERNAL, 'The runtime failed.'
      )

    self.runner.before = before
    request = runner_pb2.CreateChatRequest()
    request.CopyFrom(_CHAT)
    request.messages.add(role=runner_pb2.ChatMessage.USER, content='Hi')
    request.messages.add(role=runner_pb2.ChatMessage.SYSTEM, content='No')
    self.assert_refused(
        self.stub.CreateChat,
        request,
        grpc.StatusCode.INVALID_ARGUMENT,
        'A system message may only be the first message.',
    )
    self.assertNotIn('CreateChat', reached)

  def test_a_create_chat_replaces_the_chat(self):
    """The chat before is closed: its captures go, turns count anew."""
    runner = fake_runner.FakeRunner(
        lambda turn, prompt: (
            [runner_pb2.Chunk(content='Hi')],
            runner_pb2.Completion(),
            {'turn.txt': str(turn).encode()},
        ),
        store={_SHA256},
    ).start()
    self.addCleanup(runner.stop)
    # pylint: disable=line-too-long
    channel = grpc.insecure_channel(runner.address)
    # pylint: enable=line-too-long
    self.addCleanup(channel.close)
    stub = runner_pb2_grpc.RunnerStub(channel)
    stub.Load(_LOAD, timeout=_TIMEOUT)
    stub.CreateChat(_CHAT, timeout=_TIMEOUT)
    send = runner_pb2.SendRequest(
        message=runner_pb2.ChatMessage(
            role=runner_pb2.ChatMessage.USER, content='Hello'
        )
    )
    list(stub.Send(send, timeout=_TIMEOUT))
    list(stub.Send(send, timeout=_TIMEOUT))
    self.assertEqual(sorted(runner.captures), [1, 2])
    second = _chat(max_completion_tokens=4)
    stub.CreateChat(second, timeout=_TIMEOUT)
    self.assertEqual(runner.chat_request, second)
    self.assertEqual(runner.chats, [_CHAT, second])
    self.assertEqual(runner.captures, {})
    list(stub.Send(send, timeout=_TIMEOUT))
    self.assertEqual(runner.captures, {1: {'turn.txt': b'1'}})

  def test_a_failed_create_chat_closes_the_chat_before(self):
    """The chat it would replace is gone all the same."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    self.stub.CreateChat(_CHAT, timeout=_TIMEOUT)
    self.runner.before = _fail(
        'CreateChat', grpc.StatusCode.INTERNAL, 'The runtime failed.'
    )
    self.assert_refused(
        self.stub.CreateChat,
        _CHAT,
        grpc.StatusCode.INTERNAL,
        'The runtime failed.',
    )
    self.assertIsNone(self.runner.chat_request)

  def test_a_create_chat_during_a_send_is_refused(self):
    """FAILED_PRECONDITION, and the chat and its Send go on."""
    sending, release = threading.Event(), threading.Event()

    def respond(turn, prompt):
      del turn, prompt  # Unused.

      def chunks():
        sending.set()
        release.wait(_TIMEOUT)
        yield runner_pb2.Chunk(content='Hi')

      return chunks(), runner_pb2.Completion(), {}

    runner = fake_runner.FakeRunner(respond, store={_SHA256}).start()
    self.addCleanup(runner.stop)
    # pylint: disable=line-too-long
    channel = grpc.insecure_channel(runner.address)
    # pylint: enable=line-too-long
    self.addCleanup(channel.close)
    stub = runner_pb2_grpc.RunnerStub(channel)
    stub.Load(_LOAD, timeout=_TIMEOUT)
    stub.CreateChat(_CHAT, timeout=_TIMEOUT)
    reply = stub.Send(
        runner_pb2.SendRequest(
            message=runner_pb2.ChatMessage(
                role=runner_pb2.ChatMessage.USER, content='Hello'
            )
        ),
        timeout=_TIMEOUT,
    )
    replied = []
    reader = threading.Thread(target=lambda: replied.extend(reply))
    reader.start()
    self.assertTrue(sending.wait(_TIMEOUT))
    with self.assertRaises(grpc.RpcError) as failed:
      stub.CreateChat(_chat(max_completion_tokens=4), timeout=_TIMEOUT)
    self.assertEqual(
        (failed.exception.code(), failed.exception.details()),
        (grpc.StatusCode.FAILED_PRECONDITION, 'A Send is in progress.'),
    )
    release.set()
    reader.join(_TIMEOUT)
    self.assertEqual(runner.chat_request, _CHAT)
    self.assertEqual(sorted(runner.captures), [1])

  def test_a_cancelled_create_chat_is_discarded(self):
    """A CreateChat cancelled in the runtime makes no chat."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    self.cancel('CreateChat', self.stub.CreateChat, _CHAT)
    self.assertIsNone(self.runner.chat_request)
    self.stub.CreateChat(_CHAT, timeout=_TIMEOUT)
    self.assertEqual(self.runner.chat_request, _CHAT)

  @parameterized.parameters(
      (
          {'max_completion_tokens': 0},
          'ChatOptions.max_completion_tokens is 0; it must be at least 1.',
      ),
      (
          {'temperature': -0.5},
          'ChatOptions.temperature is -0.5; it must be at least 0.',
      ),
      (
          {'temperature': float('nan')},
          'ChatOptions.temperature is nan; it must be at least 0.',
      ),
      (
          {'top_p': 1.5},
          'ChatOptions.top_p is 1.5; it must be from 0 to 1.',
      ),
      (
          {'top_p': float('nan')},
          'ChatOptions.top_p is nan; it must be from 0 to 1.',
      ),
      ({'top_k': 0}, 'ChatOptions.top_k is 0; it must be at least 1.'),
  )
  def test_chat_options_out_of_range_are_refused(self, options, text):
    """CheckChatOptions' ranges, NaN included; the refusal makes no chat."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    self.assert_refused(
        self.stub.CreateChat,
        _chat(**options),
        grpc.StatusCode.INVALID_ARGUMENT,
        text,
    )
    self.assertIsNone(self.runner.chat_request)

  def test_chat_options_at_their_limits_create_the_chat(self):
    """0 for temperature and top_p, 1 for top_k and tokens, are taken."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    request = _chat(max_completion_tokens=1, temperature=0, top_p=1, top_k=1)
    self.stub.CreateChat(request, timeout=_TIMEOUT)
    self.assertEqual(self.runner.chat_request, request)

  # ---- runtime_options, as the runtime's schemas allow them

  @parameterized.parameters(
      ({'cpuThreads': 4}, None),
      ({'cpuThreads': 4.0, 'visionBackend': 'GPU'}, None),
      (
          {'cpuThreads': 0},
          (
              'LoadRequest.runtime_options.cpuThreads is 0.0; it must be at '
              'least 1.'
          ),
      ),
      (
          {'cpuThreads': 2.5},
          (
              'LoadRequest.runtime_options.cpuThreads is 2.5; it must be of '
              'type "integer".'
          ),
      ),
      (
          {'enableYnnpack': 1},
          (
              'LoadRequest.runtime_options.enableYnnpack is 1.0; it must be of '
              'type "boolean".'
          ),
      ),
      (
          {'visionBackend': 'NPU'},
          (
              'LoadRequest.runtime_options.visionBackend is "NPU"; it must be '
              'one of ["CPU","GPU"].'
          ),
      ),
      (
          {'noSuchOption': True},
          (
              'LoadRequest.runtime_options has noSuchOption, which is not an '
              'option of this runtime.'
          ),
      ),
      # The first refusal in key order is the one answered.
      (
          {'zeta': True, 'cpuThreads': 0},
          (
              'LoadRequest.runtime_options.cpuThreads is 0.0; it must be at '
              'least 1.'
          ),
      ),
      # Non-finite numbers are refused before any other check.
      (
          {'audioBackend': 'NPU', 'cpuThreads': float('inf')},
          'LoadRequest.runtime_options.cpuThreads is not a finite number.',
      ),
  )
  def test_load_runtime_options(self, options, text):
    """The default load schema, as ValidateOptions checks it."""
    request = runner_pb2.LoadRequest()
    request.CopyFrom(_LOAD)
    request.runtime_options.CopyFrom(_options(**options))
    if text is None:
      self.stub.Load(request, timeout=_TIMEOUT)
      self.assertEqual(self.runner.load_request, request)
      return
    self.assert_refused(
        self.stub.Load, request, grpc.StatusCode.INVALID_ARGUMENT, text
    )
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        _LOAD_FAILED,
    )

  def test_load_runtime_options_are_checked_before_the_store(self):
    """Options a Load would be refused for ask for no upload."""
    self.runner.store.clear()
    request = runner_pb2.LoadRequest()
    request.CopyFrom(_LOAD)
    request.runtime_options.CopyFrom(_options(cpuThreads=0))
    self.assert_refused(
        self.stub.Load,
        request,
        grpc.StatusCode.INVALID_ARGUMENT,
        'LoadRequest.runtime_options.cpuThreads is 0.0; it must be at least 1.',
    )

  @parameterized.named_parameters(
      (
          'mac',
          runner_pb2.HostInfo.MACOS,
          (
              'The GPU libraries are not beside the LiteRT-LM library, in '
              f'{fake_runner.LIBRARY_DIRECTORY}.'
          ),
      ),
      (
          'ios_simulator',
          runner_pb2.HostInfo.IOS,
          fake_runner.IOS_SIMULATOR_GPU_REFUSAL,
      ),
  )
  def test_a_gpu_load_where_no_gpu_is_reported_is_refused_first(
      self, platform, text
  ):
    """Before the options and the store, as CanRunOnGpu says; final."""
    self.runner.info = runner_pb2.RunnerInfo()
    self.runner.info.CopyFrom(fake_runner.DEFAULT_INFO)
    self.runner.info.host.platform = platform
    self.runner.info.host.ClearField('gpu')
    self.runner.store.clear()
    request = runner_pb2.LoadRequest()
    request.CopyFrom(_LOAD)
    request.device = runner_pb2.GPU
    request.runtime_options.CopyFrom(_options(cpuThreads=0))
    self.assert_refused(
        self.stub.Load, request, grpc.StatusCode.FAILED_PRECONDITION, text
    )
    self.runner.store.add(_SHA256)
    self.assert_refused(
        self.stub.Load,
        _LOAD,
        grpc.StatusCode.FAILED_PRECONDITION,
        _LOAD_FAILED,
    )
    self.assertIsNone(self.runner.load_request)

  def test_a_gpu_load_where_a_gpu_is_reported_loads(self):
    """The default host reports its GPU."""
    request = runner_pb2.LoadRequest()
    request.CopyFrom(_LOAD)
    request.device = runner_pb2.GPU
    self.stub.Load(request, timeout=_TIMEOUT)
    self.assertEqual(self.runner.load_request, request)

  @parameterized.parameters(
      ({'thinking': True, 'thinking_token_budget': -1}, None),
      (
          {'thinking_token_budget': -2},
          (
              'ChatOptions.runtime_options.thinking_token_budget is -2.0; it '
              'must be at least -1.'
          ),
      ),
      (
          {'temperature': 1},
          (
              'ChatOptions.runtime_options has temperature, which is not an '
              'option of this runtime.'
          ),
      ),
  )
  def test_chat_runtime_options(self, options, text):
    """The default chat schema, after the ChatOptions' ranges."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    request = _chat()
    request.options.runtime_options.CopyFrom(_options(**options))
    if text is None:
      self.stub.CreateChat(request, timeout=_TIMEOUT)
      self.assertEqual(self.runner.chat_request, request)
      return
    self.assert_refused(
        self.stub.CreateChat,
        request,
        grpc.StatusCode.INVALID_ARGUMENT,
        text,
    )
    self.assertIsNone(self.runner.chat_request)

  def test_chat_options_ranges_come_before_runtime_options(self):
    """CheckChatOptions answers first."""
    self.stub.Load(_LOAD, timeout=_TIMEOUT)
    request = _chat(top_k=0)
    request.options.runtime_options.CopyFrom(_options(nothing=True))
    self.assert_refused(
        self.stub.CreateChat,
        request,
        grpc.StatusCode.INVALID_ARGUMENT,
        'ChatOptions.top_k is 0; it must be at least 1.',
    )

  @parameterized.parameters(
      (
          {'type': 'object', 'properties': {'p': {'maximum': 1.5}}},
          {'p': 2},
          grpc.StatusCode.INVALID_ARGUMENT,
          'LoadRequest.runtime_options.p is 2.0; it must be at most 1.5.',
      ),
      (
          {'properties': {'p': {'maximum': 1}}},
          {'p': 1e15},
          grpc.StatusCode.INVALID_ARGUMENT,
          'LoadRequest.runtime_options.p is 1e+15; it must be at most 1.',
      ),
      (
          {'properties': {'p': {'minimum': 0}}},
          {'p': -1e-05},
          grpc.StatusCode.INVALID_ARGUMENT,
          'LoadRequest.runtime_options.p is -1e-05; it must be at least 0.',
      ),
      (
          {'properties': {'p': {'type': ['string', 'null']}}},
          {'p': [1, 'a']},
          grpc.StatusCode.INVALID_ARGUMENT,
          (
              'LoadRequest.runtime_options.p is [1.0,"a"]; it must be of type '
              '["string","null"].'
          ),
      ),
      (
          {'type': 'object', 'required': ['p']},
          {},
          grpc.StatusCode.INVALID_ARGUMENT,
          'LoadRequest.runtime_options lacks p, which is required.',
      ),
      (
          {'additionalProperties': {'type': 'boolean'}},
          {'q': 'x'},
          grpc.StatusCode.INVALID_ARGUMENT,
          'LoadRequest.runtime_options.q is "x"; it must be of type "boolean".',
      ),
      (
          {'properties': {'p': {'pattern': 'a*'}}},
          {},
          grpc.StatusCode.INTERNAL,
          (
              "The runtime's options schema uses a keyword at "
              '#/properties/p/pattern, which the runner does not check.'
          ),
      ),
      (
          {'type': 'text'},
          {},
          grpc.StatusCode.INTERNAL,
          (
              "The runtime's options schema names no type it knows at #/type, "
              'which the runner does not check.'
          ),
      ),
      (
          'not JSON',
          {},
          grpc.StatusCode.INTERNAL,
          "The runtime's schema for LoadRequest.runtime_options is not JSON.",
      ),
  )
  def test_the_schema_the_runner_reports_decides(
      self, schema, options, code, text
  ):
    """Keywords, types and numbers as ValidateOptions checks and writes."""
    self.runner.info = _schema_info(schema)
    request = runner_pb2.LoadRequest()
    request.CopyFrom(_LOAD)
    request.runtime_options.CopyFrom(_options(**options))
    self.assert_refused(self.stub.Load, request, code, text)


if __name__ == '__main__':
  test_main.main()
