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

"""Native request, capability and result contracts; no model execution."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from model_explorer_debugger.runtime.runner_device import (
    RunnerDevices,
    validate_native_request,
)
from model_explorer_debugger.runtime.runner_discovery import descriptor


class NativeDemoBackendTests(unittest.TestCase):

  def test_old_cpu_runner_cannot_silently_accept_a_gpu_or_long_context_request(
      self,
  ):
    connection = SimpleNamespace(hello={})
    RunnerDevices.validate_connection_capabilities(
        connection, {'backend': 'CPU', 'contextLength': 1024}
    )
    for run in ({'backend': 'GPU'}, {'backend': 'CPU', 'contextLength': 4096}):
      with self.assertRaisesRegex(ValueError, 'matching rebuilt Runner'):
        RunnerDevices.validate_connection_capabilities(connection, run)
    connection.hello = {
        'capabilities': {
            'backends': ['CPU', 'GPU'],
            'contextLengths': [1024, 4096],
        }
    }
    RunnerDevices.validate_connection_capabilities(
        connection, {'backend': 'GPU', 'contextLength': 4096}
    )

  def test_real_eval_prompt_and_gpu_context_are_admitted_without_truncation(
      self,
  ):
    prompt = (
        Path(__file__).resolve().parent / 'fixtures/real-eval-case-prompt.txt'
    ).read_text()
    request = dict(
        operation='generate',
        run=dict(backend='GPU', contextLength='4096'),
        prompt=prompt,
        manifest={'verified': True},
        generation={'maxOutputTokens': 32},
    )
    before = deepcopy(request)
    validate_native_request(request)
    self.assertEqual(request, before)

  def test_gpu_capability_is_preserved_for_the_native_runtime_only(self):
    value = dict(
        version=1,
        protocolVersion=4,
        runnerInstance=str(uuid4()),
        controlConnected=False,
        busy=False,
        residentSessions=0,
        capabilities=dict(
            runtime='LiteRT-LM',
            backends=['CPU', 'GPU'],
            modalities=['text'],
            tensorCapture=True,
            persistentConversation=True,
            contextLengths=[1024, 4096],
            maxOutputTokens=32,
            maxCapturePoints=16,
        ),
        runtimes=[
            dict(id='LiteRT-LM', backends=['CPU', 'GPU'], transport='native')
        ],
        environment=dict(
            capturedAt='synthetic',
            platform='macOS',
            operatingSystem='synthetic',
            architecture='arm64',
        ),
        state=dict(capturedAt='synthetic'),
    )
    self.assertEqual(
        descriptor(value)['capabilities']['backends'], ['CPU', 'GPU']
    )
    value['runtimes'] = [
        dict(id='PyTorch', backends=['GPU'], transport='localFiles')
    ]
    with self.assertRaises(ValueError):
      descriptor(value)

  def test_effective_result_must_match_the_requested_backend(self):
    identity = str(uuid4())
    job = dict(
        modelSHA256='a' * 64,
        backend='GPU',
        prompt='hello',
        contextLength=4096,
        maxOutputTokens=2,
    )
    result = dict(
        jobID=identity,
        modelSHA256=job['modelSHA256'],
        platform='macOS',
        backend='GPU',
        debuggerEnabled=True,
        input=job['prompt'],
        contextLength=4096,
        maxOutputTokens=2,
        thinkingEnabled=False,
        speculativeDecodingEnabled=False,
        sampler={'seed': 0, 'temperature': 0, 'topK': 1, 'topP': 1},
        backendEvidence={
            'requestedBackend': 'GPU',
            'effectiveBackend': 'GPU',
            'engineInitialized': True,
        },
    )
    adapter = RunnerDevices('/unused')
    adapter.platform = 'macOS'
    adapter._validate_result(result, identity, job)
    result['backend'] = 'CPU'
    with self.assertRaises(ValueError):
      adapter._validate_result(result, identity, job)


if __name__ == '__main__':
  unittest.main()
