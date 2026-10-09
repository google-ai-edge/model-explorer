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

"""Constants of the Server <-> Runner protocol shared by every transport module.

v5 adds to v4, without changing any v4 message: binary file frames, pipelined
file
requests and, for a Mac Runner sharing this filesystem, file links. A v4 Runner
is
still driven with base64 chunks, one at a time. src/contracts holds the wire
schema.

Swift mirrors: AppleRunnerTransport.swift (protocolVersion, inspection
subprotocol),
RunnerDescriptor.swift (descriptor version), CaptureJob.swift and
RunnerEnvironment.swift
(native limits). Change them together.
"""

PROTOCOL_VERSION = 5
SUPPORTED_PROTOCOL_VERSIONS = (4, 5)
FILE_CHUNK = 512 * 1024
FILE_WINDOW = 8
DESCRIPTOR_VERSION = 1
INSPECTION_PROTOCOL = 'model-debugger-inspect-v1'
NATIVE_BACKENDS = ('CPU', 'GPU')
NATIVE_CONTEXT_LENGTHS = (1024, 4096)
NATIVE_MAX_OUTPUT_TOKENS = 32
NATIVE_MAX_CAPTURE_POINTS = 16
NATIVE_MAX_PROMPT_BYTES = 65536
