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

"""Synthetic native manifests shaped like real ones.

They are shaped so that schema validation accepts them.
"""


def synthetic_tap(signature='prefill', output=0, **extra):
  return {
      'signature': signature,
      'subgraph': 0,
      'op': output,
      'output': output,
      'tensor': output,
      'tensor_name': f'model/tensor{output}',
      'output_name': f'out{output}',
      'shape': [1, 2],
      'tensor_type': 0,
      **extra,
  }


def synthetic_manifest(taps=None):
  return {
      'format_version': 1,
      'source_sha256': '1' * 64,
      'tapped_sha256': '2' * 64,
      'taps': taps if taps is not None else [synthetic_tap()],
  }
