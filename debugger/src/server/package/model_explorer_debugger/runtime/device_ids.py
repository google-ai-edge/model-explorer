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

"""Device identifier resolution shared by routing, discovery and launch."""

from . import macos_device

LOCAL_DEVICE = 'local'


def resolve_device(root, value):
  if value not in (None, '', 'Server host', LOCAL_DEVICE):
    return value
  # Looked up through the module so one patch point
  # (macos_device.configurations) serves every caller.
  matches = [
      key
      for key, config in macos_device.configurations(root).items()
      if config['host'] == '127.0.0.1'
  ]
  if len(matches) != 1:
    raise ValueError(
        'Open the local Runner App and select This computer. No unique local'
        ' Runner is configured.'
    )
  return 'macos:' + matches[0]
