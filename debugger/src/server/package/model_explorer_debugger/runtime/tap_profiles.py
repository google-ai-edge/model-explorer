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

"""Reviewed tap profiles.

Operator coordinates are bound to exact section bytes.
"""

import json
from pathlib import Path

PROFILE_ID = 'gemma4-e2b-layer0-rmsnorm-v1'
CUSTOM_PROFILE = 'custom-outputs-v1'
PROFILE_PATH = Path(__file__).with_name('gemma4_e2b_taps.json')


def profile_config(payload):
  profile = payload.get('tap_profile') or ''
  if profile not in ('', PROFILE_ID, CUSTOM_PROFILE):
    raise ValueError('Unknown tensor capture profile')
  points = payload.get('tap_points', {}) if profile == CUSTOM_PROFILE else {}
  if not isinstance(points, dict) or len(points) > 2:
    raise ValueError('Choose capture points for at most two model files')
  for artifact, ids in points.items():
    if (
        not isinstance(artifact, str)
        or not isinstance(ids, list)
        or not 1 <= len(ids) <= 16
        or any(not isinstance(i, str) for i in ids)
        or len(set(ids)) != len(ids)
    ):
      raise ValueError('Select 1–16 unique outputs per model')
  return {'tap_profile': profile, 'tap_prepared': False, 'tap_points': points}


def expected_manifest():
  return json.loads(PROFILE_PATH.read_text())


def available_profiles(root):
  if (
      not root
      or not (root / 'src/debugger_tap/tap.py').is_file()
      or not (root / '.venv/bin/litert-lm-builder').is_file()
  ):
    return []
  return [{
      'id': PROFILE_ID,
      'name': 'Gemma 4 E2B · Layer 0 RMSNorm',
      'description': (
          'Capture the pre-attention RMSNorm output. Only the reviewed model'
          ' build is supported.'
      ),
  }]
