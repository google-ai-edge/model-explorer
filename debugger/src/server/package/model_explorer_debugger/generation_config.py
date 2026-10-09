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

"""Validated per-chat generation requests.

These are never interpreted as captured values.
"""

import math

MAX_OUTPUT_TOKENS = 256


def generation_config(value):
  if not isinstance(value, dict) or set(value) != {'ref', 'target'}:
    raise ValueError('Reference and Target generation settings are required')
  result = {}
  ranges = {
      'temperature': (0, 100, False),
      'topK': (1, 2147483647, True),
      'topP': (0, 1, False),
      'seed': (0, 4294967295, True),
      'thinkingBudget': (-1, 2147483647, True),
      'maxOutputTokens': (1, MAX_OUTPUT_TOKENS, True),
  }
  for run, config in value.items():
    if not isinstance(config, dict):
      raise ValueError('Invalid generation settings')
    unknown = set(config) - set(ranges) - {'thinking', 'systemPrompt'}
    if unknown:
      raise ValueError(
          'Unsupported generation settings: ' + ', '.join(sorted(unknown))
      )
    clean = {}
    for field, (low, high, integer) in ranges.items():
      number = config.get(field)
      if number is None:
        continue
      if (
          isinstance(number, bool)
          or not isinstance(number, (int, float))
          or not math.isfinite(number)
          or not low <= number <= high
          or (integer and int(number) != number)
      ):
        raise ValueError(f'{field} must be between {low} and {high}')
      clean[field] = int(number) if integer else number
    thinking = config.get('thinking')
    if thinking not in (None, 'default', 'on', 'off'):
      raise ValueError('Invalid Thinking setting')
    if thinking and thinking != 'default':
      clean['thinking'] = thinking
    prompt = config.get('systemPrompt', '')
    if not isinstance(prompt, str) or len(prompt) > 16000:
      raise ValueError('System prompt exceeds 16000 characters')
    clean['systemPrompt'] = prompt
    result[run] = clean
  return result
