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

"""Recover token links from recorded sampling evidence.

Captures are never rewritten.
"""


def link_token_batches(session, telemetry, turn, runtimes):
  completed = {
      (r['run'], r['forward_id'])
      for r in telemetry.get('forwards', [])
      if r.get('turn') == turn and r.get('status') == 'completed'
  }
  batches = {}
  for batch in session.get('batches', []):
    if batch['turn'] == turn:
      for run, runtime in runtimes.items():
        source = (
            batch.get('forward_ids', {}).get(run)
            if 'forward_ids' in batch
            else batch.get('forward_id')
        )
        if source is not None and batch.get('runtime') == runtime:
          batches.setdefault((run, runtime, source), []).append(batch['batch'])
  for conversation in session.get('conversation', []):
    run = conversation['run']
    if conversation['turn'] != turn or runtimes.get(run) not in (
        'PyTorch',
        'LiteRT-LM',
    ):
      continue
    observed = {}
    for event in telemetry.get('token_records', []):
      if (
          event.get('turn') != turn
          or event.get('run') != run
          or event.get('candidate') != 0
      ):
        continue
      start = event.get('output_index')
      if (
          type(start) is not int
          or (run, event.get('forward_id')) not in completed
      ):
        continue
      for offset, token_id in enumerate(event.get('token_ids', [])):
        observed.setdefault((start + offset, token_id), set()).add(
            event['forward_id']
        )
    for token in conversation.get('tokens', []):
      token.pop('batch', None)
      sources = observed.get((token.get('step'), token.get('id')), set())
      source = next(iter(sources)) if len(sources) == 1 else None
      if source is None or token.get('source_forward_id', source) != source:
        continue
      token['source_forward_id'] = source
      linked = batches.get((run, runtimes[run], source), [])
      if len(linked) == 1:
        token['batch'] = linked[0]
