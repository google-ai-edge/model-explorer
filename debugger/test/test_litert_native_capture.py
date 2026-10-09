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

"""Synthetic native identity fixtures; no model inference is performed here."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from model_explorer_debugger.capture_importer import trace_tokens
from model_explorer_debugger.capture_telemetry import (
    empty_telemetry,
    pair_forward_evidence,
)
from model_explorer_debugger.node_details import file_digest
from model_explorer_debugger.runtime.litert_trace import (
    read_events,
    verify_trace_index,
)
from model_explorer_debugger.token_analysis import analyze_tokens
from model_explorer_debugger.token_sources import link_token_batches
import test_token_analysis as token_fixtures


class NativeCaptureTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)

  def test_multi_id_callbacks_and_missing_text_preserve_every_id(self):
    path = self.root / 'tokens.jsonl'
    rows = [
        dict(token_ids=[[1, 2]], texts=['combined']),
        dict(token_ids=[[3]], texts=[]),
        dict(token_ids=[[4]], texts=['tail']),
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    result = trace_tokens(path)
    self.assertEqual(
        [(row['id'], row['step']) for row in result],
        [(1, 0), (2, 1), (3, 2), (4, 3)],
    )
    self.assertEqual(
        [row['text'] for row in result], [None, None, None, 'tail']
    )

  def test_candidate_isolation_and_invalid_token_ids(self):
    path = self.root / 'tokens.jsonl'
    path.write_text(json.dumps(dict(token_ids=[[1], [2]], texts=['a', 'b'])))
    self.assertEqual(trace_tokens(path), [])
    for invalid in (-1, True, '3'):
      path.write_text(json.dumps(dict(token_ids=[[invalid]], texts=['bad'])))
      with self.assertRaises(ValueError):
        trace_tokens(path)

  def trace_index(self):
    before = dict(
        processed_token_count=2,
        runtime_step=3,
        processed_token_ids=[[1, 2]],
        pending_token_ids=[3],
    )
    after = dict(
        processed_token_count=3,
        runtime_step=4,
        processed_token_ids=[[1, 2, 3]],
        pending_token_ids=[],
    )
    admission = dict(
        event='admission',
        admission_id=1,
        token_ids=[[1, 2, 3]],
        state_before=before,
        supported=True,
    )
    forward = dict(
        event='graph_post',
        invocation_id=8,
        signature='decode',
        phase='decode',
        runtime_step=3,
        consumed_token_ids=[[3]],
        input_positions=[2],
        input_rows=[0],
        state_before=before,
        state_after=after,
        status='completed',
        capture_complete=True,
        supported=True,
    )
    sample = dict(
        event='sample',
        sample_id=1,
        source_invocation_id=8,
        token_ids=[[4]],
        supported=True,
        state_after={**after, 'pending_token_ids': [4]},
    )
    events = [
        admission,
        forward,
        sample,
        dict(event='terminal', task_state='done'),
    ]
    for i, row in enumerate(events):
      row.update(format_version=1, session_id=0, sequence=i + 1)
    path = self.root / 'runtime_trace.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in events) + '\n')
    return dict(
        native_trace=dict(path=path.name, sha256=file_digest(path)),
        forwards=[{**forward, 'forward_id': 8}],
        admitted_input_ids=[1, 2, 3],
        generation={'admissions': [admission]},
        tokens=[dict(id=4, step=0, text=None, source_forward_id=8)],
        token_records=[
            dict(
                forward_id=8,
                output_index=0,
                candidate=0,
                token_ids=[4],
                sample_id=1,
            )
        ],
    )

  def test_native_index_cannot_rewrite_admitted_or_consumed_ids(self):
    original = self.trace_index()
    verify_trace_index(original, self.root)
    for field in ('admission', 'forward', 'token', 'sample_source'):
      with self.subTest(field=field):
        index = deepcopy(original)
        if field == 'admission':
          index['admitted_input_ids'][0] = 99
        elif field == 'forward':
          index['forwards'][0]['state_before']['processed_token_ids'][0][0] = 99
        elif field == 'token':
          index['tokens'][0]['id'] = 99
        else:
          index['token_records'][0]['forward_id'] = 99
        with self.assertRaises(ValueError):
          verify_trace_index(index, self.root)

  def test_native_trace_sequence_and_session_are_checked(self):
    self.trace_index()
    path = self.root / 'runtime_trace.jsonl'
    original = [json.loads(row) for row in path.read_text().splitlines()]
    for field, value in [('sequence', 1), ('session_id', 1)]:
      rows = deepcopy(original)
      rows[1][field] = value
      path.write_text('\n'.join(json.dumps(row) for row in rows))
      with self.assertRaises(ValueError):
        read_events(path)

  def test_different_invocation_ids_pair_by_unique_verified_context(self):
    data = empty_telemetry()
    current = {'ref': {}, 'target': {}}
    for role, first in [('ref', 8), ('target', 4)]:
      for i, proof in enumerate((
          'context-before-the',
          'context-before-divergence',
          'different-' + role,
      )):
        forward = dict(
            forward_id=first + i,
            phase='decode',
            runtime_turn=1,
            input_identity=proof,
            comparison_eligible=True,
            status='completed',
            sample=None,
        )
        current[role][first + i] = forward
        data['resources'].append(
            dict(
                run=role,
                turn=1,
                scope='boundary',
                forward_id=first + i,
                sample=None,
            )
        )
    pair_forward_evidence(
        data,
        current,
        {'ref': 'LiteRT-LM', 'target': 'LiteRT-LM'},
        {role: {'model_sha256': 'same'} for role in current},
        1,
    )
    self.assertEqual(
        current['ref'][8]['sample'], current['target'][4]['sample']
    )
    self.assertIsNotNone(current['ref'][9]['sample'])
    self.assertEqual(
        current['ref'][9]['sample'], current['target'][5]['sample']
    )
    self.assertIsNone(current['ref'][10]['sample'])
    self.assertIsNone(current['target'][6]['sample'])

  def test_native_batches_keep_per_run_forward_identity(self):
    session = dict(
        batches=[
            dict(
                turn=1,
                batch=7,
                runtime='LiteRT-LM',
                forward_ids={'ref': 9, 'target': 5},
            )
        ],
        conversation=[
            dict(
                turn=1,
                run=role,
                tokens=[dict(id=token, step=0, source_forward_id=forward)],
            )
            for role, token, forward in [('ref', 10, 9), ('target', 11, 5)]
        ],
    )
    telemetry = dict(
        forwards=[
            dict(turn=1, run=role, forward_id=forward, status='completed')
            for role, forward in [('ref', 9), ('target', 5)]
        ],
        token_records=[
            dict(
                turn=1,
                run=role,
                forward_id=forward,
                candidate=0,
                output_index=0,
                token_ids=[token],
            )
            for role, token, forward in [('ref', 10, 9), ('target', 11, 5)]
        ],
    )
    link_token_batches(
        session, telemetry, 1, {'ref': 'LiteRT-LM', 'target': 'LiteRT-LM'}
    )
    self.assertEqual(
        [row['tokens'][0]['batch'] for row in session['conversation']], [7, 7]
    )

  def test_native_token_metrics_stop_pairing_after_history_divergence(self):
    store = token_fixtures.TokenAnalysisTests().store()
    for forward in store._telemetry['forwards']:
      forward.update(runtime='LiteRT-LM', sample=None)
    result = analyze_tokens(store, dict(turn=1, pairs=[dict(ref=0, target=0)]))[
        'pairs'
    ][0]
    self.assertIsNone(result['metrics']['kl']['value'])
    self.assertFalse(result['distribution']['compatible'])
    self.assertIsNotNone(result['distribution']['ref']['entropy'])
    self.assertIn('logical token contexts', result['distribution']['reason'])


if __name__ == '__main__':
  unittest.main()
