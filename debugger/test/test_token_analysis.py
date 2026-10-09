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

"""Analytic synthetic vectors verify token analysis contracts.

They cover the full-vocabulary and token-source contracts.
"""

import json
from types import SimpleNamespace
import unittest
from model_explorer_debugger.token_analysis import (
    analyze_tokens,
    vector_metrics,
)
import numpy as np


class TokenAnalysisTests(unittest.TestCase):

  def store(self, ref=(0.0, 0.0), target=(0.0, np.log(3)), vocab='v1'):
    values = {'ref': np.array([[ref]]), 'target': np.array([[target]])}
    forwards = [
        dict(
            turn=1,
            run=run,
            forward_id=9,
            status='completed',
            model_sha256='same',
            input_proof={
                'tokenizer': {'vocab_sha256': 'v1' if run == 'ref' else vocab}
            },
            outputs=[{'output_path': ['output', 'logits'], 'resource_id': run}],
        )
        for run in values
    ]
    return SimpleNamespace(
        session={
            'conversation': [
                {
                    'turn': 1,
                    'run': run,
                    'tokens': [{
                        'step': 0,
                        'id': 1,
                        'text': 'B',
                        'source_forward_id': 9,
                    }],
                }
                for run in values
            ]
        },
        _telemetry={
            'forwards': forwards,
            'token_records': [
                dict(
                    turn=1,
                    run=run,
                    forward_id=9,
                    candidate=0,
                    output_index=0,
                    token_ids=[1],
                )
                for run in values
            ],
        },
        load_resource=lambda identity: values[identity],
    )

  def analyze(self, store, ref=0, target=0):
    return analyze_tokens(
        store, {'turn': 1, 'pairs': [{'ref': ref, 'target': target}]}
    )['pairs'][0]

  def test_full_distribution_and_units(self):
    result = self.analyze(self.store())
    self.assertAlmostEqual(
        result['metrics']['kl']['value'], 0.5 * np.log(4 / 3)
    )
    p, q = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    m = (p + q) / 2
    self.assertAlmostEqual(
        result['metrics']['js']['value'],
        0.5 * sum(p * np.log2(p / m)) + 0.5 * sum(q * np.log2(q / m)),
    )
    d = result['distribution']
    self.assertTrue(d['compatible'])
    self.assertEqual(d['ref']['entropy'], 1)
    self.assertEqual(d['rows'][0]['id'], 1)
    self.assertAlmostEqual(d['rows'][0]['delta'], 25)
    self.assertEqual(d['target']['selected_rank'], 1)
    self.assertIsNone(result['metrics']['relative_l2']['value'])
    self.assertIn('not captured', result['metrics']['relative_l2']['reason'])
    json.dumps(result, allow_nan=False)

  def test_vocab_mismatch_retains_independent_entropy(self):
    r = self.analyze(self.store(vocab='other'))
    self.assertIsNone(r['metrics']['kl']['value'])
    self.assertFalse(r['distribution']['compatible'])
    self.assertEqual(r['distribution']['rows'], [])
    self.assertEqual(r['distribution']['ref']['entropy'], 1)

  def test_one_sided_does_not_fabricate_target(self):
    r = self.analyze(self.store(), target=None)
    self.assertIsNone(r['metrics']['js']['value'])
    self.assertTrue(r['distribution']['rows'])
    self.assertTrue(
        all(
            row['target'] is None and row['delta'] is None
            for row in r['distribution']['rows']
        )
    )

  def test_top_k_boundary_is_not_a_full_distribution(self):
    s = self.store()
    for forward in s._telemetry['forwards']:
      forward['outputs'][0]['output_path'] = ['output', 'top_k_logits']
    result = self.analyze(s)
    self.assertIsNone(result['metrics']['kl']['value'])
    self.assertIsNone(result['metrics']['js']['value'])
    self.assertFalse(result['distribution']['compatible'])
    self.assertEqual(result['distribution']['rows'], [])
    self.assertIn(
        'Full-vocabulary logits not captured',
        result['distribution']['ref']['reason'],
    )

  def test_source_conflicts_and_chunks_are_unavailable(self):
    for mutation in ('conflict', 'chunk', 'duplicate'):
      s = self.store()
      if mutation == 'conflict':
        s.session['conversation'][0]['tokens'][0]['source_forward_id'] = 0
      elif mutation == 'chunk':
        s._telemetry['token_records'][0]['token_ids'].append(0)
      else:
        s._telemetry['token_records'].append(
            s._telemetry['token_records'][0].copy()
        )
      self.assertIsNone(self.analyze(s)['metrics']['kl']['value'])

  def test_masked_logits_and_invalid_numbers(self):
    r = self.analyze(self.store(ref=(0, -np.inf), target=(-np.inf, 0)))
    self.assertIsNone(r['metrics']['kl']['value'])
    self.assertEqual(r['metrics']['js']['value'], 1)
    json.dumps(r, allow_nan=False)
    for values in ((np.nan, 0), (np.inf, 0), (-np.inf, -np.inf)):
      self.assertIsNone(
          self.analyze(self.store(ref=values))['metrics']['js']['value']
      )

  def test_activation_metrics_are_percent_and_zero_is_unknown(self):
    r = vector_metrics(np.array([3.0, 4.0]), np.array([6.0, 8.0]))
    self.assertEqual(r['relative_l2']['value'], 100)
    self.assertEqual(r['norm_ratio']['value'], 2)
    self.assertEqual(r['max_abs_error']['value'], 4)
    self.assertAlmostEqual(r['cosine_distance']['value'], 0)
    self.assertIsNone(
        vector_metrics(np.zeros(2), np.ones(2))['relative_l2']['value']
    )
    self.assertIsNone(
        vector_metrics(np.ones(2), np.ones(3))['norm_ratio']['value']
    )

  def test_only_explicit_activation_boundary_is_used(self):
    s = self.store()
    load = s.load_resource
    s.load_resource = (
        lambda identity: np.array([[[3.0, 4.0]]])
        if identity.endswith('-hidden')
        else load(identity)
    )
    for f in s._telemetry['forwards']:
      f['outputs'].append({
          'output_path': ['output', 'last_hidden_state'],
          'resource_id': f['run'] + '-hidden',
      })
    self.assertEqual(self.analyze(s)['metrics']['relative_l2']['value'], 0)

  def test_request_bound_and_original_steps(self):
    for payload in (
        {'turn': True, 'pairs': []},
        {'turn': 1, 'pairs': [{'ref': True, 'target': 0}]},
        {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}] * 129},
    ):
      with self.assertRaises(ValueError):
        analyze_tokens(self.store(), payload)
    self.assertIsNone(
        self.analyze(self.store(), ref=9)['metrics']['kl']['value']
    )


if __name__ == '__main__':
  unittest.main()
