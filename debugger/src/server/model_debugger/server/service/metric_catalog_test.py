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

"""Tests that the catalog describes what the analyses compute."""

import json

from absl.testing import absltest
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import token_analysis
from model_debugger.server.service import metric_catalog
from model_debugger.server.service import session_view
from model_debugger.server.testing import test_main

_KEYS = {
    'id',
    'label',
    'unit',
    'unit_label',
    'measures',
    'formula',
    'range',
    'reading',
    'larger_is_better',
    'ideal',
    'bounds',
    'format',
    'color',
    'inputs',
    'analyses',
}


class MetricCatalogTest(absltest.TestCase):
  """Tests that the catalog describes what the analyses compute."""

  def setUp(self):
    """The catalog, its metrics and its analyses by id."""
    super().setUp()
    self.catalog = metric_catalog.catalog()
    self.metrics = {m['id']: m for m in self.catalog['metrics']}
    self.analyses = {a['id']: a for a in self.catalog['analyses']}

  def test_every_computed_metric_is_described(self):
    """Each key an analysis answers with has a description."""
    computed = (
        set(metrics.VECTOR_KEYS)
        | set(kv_analysis.CHART_METRICS)
        | set(kv_analysis.HEAD_METRICS)
        | {'kl', 'js'}
        | set(metrics.ACTIVATION_KEYS)
        | {'entropy', 'selected_probability', 'margin'}
    )
    self.assertEqual(computed - set(self.metrics), set())

  def test_each_description_is_complete(self):
    """Every description has every field."""
    for key, metric in self.metrics.items():
      self.assertEqual(set(metric), _KEYS, key)
      self.assertTrue(metric['label'], key)
      self.assertIn(metric['format']['style'], ('fixed', 'significant'))
      thresholds = metric['color']['thresholds']
      if thresholds is not None:
        self.assertLen(thresholds, 4, key)
        self.assertEqual(thresholds, sorted(thresholds), key)
    # The catalog is plain JSON.
    json.dumps(self.catalog, allow_nan=False)

  def test_relative_l2_is_a_percent_and_cosine_a_similarity(self):
    """The one naming: relative L2 in percent, cosine as similarity."""
    relative = self.metrics['relative_l2']
    self.assertEqual(relative['unit'], 'percent')
    self.assertNotIn('scale', relative['format'])
    self.assertEqual(relative['format']['suffix'], '%')
    self.assertEqual(self.metrics['cosine_similarity']['bounds'], [-1, 1])
    self.assertNotIn('max_abs_error', self.metrics)
    for analysis in self.analyses.values():
      self.assertNotIn('cosine_distance', analysis['metrics'])

  def test_a_heat_legend_reads_where_the_heat_is_full(self):
    """heat_full, or the ideal ± the distance when both sides reach it."""
    labels = {
        key: metric['color']['heat_labels']
        for key, metric in self.metrics.items()
    }
    # Full at 0.8 and at 1.2: both sides of the ideal 1.
    self.assertEqual(labels['norm_ratio'], {'none': '1', 'full': '1 ± 20%'})
    # Bounded at the ideal: one side.
    self.assertEqual(labels['cosine_similarity'], {'none': '1', 'full': '0.98'})
    self.assertEqual(labels['kl'], {'none': '0', 'full': '0.01 nats'})
    self.assertEqual(labels['relative_l2'], {'none': '0', 'full': '1 %'})
    self.assertEqual(labels['max_abs'], {'none': '0', 'full': '0.1'})
    # No ideal, or no heat: no legend.
    self.assertIsNone(labels['entropy'])

  def test_analyses_offer_described_metrics(self):
    """Each analysis offers described metrics; its default is one."""
    for analysis in self.analyses.values():
      self.assertTrue(analysis['metrics'], analysis['id'])
      self.assertEqual(set(analysis['metrics']) - set(self.metrics), set())
      self.assertIn(analysis['default_metric'], analysis['metrics'])
      self.assertEqual(set(analysis['groups']) - set(self.metrics), set())
      for key in analysis['metrics']:
        self.assertIn(analysis['id'], self.metrics[key]['analyses'], key)
    self.assertEqual(
        self.analyses['kv']['default_head'], kv_analysis.DEFAULT_HEAD
    )
    self.assertEqual(kv_analysis.DEFAULT_HEAD, 'min')
    self.assertLessEqual(
        set(self.analyses['kv']['metrics']), set(kv_analysis.CHART_METRICS)
    )
    self.assertLessEqual(
        set(self.analyses['kv_head']['metrics']),
        set(kv_analysis.HEAD_METRICS),
    )
    self.assertLessEqual(
        set(self.analyses['token_diff']['metrics']),
        {'kl', 'js'} | set(metrics.ACTIVATION_KEYS),
    )

  def test_every_reason_code_is_listed_once(self):
    """Each module's codes are listed, each once, each with words."""
    codes = [reason['code'] for reason in self.catalog['reasons']]
    self.assertLen(set(codes), len(codes))
    for table in (
        metrics.REASONS,
        token_analysis.REASONS,
        kv_analysis.REASONS,
        session_view.REASONS,
    ):
      self.assertLessEqual(set(table), set(codes))
    for reason in self.catalog['reasons']:
      self.assertEqual(set(reason), {'code', 'text'})
      self.assertTrue(reason['text'], reason['code'])
    # What kv-head gives as a code alone: each metric's status.
    self.assertLessEqual(
        {
            metrics.NON_FINITE,
            metrics.ZERO_NORM,
            metrics.OVERFLOW,
            'unavailable',
            'position_unavailable',
        },
        set(codes),
    )

  def test_a_stage_offers_the_analyses_of_the_catalog(self):
    """The analyses a stage offers are analyses the catalog describes."""
    self.assertLessEqual(set(session_view.ANALYSES), set(self.analyses))

  def test_sampling_says_what_the_server_does(self):
    """Sampling is the softmax of the logits at the sampler temperature."""
    inputs = self.analyses['sampling']['inputs']
    self.assertIn('softmax', inputs)
    self.assertIn('temperature', inputs)
    self.assertNotIn('rebuilt', inputs)


if __name__ == '__main__':
  test_main.main()
