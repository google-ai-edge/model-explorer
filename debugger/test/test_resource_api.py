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

"""Exercise saved raw resources through the public loopback API."""

import json
import unittest
from urllib.parse import urlencode

from inline_asgi import client
from model_explorer_debugger.resource_views import preview_resource
import numpy as np
import test_pytorch_telemetry as fixtures


class ResourceApiTests(unittest.TestCase):

  def setUp(self):
    fixture = fixtures.PyTorchTelemetryTests()
    fixture.setUp()
    self.addCleanup(fixture.doCleanups)
    job, results = fixture.fixture()
    self.store = fixture.publish(job, results)
    self.ids = fixture.pair(
        self.store.telemetry(), scope='kv', moment='terminal', kind='key'
    )
    self.http = client(self.store)
    self.http.__enter__()
    self.addCleanup(self.close)

  def close(self):
    self.http.__exit__(None, None, None)

  def get(self, path):
    response = self.http.get('/api/' + path)
    self.assertEqual(response.status_code, 200, response.text)
    return response.json()

  def test_actual_saved_resources_round_trip_and_compare(self):
    self.assertEqual(self.store.session['turns'][0]['prefill_tokens'], 2)
    self.assertEqual(
        [
            row['input_token_count']
            for row in self.store.session['conversation']
        ],
        [2, 2],
    )
    telemetry = self.get('telemetry?turn=1&run=ref')
    self.assertEqual({row['run'] for row in telemetry['kv_snapshots']}, {'ref'})
    self.assertEqual(len(telemetry['forwards']), 2)
    query = urlencode(dict(resource_id=self.ids[0], offset=2, limit=3))
    preview = self.get('resource?' + query)
    self.assertEqual(
        (
            preview['shape'],
            preview['offset'],
            preview['values'],
            preview['total'],
        ),
        ([1, 1, 3, 2], 2, [1.0, 1.0, 1.0], 6),
    )
    comparison = self.http.post(
        '/api/resources/compare',
        json=dict(reference=self.ids[0], target=self.ids[1]),
    ).json()
    self.assertEqual(comparison['status'], 'ok')
    self.assertEqual(comparison['metrics']['Max abs error']['value'], 1.0)

  def test_invalid_bounds_unknown_identity_and_origin_are_rejected(self):
    for suffix in ('offset=-1', 'limit=257', 'offset=7', 'offset=NaN'):
      with self.subTest(suffix=suffix):
        self.assertEqual(
            self.http.get(
                '/api/resource?'
                + urlencode(dict(resource_id=self.ids[0]))
                + '&'
                + suffix
            ).status_code,
            400,
        )
    self.assertEqual(
        self.http.get(
            '/api/resource?resource_id=../../workspace.json'
        ).status_code,
        404,
    )
    self.assertEqual(
        self.http.post(
            '/api/resources/compare',
            json={},
            headers={'Origin': 'https://unrelated.example'},
        ).status_code,
        403,
    )

  def test_preview_preserves_nonfinite_values_as_explicit_strings(self):
    class Store:

      def load_resource(self, _identity):
        return np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)

    result = preview_resource(Store(), 'test')
    self.assertEqual(result['values'], ['nan', 'inf', '-inf', 0.0])
    json.dumps(result, allow_nan=False)

  def test_browser_preview_preserves_exact_large_signed_and_unsigned_integers(
      self,
  ):
    for dtype, values in (
        (np.int64, [-9007199254740993, 9007199254740991, 9007199254740993]),
        (np.uint64, [18446744073709551615]),
    ):

      class Store:

        def load_resource(self, _identity):
          return np.array(values, dtype=dtype)

      preview = preview_resource(Store(), 'large-integers')
      self.assertEqual([int(value) for value in preview['values']], values)
      self.assertTrue(
          all(
              isinstance(value, str)
              for value in preview['values']
              if abs(int(value)) > 2**53 - 1
          )
      )


if __name__ == '__main__':
  unittest.main()
