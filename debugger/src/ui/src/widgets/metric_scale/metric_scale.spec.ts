/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

import {METRIC_CATALOG_FIXTURE} from '../../server/testing/metric_catalog_fixture';
import {metricHeat, metricScaleLabels, steppedScale, tokenHeat} from './metric_scale';

describe('metric scale', () => {
  const metric = (id: string) => METRIC_CATALOG_FIXTURE.metrics.find((m) => m.id === id);

  it('cosine similarity is colored over its bounds, from 1 to −1', () => {
    const cosine = metric('cosine_similarity');
    for (const mode of ['auto', 'thresholds'] as const) {
      expect(metricHeat(cosine, 1, mode, 5)).toBe(0);
      expect(metricHeat(cosine, 0, mode, 5)).toBe(0.5);
      expect(metricHeat(cosine, -1, mode, 5)).toBe(1);
      expect(steppedScale(cosine, mode)).toBe(false);
    }
    expect(metricScaleLabels(cosine, 'thresholds', null)).toStrictEqual([
      '1',
      '0.6',
      '0.2',
      '−0.2',
      '−0.6',
      '−1',
    ]);
  });

  it('relative L2 steps at its percent thresholds or scales by the largest value', () => {
    const relative = metric('relative_l2');
    expect(metricHeat(relative, 0.005, 'thresholds', 1)).toBe(0);
    expect(metricHeat(relative, 0.5, 'thresholds', 1)).toBe(0.5);
    expect(metricHeat(relative, 50, 'thresholds', 1)).toBe(1);
    expect(metricHeat(relative, 2, 'auto', 4)).toBe(0.5);
    expect(metricHeat(relative, null, 'auto', 4)).toBe(null);
    expect(metricScaleLabels(relative, 'thresholds', null)).toStrictEqual([
      '0',
      '0.01',
      '0.1',
      '1',
      '10',
      '+',
    ]);
    expect(metricScaleLabels(relative, 'auto', 5)).toStrictEqual(['0', '1', '2', '3', '4', '5']);
    expect(metricScaleLabels(relative, 'auto', null)).toStrictEqual([]);
    expect(metricScaleLabels(metric('max_abs'), 'thresholds', null)).toStrictEqual([
      '0',
      '1e−5',
      '1e−4',
      '1e−3',
      '0.01',
      '+',
    ]);
  });

  it('a token heat is the distance from the ideal value relative to heat_full', () => {
    expect(tokenHeat(metric('kl'), 0.005)).toBe(0.5);
    expect(tokenHeat(metric('kl'), 1)).toBe(1);
    expect(Math.abs(tokenHeat(metric('cosine_similarity'), 0.99)! - 0.5) < 1e-9).toBeTruthy();
    // A norm ratio of 1 is equal norms: no heat, whichever side is larger.
    expect(tokenHeat(metric('norm_ratio'), 1)).toBe(0);
    expect(Math.abs(tokenHeat(metric('norm_ratio'), 0.9)! - 0.5) < 1e-9).toBeTruthy();
    expect(tokenHeat(metric('entropy'), 1)).toBe(null);
  });
});
