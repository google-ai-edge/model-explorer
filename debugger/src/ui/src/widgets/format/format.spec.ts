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
import {
  formatCount,
  formatGiB,
  formatMetric,
  formatMetricValue,
  formatPercent,
  formatScaleLabel,
  formatSignificant,
  formatTensorValue,
  metricDisplayRules,
} from './format';

describe('format', () => {
  const metric = (id: string) => METRIC_CATALOG_FIXTURE.metrics.find((m) => m.id === id);

  const format = (id: string, value: number | null | undefined) =>
    formatMetricValue(metric(id), value);

  it('significant digits, with placeholders for missing values', () => {
    expect(formatSignificant(0.123456)).toBe('0.1235');
    expect(formatSignificant(1.5, 5)).toBe('1.5');
    expect(formatSignificant(null)).toBe('—');
    expect(formatSignificant(NaN, 4, 'Unavailable')).toBe('Unavailable');
  });

  it('metric readouts switch to exponent notation outside the readable range and keep −0', () => {
    expect(formatMetric(0.123456)).toBe('0.1235');
    expect(formatMetric(0.0001234)).toBe('1.23e-4');
    expect(formatMetric(12345.6)).toBe('1.23e+4');
    expect(formatMetric(0)).toBe('0');
    expect(formatMetric(-0)).toBe('−0');
    expect(formatMetric(Infinity)).toBe('Unavailable');
    expect(formatMetric(null, {placeholder: '—'})).toBe('—');
    expect(formatMetric(0.00005, {digits: 4, exponentDigits: 3, small: 0.0001, large: 1000})).toBe(
      '5.000e-5',
    );
    expect(formatMetric(999.99, {small: 0.0001, large: 1000})).toBe('1000');
  });

  it('percentages scale ratios and never append % to a placeholder', () => {
    expect(formatPercent(0.1234)).toBe('12.34%');
    expect(formatPercent(0.0000012)).toBe('1.20e-4%');
    expect(formatPercent(null)).toBe('Unavailable');
    expect(formatPercent(undefined, {placeholder: '—'})).toBe('—');
  });

  it('bytes and counts use en-US grouping', () => {
    expect(formatGiB(16 * 1024 ** 3)).toBe('16 GiB');
    expect(formatGiB(1.5 * 1024 ** 3, 2)).toBe('1.5 GiB');
    expect(formatGiB(20 * 1024 ** 3 + 123456789, 2)).toBe('20.11 GiB');
    expect(formatCount(1234567)).toBe('1,234,567');
  });

  it('metric values follow the format the server describes', () => {
    expect(format('cosine_similarity', 1)).toBe('1.000000');
    expect(format('cosine_similarity', 0.99999999)).toBe('≈1');
    expect(format('cosine_similarity', -0.99999999)).toBe('≈−1');
    expect(format('cosine_similarity', 1e-8)).toBe('1.00000e-8');
    expect(format('cosine_similarity', -0)).toBe('0.000000');
    // Relative L2 is a percent already; probabilities are scaled for display.
    expect(format('relative_l2', 12.3456)).toBe('12.35%');
    expect(format('relative_l2', 1.23e-5)).toBe('1.23e-5%');
    expect(format('selected_probability', 0.5)).toBe('50.00%');
    expect(format('selected_probability', 0.99999999)).toBe('100.00%');
    expect(format('margin', 0.125)).toBe('12.50 pp');
    expect(format('max_abs', 1.2345678e-7)).toBe('1.23457e-7');
    expect(format('max_abs', 1234567)).toBe('1.23457e+6');
    expect(format('entropy', 2.41038)).toBe('2.41038');
    expect(format('max_abs', -0)).toBe('0');
    expect(format('max_abs', null)).toBe('—');
    expect(formatTensorValue(0.1234567)).toBe('0.123457');
  });

  it('display rules describe the format in words', () => {
    expect(metricDisplayRules(metric('cosine_similarity')).precision).toBe('6 decimal places');
    expect(metricDisplayRules(metric('relative_l2')).precision).toBe('2 decimal places');
    expect(metricDisplayRules(metric('margin')).rounding).toMatch(/multiplying by 100/);
    expect(metricDisplayRules(metric('kl')).precision).toBe('6 significant digits');
    expect(metricDisplayRules(undefined).precision).toBe('6 significant digits');
  });

  it('scale labels are short and use the minus sign', () => {
    expect(formatScaleLabel(0.00001)).toBe('1e−5');
    expect(formatScaleLabel(0.001)).toBe('1e−3');
    expect(formatScaleLabel(0.01)).toBe('0.01');
    expect(formatScaleLabel(-0.6)).toBe('−0.6');
    expect(formatScaleLabel(10)).toBe('10');
  });
});
