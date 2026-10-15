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

/** Number formatting shared by metric tables, charts, tooltips and the session editor. */
export const EM_DASH = '—';
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

/** `digits` significant digits without trailing zeros: 0.123456 → "0.1235". */
export function formatSignificant(
  value: number | null | undefined,
  digits = 4,
  placeholder = EM_DASH,
): string {
  return finite(value) ? Number(value.toPrecision(digits)).toString() : placeholder;
}
/** Fixed decimals: 1.5 → "1.500" with 3 digits. */
export function formatFixed(
  value: number | null | undefined,
  digits: number,
  placeholder = EM_DASH,
): string {
  return finite(value) ? value.toFixed(digits) : placeholder;
}
export interface MetricFormat {
  /** Significant digits inside [small, large). */
  digits?: number;
  /** Fraction digits of the exponent notation used outside that range. */
  exponentDigits?: number;
  small?: number;
  large?: number;
  placeholder?: string;
}
/** A metric readout: significant digits in the readable range, exponent notation outside it, −0 kept. */
export function formatMetric(
  value: number | null | undefined,
  {
    digits = 4,
    exponentDigits = 2,
    small = 0.001,
    large = 10000,
    placeholder = 'Unavailable',
  }: MetricFormat = {},
): string {
  if (!finite(value)) return placeholder;
  if (Object.is(value, -0)) return '−0';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  return magnitude < small || magnitude >= large
    ? value.toExponential(exponentDigits)
    : Number(value.toPrecision(digits)).toString();
}
/** A ratio shown as a percentage with the metric rules: 0.1234 → "12.34%". */
export function formatPercent(
  ratio: number | null | undefined,
  options: MetricFormat = {},
): string {
  if (!finite(ratio)) return options.placeholder ?? 'Unavailable';
  return formatMetric(ratio * 100, options) + '%';
}
export function formatGiB(bytes: number, maximumFractionDigits = 1): string {
  return `${(bytes / 1024 ** 3).toLocaleString('en-US', {maximumFractionDigits})} GiB`;
}
export function formatCount(count: number): string {
  return count.toLocaleString('en-US');
}

const COSINE_DECIMALS = 6;
const PERCENT_DECIMALS = 2;
const METRIC_SIGNIFICANT_DIGITS = 6;
const PERCENT_METRICS = ['relative_l2', 'selected_probability', 'margin'];

/** Describes the same display rules used by formatAnalysisMetric, not computation accuracy. */
export function analysisMetricDisplayFormat(metric: string) {
  if (metric === 'cosine_similarity')
    return {
      precision: `${COSINE_DECIMALS} decimal places`,
      notation: `Scientific notation (${COSINE_DECIMALS} significant digits) if a nonzero value rounds to 0`,
      rounding: '≈1 / ≈−1 when rounding reaches an endpoint; exact endpoints retain decimals',
    };
  if (PERCENT_METRICS.includes(metric))
    return {
      precision: `${PERCENT_DECIMALS} decimal places`,
      notation: `Scientific notation (${PERCENT_DECIMALS + 1} significant digits) if a nonzero percentage rounds to 0`,
      rounding: 'Applied after conversion to percent or percentage points',
    };
  return {
    precision: `${METRIC_SIGNIFICANT_DIGITS} significant digits`,
    notation: 'Scientific notation when 0 < |value| < 0.0001 or |value| ≥ 1,000,000',
    rounding: 'Trailing zeros omitted',
  };
}

/** Accepted analysis-metric-spec.md display contract; input values use canonical raw units. */
export function formatAnalysisMetric(metric: string, value: number | null | undefined): string {
  if (!finite(value)) return EM_DASH;
  const normalized = value === 0 ? 0 : value;
  if (metric === 'cosine_similarity') {
    const fixed = normalized.toFixed(COSINE_DECIMALS);
    if (Math.abs(normalized) !== 1 && Math.abs(Number(fixed)) === 1)
      return normalized < 0 ? '≈−1' : '≈1';
    return normalized !== 0 && Number(fixed) === 0
      ? normalized.toExponential(COSINE_DECIMALS - 1)
      : fixed;
  }
  if (PERCENT_METRICS.includes(metric)) {
    const display = normalized * 100;
    const fixed = display.toFixed(PERCENT_DECIMALS);
    const formatted =
      display !== 0 && Number(fixed) === 0 ? display.toExponential(PERCENT_DECIMALS) : fixed;
    return formatted + (metric === 'margin' ? ' pp' : '%');
  }
  return formatMetric(normalized, {
    digits: METRIC_SIGNIFICANT_DIGITS,
    exponentDigits: METRIC_SIGNIFICANT_DIGITS - 1,
    small: 0.0001,
    large: 1000000,
    placeholder: EM_DASH,
  });
}
