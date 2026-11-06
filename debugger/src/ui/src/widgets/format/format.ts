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

import type {
  MetricDescription,
  MetricFormat as DescribedFormat,
} from '../../server/contracts/metrics';

/** Number formatting shared by metric tables, charts, tooltips and the session editor. */
export const EM_DASH = '—';
function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** `digits` significant digits without trailing zeros: 0.123456 → "0.1235". */
export function formatSignificant(
  value: number | null | undefined,
  digits = 4,
  placeholder = EM_DASH,
): string {
  return finite(value) ? Number(value.toPrecision(digits)).toString() : placeholder;
}
/** Formatting options for numeric metric readouts. */
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
  if (!finite(value)) {
    return placeholder;
  }
  if (Object.is(value, -0)) {
    return '−0';
  }
  if (value === 0) {
    return '0';
  }
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
  if (!finite(ratio)) {
    return options.placeholder ?? 'Unavailable';
  }
  return formatMetric(ratio * 100, options) + '%';
}
/** Formats a byte count in gibibytes (GiB). */
export function formatGiB(bytes: number, maximumFractionDigits = 1): string {
  return `${(bytes / 1024 ** 3).toLocaleString('en-US', {maximumFractionDigits})} GiB`;
}
/** Formats a count with US English locale thousand separators. */
export function formatCount(count: number): string {
  return count.toLocaleString('en-US');
}

const DEFAULT_FORMAT: DescribedFormat = {style: 'significant', digits: 6};
const SMALL = 0.0001;
const LARGE = 1000000;

/** The digits of the scientific notation a fixed format falls back to. */
function exponentDigits(decimals: number): number {
  return Math.max(3, decimals);
}

/** The rules formatMetricValue writes a metric's values by, in words; not its accuracy. */
export function metricDisplayRules(description: MetricDescription | undefined): {
  precision: string;
  notation: string;
  rounding: string;
} {
  const format = description?.format ?? DEFAULT_FORMAT;
  if (format.style === 'fixed') {
    const decimals = format.decimals ?? 2;
    return {
      precision: `${decimals} decimal places`,
      notation: `Scientific notation (${exponentDigits(decimals)} significant digits) if a nonzero value rounds to 0`,
      rounding: format.scale
        ? `Applied after multiplying by ${format.scale}`
        : '≈ and the bound when rounding reaches a bound; exact bounds retain decimals',
    };
  }
  const digits = format.digits ?? 6;
  return {
    precision: `${digits} significant digits`,
    notation: `Scientific notation when 0 < |value| < ${SMALL} or |value| ≥ ${LARGE.toLocaleString('en-US')}`,
    rounding: 'Trailing zeros omitted',
  };
}

/**
 * A metric's value as its description says to write it (format, scale, suffix); a value with
 * no description is written with six significant digits.
 */
export function formatMetricValue(
  description: MetricDescription | undefined,
  value: number | null | undefined,
): string {
  if (!finite(value)) {
    return EM_DASH;
  }
  const format = description?.format ?? DEFAULT_FORMAT;
  const scaled = (value === 0 ? 0 : value) * (format.scale ?? 1);
  const suffix = format.suffix ?? '';
  if (format.style === 'fixed') {
    const decimals = format.decimals ?? 2;
    const fixed = scaled.toFixed(decimals);
    if (!format.scale) {
      for (const bound of description?.bounds ?? []) {
        if (bound !== null && bound !== 0 && scaled !== bound && Number(fixed) === bound) {
          return '≈' + String(bound).replaceAll('-', '−') + suffix;
        }
      }
    }
    return (
      (scaled !== 0 && Number(fixed) === 0
        ? scaled.toExponential(exponentDigits(decimals) - 1)
        : fixed) + suffix
    );
  }
  const digits = format.digits ?? 6;
  return (
    formatMetric(scaled, {
      digits,
      exponentDigits: digits - 1,
      small: SMALL,
      large: LARGE,
      placeholder: EM_DASH,
    }) + suffix
  );
}

/** A captured tensor value, not a metric: six significant digits. */
export function formatTensorValue(value: number | null | undefined): string {
  return formatMetricValue(undefined, value);
}

/** A color-scale tick: three significant digits, a small one as a power of ten, − for minus. */
export function formatScaleLabel(value: number): string {
  const text =
    value !== 0 && Math.abs(value) <= 0.001 ? value.toExponential(0) : formatSignificant(value, 3);
  return text.replaceAll('-', '−');
}
