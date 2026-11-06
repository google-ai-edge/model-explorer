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

import type {MetricDescription} from '../../server/contracts/metrics';
import {formatScaleLabel, formatSignificant} from '../format/format';

/** How a panel colors a metric: over the values it shows, or by the metric's four thresholds. */
export type MetricScaleMode = 'auto' | 'thresholds';

function clamp(fraction: number): number {
  return Math.min(1, Math.max(0, fraction));
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** A metric bounded on both sides with no thresholds, as cosine similarity, is colored over
 *  its bounds whatever the mode. */
export function boundedScale(
  description: MetricDescription | undefined,
): description is MetricDescription & {ideal: number; bounds: [number, number]} {
  const [low, high] = description?.bounds ?? [null, null];
  return (
    !!description &&
    description.color.thresholds === null &&
    description.ideal !== null &&
    low !== null &&
    high !== null
  );
}

/** Whether the four thresholds color the metric in this mode. */
export function steppedScale(
  description: MetricDescription | undefined,
  mode: MetricScaleMode,
): description is MetricDescription & {
  color: MetricDescription['color'] & {thresholds: number[]};
} {
  return mode === 'thresholds' && !boundedScale(description) && !!description?.color.thresholds;
}

/** How far a value is from the metric's ideal: the distance a color scale measures. */
export function idealDistance(
  description: MetricDescription | undefined,
  value: number | null | undefined,
): number | null {
  return finite(value) ? Math.abs(value - (description?.ideal ?? 0)) : null;
}

/** The bound farthest from the ideal value, of a bounded scale. */
function farBound(
  description: MetricDescription & {ideal: number; bounds: [number, number]},
): number {
  const ideal = description.ideal;
  const [low, high] = description.bounds;
  return Math.abs(low - ideal) >= Math.abs(high - ideal) ? low : high;
}

/** A value's place on the metric's color scale, from 0 (the ideal value) to 1; null without a
 *  value. `maximum` is the largest distance from the ideal value among the values shown, which
 *  the auto mode scales by. */
export function metricHeat(
  description: MetricDescription | undefined,
  value: number | null | undefined,
  mode: MetricScaleMode,
  maximum: number,
): number | null {
  const distance = idealDistance(description, value);
  if (distance === null) {
    return null;
  }
  if (boundedScale(description)) {
    return clamp(distance / Math.abs(farBound(description) - description.ideal));
  }
  if (steppedScale(description, mode)) {
    const {thresholds} = description.color;
    return thresholds.filter((threshold) => distance >= threshold).length / thresholds.length;
  }
  return clamp(distance / maximum);
}

/** The value a fraction of the way along a continuous color scale: from the ideal value to the
 *  far bound of a bounded scale, else from 0 to `maximum`. */
export function scaleValue(
  description: MetricDescription | undefined,
  fraction: number,
  maximum: number,
): number {
  if (!boundedScale(description)) {
    return fraction * maximum;
  }
  return description.ideal + fraction * (farBound(description) - description.ideal);
}

/** The labels along a color scale: six evenly spaced values (auto, bounded), or 0, the four
 *  thresholds and + (stepped). An auto scale with no maximum yet has none. */
export function metricScaleLabels(
  description: MetricDescription | undefined,
  mode: MetricScaleMode,
  maximum: number | null,
): string[] {
  const fractions = [0, 0.2, 0.4, 0.6, 0.8, 1];
  if (boundedScale(description)) {
    return fractions.map((f) => formatScaleLabel(scaleValue(description, f, 0)));
  }
  if (steppedScale(description, mode)) {
    return ['0', ...description.color.thresholds.map(formatScaleLabel), '+'];
  }
  return maximum === null ? [] : fractions.map((f) => formatSignificant(f * maximum, 3));
}

/** A token's heat for its metric, 0 to 1: the distance from the ideal value relative to the
 *  metric's `heat_full`; null without a value or a heat scale. */
export function tokenHeat(
  description: MetricDescription | undefined,
  value: number | null | undefined,
): number | null {
  const distance = idealDistance(description, value);
  const full = description?.color.heat_full;
  if (!description || distance === null || full == null) {
    return null;
  }
  return clamp(distance / Math.abs(full - (description.ideal ?? 0)));
}
