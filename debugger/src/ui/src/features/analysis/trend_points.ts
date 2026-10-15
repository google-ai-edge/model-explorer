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

import type {KvRangeBin, KvRangeMetricSummary} from '../../data/contracts/kv';
import type {HeatmapReduction} from './heatmap_model';

/** One point per shared Position bin; missing data and sparse intervals break the line. */
export function trendPoints(
  bins: readonly KvRangeBin[],
  summaries: readonly (KvRangeMetricSummary | null | undefined)[],
  reduction: HeatmapReduction,
) {
  const points: ({position: number; value: number; bin: number} | null)[] = [];
  bins.forEach((bin, index) => {
    if (index && bins[index - 1].end !== bin.start) points.push(null);
    const summary = summaries[index];
    const value = summary && summary.valid_count > 0 ? summary[reduction] : null;
    points.push(
      value == null || !Number.isFinite(value)
        ? null
        : {
            position: (bin.start + bin.end - 1) / 2,
            value,
            bin: index,
          },
    );
  });
  return points;
}
