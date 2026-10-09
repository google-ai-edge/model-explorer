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

import {ChangeDetectionStrategy, Component, input} from '@angular/core';
import type {KvHeadMetric} from '../../../data/contracts/kv';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {formatMetric, formatPercent} from '../../../shared/format/format';

export type KvDetailMetric = Extract<
  KvHeadMetric,
  'cosine_similarity' | 'relative_l2' | 'rmse' | 'max_abs'
>;
export const KV_DETAIL_METRICS: readonly {
  key: KvDetailMetric;
  label: string;
  description: string;
}[] = [
  {
    key: 'cosine_similarity',
    label: 'CosSim',
    description: 'Cosine similarity; higher is more similar.',
  },
  {
    key: 'relative_l2',
    label: 'Relative L2',
    description:
      'L2 norm of Target − Ref divided by the Ref L2 norm; shown as a percentage.',
  },
  {key: 'rmse', label: 'RMSE', description: 'Root mean squared error.'},
  {
    key: 'max_abs',
    label: 'Max |Δ|',
    description: 'Largest absolute Target − Ref value.',
  },
];

export function formatKvMetric(
  key: KvHeadMetric,
  value: number | null | undefined,
): string {
  return key === 'relative_l2' ? formatPercent(value) : formatMetric(value);
}

/** The same numerical readout is used for a token tensor and an individual head. */
@Component({
  selector: 'kv-metric-values',
  standalone: true,
  imports: [ExplorerInfoValue],
  template: `<table class="metric-table" [attr.aria-label]="label()">
    <tbody>
      @for (metric of definitions; track metric.key) {
        <tr>
          <th scope="row">
            <explorer-info-value [label]="metric.description" [value]="metric.label" />
          </th>
          <td [class.unavailable]="!hasValue(metric.key)">
            <span
              class="metric-value"
              tabindex="0"
              [title]="tooltip(metric)"
              [attr.aria-label]="metric.label + ': ' + text(metric.key) + '. ' + reason(metric.key)"
              >{{ text(metric.key) }}</span
            >
          </td>
        </tr>
      }
    </tbody>
  </table>`,
  styleUrl: './kv_metric_values.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvMetricValues {
  readonly metrics = input<Partial<Record<KvHeadMetric, number | null>> | null>(
    null,
  );
  readonly metricStatus = input<Partial<Record<KvHeadMetric, string>> | null>(
    null,
  );
  readonly label = input.required<string>();
  readonly definitions = KV_DETAIL_METRICS;

  hasValue(key: KvHeadMetric) {
    const value = this.metrics()?.[key];
    return typeof value === 'number' && Number.isFinite(value);
  }
  text(key: KvHeadMetric) {
    return formatKvMetric(key, this.metrics()?.[key]);
  }
  reason(key: KvHeadMetric) {
    if (this.hasValue(key)) return '';
    switch (this.metricStatus()?.[key]) {
      case 'undefined_zero_norm':
        return 'Undefined because a required vector norm is zero.';
      case 'non_finite':
      case 'non_finite_tensor':
        return 'Unavailable because the values include NaN or infinity.';
      case 'numeric_overflow':
        return 'The comparison exceeds the supported numerical range.';
      default:
        return 'Unavailable for this selection.';
    }
  }
  tooltip(metric: {key: KvHeadMetric; description: string}) {
    return [metric.description, this.reason(metric.key)]
      .filter(Boolean)
      .join(' ');
  }
}
