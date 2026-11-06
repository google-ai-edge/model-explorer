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

import {ChangeDetectionStrategy, Component, computed, inject, input} from '@angular/core';
import {ExplorerInfoValue} from '../../widgets/explorer_info_value/explorer_info_value';
import type {KvHeadMetric} from '../../server/contracts/kv';
import {MetricCatalogStore, type MetricChoice} from '../../server/metric_catalog';
import {formatMetricValue} from '../../widgets/format/format';

/** The same numerical readout is used for a token tensor and an individual head. */
@Component({
  selector: 'kv-metric-values',
  standalone: true,
  imports: [ExplorerInfoValue],
  template: `<table class="metric-table" [attr.aria-label]="label()">
    <tbody>
      @for (metric of definitions(); track metric.id) {
        <tr>
          <th scope="row">
            <explorer-info-value [label]="metric.description.measures" [value]="metric.label" />
          </th>
          <td [class.unavailable]="!hasValue(metric.id)">
            <span
              class="metric-value"
              tabindex="0"
              [title]="tooltip(metric)"
              [attr.aria-label]="metric.label + ': ' + text(metric.id) + '. ' + reason(metric.id)"
              >{{ text(metric.id) }}</span
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
  readonly metrics = input<Partial<Record<KvHeadMetric, number | null>> | null>(null);
  readonly metricStatus = input<Partial<Record<KvHeadMetric, string>> | null>(null);
  readonly label = input.required<string>();
  private readonly catalog = inject(MetricCatalogStore);
  /** The metrics the catalog's KV head analysis offers, in its order. */
  readonly definitions = computed(() => this.catalog.lookup().choices('kv_head'));

  hasValue(key: string): boolean {
    const value = this.metrics()?.[key as KvHeadMetric];
    return typeof value === 'number' && Number.isFinite(value);
  }
  text(key: string): string {
    return formatMetricValue(
      this.catalog.lookup().metric(key),
      this.metrics()?.[key as KvHeadMetric],
    );
  }
  /** The catalog's words for the metric's status, a reason code the head gives alone. */
  reason(key: string): string {
    if (this.hasValue(key)) {
      return '';
    }
    const status = this.metricStatus()?.[key as KvHeadMetric];
    return status ? this.catalog.lookup().reason(status) : '';
  }
  tooltip(metric: MetricChoice): string {
    return [metric.description.measures, this.reason(metric.id)].filter(Boolean).join(' ');
  }
}
