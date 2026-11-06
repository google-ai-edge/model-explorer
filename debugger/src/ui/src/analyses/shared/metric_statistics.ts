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

import {NgTemplateOutlet} from '@angular/common';
import {Component, input} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {ExplorerStatisticTable} from '../../widgets/explorer_statistic_table/explorer_statistic_table';
import {
  TensorNumericCell,
  type AnalysisDetails,
  type StatisticsSection,
} from './analysis_details';

/** A details body's Statistics table of a metric: one sortable row per group, its minimum,
 *  maximum and mean, and locators to the positions of the extrema. */
@Component({
  selector: 'metric-statistics',
  imports: [ExplorerStatisticTable, MatIconModule, NgTemplateOutlet, TensorNumericCell],
  templateUrl: './metric_statistics.ng.html',
  styles: [
    `
      @use '../../theme/analysis_details' as details;
      :host {
        display: contents;
      }
      @include details.body;
      // The table and its loading note hang under their Statistics item, as the body's own do.
      :host-context(.analysis-item) > .layer-table,
      :host-context(.analysis-item) > .statistic-table-note {
        margin: 2px 0 0 18px;
        width: calc(100% - 18px);
        max-width: calc(100% - 18px);
      }
    `,
  ],
})
export class MetricStatistics {
  readonly details = input.required<AnalysisDetails>();
  readonly section = input.required<StatisticsSection>();
}
