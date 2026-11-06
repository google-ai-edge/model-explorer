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

import {Component, input} from '@angular/core';
import {ExplorerStatisticTable} from '../../widgets/explorer_statistic_table/explorer_statistic_table';
import type {KvRangeMetricSummary} from '../../server/contracts/kv';
import {TensorNumericCell, type AnalysisDetails} from './analysis_details';

/** A details body's Statistics table of an inspected tensor: its minimum, maximum and mean on
 *  each side and of their difference, with locators to the elements of the extrema. */
@Component({
  selector: 'element-statistics',
  imports: [ExplorerStatisticTable, TensorNumericCell],
  templateUrl: './element_statistics.ng.html',
  styles: [
    `
      @use '../../theme/analysis_details' as details;
      :host {
        display: contents;
      }
      @include details.body;
      // The table hangs under its Statistics item, as the body's own tables do.
      :host-context(.analysis-item) > .inspector-table {
        margin: 2px 0 0 18px;
        width: calc(100% - 18px);
        max-width: calc(100% - 18px);
      }
    `,
  ],
})
export class ElementStatistics {
  readonly details = input.required<AnalysisDetails>();
  /** The tensor's name, as the item and its locators say it. */
  readonly label = input.required<string>();
  readonly rows = input.required<{label: string; stats: KvRangeMetricSummary}[]>();
  /** The tensor's kind, which the body's locator is told. */
  readonly kind = input.required<string>();
}
