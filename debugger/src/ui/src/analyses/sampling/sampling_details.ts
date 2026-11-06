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

import {Component, DestroyRef, computed, effect, inject, input, output} from '@angular/core';
import {
  ExplorerInfoList,
  type ExplorerInfoItem,
} from '../../widgets/explorer_info_list/explorer_info_list';
import {ExplorerInfoSection} from '../../widgets/explorer_info_section/explorer_info_section';
import {ExplorerLocator} from '../../widgets/explorer_locator/explorer_locator';
import {CaptureStore} from '../../server/capture_store';
import type {ComparisonAlignment} from '../../server/contracts/capture';
import type {KvRangeMetricSummary} from '../../server/contracts/kv';
import type {
  SamplingDistribution,
  TokenRangeRequest,
  TokenRangeResponse,
} from '../../server/contracts/token_analysis';
import {LatestRequest} from '../../server/latest_request';
import {TokenAnalysisApi} from '../../server/token_analysis_api';
import {
  AnalysisDetails,
  EMPTY_STATISTIC,
  type StagePoint,
  type SummaryRow,
} from '../shared/analysis_details';
import {MetricStatistics} from '../shared/metric_statistics';
import {samplingDistributionMetadata, samplingMetricMetadata} from './sampling_analysis_info';
import {SamplingDistributionInspection} from './sampling_distribution_inspection';

/**
 * The details of a Sampling Distribution panel: each side's statistics of the metric over the
 * range, the distributions' metadata and, at one position, the two distributions.
 */
@Component({
  selector: 'sampling-details',
  imports: [
    ExplorerInfoList,
    ExplorerInfoSection,
    ExplorerLocator,
    MetricStatistics,
    SamplingDistributionInspection,
  ],
  templateUrl: './sampling_details.ng.html',
  styles: [
    `
      @use '../../theme/analysis_details' as details;
      :host {
        display: contents;
      }
      @include details.body;
    `,
  ],
})
export class SamplingDetails extends AnalysisDetails {
  readonly alignment = input<ComparisonAlignment>('content');
  /** The range's points: at one position, its distributions. */
  readonly points = input<StagePoint[]>([]);
  readonly locateView = output<void>();
  private readonly state = inject(CaptureStore);
  private readonly api = inject(TokenAnalysisApi);
  /** Each side's metric over the range, one bin, with what the range compared. */
  readonly rangeData = new LatestRequest<TokenRangeResponse>();
  readonly locateFinal = output<string>();
  protected override readonly metricWraps = true;
  protected override readonly groupsByLabel = true;
  readonly metric = computed(() => this.panel().logit?.metric ?? 'entropy');
  protected readonly metricAnalysis = () => 'sampling';
  protected override dashesMissing(): boolean {
    return true;
  }
  override rowLabel(row: SummaryRow): string {
    return row.label;
  }
  locateDefinition(item: ExplorerInfoItem): void {
    if (item.label === 'Name') {
      this.locateAnalysis.emit();
    } else if (item.label === 'Mode') {
      this.locateView.emit();
    } else if (['Token Range', 'Positions', 'Selected position'].includes(item.label)) {
      this.locateScope.emit('positions');
    } else if (item.label === 'Metric') {
      this.locateMetric.emit();
    }
  }
  locateStatistic(label: string): void {
    this.locateFinal.emit(label);
  }
  readonly samplingInfoItems = computed<ExplorerInfoItem[]>(() => [
    {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
    {
      label: 'Mode',
      value: 'Distribution Comparison',
      locateLabel: 'Locate View control',
    },
    {
      label: 'Stage',
      value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`,
    },
    {
      label: 'Token Range',
      value: `${this.range().start}–${this.range().end - 1}`,
      locateLabel: 'Locate Token Range control',
    },
    this.metricDefinition(),
  ]);
  /** The row the panel's selection stands for, or null (AnalysisPanels.inspectedRow). */
  readonly inspectedRow = input<number | null>(null);
  /**
   * The position whose two distributions Distribution Inspection lists: the range's one
   * position; else the selected row, while it is in the range; else the range's first row whose
   * distributions the server has answered. A distribution is of one position, never a range.
   */
  readonly inspectedPoint = computed(() => {
    const inRange = this.points().filter(
      (point) => this.sideInRange(point, 'ref') || this.sideInRange(point, 'target'),
    );
    const row = this.inspectedRow();
    return (
      (this.range().end - this.range().start === 1 ? inRange[0] : undefined) ??
      inRange.find((point) => point.index === row) ??
      inRange.find((point) => point.analysis?.sampling_distribution) ??
      null
    );
  });
  /** The inspected position's column, as the chart's axis names it. */
  readonly inspectedPosition = computed(() => this.inspectedPoint()?.position ?? null);
  samplingInspectionSource(side: 'ref' | 'target'): SamplingDistribution | undefined {
    const point = this.inspectedPoint();
    return point && (side === 'ref' ? point.hasRef : point.hasTarget)
      ? point.analysis?.sampling_distribution
      : undefined;
  }
  samplingSourceItems(side: 'ref' | 'target'): ExplorerInfoItem[] {
    const metadata = this.rangeData.data()?.metadata[side];
    return metadata ? samplingDistributionMetadata(metadata) : [];
  }
  readonly samplingMetricMetadata = computed(() =>
    samplingMetricMetadata(
      this.metricDescription(),
      this.range(),
      this.alignment(),
      this.rangeData.data()?.metadata.output ?? null,
    ),
  );
  sideInRange(point: StagePoint, side: 'ref' | 'target'): boolean {
    const position = side === 'ref' ? point.refPosition : point.targetPosition;
    return position !== null && position >= this.range().start && position < this.range().end;
  }
  /** A side's statistic of the metric over the range, as the server gives it. */
  sideStats(side: 'ref' | 'target'): KvRangeMetricSummary {
    return this.rangeData.data()?.rows.find((row) => row.side === side)?.bins[0] ?? EMPTY_STATISTIC;
  }
  readonly sections = computed(() => {
    const label = 'Sampling distribution';
    return [
      {
        label,
        loading: this.rangeData.loading(),
        error: this.rangeData.error(),
        rows: this.sort(
          [
            {label: 'Ref', layer: null, kind: 'ref', stats: this.sideStats('ref')},
            {label: 'Target', layer: null, kind: 'target', stats: this.sideStats('target')},
          ],
          label,
        ),
      },
    ];
  });
  /**
   * The token-range request of the range, of the fields that change its answer alone; none
   * without a capture, a distribution or a range.
   */
  private readonly rangeQuery = computed(
    () => {
      const capture = this.state.captureId();
      const {turn, stage} = this.panel();
      const range = this.range();
      if (!capture || range.end <= range.start) {
        return null;
      }
      const request: TokenRangeRequest = {
        turn,
        stage,
        alignment: this.alignment(),
        analysis: 'sampling',
        metric: this.metric(),
        start: range.start,
        end: range.end,
        bins: 1,
      };
      return {capture, request};
    },
    {equal: (a, b) => JSON.stringify(a) === JSON.stringify(b)},
  );
  constructor() {
    super();
    effect((onCleanup) => {
      const query = this.rangeQuery();
      onCleanup(
        this.rangeData.load(
          query ? (signal) => this.api.tokenRange(query.capture, query.request, signal) : null,
        ),
      );
    });
    inject(DestroyRef).onDestroy(() => this.rangeData.destroy());
  }
}
