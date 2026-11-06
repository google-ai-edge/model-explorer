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
import {
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  input,
  linkedSignal,
  output,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {ExplorerPagination} from '../../widgets/explorer_pagination/explorer_pagination';
import {ExplorerInfoValue} from '../../widgets/explorer_info_value/explorer_info_value';
import {ExplorerLocator} from '../../widgets/explorer_locator/explorer_locator';
import {ExplorerStatisticTable} from '../../widgets/explorer_statistic_table/explorer_statistic_table';
import {
  ExplorerInfoList,
  type ExplorerInfoItem,
} from '../../widgets/explorer_info_list/explorer_info_list';
import {ExplorerInfoSection} from '../../widgets/explorer_info_section/explorer_info_section';
import {CaptureStore} from '../../server/capture_store';
import type {ComparisonAlignment} from '../../server/contracts/capture';
import type {TokenRangeRequest, TokenRangeResponse} from '../../server/contracts/token_analysis';
import {LatestRequest} from '../../server/latest_request';
import {reasonText} from '../../server/reasons';
import {TokenAnalysisApi} from '../../server/token_analysis_api';
import {
  AnalysisDetails,
  EMPTY_STATISTIC,
  elementStatisticRows,
  type StatisticsSection,
  type SummaryRow,
} from '../shared/analysis_details';
import {ElementStatistics} from '../shared/element_statistics';
import {catalogMetricId} from '../shared/logit_panel_state';
import {MetricStatistics} from '../shared/metric_statistics';
import type {FinalLogitsInspection} from './final_logits_inspection';
import {logitSourceItems} from './logit_analysis_info';
import {generatedSteps} from '../shared/column_steps';
import {stepRows} from '../../widgets/format/step_labels';

/**
 * The details of a Final Logits panel: in Tensor Comparison, the metric's statistics over the
 * range and the logits' metadata; in Tensor Inspection, the inspected vector's statistics,
 * metadata and values.
 */
@Component({
  selector: 'final-logits-details',
  host: {'[style.--info-inspector-identity-width]': 'tensorIdentityWidth() + "px"'},
  imports: [
    ElementStatistics,
    ExplorerPagination,
    ExplorerInfoList,
    ExplorerInfoValue,
    ExplorerStatisticTable,
    ExplorerInfoSection,
    ExplorerLocator,
    MatIconModule,
    MetricStatistics,
    NgTemplateOutlet,
  ],
  templateUrl: './final_logits_details.ng.html',
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
export class FinalLogitsDetails extends AnalysisDetails {
  /** The server's text of a response's reasons. */
  readonly reasonText = reasonText;
  readonly alignment = input<ComparisonAlignment>('content');
  readonly finalLogits = input<FinalLogitsInspection>();
  readonly locateView = output<void>();
  readonly locateFinal = output<string>();
  readonly locateLogitElement = output<number>();
  readonly locateTensorPosition = output<void>();
  private readonly state = inject(CaptureStore);
  private readonly api = inject(TokenAnalysisApi);
  /** The metric over the range, one bin, with what the range compared. */
  readonly rangeData = new LatestRequest<TokenRangeResponse>();
  protected override readonly metricWraps = true;
  readonly metric = computed(() => this.panel().logit?.metric ?? 'logit_rmse');
  protected readonly metricAnalysis = () => 'final_logits';
  override rowLabel(_row: SummaryRow): string {
    return this.logitGroupLabel();
  }
  protected override positionAlignmentLabel(): 'Context' | 'Steps' {
    return this.logitMetricInfo()
      ? this.alignment() === 'content'
        ? 'Context'
        : 'Steps'
      : 'Steps';
  }
  logitIndexSegments(row: {index: number; tokenLabel: string}) {
    return [
      {
        text: String(row.index),
        help: {
          title: 'Token',
          text: row.tokenLabel,
          rows: [],
        },
      },
    ];
  }
  selectLogitRow(event: KeyboardEvent, index: number): void {
    if (
      event.target !== event.currentTarget ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      (event.key !== 'Enter' && event.key !== ' ')
    ) {
      return;
    }
    event.preventDefault();
    this.finalLogits()?.selectElement(index);
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
    } else if (item.label === 'Token position') {
      this.locateTensorPosition.emit();
    }
  }
  locateStatistic(label: string): void {
    this.locateFinal.emit(label);
  }
  readonly logitInspection = computed(
    () => this.panel().kind === 'tokens' && this.panel().logit?.logitView === 'inspection',
  );
  readonly tensorIdentityWidth = linkedSignal({
    source: () => {
      const source = this.logitInspection() ? this.finalLogits() : undefined;
      return {
        source,
        panel: this.panel().id,
        capture: this.state.captureId(),
        position: source?.position(),
        rows: source?.rows() ?? [],
        pending: source?.valuesLoading() || !!source?.valuesError(),
      };
    },
    computation: (current, previous) => {
      if (
        current.pending &&
        !current.rows.length &&
        previous?.source.source === current.source &&
        previous?.source.panel === current.panel &&
        previous?.source.capture === current.capture &&
        previous?.source.position === current.position
      ) {
        return previous.value;
      }
      const digits = Math.max(0, ...current.rows.map((row) => String(row.index).length));
      // Fit the current page plus the 12px help and locator icons. Both tables
      // share the width, which stays stable while the next page is loading.
      return Math.max(48, Math.ceil(digits * 6.75 + 24));
    },
  });
  readonly logitMetricInfo = computed(
    () => this.panel().kind === 'tokens' && !this.logitInspection(),
  );
  readonly logitGroupLabel = computed(() => 'Final logits');
  readonly logitInfoItems = computed<ExplorerInfoItem[]>(() => {
    const context = this.rangeContext();
    return [
      {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
      {
        label: 'Mode',
        value: 'Tensor Comparison',
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
      ...(context ? [{label: 'Context', value: context}] : []),
      this.metricDefinition(),
    ];
  });
  /**
   * Why the range's columns after the divergence still compare, in the server's words; null
   * where none of them reads different tokens on the two sides.
   */
  private readonly rangeContext = computed(() => {
    const panel = this.panel();
    const comparison = this.state.session()?.turns.find((turn) => turn.n === panel.turn)?.comparison;
    const positions =
      panel.stage === 'prefill'
        ? comparison?.prefill.positions
        : comparison?.alignments[this.alignment()]?.positions;
    const context = positions?.columns?.final_logits?.context;
    return context && this.range().end > context.start ? context.reason.text : null;
  });
  readonly logitInspectionInfoItems = computed<ExplorerInfoItem[]>(() => [
    {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
    {label: 'Mode', value: 'Tensor Inspection', locateLabel: 'Locate View control'},
    {
      label: 'Stage',
      value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`,
    },
    {
      label: 'Token position',
      value: String(
        this.finalLogits()?.position() ?? this.panel().logit?.logitPosition ?? this.range().start,
      ),
      locateLabel: 'Locate Token Position control',
    },
    ...stepRows(this.inspectedSteps()),
  ]);
  /** Which step produced the inspected row's tokens, in the server's words. */
  private readonly inspectedSteps = computed(() => {
    const pair = this.finalLogits()?.analysis();
    return pair
      ? generatedSteps(
          this.state.session(),
          this.panel().turn,
          pair.ref_step,
          pair.target_step,
          reasonText(pair.distribution?.notes ?? []),
        )
      : null;
  });
  readonly logitInspectionData = computed(() => {
    const source = this.finalLogits();
    return source && !source.loading() && !source.error() ? source.data() : null;
  });
  /** The inspected vectors' statistics, as the server gives them with the vectors. */
  readonly logitInspectionStatistics = computed(() => {
    const source = this.finalLogits();
    return elementStatisticRows(
      source && !source.loading() && !source.error() ? source.statistics() : null,
    );
  });
  override locateTensorStatistic(kind: string, index: number): void {
    if (kind === 'logits') {
      this.locateLogitElement.emit(index);
    }
  }
  logitInspectionMetadata(side: 'ref' | 'target'): ExplorerInfoItem[] {
    const source = this.logitInspectionData()?.[side];
    const metadata = source?.logits_metadata;
    const size = metadata
      ? metadata.comparison_shape.length === 1
        ? metadata.comparison_shape[0]
        : undefined
      : source?.size;
    const shape = size != null ? `[${size}]` : '—';
    return [
      {
        label: 'Dimensions',
        value: '[Vocabulary]',
        segments: [
          {
            text: '[Vocabulary]',
            help: {
              title: 'Vocabulary',
              rows: [
                {label: 'Dimension', value: 'Vocabulary token IDs in the captured logits vector'},
                {label: 'Identity', value: metadata?.vocab_identity ?? 'Not reported'},
              ],
            },
          },
        ],
      },
      {
        label: 'Shape',
        value: shape,
      },
      {
        label: 'Dtype',
        value: metadata?.source_dtype ?? '—',
      },
    ];
  }
  logitSourceItems(side: 'ref' | 'target'): ExplorerInfoItem[] {
    const metadata = this.rangeData.data()?.metadata[side];
    return metadata ? logitSourceItems(metadata) : [];
  }
  readonly logitMetricMetadata = computed(() =>
    this.metadataItems(this.rangeData.data()?.metadata.output ?? null, true),
  );
  readonly sections = computed<StatisticsSection[]>(() => {
    const rows: SummaryRow[] = [
      {
        label: 'Final',
        layer: null,
        kind: 'logits',
        stats: this.rangeData.data()?.rows[0]?.bins[0] ?? EMPTY_STATISTIC,
      },
    ];
    return [
      {
        label: this.logitGroupLabel(),
        loading: this.rangeData.loading(),
        error: this.rangeData.error(),
        rows,
      },
    ];
  });
  /**
   * The token-range request of the range, of the fields that change its answer alone; none
   * without a capture, a metric or a range.
   */
  private readonly rangeQuery = computed(
    () => {
      const capture = this.state.captureId();
      const {turn, stage} = this.panel();
      const range = this.range();
      if (!capture || !this.logitMetricInfo() || range.end <= range.start) {
        return null;
      }
      const request: TokenRangeRequest = {
        turn,
        stage,
        alignment: this.alignment(),
        analysis: 'final_logits',
        metric: catalogMetricId(this.metric()),
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
