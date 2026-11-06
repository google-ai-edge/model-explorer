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
  type WritableSignal,
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
import {KvApi} from '../../server/kv_api';
import {CaptureStore} from '../../server/capture_store';
import {LatestRequest} from '../../server/latest_request';
import type {
  KvAnalysisContext,
  KvMetric,
  KvPositionAlignment,
  KvRangeBin,
  KvRangeMetricSummary,
  KvRangeResponse,
  KvRangeRow,
} from '../../server/contracts/kv';
import {reasonText} from '../../server/reasons';
import type {RangeSelection} from '../../widgets/charts/range_selection';
import {type ColumnSteps, stepRows} from '../../widgets/format/step_labels';
import {AnalysisDetails, EMPTY_STATISTIC, elementStatisticRows} from '../shared/analysis_details';
import {ElementStatistics} from '../shared/element_statistics';
import {MetricStatistics} from '../shared/metric_statistics';
import {
  sortTensorValueRows,
  type TensorValueSortColumn,
  type TensorValueSortDirection,
} from '../shared/tensor_value_sort';
import type {KvTensorHead} from './kv_tensor_data';

/** Which KV cache tensor is selected or summarized. */
export type KvCacheKind = 'key' | 'value';

/** Metadata and request state for one KV cache tensor group in Tensor Inspection. */
export interface KvTensorGroup {
  kind: KvCacheKind;
  label: string;
  request: Pick<LatestRequest<KvTensorHead[]>, 'loading' | 'error'>;
}

/** Structural contract for the KV panel read by `KvDetails`. */
export interface KvPanelSource {
  metric(): KvMetric;
  tensorView(): boolean;
  color(value: number | null): string;
  positionAlignment(): KvPositionAlignment | null;
  visibleMetricLayers(): number[];
  locateView(): void;
  view(): 'heatmap' | 'lineChart';
  loading(): boolean;
  head(): string;
  selectedHeads(): number[];
  tensorPosition(): number | null;
  tensorLayer(): number | null;
  tensorHead(): number | null;
  columnSteps(column: number): ColumnSteps | null;
  locateTensorPosition(): void;
  locateTensorIndex(): void;
  locateTensorElement(kind: KvCacheKind, index: number): void;
  locateTensor(kind: KvCacheKind): void;
  tensorGroups(): readonly KvTensorGroup[];
  tensorEvidence(kind: KvCacheKind): KvTensorHead | null;
  tensorSelectedRange(): KvRangeBin | null;
  tensorChannels(): Readonly<Record<string, number | null>>;
  readonly tensorRetry: WritableSignal<number>;
  readonly incrementTensorRetry: (value: number) => number;
  context(): KvAnalysisContext | undefined;
  aggregationHeads(): number[] | null;
}

const KV_METRICS: readonly KvMetric[] = [
  'relative_l2',
  'max_abs',
  'cosine_distance',
  'cosine_similarity',
];

function isKvMetric(value: string): value is KvMetric {
  return (KV_METRICS as readonly string[]).includes(value);
}

/**
 * The details of a KV panel: in Tensor Comparison, the metric's statistics of the Key and Value
 * caches by layer and their metadata; in Tensor Inspection, the inspected head's statistics,
 * metadata and values.
 */
@Component({
  selector: 'kv-details',
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
  templateUrl: './kv_details.ng.html',
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
export class KvDetails extends AnalysisDetails {
  readonly kv = input<KvPanelSource>();
  readonly kvSelection = input<(RangeSelection & {kind: string | null}) | null>(null);
  readonly locateCache = output<'key' | 'value'>();
  readonly locateAggregation = output<void>();
  readonly metric = computed<KvMetric>(() => {
    const panelMetric = this.panel().logit?.metric;
    return (
      this.kv()?.metric() ??
      (panelMetric && isKvMetric(panelMetric) ? panelMetric : 'cosine_similarity')
    );
  });
  protected readonly metricAnalysis = computed(() => (this.kv()?.tensorView() ? 'kv_head' : 'kv'));
  protected override metricColor(value: number): string | null {
    return this.kv()?.color(value) ?? null;
  }
  protected override positionAlignmentLabel(): 'Context' | 'Steps' {
    return this.kv()?.positionAlignment()?.alignment === 'content' ? 'Context' : 'Steps';
  }
  readonly visibleKvLayers = computed(() => {
    const kv = this.kv();
    return kv?.visibleMetricLayers() ?? [];
  });
  locateDefinition(item: ExplorerInfoItem): void {
    if (item.label === 'Name') {
      this.locateAnalysis.emit();
    } else if (item.label === 'Mode') {
      this.kv()?.locateView();
    } else if (['Token Range', 'Positions', 'Selected position'].includes(item.label)) {
      this.locateScope.emit('positions');
    } else if (item.label === 'Groups') {
      this.locateScope.emit('layers');
    } else if (item.label === 'Metric') {
      this.locateMetric.emit();
    } else if (item.label === 'Aggregation') {
      this.locateAggregation.emit();
    }
  }
  statisticView(): string {
    return this.kv()?.view() === 'heatmap' ? 'Heatmap' : 'Line chart';
  }
  locateStatistic(label: string): void {
    this.locateCache.emit(label === 'Key cache' ? 'key' : 'value');
  }
  private readonly api = inject(KvApi);
  private readonly state = inject(CaptureStore);
  readonly keyData = new LatestRequest<KvRangeResponse>();
  readonly valueData = new LatestRequest<KvRangeResponse>();
  readonly kvMetricInfo = computed(() => !this.kv()?.tensorView());
  readonly analysisKvLayers = computed(() => {
    const selected = this.kvSelection()?.layer;
    return selected == null || !this.visibleKvLayers().includes(selected)
      ? this.visibleKvLayers()
      : [selected];
  });
  readonly metricInfoItems = computed<ExplorerInfoItem[]>(() => {
    const panel = this.panel();
    const layers = this.analysisKvLayers();
    return [
      {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
      {label: 'Mode', value: 'Tensor Comparison', locateLabel: 'Locate View control'},
      {
        label: 'Stage',
        value: `Turn ${panel.turn} · ${panel.stage === 'prefill' ? 'Prefill' : 'Decode'}`,
      },
      {
        label: 'Token Range',
        value: `${this.range().start}–${this.range().end - 1}`,
        locateLabel: 'Locate Token Range control',
      },
      {
        label: 'Groups',
        locateLabel: 'Locate Groups control',
        value: `Transformer Layers [${layers.map((layer) => `L${layer}`).join(', ')}]`,
        expandable: true,
      },
      this.metricDefinition(),
      this.aggregationDefinition(),
    ];
  });
  /**
   * Each cache's inputs and output, as the server's kv-range answer of the range, its heads
   * and its layers describes them: while a new answer loads, the last one; the server's reason
   * when it fails; loading only before any answer arrived.
   */
  readonly kvMetadata = computed(() =>
    [
      {kind: 'key', request: this.keyData},
      {kind: 'value', request: this.valueData},
    ].map(({kind, request}) => {
      const error = request.error();
      const metadata = error ? undefined : (request.data() ?? request.last())?.metadata;
      return {
        kind,
        label: kind === 'key' ? 'Key cache' : 'Value cache',
        error,
        loading: !error && !metadata && (!!this.kv()?.loading() || request.loading()),
        sources: (['ref', 'target'] as const).map((side) => ({
          label: side === 'ref' ? 'Reference' : 'Target',
          metadata: metadata?.[side] ?? null,
        })),
        output: metadata?.output ?? null,
      };
    }),
  );
  readonly aggregationDefinition = computed<ExplorerInfoItem>(() =>
    this.aggregationItem(this.kv()?.head() ?? 'min', 'Heads', this.kv()?.selectedHeads() ?? []),
  );
  readonly inspectionInfoItems = computed<ExplorerInfoItem[]>(() => {
    const source = this.kv();
    const pos = source?.tensorPosition();
    const layer = source?.tensorLayer();
    const head = source?.tensorHead();
    return [
      {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
      {label: 'Mode', value: 'Tensor Inspection', locateLabel: 'Locate View control'},
      {
        label: 'Stage',
        value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`,
      },
      {
        label: 'Token position',
        value: String(pos ?? '-'),
        locateLabel: 'Locate Token Position control',
      },
      ...(source && pos != null ? stepRows(source.columnSteps(pos)) : []),
      {
        label: 'Tensor Index',
        locateLabel: 'Locate Tensor Index control',
        value: `[Layer: ${layer ?? '-'}, Head: ${head ?? '-'}]`,
        segments: [
          {text: '['},
          {
            text: 'Layer',
            help: {
              title: 'Layer',
              rows: [
                {label: 'Dimension', value: 'Transformer layer'},
                {label: 'Index', value: String(layer ?? '-')},
                {label: 'Indexing', value: 'Zero-based'},
              ],
            },
          },
          {text: `: ${layer ?? '-'}, `},
          {
            text: 'Head',
            help: {
              title: 'Head',
              rows: [
                {label: 'Dimension', value: 'KV attention head'},
                {label: 'Index', value: String(head ?? '-')},
                {label: 'Layer', value: String(layer ?? '-')},
                {label: 'Indexing', value: 'Zero-based within this layer'},
              ],
            },
          },
          {text: `: ${head ?? '-'}]`},
        ],
        expandable: true,
      },
    ];
  });
  locateInspection(item: ExplorerInfoItem): void {
    if (item.label === 'Name') {
      this.locateAnalysis.emit();
    } else if (item.label === 'Mode') {
      this.kv()?.locateView();
    } else if (item.label === 'Token position') {
      this.kv()?.locateTensorPosition();
    } else if (item.label === 'Tensor Index') {
      this.kv()?.locateTensorIndex();
    }
  }
  override locateTensorStatistic(kind: string, index: number): void {
    if (kind === 'key' || kind === 'value') {
      this.kv()?.locateTensorElement(kind, index);
    }
  }
  inspectionMetadata(kind: KvCacheKind, side: 'ref' | 'target'): ExplorerInfoItem[] {
    const source = this.kv();
    const request = source?.tensorGroups().find((group) => group.kind === kind)?.request;
    const loading = request?.loading();
    const failed = request?.error() || source?.tensorEvidence(kind)?.error;
    const metadata =
      !loading && !failed ? source?.tensorEvidence(kind)?.evidence?.sources?.[side] : null;
    return [
      {
        label: 'Shape',
        value: metadata ? '[' + metadata.comparison_shape.join(', ') + ']' : '-',
        expandable: true,
      },
      {label: 'Dtype', value: metadata?.comparison_dtype ?? '-'},
    ];
  }
  readonly tensorInspectionRows = computed(() =>
    Object.fromEntries(
      (this.kv()?.tensorGroups() ?? []).map(({kind, request}) => [
        kind,
        (request.loading() || request.error()
          ? []
          : (this.kv()?.tensorEvidence(kind)?.evidence?.channels ?? [])
        )
          .map((element) => ({
            label: String(element.channel),
            index: element.channel,
            values: [element.ref, element.target, element.delta, element.abs_delta],
          }))
          .sort((a, b) => a.index - b.index),
      ]),
    ),
  );
  /** Each cache's inspected head's statistics, as the server gives them. */
  readonly tensorStatistics = computed(() =>
    Object.fromEntries(
      (this.kv()?.tensorGroups() ?? []).map(({kind, request}) => [
        kind,
        elementStatisticRows(
          request.loading() || request.error()
            ? null
            : (this.kv()?.tensorEvidence(kind)?.evidence?.statistics ?? null),
        ),
      ]),
    ),
  );
  readonly tensorValueSorts = linkedSignal<
    Record<string, {column: TensorValueSortColumn; direction: TensorValueSortDirection}>
  >(() => {
    this.statisticPanelId();
    return {};
  });
  tensorValueSortDirection(kind: string, column: TensorValueSortColumn): TensorValueSortDirection | 'none' {
    const sort = this.tensorValueSorts()[kind] ?? {column: 'index', direction: 'ascending'};
    return sort.column === column ? sort.direction : 'none';
  }
  sortTensorValues(kind: string, column: TensorValueSortColumn): void {
    const pages = this.tensorPages();
    const direction =
      this.tensorValueSortDirection(kind, column) === 'ascending' ? 'descending' : 'ascending';
    this.tensorValueSorts.update((sorts) => ({...sorts, [kind]: {column, direction}}));
    this.tensorPages.set({...pages, [kind]: 0});
  }
  readonly sortedTensorValueRows = computed(() =>
    Object.fromEntries(
      Object.entries(this.tensorInspectionRows()).map(([kind, rows]) => {
        const sort = this.tensorValueSorts()[kind] ?? {column: 'index', direction: 'ascending'};
        return [kind, sortTensorValueRows(rows, sort.column, sort.direction)];
      }),
    ),
  );
  readonly tensorPages = linkedSignal<Record<string, number>>(() => {
    // Reveal the selected representative without shrinking the highlighted bin.
    this.panel().id;
    this.kv()?.tensorPosition();
    this.kv()?.tensorLayer();
    this.kv()?.tensorHead();
    const selection = this.kv()?.tensorSelectedRange();
    const representatives = this.kv()?.tensorChannels();
    return Object.fromEntries(
      Object.entries(this.sortedTensorValueRows()).map(([kind, rows]) => {
        const representative = representatives?.[kind];
        const chosen =
          selection &&
          representative != null &&
          representative >= selection.start &&
          representative < selection.end
            ? rows.findIndex((row) => row.index === representative)
            : -1;
        const first =
          chosen >= 0
            ? chosen
            : selection
              ? rows.findIndex((row) => row.index >= selection.start && row.index < selection.end)
              : 0;
        return [kind, Math.floor(Math.max(0, first) / this.tensorPageSize)];
      }),
    );
  });
  tensorInspectionPage(kind: string): {
    index: number;
    pages: number;
    rows: Array<{label: string; index: number; values: Array<number | string | null>}>;
  } {
    const rows = this.sortedTensorValueRows()[kind] ?? [];
    const pages = Math.max(1, Math.ceil(rows.length / this.tensorPageSize));
    const index = Math.min(pages - 1, Math.max(0, this.tensorPages()[kind] ?? 0));
    return {
      index,
      pages,
      rows: rows.slice(index * this.tensorPageSize, (index + 1) * this.tensorPageSize),
    };
  }
  changeTensorPage(kind: string, delta: number): void {
    const page = this.tensorInspectionPage(kind);
    const index = Math.max(0, Math.min(page.pages - 1, page.index + delta));
    this.tensorPages.update((pages) => ({...pages, [kind]: index}));
  }
  tensorElementSelected(index: number): boolean {
    const range = this.kv()?.tensorSelectedRange();
    return range != null && index >= range.start && index < range.end;
  }
  readonly sections = computed(() =>
    [
      {label: 'Key cache', kind: 'key' as const, request: this.keyData},
      {label: 'Value cache', kind: 'value' as const, request: this.valueData},
    ].map((section) => {
      const data = section.request.data() ?? section.request.last();
      return {
        label: section.label,
        loading: !!this.kv()?.loading() || section.request.loading(),
        error: section.request.error(),
        rows: this.sort(
          (data?.rows ?? [])
            .filter(
              (r) =>
                this.visibleKvLayers().includes(r.layer) &&
                this.analysisKvLayers().includes(r.layer),
            )
            .map((r) => ({
              label: `Layer ${r.layer}`,
              layer: r.layer,
              kind: section.kind,
              stats: this.stats(r),
              reason: reasonText(r.reasons),
            })),
          section.label,
        ),
      };
    }),
  );
  stats(row: KvRangeRow): KvRangeMetricSummary {
    return row.bins[0]?.metrics[this.metric()] ?? EMPTY_STATISTIC;
  }
  constructor() {
    super();
    effect((onCleanup) => {
      const panel = this.kv();
      const context = panel?.context();
      const range = this.range();
      const head = panel?.head();
      const heads = panel?.aggregationHeads();
      const layers = this.analysisKvLayers();
      const alignment = panel?.positionAlignment();
      const capture = this.state.captureId();
      const read = (kind: KvCacheKind) =>
        context && !panel?.tensorView()
          ? (signal: AbortSignal) =>
              this.api.kvRange(
                capture,
                this.panel().turn,
                context.id,
                range.start,
                range.end,
                1,
                kind,
                head ?? 'min',
                signal,
                alignment,
                heads,
                [...layers].sort((a, b) => a - b),
              )
          : null;
      onCleanup(this.keyData.load(read('key')));
      onCleanup(this.valueData.load(read('value')));
    });
    inject(DestroyRef).onDestroy(() => {
      this.keyData.destroy();
      this.valueData.destroy();
    });
  }
}
