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
import {Component, computed, input, linkedSignal, output} from '@angular/core';
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
import {metricDisplayRules} from '../../widgets/format/format';
import {stepRows} from '../../widgets/format/step_labels';
import type {KvMetric, KvRangeMetricSummary, KvRangeRow} from '../../server/contracts/kv';
import {reasonText} from '../../server/reasons';
import type {RangeSelection} from '../../widgets/charts/range_selection';
import {AnalysisDetails, type StatisticsSection} from '../shared/analysis_details';
import {ElementStatistics} from '../shared/element_statistics';
import {MetricStatistics} from '../shared/metric_statistics';
import {
  sortTensorValueRows,
  type TensorValueSortColumn,
  type TensorValueSortDirection,
} from '../shared/tensor_value_sort';
import type {GraphMetricPanel} from './graph_metric_panel';
import {tensorPointLabel} from './tensor_labels';

/** What a statistics row shows where the server gave no statistic. */
const NO_STATISTIC: KvRangeMetricSummary = {
  min: null,
  max: null,
  mean: null,
  min_position: null,
  max_position: null,
  valid_count: 0,
};

function isKvMetric(metric: string): metric is KvMetric {
  return (
    metric === 'relative_l2' ||
    metric === 'max_abs' ||
    metric === 'cosine_distance' ||
    metric === 'cosine_similarity'
  );
}

/**
 * The details of an Intermediate Tensors panel: in Tensor Comparison, the metric's statistics
 * by layer and its metadata; in Tensor Inspection, the inspected tensor's statistics, metadata
 * and values; in the Layer Graph, the subgraph and the panel's own details.
 */
@Component({
  selector: 'intermediate-tensors-details',
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
  templateUrl: './intermediate_tensors_details.ng.html',
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
export class IntermediateTensorsDetails extends AnalysisDetails {
  readonly graph = input<GraphMetricPanel>();
  readonly graphSelection = input<RangeSelection | null>(null);
  readonly locateView = output<void>();
  readonly locateGraph = output<void>();
  readonly locateAggregation = output<void>();
  readonly metric = computed(() => this.graph()?.metric() ?? this.panel().logit?.metric ?? 'kl');
  protected readonly metricAnalysis = () => 'intermediate_tensors';
  protected override metricColor(value: number): string | null {
    return this.graph()?.color(value) ?? null;
  }
  protected override dashesMissing(): boolean {
    return true;
  }
  override statisticsEmptyText(_section: StatisticsSection): string {
    return '-';
  }
  locateDefinition(item: ExplorerInfoItem): void {
    if (item.label === 'Name') {
      this.locateAnalysis.emit();
    } else if (item.label === 'Mode') {
      this.locateView.emit();
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
    return this.graph()?.view() === 'heatmap' ? 'Heatmap' : 'Line chart';
  }
  locateStatistic(_label: string): void {
    this.locateGraph.emit();
  }
  /** A sync point's label: the server's, through the panel. */
  pointLabel(anchor: string | null | undefined): string {
    return this.graph()?.anchorLabel(anchor) ?? tensorPointLabel(anchor);
  }
  readonly activationInspectionMode = computed(() => !!this.graph()?.inspectionMode?.());
  readonly activationInspection = computed(() =>
    this.activationInspectionMode() ? this.graph()?.inspection() : undefined,
  );
  readonly activationInspectionData = computed(() => {
    const source = this.activationInspection();
    return source && !source.loading() && !source.error() ? source.data() : null;
  });
  readonly activationSelectedRange = computed(() => {
    const source = this.activationInspection();
    const range = source?.channelSelection();
    const index = source?.selectedIndex();
    return range ?? (index == null ? null : {start: index, end: index + 1});
  });
  readonly activationInspectionInfoItems = computed<ExplorerInfoItem[]>(() => {
    const source = this.activationInspection();
    return [
      {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
      {label: 'Mode', value: 'Tensor Inspection', locateLabel: 'Locate View control'},
      {
        label: 'Stage',
        value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`,
      },
      {
        label: 'Token position',
        value: String(source?.position() ?? '-'),
        locateLabel: 'Locate Token Position control',
      },
      ...(source ? stepRows(this.graph()?.columnSteps(source.position())) : []),
      {
        label: 'Tensor Index',
        value: `[Layer: ${source?.layer() ?? '-'}, Sync point: ${this.pointLabel(source?.anchor())}]`,
        locateLabel: 'Locate Tensor Index control',
        segments: [
          {text: '['},
          {
            text: 'Layer',
            help: {
              title: 'Layer',
              rows: [
                {label: 'Dimension', value: 'Transformer layer'},
                {label: 'Index', value: String(source?.layer() ?? '-')},
                {label: 'Indexing', value: 'Zero-based'},
              ],
            },
          },
          {text: `: ${source?.layer() ?? '-'}, `},
          {
            text: 'Sync point',
            help: {
              title: 'Sync point',
              rows: [
                {label: 'Dimension', value: 'Tensor capture anchor'},
                {label: 'Selection', value: this.pointLabel(source?.anchor())},
              ],
            },
          },
          {text: `: ${this.pointLabel(source?.anchor())}]`},
        ],
        expandable: true,
      },
    ];
  });
  locateActivationInspection(item: ExplorerInfoItem): void {
    const source = this.activationInspection();
    if (item.label === 'Name') {
      this.locateAnalysis.emit();
    } else if (item.label === 'Mode') {
      this.locateView.emit();
    } else if (item.label === 'Token position') {
      source?.locatePosition();
    } else if (item.label === 'Tensor Index') {
      source?.locateLayer();
    }
  }
  activationInspectionMetadata(side: 'ref' | 'target'): ExplorerInfoItem[] {
    const data = this.activationInspectionData();
    const source = side === 'ref' ? data?.reference : data?.target;
    return [
      {label: 'Dimensions', value: '[Element]'},
      {label: 'Shape', value: source ? `[${source.shape.join(', ')}]` : '-', expandable: true},
      {label: 'Dtype', value: source?.dtype ?? '-'},
      {
        label: 'Point',
        value: source ? `${source.point} · ${source.signature} step ${source.step}` : '-',
      },
    ];
  }
  readonly activationInspectionRows = computed(() => {
    const source = this.activationInspection();
    const compatible = source?.compatible() ?? false;
    return (source && !source.loading() && !source.error() ? source.channels() : []).map((row) => ({
      index: row.channel,
      label: String(row.channel),
      values: [
        row.ref,
        row.target,
        compatible ? row.delta : null,
        compatible ? row.abs_delta : null,
      ],
    }));
  });
  /** The server's statistics of the inspected tensor's elements, on each side and of Δ. */
  readonly activationInspectionStatistics = computed(() => {
    const statistics = this.activationInspectionData()?.statistics;
    return (
      [
        ['Reference', 'ref'],
        ['Target', 'target'],
        ['Δ', 'delta'],
        ['|Δ|', 'abs_delta'],
      ] as const
    ).map(([label, key]) => ({label, stats: statistics?.[key] ?? NO_STATISTIC}));
  });
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
  readonly sortedActivationRows = computed(() => {
    const sort = this.tensorValueSorts()['activations'] ?? {
      column: 'index',
      direction: 'ascending',
    };
    return sortTensorValueRows(this.activationInspectionRows(), sort.column, sort.direction);
  });
  readonly activationPageIndex = linkedSignal(() => {
    this.panel().id;
    const source = this.activationInspection();
    source?.position();
    source?.layer();
    source?.anchor();
    const selection = this.activationSelectedRange();
    const representative = source?.selectedIndex();
    const rows = this.sortedActivationRows();
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
    return Math.floor(Math.max(0, first) / this.tensorPageSize);
  });
  readonly activationInspectionPage = computed(() => {
    const rows = this.sortedActivationRows();
    const pages = Math.max(1, Math.ceil(rows.length / this.tensorPageSize));
    const index = Math.min(pages - 1, Math.max(0, this.activationPageIndex()));
    return {
      index,
      pages,
      rows: rows.slice(index * this.tensorPageSize, (index + 1) * this.tensorPageSize),
    };
  });
  changeActivationPage(delta: number): void {
    const page = this.activationInspectionPage();
    this.activationPageIndex.set(Math.max(0, Math.min(page.pages - 1, page.index + delta)));
  }
  sortActivationValues(column: TensorValueSortColumn): void {
    const direction =
      this.tensorValueSortDirection('activations', column) === 'ascending'
        ? 'descending'
        : 'ascending';
    this.tensorValueSorts.update((sorts) => ({...sorts, activations: {column, direction}}));
    this.activationPageIndex.set(0);
  }
  activationElementSelected(index: number): boolean {
    const selected = this.activationSelectedRange();
    return selected != null && index >= selected.start && index < selected.end;
  }
  override locateTensorStatistic(kind: string, index: number): void {
    if (kind === 'activations') {
      this.activationInspection()?.locateElement(index);
    }
  }
  readonly analysisGraphLayers = computed(() => {
    const graph = this.graph();
    const layers = (graph?.view() === 'lineChart' ? graph.selectedLayers() : graph?.layers()) ?? [];
    const selected = this.graphSelection()?.layer;
    return selected == null ? layers : layers.filter((layer) => layer === selected);
  });
  readonly graphArchInfoItems = computed<ExplorerInfoItem[]>(() => {
    const graph = this.graph();
    const graphPosition = graph?.graphPosition();
    const arch = graph?.archDetails().find((section) => section.label === 'Analysis Info');
    const mapping = graph?.archView?.()?.mapping();
    return [
      {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
      {label: 'Mode', value: 'Tensor Comparison', locateLabel: 'Locate View control'},
      {
        label: 'Stage',
        value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`,
      },
      {
        label: 'Token Position',
        value: String(graphPosition ?? '—'),
        locateLabel: 'Locate Token Position control',
      },
      ...(graph && graphPosition != null ? stepRows(graph.columnSteps(graphPosition)) : []),
      {
        label: 'Subgraph',
        value: mapping?.active
          ? mapping.title
          : (arch?.items.find((item) => item.label === 'Name')?.value ?? '—'),
        expandable: true,
      },
      // The layer's facts the server's skeleton gives, and the signature that ran for the column.
      ...(!mapping?.active
        ? (arch?.items ?? [])
            .filter((item) => ['Attention', 'K / V source', 'Signature'].includes(item.label))
            .map((item) => ({label: item.label, value: item.value}))
        : []),
      this.metricDefinition(),
      ...(!mapping?.active
        ? [
            {
              label: 'Summary',
              value: graph?.moduleSummary() === 'min' ? 'Min' : 'Max',
              locateLabel: 'Locate Summary control',
              segments: [
                {
                  text: graph?.moduleSummary() === 'min' ? 'Min' : 'Max',
                  help: {
                    title: 'Internal tensor summary',
                    rows: [{label: 'Scope', value: 'Internal tensors only; final output excluded'}],
                  },
                },
              ],
            },
          ]
        : []),
    ];
  });
  locateGraphArchDefinition(item: ExplorerInfoItem): void {
    if (item.label === 'Token Position') {
      this.graph()?.locateRange();
    } else if (item.label === 'Summary') {
      this.graph()?.locateSummary();
    } else {
      this.locateDefinition(item);
    }
  }
  readonly activationInfoItems = computed<ExplorerInfoItem[]>(() => [
    {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
    {label: 'Mode', value: 'Tensor Comparison', locateLabel: 'Locate View control'},
    {
      label: 'Stage',
      value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`,
    },
    {
      label: 'Token Range',
      value: `${this.range().start}–${this.range().end - 1}`,
      locateLabel: 'Locate Token Range control',
    },
    {
      label: 'Groups',
      value: `Transformer Layers [${this.analysisGraphLayers()
        .map((layer) => `L${layer}`)
        .join(', ')}]`,
      expandable: true,
      locateLabel: this.graph()?.view() === 'lineChart' ? 'Locate Groups control' : undefined,
    },
    this.metricDefinition(),
    this.aggregationDefinition(),
  ]);
  readonly activationMetricMetadata = computed<ExplorerInfoItem[]>(() => {
    const graph = this.graph();
    const available =
      graph && !graph.request.loading() && !graph.request.error() && graph.source().length > 0;
    // The metric output the server compared, not the captured tensors' dtype.
    const output = available ? graph.rangeRequest.data()?.metadata.output : undefined;
    return [
      {label: 'Dimensions', value: '[Token position, Layer]', expandable: true},
      {
        label: 'Shape',
        value: available
          ? `[${this.range().end - this.range().start}, ${this.analysisGraphLayers().length}]`
          : '-',
        expandable: true,
      },
      {label: 'Dtype', value: typeof output?.dtype === 'string' ? output.dtype : '-'},
      {
        label: 'Display precision',
        value: metricDisplayRules(this.metricDescription()).precision,
        expandable: true,
      },
    ];
  });
  readonly aggregationDefinition = computed<ExplorerInfoItem>(() => {
    const selectedAnchors = this.graph()?.selectedAnchors() ?? [];
    const choice =
      selectedAnchors.length === 1 ? selectedAnchors[0] : (this.graph()?.anchor() ?? 'min');
    return this.aggregationItem(
      choice,
      'Sync points',
      selectedAnchors.map((anchor) => this.pointLabel(anchor)),
    );
  });
  readonly sections = computed<StatisticsSection[]>(() => {
    const graph = this.graph();
    // Without its panel the details show the empty cache sections they always have.
    if (!graph) {
      return ['Key cache', 'Value cache'].map((label) => ({
        label,
        loading: false,
        error: '',
        rows: this.sort([], label),
      }));
    }
    // The server's statistics of each layer over the visible range, one bin each.
    const layers = (graph.view() === 'lineChart' ? graph.selectedLayers() : graph.layers()).filter(
      (layer) => this.graphSelection()?.layer == null || this.graphSelection()?.layer === layer,
    );
    const rows = graph.request.error()
      ? []
      : graph.totals().filter((row) => layers.includes(row.layer));
    return [
      {
        label: graph.label(),
        loading: graph.request.loading() || graph.chartPending(),
        error: graph.request.error() || graph.rangeRequest.error(),
        rows: this.sort(
          rows.map((row) => ({
            label: `Layer ${row.layer}`,
            layer: row.layer,
            kind: graph.label(),
            stats: this.stats(row),
            reason: reasonText(row.reasons),
          })),
          graph.label(),
        ),
      },
    ];
  });
  stats(row: KvRangeRow): KvRangeMetricSummary {
    const metric = this.metric();
    return (isKvMetric(metric) ? row.bins[0]?.metrics[metric] : undefined) ?? NO_STATISTIC;
  }
}
