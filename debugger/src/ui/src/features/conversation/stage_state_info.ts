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

import {SamplingDistributionInspection} from '../analysis/sampling_distribution_inspection';
import {tensorPointLabel} from '../analysis/tensor_labels';
import {ExplorerPagination} from '../../shared/explorer_pagination/explorer_pagination';
import {analysisMetricDisplayFormat} from '../../shared/format/format';
import {metricHelp} from '../analysis/metric_help';
import {
  kvInputMetadata,
  kvOutputMetadata,
  type KvLogicalMetadata,
} from '../kv/kv_panel/kv_analysis_metadata';
import type {KvTensorHead} from '../kv/kv_panel/kv_tensor_data';
import {KV_DETAIL_METRICS, type KvDetailMetric} from '../kv/kv_panel/kv_metric_values';
import type {KvRangeSelection} from '../kv/kv_panel/kv_range_types';
import {NgTemplateOutlet} from '@angular/common';
import {summarizeStageValues} from './stage_state_model';
import {logitSourceItems} from './logit_analysis_info';
import {samplingDistributionMetadata, samplingMetricMetadata} from './sampling_analysis_info';
import {
  sortTensorValueRows,
  type TensorValueSortColumn,
  type TensorValueSortDirection,
} from './tensor_value_sort';
import {FINAL_LOGIT_METRICS, isFinalLogitMetric} from './token_trends';
import type {FinalLogitsInspection} from '../analysis/final_logits_inspection';
import {TOKEN_METRICS} from './token_metric_metadata';
import type {AlignmentMode} from './token_alignment';
import {Component, computed, effect, inject, input, linkedSignal, output} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {ExplorerInfoValue} from '../../shared/explorer_info_value/explorer_info_value';
import {ExplorerLocator} from '../../shared/explorer_locator/explorer_locator';
import {ExplorerStatisticTable} from '../../shared/explorer_statistic_table/explorer_statistic_table';
import {
  ExplorerInfoList,
  type ExplorerInfoItem,
} from '../../shared/explorer_info_list/explorer_info_list';
import {ExplorerInfoSection} from '../../shared/explorer_info_section/explorer_info_section';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import {KvRequest} from '../kv/kv_panel/kv_data';
import type {KvRangeResponse, KvRangeRow, KvRangeMetricSummary} from '../../data/contracts/kv';
import type {TokenAnalysisPair} from '../../data/contracts/token_analysis';
import type {StageAnalysisPanel} from './stage_analysis';
import type {GraphMetricPanel} from './graph_metric_panel';
import type {KvPanel} from '../kv/kv_panel/kv_panel';
import {tensorRangeRows} from '../analysis/position_tensor_model';
import {formatAnalysisMetric} from '../../shared/format/format';

export interface StagePoint {
  position: number;
  index: number;
  hasRef: boolean;
  hasTarget: boolean;
  refPosition: number | null;
  targetPosition: number | null;
  analysis?: TokenAnalysisPair;
}
export interface StateLocation {
  position: number;
  layer: number | null;
  kind?: string;
  inspect: boolean;
  index?: number;
  head?: number;
  anchor?: string;
}
type StatisticSortColumn = 'layer' | 'mean' | 'min' | 'max';
interface StatisticSort {
  column: StatisticSortColumn | 'head' | KvDetailMetric;
  direction: 'ascending' | 'descending';
}
interface SummaryRow {
  label: string;
  layer: number | null;
  kind?: string;
  stats: KvRangeMetricSummary;
  reason?: string;
  children?: SummaryRow[];
  head?: number;
  anchor?: string;
}
@Component({
  selector: 'stage-state-info',
  host: {'[style.--info-inspector-identity-width]': 'tensorIdentityWidth() + "px"'},
  imports: [
    SamplingDistributionInspection,
    ExplorerPagination,
    ExplorerInfoList,
    ExplorerInfoValue,
    ExplorerStatisticTable,
    ExplorerInfoSection,
    ExplorerLocator,
    MatIconModule,
    NgTemplateOutlet,
  ],
  templateUrl: './stage_state_info.ng.html',
  styles: [
    `
      @use '../../theme/explorer_info' as info;
      @use '../../theme/token_diff' as token-diff;
      @include info.metadata;
      @include info.section-dividers;
      @include info.actions;
      @include info.statistic-table;
      :host {
        @include token-diff.activation-colors;
        container: analysis-info / inline-size;
        --info-property-gap: 0px;
        --me-font-family: Arial, Helvetica, sans-serif;
        --info-property-line-height: normal;
        --info-table-action-size: 14px;
        --info-table-action-gap: 1px;
        font-family: var(--me-font-family);
        letter-spacing: normal;
        display: block;
        min-width: 0;
        font-size: 12px;
      }
      th,
      td {
        overflow-wrap: anywhere;
        white-space: normal;
      }
      .representation-section {
        --info-action-spacer: 0 0 4px;
      }
      // Model Explorer info_panel: section attributes align with their heading.
      // The cache's nested sections add one 18px item-container indent. Inputs use
      // their metadata table 18px from the item origin, not 18px from its name.
      // Use that same 18px gutter for the index/name row to avoid the upstream
      // two-pixel difference between the name and the metadata table.
      .table-section {
        border-top: 0;
        --info-heading-color: var(--me-on-surface-low-color);
        --info-toggle-gap: 2px;
        --info-toggle-icon-size: 12px;
        --info-toggle-offset: 0px;
        --info-content-spacing: 3px;
        --info-item-gap: 0px;
      }
      .statistic-group explorer-info-section.table-section {
        // Property disclosures replace a metadata bullet without shifting its label.
        --info-toggle-offset: calc(14px - #{info.$bullet-gap} - #{info.$bullet-width} / 2);
        margin-left: calc(30px + #{info.$child-indent} - 26px);
        border-top: 0;
      }
      .table-section :is(.layer-table, .inspector-table, .statistic-table-note) {
        margin: 0 16px 0 26px;
        width: calc(100% - 42px);
        max-width: calc(100% - 42px);
      }
      .statistic-group {
        --info-compact-row-height: 16px;
        --info-property-line-height: normal;
      }
      .tensor-location-properties {
        --info-indent: calc(30px + #{info.$child-indent} - #{info.$bullet-text-indent});
        --info-key-offset: calc(#{info.$child-indent} - #{info.$bullet-text-indent});
        --info-property-gap: 0px;
      }
      .value-definition-properties,
      .selected-value-properties {
        --info-indent: 26px;
        --info-key-offset: #{info.$child-indent};
      }
      .metric-representation .statistic-group explorer-info-section.table-section {
        --info-heading-color: var(--me-on-surface-low-color);
        --info-content-spacing: 6px;
        --info-heading-size: 13px;
        --info-toggle-icon-size: 18px;
        margin-left: 4px;
        --info-toggle-offset: 0px;
      }
      .metric-representation .statistic-group explorer-info-section.value-definition {
        --info-heading-size: 12px;
      }
      .value-definition-properties {
        --info-indent: 26px;
        --info-key-offset: 0px;
      }
      .analysis-items {
        margin: 0 16px 0 12px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        font: 12px / normal var(--me-font-family);
      }
      .analysis-item {
        min-width: 0;
      }
      .item-heading {
        display: flex;
        align-items: flex-start;
        gap: 2px;
        min-height: 24px;
      }
      .item-index {
        flex: 0 0 16px;
        color: var(--me-on-surface-variant-color);
        line-height: 24px;
      }
      .item-name {
        color: var(--me-primary-color);
        line-height: 24px;
      }
      .analysis-properties {
        --info-indent: 18px;
        --info-key-offset: 0px;
        margin-right: 0;
      }
      .analysis-item .source-section {
        // Center the arrow over the single-digit index, keeping labels in the name column.
        margin-left: 0;
        --info-heading-color: var(--me-on-surface-low-color);
        --info-content-spacing: 2px;
        --info-content-bottom-spacing: 0px;
        --info-compact-row-height: 20px;
        --info-toggle-width: 18px;
        --info-toggle-offset: -6px;
        border-top: 0;
      }
      .source-section explorer-info-list {
        --info-indent: 18px;
        --info-key-offset: 0px;
        margin-right: 0;
      }
      .statistic-title {
        min-width: 0;
        flex: 1;
        line-height: 24px;
      }
      .statistic-summary {
        color: var(--me-on-surface-variant-color);
      }
      .statistic-title [explorer-locator] {
        vertical-align: middle;
      }
      .analysis-item > .layer-table,
      .analysis-item > .inspector-table,
      .analysis-item > .statistic-table-note {
        margin: 2px 0 0 18px;
        width: calc(100% - 18px);
        max-width: calc(100% - 18px);
      }
      .column-sort .sort-label {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .tensor-pagination {
        margin-left: auto;
        align-self: center;
      }
      .sampling-inspection-item {
        display: grid;
        grid-template-columns: 16px minmax(0, 1fr);
        column-gap: 2px;
      }
      .sampling-inspection-item > .item-heading {
        display: contents;
      }
      .sampling-details-table {
        grid-column: 2;
        margin-top: 2px;
        width: 100%;
      }
      .inspector-table {
        --info-table-identity-width: var(--info-inspector-identity-width, 48px);
      }
      .inspector-table th:first-child,
      .inspector-table td:first-child {
        --info-table-action-size: 12px;
        --info-table-action-gap: 0px;
        --explorer-help-icon-size: 12px;
        --explorer-help-gap: 0px;
      }
      .inspector-table tr.tensor-element-selected td,
      .inspector-table tr.tensor-element-selected:hover td,
      .inspector-table tr.tensor-head-selected td,
      .inspector-table tr.tensor-head-selected:hover td {
        background: var(--me-primary-container-color);
      }
      :is(.layer-table, .inspector-table) button.column-sort {
        display: grid;
        grid-template-columns: minmax(0, 1fr) var(--info-table-action-size);
        align-items: center;
        width: 100%;
        gap: var(--info-table-action-gap);
        padding: 0;
        text-align: inherit;
        font-weight: 400;
        border-radius: 3px;
        color: inherit;
      }
      :is(.layer-table, .inspector-table) th:not(:first-child),
      :is(.layer-table, .inspector-table) td:not(:first-child) {
        text-align: right;
      }
      .layer-table .stat-value {
        justify-content: flex-end;
      }
      :is(.layer-table, .inspector-table) .stat-locator [explorer-locator] {
        width: var(--info-table-action-size);
        flex-basis: var(--info-table-action-size);
      }
      .column-sort .sort-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: var(--info-table-action-size);
        height: 20px;
      }
      .column-sort mat-icon {
        width: 12px;
        height: 12px;
        font-size: 12px;
        line-height: 12px;
        color: inherit;
      }
      :is(.layer-table, .inspector-table)
        button.column-sort:not(:disabled):is(:hover, :focus-visible) {
        color: var(--me-on-surface-color);
      }
      .layer-table .stat-value {
        border-radius: 2px;
        color: var(--me-on-surface-color);
      }
      :is(.layer-table, .inspector-table) button.column-sort:disabled {
        opacity: 0.4;
        cursor: default;
        background: transparent;
      }
      explorer-info-value {
        flex: 1;
        min-width: 0;
        text-align: inherit;
      }
      .selection {
        color: var(--me-on-surface-variant-color);
        font-weight: 400;
      }
      .tensor-details {
        margin: 0 12px;
      }
      .note {
        margin: 4px 12px 4px var(--info-indent, 30px);
        color: var(--me-on-surface-variant-color);
        font: 12px / normal var(--me-font-family);
      }
      button:not([explorer-locator]) {
        border: 0;
        background: transparent;
        color: inherit;
        cursor: pointer;
        font: inherit;
        padding: 2px;
      }
      button:hover {
        background: var(--me-surface-container-high-color);
      }
      button:focus-visible {
        outline: 2px solid var(--me-primary-color);
      }
      .locate {
        float: right;
      }
      mat-icon {
        font-size: 16px;
        width: 16px;
        height: 16px;
        vertical-align: middle;
      }
      summary {
        cursor: pointer;
        padding: 6px 12px 6px 30px;
      }
      .position-link {
        color: var(--me-primary-color);
      }
    `,
  ],
})
export class StageStateInfo {
  readonly panel = input.required<StageAnalysisPanel>();
  readonly label = input.required<string>();
  readonly range = input.required<{start: number; end: number}>();
  readonly graph = input<GraphMetricPanel>();
  readonly alignment = input<AlignmentMode>('content');
  readonly kv = input<KvPanel>();
  readonly finalLogits = input<FinalLogitsInspection>();
  readonly kvSelection = input<(KvRangeSelection & {kind: string | null}) | null>(null);
  // Reset disclosure only when the inspected object changes, not when its metrics reload.
  private readonly kvInspectionSelection = computed(() => {
    const selected = this.kvSelection();
    return JSON.stringify([
      this.panel().id,
      selected?.start,
      selected?.end,
      selected?.layer,
      selected?.kind,
    ]);
  });
  readonly graphSelection = input<KvRangeSelection | null>(null);
  readonly statisticColumns = ['min', 'max', 'mean'] as const;
  readonly sortColumns = ['layer', ...this.statisticColumns] as const;
  private readonly statisticPanelId = computed(() => this.panel().id);
  readonly statisticSorts = linkedSignal<Record<string, StatisticSort>>(() => {
    this.statisticPanelId();
    return {};
  });
  sortState(group: string, defaultColumn: StatisticSort['column'] = 'layer'): StatisticSort {
    return this.statisticSorts()[group] ?? {column: defaultColumn, direction: 'ascending'};
  }
  sortDirection(
    group: string,
    column: StatisticSort['column'],
    defaultColumn: StatisticSort['column'] = 'layer',
  ) {
    const current = this.sortState(group, defaultColumn);
    return current.column === column ? current.direction : 'none';
  }
  sortAction(
    group: string,
    column: StatisticSort['column'],
    defaultColumn: StatisticSort['column'] = 'layer',
  ) {
    return `Sort ${group} by ${this.statisticLabel(column)} ${this.sortDirection(group, column, defaultColumn) === 'ascending' ? 'descending' : 'ascending'}`;
  }
  sortBy(
    group: string,
    column: StatisticSort['column'],
    defaultColumn: StatisticSort['column'] = 'layer',
  ) {
    const direction =
      this.sortDirection(group, column, defaultColumn) === 'ascending' ? 'descending' : 'ascending';
    this.statisticSorts.update((groups) => ({...groups, [group]: {column, direction}}));
  }
  statisticLabel(value: string) {
    if (value === 'layer') return 'Grouping';
    return (
      KV_DETAIL_METRICS.find((metric) => metric.key === value)?.label ??
      value.charAt(0).toUpperCase() + value.slice(1)
    );
  }
  readonly tensorSortColumns = ['head', ...KV_DETAIL_METRICS.map((metric) => metric.key)] as const;
  readonly tensorValueColumns = [
    'Reference value',
    'Target value',
    'Signed difference: Target − Reference',
    'Absolute difference: |Target − Reference|',
  ];
  readonly logitValueColumns = [
    {key: 'ref', label: 'Ref', title: 'Reference value'},
    {key: 'target', label: 'Target', title: 'Target value'},
    {key: 'delta', label: 'Δ', title: 'Signed difference: Target − Reference'},
    {key: 'absolute_delta', label: '|Δ|', title: 'Absolute difference: |Target − Reference|'},
  ] as const;
  readonly tensorValueSortColumns = [
    {key: 'index', label: 'Index', title: 'Element index'},
    ...this.logitValueColumns,
  ] as const;
  readonly tensorValueSorts = linkedSignal<
    Record<string, {column: TensorValueSortColumn; direction: TensorValueSortDirection}>
  >(() => {
    this.statisticPanelId();
    return {};
  });
  tensorValueSortDirection(kind: string, column: TensorValueSortColumn) {
    const sort = this.tensorValueSorts()[kind] ?? {column: 'index', direction: 'ascending'};
    return sort.column === column ? sort.direction : 'none';
  }
  sortTensorValues(kind: string, column: TensorValueSortColumn) {
    const pages = this.tensorPages();
    const direction =
      this.tensorValueSortDirection(kind, column) === 'ascending' ? 'descending' : 'ascending';
    this.tensorValueSorts.update((sorts) => ({...sorts, [kind]: {column, direction}}));
    this.tensorPages.set({...pages, [kind]: 0});
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
  selectLogitRow(event: KeyboardEvent, index: number) {
    if (
      event.target !== event.currentTarget ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      (event.key !== 'Enter' && event.key !== ' ')
    )
      return;
    event.preventDefault();
    this.finalLogits()?.selectElement(index);
  }
  readonly formatTensorValue = (value: number | string | null | undefined) =>
    typeof value === 'string'
      ? value
      : value != null && Number.isFinite(value)
        ? formatAnalysisMetric('raw', value)
        : '-';
  readonly tensorRuntimes = [
    {label: 'Reference', side: 'ref' as const},
    {label: 'Target', side: 'target' as const},
  ];
  tensorRows(rows: readonly KvTensorHead[], group: string) {
    const {column, direction} = this.sortState(group + ' heads', 'head');
    return [...rows].sort((a, b) => {
      const av = column === 'head' ? a.head : a.evidence?.metrics?.[column as KvDetailMetric];
      const bv = column === 'head' ? b.head : b.evidence?.metrics?.[column as KvDetailMetric];
      const aValid = av != null && Number.isFinite(av);
      const bValid = bv != null && Number.isFinite(bv);
      if (aValid !== bValid) return aValid ? -1 : 1;
      return (
        (aValid && bValid ? (av! - bv!) * (direction === 'ascending' ? 1 : -1) : 0) ||
        a.head - b.head
      );
    });
  }
  statistic(stats: KvRangeMetricSummary, value: string) {
    return stats[value as 'min' | 'max' | 'mean'];
  }
  statisticColor(value: number | null | undefined) {
    return value != null && Number.isFinite(value)
      ? (this.kv()?.color(value) ?? this.graph()?.color(value) ?? null)
      : null;
  }
  readonly fmtCompact = (value: number | null | undefined) =>
    (this.samplingInfo() || this.activationInfo()) && (value == null || !Number.isFinite(value))
      ? '-'
      : formatAnalysisMetric(this.metric(), value);
  metricValueTitle(value: number | null | undefined) {
    return value == null
      ? '—'
      : `${this.metric()}: ${value.toPrecision(17)}${this.metric() === 'relative_l2' ? ' (raw ratio)' : ''}`;
  }
  readonly visibleKvLayers = computed(() => {
    const kv = this.kv();
    return kv?.visibleMetricLayers() ?? [];
  });
  readonly points = input<StagePoint[]>([]);
  readonly tokenLoading = input(false);
  readonly tokenError = input('');
  readonly locateAnalysis = output<void>();
  readonly locateCache = output<'key' | 'value'>();
  readonly locateAggregation = output<void>();
  readonly locateMetric = output<void>();
  readonly locateView = output<void>();
  readonly locateLogitElement = output<number>();
  readonly locateTensorPosition = output<void>();
  readonly locateScope = output<'positions' | 'layers'>();
  locateDefinition(item: ExplorerInfoItem) {
    if (item.label === 'Name') this.locateAnalysis.emit();
    else if (item.label === 'Mode') {
      if (this.panel().kind === 'tokens' || this.samplingInfo() || this.activationInfo())
        this.locateView.emit();
      else this.kv()?.locateView();
    } else if (['Token Range', 'Positions', 'Selected position'].includes(item.label))
      this.locateScope.emit('positions');
    else if (item.label === 'Groups') this.locateScope.emit('layers');
    else if (item.label === 'Metric') this.locateMetric.emit();
    else if (item.label === 'Aggregation') this.locateAggregation.emit();
    else if (item.label === 'Token position') this.locateTensorPosition.emit();
  }
  readonly locateGraph = output<void>();
  readonly locateFinal = output<string>();
  statisticView() {
    return this.kv()?.view() === 'heatmap' || this.graph()?.view() === 'heatmap'
      ? 'Heatmap'
      : 'Line chart';
  }
  locateStatistic(label: string) {
    if (this.panel().kind === 'kv') this.locateCache.emit(label === 'Key cache' ? 'key' : 'value');
    else if (this.panel().kind === 'tokens' || this.panel().kind === 'distribution')
      this.locateFinal.emit(label);
    else this.locateGraph.emit();
  }
  readonly location = output<StateLocation>();
  private readonly api = inject(ReportApiService);
  private readonly state = inject(ReportStateService);
  readonly keyData = new KvRequest<KvRangeResponse>();
  readonly valueData = new KvRequest<KvRangeResponse>();
  readonly metric = computed(
    () =>
      this.graph()?.metric() ??
      this.kv()?.metric() ??
      this.panel().trend?.metric ??
      (this.panel().kind === 'tokens' ? 'logit_rmse' : this.samplingInfo() ? 'entropy' : 'kl'),
  );
  readonly metricLabel = computed(
    () =>
      this.graph()
        ?.metrics.find((m) => m.id === this.metric())
        ?.label.replace(/ \(%\)$/, '') ??
      this.kv()?.metricLabel() ??
      FINAL_LOGIT_METRICS.find((metric) => metric.key === this.metric())?.label ??
      (
        {
          kl: TOKEN_METRICS.find((metric) => metric.key === 'kl')!.label.replace('Logits · ', ''),
          js: TOKEN_METRICS.find((metric) => metric.key === 'js')!.label.replace('Logits · ', ''),
          entropy: 'Entropy',
          selected_probability: 'Selected probability',
          margin: 'Top-2 margin',
        } as Record<string, string>
      )[this.metric()] ??
      this.metric(),
  );
  readonly metricDefinition = computed<ExplorerInfoItem>(() => {
    const help = metricHelp(this.metric(), this.metricLabel());
    const samplingHelp =
      this.samplingInfo() && help
        ? {
            ...help,
            rows: [
              ...help.rows,
              {
                label: 'Inputs',
                value: 'Final sampling probabilities after configured sampling rules',
              },
              {label: 'Series', value: 'Each runtime uses its own distribution and selected token'},
            ],
          }
        : help;
    return {
      label: 'Metric',
      value: this.metricLabel(),
      segments: [{text: this.metricLabel(), help: samplingHelp}],
      locateLabel: 'Locate Metric control',
      expandable: true,
      wrap: this.panel().kind === 'tokens' || this.samplingInfo(),
    };
  });
  readonly kvMetricInfo = computed(() => this.panel().kind === 'kv' && !this.kv()?.tensorView());
  readonly activationInfo = computed(() => ['activations', 'graph'].includes(this.panel().kind));
  readonly activationInspectionMode = computed(
    () => this.activationInfo() && !!this.graph()?.inspectionMode?.(),
  );
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
      {
        label: 'Tensor Index',
        value: `[Layer: ${source?.layer() ?? '-'}, Sync point: ${tensorPointLabel(source?.anchor())}]`,
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
                {label: 'Selection', value: tensorPointLabel(source?.anchor())},
              ],
            },
          },
          {text: `: ${tensorPointLabel(source?.anchor())}]`},
        ],
        expandable: true,
      },
    ];
  });
  locateActivationInspection(item: ExplorerInfoItem) {
    const source = this.activationInspection();
    if (item.label === 'Name') this.locateAnalysis.emit();
    else if (item.label === 'Mode') this.locateView.emit();
    else if (item.label === 'Token position') source?.locatePosition();
    else if (item.label === 'Tensor Index') source?.locateLayer();
  }
  activationInspectionMetadata(side: 'ref' | 'target'): ExplorerInfoItem[] {
    const data = this.activationInspectionData();
    const source = side === 'ref' ? data?.reference : data?.target;
    return [
      {label: 'Dimensions', value: '[Element]'},
      {label: 'Shape', value: source ? `[${source.shape.join(', ')}]` : '-', expandable: true},
      {label: 'Dtype', value: source?.dtype ?? '-'},
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
  readonly activationInspectionStatistics = computed(() =>
    ['Reference', 'Target', 'Δ', '|Δ|'].map((label, index) => ({
      label,
      stats: summarizeStageValues(
        this.activationInspectionRows().map((row) => ({
          position: row.index,
          value: typeof row.values[index] === 'number' ? (row.values[index] as number) : null,
        })),
      ),
    })),
  );
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
  changeActivationPage(delta: number) {
    const page = this.activationInspectionPage();
    this.activationPageIndex.set(Math.max(0, Math.min(page.pages - 1, page.index + delta)));
  }
  sortActivationValues(column: TensorValueSortColumn) {
    this.sortTensorValues('activations', column);
    this.activationPageIndex.set(0);
  }
  activationElementSelected(index: number) {
    const selected = this.activationSelectedRange();
    return selected != null && index >= selected.start && index < selected.end;
  }
  readonly analysisGraphLayers = computed(() => {
    const graph = this.graph();
    const layers = (graph?.view() === 'trends' ? graph.selectedLayers() : graph?.layers()) ?? [];
    const selected = this.graphSelection()?.layer;
    return selected == null ? layers : layers.filter((layer) => layer === selected);
  });
  readonly graphArchInfoItems = computed<ExplorerInfoItem[]>(() => {
    const graph = this.graph();
    const arch = graph?.archDetails().find(section => section.label === 'Analysis Info');
    const mapping = graph?.archView?.()?.mapping();
    return [
      {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
      {label: 'Mode', value: 'Tensor Comparison', locateLabel: 'Locate View control'},
      {label: 'Stage', value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`},
      {label: 'Token Position', value: String(graph?.graphPosition() ?? '—'), locateLabel: 'Locate Token Position control'},
      {label: 'Subgraph', value: mapping?.active ? mapping.title : arch?.items.find(item => item.label === 'Name')?.value ?? '—', expandable: true},
      this.metricDefinition(),
      ...(!mapping?.active ? [{label: 'Summary', value: graph?.moduleSummary() === 'min' ? 'Min' : 'Max', locateLabel: 'Locate Summary control',
        segments: [{text: graph?.moduleSummary() === 'min' ? 'Min' : 'Max', help: {
          title: 'Internal tensor summary', rows: [{label: 'Scope', value: 'Internal tensors only; final output excluded'}],
        }}]}] : []),
    ];
  });
  locateGraphArchDefinition(item: ExplorerInfoItem) {
    if (item.label === 'Token Position') this.graph()?.locateRange();
    else if (item.label === 'Summary') this.graph()?.locateSummary();
    else this.locateDefinition(item);
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
      locateLabel: this.graph()?.view() === 'trends' ? 'Locate Groups control' : undefined,
    },
    this.metricDefinition(),
    this.aggregationDefinition(),
  ]);
  readonly activationMetricMetadata = computed<ExplorerInfoItem[]>(() => {
    const graph = this.graph();
    const available =
      graph && !graph.request.loading() && !graph.request.error() && graph.source().length > 0;
    return [
      {label: 'Dimensions', value: '[Token position, Layer]', expandable: true},
      {
        label: 'Shape',
        value: available
          ? `[${this.range().end - this.range().start}, ${this.analysisGraphLayers().length}]`
          : '-',
        expandable: true,
      },
      // position_analysis computes comparison metrics from float64 vectors.
      // This describes metric output, not the captured tensor's source dtype.
      {label: 'Dtype', value: available ? 'float64' : '-'},
      {
        label: 'Display precision',
        value: analysisMetricDisplayFormat(this.metric()).precision,
        expandable: true,
      },
    ];
  });
  readonly samplingInfo = computed(() => this.panel().kind === 'distribution');
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
  readonly samplingSinglePosition = computed(() => this.range().end - this.range().start === 1);
  samplingInspectionSource(side: 'ref' | 'target') {
    if (!this.samplingSinglePosition()) return undefined;
    return this.points().find((point) => this.sideInRange(point, side))?.analysis
      ?.sampling_distribution;
  }
  samplingSourceItems(side: 'ref' | 'target') {
    return samplingDistributionMetadata(
      this.points().filter((point) => this.sideInRange(point, side)),
      side,
    );
  }
  readonly samplingMetricMetadata = computed(() =>
    samplingMetricMetadata(
      this.metric() as 'entropy' | 'selected_probability' | 'margin',
      this.range(),
      this.alignment(),
    ),
  );
  readonly logitInspection = computed(
    () => this.panel().kind === 'tokens' && this.panel().trend?.logitView === 'inspection',
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
      )
        return previous.value;
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
  readonly logitInfoItems = computed<ExplorerInfoItem[]>(() => [
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
    this.metricDefinition(),
  ]);
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
        this.finalLogits()?.position() ?? this.panel().trend?.logitPosition ?? this.range().start,
      ),
      locateLabel: 'Locate Token Position control',
    },
  ]);
  readonly logitInspectionData = computed(() => {
    const source = this.finalLogits();
    return source && !source.loading() && !source.error() ? source.data() : null;
  });
  private readonly logitInspectionCompatible = computed(
    () => this.logitInspectionData()?.compatible ?? false,
  );
  readonly logitInspectionStatistics = computed(() => {
    const source = this.finalLogits();
    const elements = source && !source.loading() && !source.error() ? source.channels() : [];
    const compatible = this.logitInspectionCompatible();
    return (['ref', 'target', 'delta', 'abs_delta'] as const).map((key, index) => ({
      label: ['Reference', 'Target', 'Δ', '|Δ|'][index],
      stats: summarizeStageValues(
        elements.map((element) => {
          const value = element[key];
          return {
            position: element.channel,
            value: (index < 2 || compatible) && typeof value === 'number' ? value : null,
          };
        }),
      ),
    }));
  });
  locateTensorStatistic(kind: string, index: number) {
    if (kind === 'logits') this.locateLogitElement.emit(index);
    else if (kind === 'activations') this.activationInspection()?.locateElement(index);
    else if (kind === 'key' || kind === 'value') this.kv()?.locateTensorElement(kind, index);
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
  logitSourceItems(side: 'ref' | 'target') {
    return logitSourceItems(this.points(), side);
  }
  readonly logitMetricMetadata = computed(() =>
    this.metadataItems(
      {dimensions: 'Position', shape: `[${this.points().length}]`, dtype: 'float64', layers: []},
      true,
    ),
  );
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
  readonly kvMetadata = computed(() =>
    ['key', 'value'].map((kind) => {
      const rows = (this.kv()?.context()?.layers ?? [])
        .filter((row) => row.kind === kind && this.analysisKvLayers().includes(row.layer))
        .sort((a, b) => a.layer - b.layer);
      const scope = {
        range: this.range(),
        alignment: this.kv()?.positionAlignment(),
        heads: this.kv()?.selectedHeads() ?? [],
      };
      return {
        kind,
        label: kind === 'key' ? 'Key cache' : 'Value cache',
        sources: (['ref', 'target'] as const).map((side) => ({
          label: side === 'ref' ? 'Reference' : 'Target',
          ...kvInputMetadata(rows, side, scope),
        })),
        output: kvOutputMetadata(rows, scope),
      };
    }),
  );
  readonly valueDefinition = computed<ExplorerInfoItem[]>(() => [
    this.metricDefinition(),
    {label: 'Index', value: 'Token position'},
    {
      label: 'Grouping',
      value: ['kv', 'graph', 'activations'].includes(this.panel().kind) ? 'Layer' : '—',
    },
    this.aggregationDefinition(),
  ]);
  readonly aggregationDefinition = computed<ExplorerInfoItem>(() => {
    const kind = this.panel().kind;
    if (!['kv', 'graph', 'activations'].includes(kind)) return {label: 'Aggregation', value: '-'};
    const selectedAnchors = this.graph()?.selectedAnchors() ?? [];
    const choice =
      kind === 'kv'
        ? (this.kv()?.head() ?? 'min')
        : selectedAnchors.length === 1
          ? selectedAnchors[0]
          : (this.graph()?.anchor() ?? 'min');
    const functions: Record<string, string> = {min: 'Minimum', max: 'Maximum', mean: 'Mean'};
    const dimension = kind === 'kv' ? 'Heads' : 'Sync points';
    const selection =
      kind === 'kv' ? (this.kv()?.selectedHeads() ?? []) : selectedAnchors.map(tensorPointLabel);
    const domain = `${dimension} [${selection.join(', ')}]`;
    return {
      label: 'Aggregation',
      value: choice in functions ? `${functions[choice]} · ${domain}` : domain,
      segments: [
        ...(choice in functions
          ? [
              {
                text: functions[choice],
                help: {
                  title: functions[choice],
                  rows: [
                    {label: 'Input', value: `Metric values of selected ${dimension.toLowerCase()}`},
                    {label: 'Order', value: `Metric → ${functions[choice]}`},
                    {label: 'Output', value: 'One value per token position and layer'},
                  ],
                },
              },
              {text: ' · '},
            ]
          : []),
        {
          text: domain,
          help: {
            title: dimension,
            rows: [
              {label: 'Selection', value: `[${selection.join(', ')}]`},
              {label: 'Dimension', value: kind === 'kv' ? 'KV attention heads' : 'Sync points'},
              {label: 'Participation', value: 'Present in both Reference and Target'},
            ],
          },
        },
      ],
      locateLabel: kind === 'kv' ? 'Locate Heads control' : 'Locate Sync points control',
      expandable: true,
    };
  });
  metadataItems(metadata: KvLogicalMetadata, output = false): ExplorerInfoItem[] {
    const dimensions = metadata.dimensions
      .split(' × ')
      .map((axis) => (axis === 'Position' ? 'Token position' : axis));
    const sizes = metadata.shape.startsWith('[')
      ? metadata.shape.slice(1, -1).split(', ')
      : dimensions.map(() => '-');
    const range = this.range();
    const input = dimensions.includes('Head');
    const dimensionSegments = dimensions.flatMap((text, index) => {
      const size = sizes[index] ?? '-';
      const rows =
        text === 'Token position'
          ? [
              {
                label: 'Alignment',
                value: this.logitMetricInfo()
                  ? this.alignment() === 'content'
                    ? 'Context'
                    : 'Steps'
                  : this.kv()?.positionAlignment()
                    ? 'Context'
                    : 'Steps',
              },
              {label: 'Range', value: `${range.start}–${range.end - 1}`},
              {label: 'Count', value: size},
              {
                label: 'Counting',
                value: input
                  ? 'Stored positions; alignment gaps excluded'
                  : 'All selected aligned positions',
              },
            ]
          : text === 'Layer'
            ? [
                {label: 'Dimension', value: 'Transformer layers'},
                {
                  label: 'Selection',
                  value: `[${metadata.layers.map((layer) => layer.label).join(', ')}]`,
                },
                {label: 'Count', value: size},
              ]
            : text === 'Head'
              ? [
                  {label: 'Dimension', value: 'KV attention heads'},
                  {label: 'Count', value: size},
                  {label: 'Scope', value: 'Selected heads available in each layer'},
                ]
              : [
                  {label: 'Dimension', value: 'Components in one KV head vector'},
                  {label: 'Size', value: size},
                ];
      return [{text: index ? ', ' : '['}, {text, help: {title: text, rows}}];
    });
    dimensionSegments.push({text: ']'});
    return [
      {
        label: 'Dimensions',
        value: `[${dimensions.join(', ')}]`,
        expandable: true,
        segments: dimensionSegments,
      },
      {
        label: 'Shape',
        value: metadata.shape,
        expandable: true,
        segments: metadata.shape.startsWith('[')
          ? [
              ...sizes.flatMap((text, index) => [
                {text: index ? ', ' : '['},
                {
                  text,
                  help:
                    text === 'Mixed'
                      ? {
                          title: 'Mixed',
                          rows: [
                            {label: 'Dimension', value: dimensions[index]},
                            {label: 'Meaning', value: 'Sizes differ across selected layers'},
                          ],
                        }
                      : undefined,
                },
              ]),
              {text: ']'},
            ]
          : undefined,
      },
      {label: 'Dtype', value: metadata.dtype},
      ...(output
        ? [
            {
              label: 'Display precision',
              value: analysisMetricDisplayFormat(this.metric()).precision,
              expandable: true,
            },
          ]
        : []),
    ];
  }
  readonly inspectionInfoItems = computed<ExplorerInfoItem[]>(() => {
    const source = this.kv();
    return [
      {label: 'Name', value: this.label(), locateLabel: 'Locate analysis view'},
      {label: 'Mode', value: 'Tensor Inspection', locateLabel: 'Locate View control'},
      {
        label: 'Stage',
        value: `Turn ${this.panel().turn} · ${this.panel().stage === 'prefill' ? 'Prefill' : 'Decode'}`,
      },
      {
        label: 'Token position',
        value: String(source?.tensorPosition() ?? '-'),
        locateLabel: 'Locate Token Position control',
      },
      {
        label: 'Tensor Index',
        locateLabel: 'Locate Tensor Index control',
        value: `[Layer: ${source?.tensorLayer() ?? '-'}, Head: ${source?.tensorHead() ?? '-'}]`,
        segments: [
          {text: '['},
          {
            text: 'Layer',
            help: {
              title: 'Layer',
              rows: [
                {label: 'Dimension', value: 'Transformer layer'},
                {label: 'Index', value: String(source?.tensorLayer() ?? '-')},
                {label: 'Indexing', value: 'Zero-based'},
              ],
            },
          },
          {text: `: ${source?.tensorLayer() ?? '-'}, `},
          {
            text: 'Head',
            help: {
              title: 'Head',
              rows: [
                {label: 'Dimension', value: 'KV attention head'},
                {label: 'Index', value: String(source?.tensorHead() ?? '-')},
                {label: 'Layer', value: String(source?.tensorLayer() ?? '-')},
                {label: 'Indexing', value: 'Zero-based within this layer'},
              ],
            },
          },
          {text: `: ${source?.tensorHead() ?? '-'}]`},
        ],
        expandable: true,
      },
    ];
  });
  locateInspection(item: ExplorerInfoItem) {
    if (item.label === 'Name') this.locateAnalysis.emit();
    else if (item.label === 'Mode') this.kv()?.locateView();
    else if (item.label === 'Token position') this.kv()?.locateTensorPosition();
    else if (item.label === 'Tensor Index') this.kv()?.locateTensorIndex();
  }
  inspectionMetadata(kind: string, side: 'ref' | 'target'): ExplorerInfoItem[] {
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
  readonly tensorStatistics = computed(() =>
    Object.fromEntries(
      Object.entries(this.tensorInspectionRows()).map(([kind, elements]) => [
        kind,
        ['Reference', 'Target', 'Δ', '|Δ|'].map((label, index) => ({
          label,
          stats: summarizeStageValues(
            elements.map((element) => ({
              position: element.index,
              value:
                typeof element.values[index] === 'number'
                  ? (element.values[index] as number)
                  : null,
            })),
          ),
        })),
      ]),
    ),
  );
  readonly tensorPageSize = 16;
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
  tensorInspectionPage(kind: string) {
    const rows = this.sortedTensorValueRows()[kind] ?? [];
    const pages = Math.max(1, Math.ceil(rows.length / this.tensorPageSize));
    const index = Math.min(pages - 1, Math.max(0, this.tensorPages()[kind] ?? 0));
    return {
      index,
      pages,
      rows: rows.slice(index * this.tensorPageSize, (index + 1) * this.tensorPageSize),
    };
  }
  changeTensorPage(kind: string, delta: number) {
    const page = this.tensorInspectionPage(kind);
    const index = Math.max(0, Math.min(page.pages - 1, page.index + delta));
    this.tensorPages.update((pages) => ({...pages, [kind]: index}));
  }
  tensorElementSelected(index: number) {
    const range = this.kv()?.tensorSelectedRange();
    return range != null && index >= range.start && index < range.end;
  }
  readonly logitValues = computed(() =>
    this.points().map((p) => ({
      ...p,
      value: isFinalLogitMetric(this.metric())
        ? (p.analysis?.final_logits?.raw[this.metric() === 'logit_max_abs' ? 'max_abs' : 'rmse']
            .value ?? null)
        : (p.analysis?.metrics[this.metric() as 'kl' | 'js']?.value ?? null),
    })),
  );
  readonly logitStats = computed(() => summarizeStageValues(this.logitValues()));
  sideValue(point: StagePoint, side: 'ref' | 'target') {
    if (!this.sideInRange(point, side)) return null;
    const value =
      point.analysis?.sampling_distribution?.[side][
        this.metric() as 'entropy' | 'margin' | 'selected_probability'
      ];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }
  sideInRange(point: StagePoint, side: 'ref' | 'target') {
    const position = side === 'ref' ? point.refPosition : point.targetPosition;
    return position !== null && position >= this.range().start && position < this.range().end;
  }
  sideStats(side: 'ref' | 'target') {
    return summarizeStageValues(
      this.points()
        .filter((p) => this.sideInRange(p, side))
        .map((p) => ({
          position: (side === 'ref' ? p.refPosition : p.targetPosition)!,
          value: this.sideValue(p, side),
        })),
    );
  }
  readonly sections = computed(() => {
    if (this.samplingInfo()) {
      const label = 'Sampling distribution';
      return [
        {
          label,
          loading: this.tokenLoading(),
          error: this.tokenError(),
          rows: this.sort(
            [
              {label: 'Ref', layer: null, kind: 'ref', stats: this.sideStats('ref')},
              {label: 'Target', layer: null, kind: 'target', stats: this.sideStats('target')},
            ],
            label,
          ),
        },
      ];
    }
    if (this.panel().kind === 'tokens') {
      const groups = [{label: this.logitGroupLabel(), kind: 'logits', stats: this.logitStats()}];
      return groups.map((group) => ({
        label: group.label,
        loading: this.tokenLoading(),
        error: this.tokenError(),
        rows: [{label: 'Final', layer: null, kind: group.kind, stats: group.stats}] as SummaryRow[],
      }));
    }
    const graph = this.graph();
    if (graph) {
      const rows = tensorRangeRows(
        this.activationInfo() && graph.request.error() ? [] : graph.metricSource(),
        (graph.view() === 'trends' ? graph.selectedLayers() : graph.layers()).filter(
          (layer) => this.graphSelection()?.layer == null || this.graphSelection()?.layer === layer,
        ),
        [this.range()],
        graph.anchor(),
        graph.label(),
      );
      return [
        {
          label: graph.label(),
          loading: graph.request.loading(),
          error: graph.request.error(),
          rows: this.sort(
            rows.map((row) => ({
              label: `Layer ${row.layer}`,
              layer: row.layer,
              kind: graph.label(),
              stats: this.stats(row),
              reason: row.reason,
            })),
            graph.label(),
          ),
        },
      ];
    }
    return [
      {label: 'Key cache', data: this.keyData.data(), kind: 'key', request: this.keyData},
      {label: 'Value cache', data: this.valueData.data(), kind: 'value', request: this.valueData},
    ].map((section) => ({
      label: section.label,
      loading: !!this.kv()?.loading() || section.request.loading(),
      error: section.request.error(),
      rows: this.sort(
        (section.data?.rows ?? [])
          .filter(
            (r) =>
              this.visibleKvLayers().includes(r.layer) && this.analysisKvLayers().includes(r.layer),
          )
          .map((r) => ({
            label: `Layer ${r.layer}`,
            layer: r.layer,
            kind: section.kind,
            stats: this.stats(r),
            reason: r.reason,
          })),
        section.label,
      ),
    }));
  });
  stats(row: KvRangeRow) {
    return row.bins[0]?.metrics[this.metric() as 'cosine_similarity'] ?? summarizeStageValues([]);
  }
  sort(rows: SummaryRow[], group: string) {
    const {column, direction} = this.sortState(group);
    return rows.sort((a, b) => {
      if (this.samplingInfo() && column === 'layer')
        return a.label.localeCompare(b.label) * (direction === 'ascending' ? 1 : -1);
      const av = column === 'layer' ? a.layer : a.stats[column as 'mean' | 'min' | 'max'];
      const bv = column === 'layer' ? b.layer : b.stats[column as 'mean' | 'min' | 'max'];
      // Missing values stay last in either direction; ties remain in layer order.
      const aValid = av != null && Number.isFinite(av);
      const bValid = bv != null && Number.isFinite(bv);
      if (aValid !== bValid) return aValid ? -1 : 1;
      const difference = aValid && bValid ? av! - bv! : 0;
      return difference * (direction === 'ascending' ? 1 : -1) || (a.layer ?? 0) - (b.layer ?? 0);
    });
  }
  goPosition(row: SummaryRow, position: number, inspect: boolean) {
    this.location.emit({
      position,
      layer: row.layer,
      kind: row.kind,
      head: row.head,
      anchor: row.anchor,
      inspect,
    });
  }
  constructor() {
    effect((onCleanup) => {
      const panel = this.kv(),
        context = panel?.context(),
        range = this.range(),
        head = panel?.head(),
        heads = panel?.aggregationHeads(),
        alignment = panel?.positionAlignment(),
        capture = this.state.captureId();
      const read = (kind: string) =>
        context && !panel?.tensorView() && this.panel().kind === 'kv'
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
                '',
                signal,
                alignment,
                heads,
              )
          : null;
      onCleanup(this.keyData.load(read('key')));
      onCleanup(this.valueData.load(read('value')));
    });
  }
}
