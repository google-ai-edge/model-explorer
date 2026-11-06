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

import {
  ChangeDetectionStrategy,
  Component,
  Directive,
  computed,
  inject,
  input,
  linkedSignal,
  output,
} from '@angular/core';
import type {ExplorerInfoItem} from '../../widgets/explorer_info_list/explorer_info_list';
import {ExplorerInfoValue} from '../../widgets/explorer_info_value/explorer_info_value';
import {ExplorerLocator} from '../../widgets/explorer_locator/explorer_locator';
import {
  formatMetricValue,
  formatTensorValue as formatTensor,
  metricDisplayRules,
} from '../../widgets/format/format';
import {MetricCatalogStore} from '../../server/metric_catalog';
import type {
  ElementStatistics,
  KvHeadMetric,
  KvRangeMetricSummary,
  LogicalTensor,
  ShapeAxis,
} from '../../server/contracts/kv';
import type {TokenAnalysisPair} from '../../server/contracts/token_analysis';
import {catalogMetricId, type LogitPanelState} from './logit_panel_state';
import {metricHelp} from './metric_help';

/** One aligned position of the details' range, with each side's own position. */
export interface StagePoint {
  position: number;
  index: number;
  hasRef: boolean;
  hasTarget: boolean;
  refPosition: number | null;
  targetPosition: number | null;
  analysis?: TokenAnalysisPair;
}
/** A place in the analysis panel that a details row locates. */
export interface StateLocation {
  position: number;
  layer: number | null;
  kind?: string;
  inspect: boolean;
  index?: number;
  head?: number;
  anchor?: string;
}
/** The analysis panel the details describe, as far as a details body reads it. */
export interface DetailsPanel {
  id: string;
  turn: number;
  stage: 'prefill' | 'decode';
  kind: string;
  logit?: LogitPanelState;
}
/** The statistic of no value: what a Statistics row shows while nothing is loaded. */
export const EMPTY_STATISTIC: KvRangeMetricSummary = {
  mean: null,
  min: null,
  max: null,
  min_position: null,
  max_position: null,
  valid_count: 0,
};
/** The rows of an inspected tensor's Statistics table: each side's and their difference's, as the
 *  server gives them; empty rows without statistics. */
export function elementStatisticRows(
  statistics: ElementStatistics | null,
): {label: string; stats: KvRangeMetricSummary}[] {
  return (['ref', 'target', 'delta', 'abs_delta'] as const).map((key, index) => ({
    label: ['Reference', 'Target', 'Δ', '|Δ|'][index],
    stats: statistics?.[key] ?? EMPTY_STATISTIC,
  }));
}
/** How a Metadata Shape writes one axis: its size, Mixed where the layers' differ, a dash where a
 *  layer has none. */
function axisText(axis: ShapeAxis): string {
  return axis === null ? '—' : Array.isArray(axis) ? 'Mixed' : String(axis);
}
type StatisticSortColumn = 'layer' | 'mean' | 'min' | 'max';
interface StatisticSort {
  column: StatisticSortColumn | 'head' | KvHeadMetric;
  direction: 'ascending' | 'descending';
}
/** One row of a Statistics table: a group of the metric's values and their summary. */
export interface SummaryRow {
  label: string;
  layer: number | null;
  kind?: string;
  stats: KvRangeMetricSummary;
  reason?: string;
  children?: SummaryRow[];
  head?: number;
  anchor?: string;
}
/** One Statistics table of a details body. */
export interface StatisticsSection {
  label: string;
  loading: boolean;
  error: string;
  rows: SummaryRow[];
}

/** What every analysis' body of the details panel shares: the analysis it describes, its metric
 *  as the server's catalog describes it, the sort of its Statistics tables and how their values
 *  read. Each analysis' body extends it with its own sections. */
@Directive()
export abstract class AnalysisDetails {
  readonly panel = input.required<DetailsPanel>();
  readonly label = input.required<string>();
  readonly range = input.required<{start: number; end: number}>();
  readonly locateAnalysis = output<void>();
  readonly locateScope = output<'positions' | 'layers'>();
  readonly locateMetric = output<void>();
  readonly location = output<StateLocation>();
  /** The panel's metric, by the panel's key. */
  abstract readonly metric: () => string;
  /** The catalog's analysis the panel shows, whose inputs a metric's help names. */
  protected abstract readonly metricAnalysis: () => string;
  /** Whether the Metric item of Analysis Info wraps its value. */
  protected readonly metricWraps: boolean = false;
  /** Whether Grouping orders the Statistics rows by their labels rather than their layers. */
  protected readonly groupsByLabel: boolean = false;
  protected readonly metricCatalog = inject(MetricCatalogStore);
  readonly statisticColumns = ['min', 'max', 'mean'] as const;
  readonly sortColumns = ['layer', ...this.statisticColumns] as const;
  protected readonly statisticPanelId = computed(() => this.panel().id);
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
  ): 'ascending' | 'descending' | 'none' {
    const current = this.sortState(group, defaultColumn);
    return current.column === column ? current.direction : 'none';
  }
  sortAction(
    group: string,
    column: StatisticSort['column'],
    defaultColumn: StatisticSort['column'] = 'layer',
  ): string {
    return `Sort ${group} by ${this.statisticLabel(column)} ${this.sortDirection(group, column, defaultColumn) === 'ascending' ? 'descending' : 'ascending'}`;
  }
  sortBy(
    group: string,
    column: StatisticSort['column'],
    defaultColumn: StatisticSort['column'] = 'layer',
  ): void {
    const direction =
      this.sortDirection(group, column, defaultColumn) === 'ascending' ? 'descending' : 'ascending';
    this.statisticSorts.update((groups) => ({...groups, [group]: {column, direction}}));
  }
  statisticLabel(value: string): string {
    if (value === 'layer') {
      return 'Grouping';
    }
    const description = this.metricCatalog.lookup().metric(value);
    return description?.label ?? `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
  }
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
  readonly formatTensorValue = (value: number | string | null | undefined): string =>
    typeof value === 'string'
      ? value
      : value != null && Number.isFinite(value)
        ? formatTensor(value)
        : '-';
  readonly tensorRuntimes = [
    {label: 'Reference', side: 'ref' as const},
    {label: 'Target', side: 'target' as const},
  ];
  readonly tensorPageSize = 16;
  statistic(stats: KvRangeMetricSummary, value: string): number | null {
    return stats[value as 'min' | 'max' | 'mean'];
  }
  statisticColor(value: number | null | undefined): string | null {
    return value != null && Number.isFinite(value) ? this.metricColor(value) : null;
  }
  /** The color of a finite metric value, as the panel's chart draws it. */
  protected metricColor(_value: number): string | null {
    return null;
  }
  /** Whether a statistic that is missing or not finite reads '-'. */
  protected dashesMissing(): boolean {
    return false;
  }
  readonly fmtCompact = (value: number | null | undefined): string =>
    this.dashesMissing() && (value == null || !Number.isFinite(value))
      ? '-'
      : formatMetricValue(this.metricDescription(), value);
  metricValueTitle(value: number | null | undefined): string {
    const description = this.metricDescription();
    return value == null
      ? '—'
      : `${this.metricId()}: ${value.toPrecision(17)}${description ? ` (${description.unit})` : ''}`;
  }
  /** The catalog's id of the panel's metric; Final Logits keys name its rmse and max_abs. */
  readonly metricId = computed(() => catalogMetricId(this.metric()));
  /** The panel's metric as the server's catalog describes it. */
  readonly metricDescription = computed(() => this.metricCatalog.lookup().metric(this.metricId()));
  readonly metricLabel = computed(() => this.metricCatalog.lookup().label(this.metricId()));
  readonly metricDefinition = computed<ExplorerInfoItem>(() => {
    const help = metricHelp(
      this.metricCatalog.lookup(),
      this.metricId(),
      this.metricAnalysis(),
      this.metricLabel(),
    );
    return {
      label: 'Metric',
      value: this.metricLabel(),
      segments: [{text: this.metricLabel(), help}],
      locateLabel: 'Locate Metric control',
      expandable: true,
      wrap: this.metricWraps,
    };
  });
  /** The Aggregation item: how the metric values of the selected heads or sync points are
   *  reduced to one value per token position and layer. */
  protected aggregationItem(
    choice: string,
    dimension: 'Heads' | 'Sync points',
    selection: readonly (string | number)[],
  ): ExplorerInfoItem {
    const functions: Record<string, string> = {min: 'Minimum', max: 'Maximum', mean: 'Mean'};
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
              {
                label: 'Dimension',
                value: dimension === 'Heads' ? 'KV attention heads' : 'Sync points',
              },
              {label: 'Participation', value: 'Present in both Reference and Target'},
            ],
          },
        },
      ],
      locateLabel: dimension === 'Heads' ? 'Locate Heads control' : 'Locate Sync points control',
      expandable: true,
    };
  }
  /** How a Statistics row names its group. */
  rowLabel(row: SummaryRow): string {
    return row.layer === null ? '—' : `L${row.layer}`;
  }
  /** What a Statistics table without rows says. */
  statisticsEmptyText(section: StatisticsSection): string {
    return section.error ? '—' : 'No visible groups.';
  }
  /** Locates one element of the inspected tensor from its Statistics. */
  locateTensorStatistic(_kind: string, _index: number): void {}
  /** How the Token position axis of the Metadata is aligned. */
  protected positionAlignmentLabel(): string {
    return 'Steps';
  }
  /** The Metadata items of a logical tensor as the server describes it; none without one. */
  metadataItems(metadata: LogicalTensor | null, output = false): ExplorerInfoItem[] {
    if (!metadata) {
      return [];
    }
    const dimensions = metadata.dimensions.map((axis) =>
      axis === 'Position' ? 'Token position' : axis,
    );
    const shape = metadata.shape;
    const sizes = shape ? shape.map(axisText) : dimensions.map(() => '-');
    const dtype = metadata.dtype;
    const layers = (metadata.layers ?? []).map((layer) => `L${layer.layer}`);
    const range = this.range();
    const input = dimensions.includes('Head');
    const dimensionSegments = dimensions.flatMap((text, index) => {
      const size = sizes[index] ?? '-';
      const rows =
        text === 'Token position'
          ? [
              {label: 'Alignment', value: this.positionAlignmentLabel()},
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
                {label: 'Selection', value: `[${layers.join(', ')}]`},
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
        value: shape ? `[${sizes.join(', ')}]` : '—',
        expandable: true,
        segments: shape
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
      {label: 'Dtype', value: dtype === null ? '—' : Array.isArray(dtype) ? 'Mixed' : dtype},
      ...(output
        ? [
            {
              label: 'Display precision',
              value: metricDisplayRules(this.metricDescription()).precision,
              expandable: true,
            },
          ]
        : []),
    ];
  }
  sort(rows: readonly SummaryRow[], group: string): SummaryRow[] {
    const {column, direction} = this.sortState(group);
    return [...rows].sort((a, b) => {
      if (this.groupsByLabel && column === 'layer') {
        return a.label.localeCompare(b.label) * (direction === 'ascending' ? 1 : -1);
      }
      const av = column === 'layer' ? a.layer : a.stats[column as 'mean' | 'min' | 'max'];
      const bv = column === 'layer' ? b.layer : b.stats[column as 'mean' | 'min' | 'max'];
      // Missing values stay last in either direction; ties remain in layer order.
      const aValid = typeof av === 'number' && Number.isFinite(av);
      const bValid = typeof bv === 'number' && Number.isFinite(bv);
      if (aValid !== bValid) {
        return aValid ? -1 : 1;
      }
      const difference = aValid && bValid ? av - bv : 0;
      return difference * (direction === 'ascending' ? 1 : -1) || (a.layer ?? 0) - (b.layer ?? 0);
    });
  }
  goPosition(row: SummaryRow, position: number, inspect: boolean): void {
    this.location.emit({
      position,
      layer: row.layer,
      kind: row.kind,
      head: row.head,
      anchor: row.anchor,
      inspect,
    });
  }
}

/** Where a numeric statistics cell's locator jumps when clicked. */
export interface TensorNumericLocation {
  title: string;
  label: string;
  index: number;
  row?: SummaryRow;
  kind?: string;
}

/** Shared numeric value + locator slot used across element and metric statistics tables. */
@Component({
  selector: 'tensor-numeric-cell',
  imports: [ExplorerInfoValue, ExplorerLocator],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span class="stat-cell">
      <explorer-info-value
        class="stat-value"
        [style.background-color]="color()"
        [inline]="true"
        [label]="label()"
        [value]="value()"
      />
      <span class="stat-locator">
        @if (location(); as loc) {
          <button
            explorer-locator
            [title]="loc.title"
            [attr.aria-label]="loc.label"
            (click)="locate(loc)"
          ></button>
        }
      </span>
    </span>
  `,
  styles: `
    :host {
      display: contents;
    }
    .stat-cell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) var(--info-table-action-size, 20px);
      align-items: center;
      gap: var(--info-table-action-gap, 4px);
      min-width: 0;
    }
    .stat-value {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
      text-align: inherit;
    }
    :host-context(.layer-table) .stat-value {
      justify-content: flex-end;
      border-radius: 2px;
      color: var(--me-on-surface-color);
    }
    .stat-locator {
      flex: 0 0 var(--info-table-action-size, 20px);
      width: var(--info-table-action-size, 20px);
      height: 20px;
    }
    .stat-locator [explorer-locator] {
      width: var(--info-table-action-size, 20px);
      flex-basis: var(--info-table-action-size, 20px);
    }
  `,
})
export class TensorNumericCell {
  readonly details = input.required<AnalysisDetails>();
  readonly label = input.required<string>();
  readonly value = input.required<string>();
  readonly color = input<string | null>(null);
  readonly location = input<TensorNumericLocation | null>(null);

  locate(loc: TensorNumericLocation): void {
    if (loc.row) {
      this.details().goPosition(loc.row, loc.index, false);
    } else if (loc.kind !== undefined) {
      this.details().locateTensorStatistic(loc.kind, loc.index);
    }
  }
}
