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
  computed,
  inject,
  input,
  output,
  viewChild,
} from '@angular/core';

import {
  ANALYSES,
  FinalLogitsInspection,
  FinalLogitsLineChart,
  type LogitPanelState,
  type LogitPositionSeries,
  catalogMetricId,
  columnContext,
  finalLogitKey,
  isPerRuntimeMetric,
} from '../../../analyses/registry';
import type {AnalysisColumns} from '../../../server/contracts/capture';
import {TokenAnalysisPair} from '../../../server/contracts/token_analysis';
import {MetricCatalogStore} from '../../../server/metric_catalog';
import type {RangeSelection} from '../../../widgets/charts/range_selection';
import {runtimeSeries} from '../../../widgets/charts/runtime_series';
import type {ColumnSteps} from '../../../widgets/format/step_labels';
import {AlignmentMode, TokenPair} from '../../token_alignment';
import {pairKey} from '../../token_analysis';

/** A Final Logits or Sampling Distribution panel: its Line chart or the Final Logits inspection. */
@Component({
  selector: 'logit-panel',
  imports: [FinalLogitsInspection, ANALYSES.final_logits.panel],
  template: `@if (inspectionActive()) {
      <final-logits-inspection
        [turn]="turn()"
        [position]="inspectionPosition()"
        [fullRange]="fullPositionRange()"
        [analysis]="inspectionAnalysis()"
        [settings]="settings()"
        [expanded]="expanded()"
        [analysisLoading]="loading()"
        (settingsChange)="settingsChange.emit($event)"
        (positionChange)="inspectPosition($event)"
        (expand)="expand.emit()"
      />
    } @else {
      <final-logits-line-chart
        [turn]="turn()"
        [alignment]="alignment()"
        [analysis]="logitMode() === 'distribution' ? 'sampling' : 'final_logits'"
        [columnSteps]="columnSteps"
        [metrics]="logitMetrics()"
        [inspectionEnabled]="logitMode() !== 'distribution'"
        (inspect)="openInspection($event)"
        [analysisLabel]="logitMode() === 'distribution' ? 'Sampling Distribution' : 'Final Logits'"
        [modeLabel]="
          logitMode() === 'distribution' ? 'Distribution Comparison' : 'Tensor Comparison'
        "
        [metricLabel]="logitMetricLabel()"
        [positionSeries]="logitSeries()"
        [settings]="settings()"
        [selectedIndex]="selected()"
        [selectedRange]="selectedRange()"
        [expanded]="expanded()"
        (settingsChange)="settingsChange.emit($event)"
        (metricChange)="setLogitMetric($event)"
        (expand)="expand.emit()"
        (selectPosition)="selectPosition.emit($event)"
        (selectRange)="selectRange.emit($event)"
        (backgroundClick)="backgroundClick.emit($event)"
        (positionHover)="positionHover.emit($event)"
      />
      @if (loading()) {
        <p role="status">Loading captured metrics…</p>
      }
    }`,
  styleUrl: './logit_panel.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LogitPanel {
  readonly panelId = input('');
  readonly finalChart = viewChild(FinalLogitsLineChart);
  readonly inspection = viewChild(FinalLogitsInspection);
  readonly inspectionActive = computed(
    () => this.logitMode() === 'comparison' && this.settings().logitView === 'inspection',
  );
  private rowPosition(row: TokenPair): number {
    return (
      this.globalStart() +
      (this.alignment() === 'content' ? row.index : ((row.ref ?? row.target)?.step ?? row.index))
    );
  }
  /** Which step produced each side's token at a column, in the server's words; and, after the
   *  divergence, why its two sides' values still compare. */
  readonly columnSteps = (column: number): ColumnSteps | null => {
    const row = this.rows().find((row) => this.rowPosition(row) === column);
    if (!row) {
      return null;
    }
    const context = columnContext(this.columns(), column, !!row.ref && !!row.target);
    return {
      ref: row.ref?.produced_text,
      target: row.target?.produced_text,
      ...(context ? {context} : {}),
    };
  };
  readonly fullPositionRange = computed(() => {
    const positions = this.rows().map((row) => this.rowPosition(row));
    return positions.length
      ? {start: Math.min(...positions), end: Math.max(...positions) + 1}
      : {start: this.globalStart(), end: this.globalStart() + 1};
  });
  readonly inspectionPosition = computed(() => {
    const full = this.fullPositionRange();
    const preferred =
      this.settings().logitPosition ??
      this.selectedComparisonPosition() ??
      this.settings().positionRange?.start ??
      full.start;
    return Math.max(full.start, Math.min(full.end - 1, preferred));
  });
  private readonly selectedComparisonPosition = computed(() => {
    const range = this.selectedRange();
    const index = (range ? this.finalChart()?.representativeIndex(range) : null) ?? this.selected();
    const row = this.rows().find((row) => row.index === index);
    const position = row ? this.rowPosition(row) : null;
    if (position !== null && (!range || (position >= range.start && position < range.end))) {
      return position;
    }
    return range?.start;
  });
  readonly inspectionAnalysis = computed(() => {
    const row = this.rows().find((row) => this.rowPosition(row) === this.inspectionPosition());
    return row ? this.analysis().get(pairKey(this.turn(), row)) : undefined;
  });
  openInspection(plot: 'index' | 'scatter' = 'index'): void {
    const full = this.fullPositionRange();
    const preferred = this.selectedComparisonPosition() ?? this.inspectionPosition();
    const position = Math.max(full.start, Math.min(full.end - 1, preferred));
    if (this.logitMode() === 'distribution') {
      return;
    }
    this.settingsChange.emit({
      ...this.settings(),
      logitView: 'inspection',
      logitPlot: plot,
      logitPosition: position,
      logitOffset: 0,
    });
  }
  inspectPosition(position: number): void {
    this.settingsChange.emit({...this.settings(), logitPosition: position, logitOffset: 0});
    const row = this.rows().find((row) => this.rowPosition(row) === position);
    if (row) {
      this.selectRange.emit({
        index: row.index,
        range: {start: position, end: position + 1, layer: null},
      });
    }
  }
  locateView(): void {
    const inspection = this.inspection();
    if (inspection) {
      inspection.locateView();
    } else {
      this.finalChart()?.locateView();
    }
  }
  locateRange(): void {
    const inspection = this.inspection();
    if (inspection) {
      inspection.locatePosition();
    } else {
      this.finalChart()?.locateRange();
    }
  }
  locateMetric(): void {
    this.finalChart()?.locateMetric();
  }
  locateFinal(side?: string, position?: number): void {
    this.finalChart()?.locate(side, position);
  }
  readonly expanded = input(false);
  readonly expand = output<void>();
  readonly globalStart = input(0);
  readonly positionHover = output<RangeSelection | null>();
  readonly logitMode = input<'comparison' | 'distribution'>('comparison');
  private readonly catalog = inject(MetricCatalogStore);
  /** The metrics the catalog's Sampling or Final Logits analysis offers, by the panel's keys. */
  readonly logitMetrics = computed(() =>
    this.catalog
      .lookup()
      .choices(this.logitMode() === 'distribution' ? 'sampling' : 'final_logits')
      .map((choice) => ({id: finalLogitKey(choice.id), label: choice.description.label})),
  );
  readonly description = computed(() =>
    this.catalog.lookup().metric(catalogMetricId(this.settings().metric)),
  );
  readonly logitMetricLabel = computed(() => {
    const label = this.logitMetrics().find((m) => m.id === this.settings().metric)?.label ?? '';
    const unit = this.description()?.unit_label;
    return label + (unit ? ' (' + unit + ')' : '');
  });
  setLogitMetric(metric: string): void {
    const found = this.logitMetrics().find((m) => m.id === metric);
    if (found) {
      this.settingsChange.emit({...this.settings(), metric: found.id});
    }
  }
  /** The lines of the chart: one per runtime for a per-runtime metric, each at the rows its side
   *  has a token at; else one, at every row with a token. The server bins their values. */
  readonly logitSeries = computed<LogitPositionSeries[]>(() => {
    const start = this.globalStart();
    if (isPerRuntimeMetric(this.settings().metric)) {
      return (['ref', 'target'] as const).map((side) => ({
        name: runtimeSeries(side).name,
        runtime: side,
        legendLabel: runtimeSeries(side).label,
        color: runtimeSeries(side).color,
        points: this.rows().flatMap((row, index) => {
          const token = row[side];
          return token
            ? [
                {
                  position: start + (this.alignment() === 'content' ? row.index : token.step),
                  index,
                },
              ]
            : [];
        }),
      }));
    }
    return [
      {
        name: this.logitMode() === 'distribution' ? 'Distribution comparison' : 'Final logits',
        color: '#4285f4',
        points: this.rows().flatMap((row, index) => {
          const token = row.ref ?? row.target;
          return token
            ? [{position: start + (this.alignment() === 'content' ? row.index : token.step), index}]
            : [];
        }),
      },
    ];
  });
  readonly turn = input.required<number>();
  /** The stage's rows; the panel shows those of the columns the server gives it (`columns`). */
  readonly allRows = input.required<TokenPair[]>({alias: 'rows'});
  /** Which of the stage's columns the analysis has, as the server says; null for every one. */
  readonly columns = input<AnalysisColumns | null>(null);
  readonly rows = computed(() => {
    const columns = this.columns();
    const rows = this.allRows();
    if (!columns) {
      return rows;
    }
    return rows.filter((row) => {
      const column = this.rowPosition(row);
      return column >= columns.start && column < columns.end;
    });
  });
  readonly analysis = input.required<Map<string, TokenAnalysisPair>>();
  readonly settings = input.required<LogitPanelState>();
  readonly selected = input<number | null>(null);
  readonly loading = input(false);
  /** Whether the panel waits for its first data: the turn's token analysis, then every answer its
   *  chart draws, of both alignments, or its inspection's summary. */
  readonly waiting = computed(
    () => this.loading() || !!this.finalChart()?.busy() || !!this.inspection()?.loading(),
  );
  readonly alignment = input<AlignmentMode>('content');
  readonly settingsChange = output<LogitPanelState>();
  readonly selectPosition = output<number>();
  readonly selectedRange = input<RangeSelection | null>(null);
  readonly selectRange = output<{index: number; range: RangeSelection}>();
  readonly backgroundClick = output<MouseEvent>();
}
