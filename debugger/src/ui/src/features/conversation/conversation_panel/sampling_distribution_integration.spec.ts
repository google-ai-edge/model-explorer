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

import {signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import type {
  SamplingDistributionSide,
  TokenAnalysisPair,
} from '../../../data/contracts/token_analysis';
import {ReportApiService} from '../../../data/report_api_service';
import {ReportStateService} from '../../../data/report_state_service';
import type {StageAnalysisPanel} from '../stage_analysis';
import {StageStateInfo, type StagePoint} from '../stage_state_info';
import type {TokenPair} from '../token_alignment';
import {pairKey} from '../token_analysis';
import type {TokenTrendState} from '../token_trends';
import {ConversationPanel} from './conversation_panel';
import {TokenTrends} from './token_trends';

const rows: TokenPair[] = [
  {index: 0, ref: {step: 0, text: 'a'}, target: {step: 0, text: 'a'}, match: true},
  {index: 1, target: {step: 1, text: 'extra'}, match: false},
  {index: 2, ref: {step: 2, text: 'b'}, target: {step: 2, text: 'b'}, match: true},
  {index: 3, ref: {step: 3, text: 'c'}, target: {step: 3, text: 'c'}, match: true},
];
const settings: TokenTrendState = {
  open: true,
  metric: 'entropy',
  start: 0,
  end: 3,
  positionRange: {start: 100, end: 104},
};
const panel: StageAnalysisPanel = {
  id: 'analysis-1',
  kind: 'distribution',
  turn: 2,
  stage: 'decode',
  open: true,
  trend: settings,
};
function side(): SamplingDistributionSide {
  return {
    entropy: 0,
    selected_probability: 1,
    margin: 1,
    size: 262144,
    candidate_count: 1,
    source_forward_id: 0,
    basis: 'reconstructed_greedy',
    sampler: {temperature: 0, top_k: 1, top_p: 1, seed: 0},
  };
}
function record(row: TokenPair, sampling = true): TokenAnalysisPair {
  return {
    ref_step: row.ref?.step ?? null,
    target_step: row.target?.step ?? null,
    metrics: Object.fromEntries(
      ['kl', 'js', 'relative_l2', 'cosine_distance', 'norm_ratio', 'max_abs_error'].map((key) => [
        key,
        {value: 0.5, reason: null},
      ]),
    ) as TokenAnalysisPair['metrics'],
    distribution: {
      compatible: true,
      reason: null,
      ref: {entropy: 5, selected_probability: 0.2, margin: 0.3},
      target: {entropy: 6, selected_probability: 0.4, margin: 0.5},
      rows: [],
    },
    ...(sampling
      ? {
          sampling_distribution: {
            context: 'different' as const,
            vocabulary_compatible: true,
            rows_complete: true,
            reason: null,
            ref: side(),
            target: side(),
            rows: [],
          },
        }
      : {}),
  };
}
function data() {
  return new Map([
    [pairKey(2, rows[0]), record(rows[0])],
    [pairKey(2, rows[1]), record(rows[1])],
    [pairKey(2, rows[2]), record(rows[2], false)],
  ]);
}
function points(analysis = data()): StagePoint[] {
  return rows.map((row) => ({
    index: row.index,
    position: 100 + row.index,
    hasRef: !!row.ref,
    hasTarget: !!row.target,
    refPosition: row.ref ? 100 + row.ref.step : null,
    targetPosition: row.target ? 100 + row.target.step : null,
    analysis: analysis.get(pairKey(2, row)),
  }));
}

describe('Sampling Distribution panel and details', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({
      matches: true,
      addEventListener() {},
      removeEventListener() {},
    }));
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: 0}),
    });
    TestBed.configureTestingModule({
      providers: [
        {provide: ReportApiService, useValue: {}},
        {provide: ReportStateService, useValue: {captureId: signal('capture')}},
      ],
    });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });

  function trends() {
    const fixture = TestBed.createComponent(TokenTrends);
    for (const [name, value] of Object.entries({
      turn: 2,
      rows,
      analysis: data(),
      settings,
      globalStart: 100,
      logitsOnly: true,
      logitMode: 'distribution',
    }))
      fixture.componentRef.setInput(name, value);
    return fixture;
  }
  function details() {
    const fixture = TestBed.createComponent(StageStateInfo);
    fixture.componentRef.setInput('panel', panel);
    fixture.componentRef.setInput('label', 'Sampling Distribution Analysis 0');
    fixture.componentRef.setInput('range', {start: 100, end: 104});
    fixture.componentRef.setInput('points', points());
    fixture.detectChanges();
    return fixture;
  }

  it.each(['entropy', 'selected_probability', 'margin'] as const)(
    'uses final sampling %s per side without replacing missing evidence with T=1 values',
    (metric) => {
      const fixture = trends();
      fixture.componentRef.setInput('settings', {...settings, metric});
      const view = fixture.componentInstance;
      const value = metric === 'entropy' ? 0 : 1;
      expect(view.logitMetricLabel()).toBe(
        {
          entropy: 'Entropy (bits)',
          selected_probability: 'Selected probability (%)',
          margin: 'Top-2 margin (pp)',
        }[metric],
      );
      expect(view.sideSeries()).toEqual({
        ref: [value, null, null, null],
        target: [value, value, null, null],
      });
      expect(
        view.logitSeries().map((series) => series.points.map((p) => [p.position, p.value])),
      ).toEqual([
        [
          [100, value],
          [102, null],
          [103, null],
        ],
        [
          [100, value],
          [101, value],
          [102, null],
          [103, null],
        ],
      ]);
      fixture.destroy();
    },
  );

  it('shows no sampling notice for partial, absent, complete, or loading data', () => {
    const fixture = trends();
    const view = fixture.componentInstance;
    fixture.detectChanges();
    const chart = view.finalChart()!;
    expect(view.logitMetrics().map((metric) => metric.id)).toEqual([
      'selected_probability',
      'entropy',
      'margin',
    ]);
    expect(chart.visualizationOptions()).toEqual(['Distribution Comparison · Line chart']);
    expect(chart.inspectionEnabled()).toBe(false);
    expect(fixture.nativeElement.querySelector('.metric-note')).toBeNull();
    fixture.componentRef.setInput(
      'analysis',
      new Map([[pairKey(2, rows[0]), record(rows[0], false)]]),
    );
    fixture.detectChanges();
    expect(view.sideSeries()?.ref).toEqual([null, null, null, null]);
    expect(fixture.nativeElement.querySelector('.metric-note')).toBeNull();
    fixture.componentRef.setInput(
      'analysis',
      new Map(rows.map((row) => [pairKey(2, row), record(row)])),
    );
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.metric-note')).toBeNull();
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.metric-note')).toBeNull();
    fixture.destroy();
  });

  it('uses unified Analysis Info and per-side statistics while exposing sampling source metadata', () => {
    const fixture = details();
    const view = fixture.componentInstance;
    const el: HTMLElement = fixture.nativeElement;
    expect(
      Array.from(el.querySelectorAll(':scope > explorer-info-section')).map((section) =>
        section.querySelector('section')?.getAttribute('aria-label'),
      ),
    ).toEqual([
      'Analysis Info',
      'Statistics',
      'Distribution Metadata',
      'Metric Metadata',
      'Distribution Inspection',
    ]);
    expect(view.samplingInfoItems()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({label: 'Mode', value: 'Distribution Comparison'}),
        expect.objectContaining({label: 'Token Range', value: '100–103'}),
      ]),
    );
    const table = el.querySelector('table[aria-label="Sampling distribution statistics"]')!;
    const inspectionTable = el.querySelector(
      'table[aria-label="Sampling distribution probabilities"]',
    )!;
    for (const sharedTable of [table, inspectionTable]) {
      expect(sharedTable.matches('table[explorer-statistic-table]:not(.equal-columns)')).toBe(true);
      expect(sharedTable.querySelectorAll('col')).toHaveLength(5);
      expect(
        Array.from(sharedTable.querySelectorAll('th')).map(
          (th) => th.getAttribute('colspan') ?? '1',
        ),
      ).toEqual(['1', '1', '1', '2']);
    }
    expect(
      Array.from(table.querySelectorAll('th .sort-label')).map((cell) => cell.textContent?.trim()),
    ).toEqual(['Grouping', 'Min', 'Max', 'Mean']);
    expect(
      Array.from(table.querySelectorAll('tbody tr td:first-child')).map((cell) =>
        cell.textContent?.trim(),
      ),
    ).toEqual(['Ref', 'Target']);
    const metadata = el.querySelector('section[aria-label="Distribution Metadata"]')!;
    for (const item of Array.from(metadata.querySelectorAll('.analysis-item'))) {
      expect(
        Array.from(item.querySelectorAll('explorer-info-list [role="listitem"]')).map((row) =>
          row.getAttribute('aria-label'),
        ),
      ).toEqual(['Vocabulary size', 'Sampling parameters']);
    }
    expect(el.querySelector('section[aria-label="Candidates"]')).toBeNull();
    expect(view.sideStats('ref')).toMatchObject({min: 0, max: 0, mean: 0, valid_count: 1});
    expect(view.sideStats('target')).toMatchObject({min: 0, max: 0, mean: 0, valid_count: 2});
    expect(el.textContent).not.toContain('Jensen–Shannon');
    expect(el.textContent).not.toContain('Kullback–Leibler');
    expect(view.metricDefinition().segments?.[0].help?.rows).toContainEqual({
      label: 'Inputs',
      value: 'Final sampling probabilities after configured sampling rules',
    });
    for (const [metric, label] of [
      ['entropy', 'Entropy'],
      ['selected_probability', 'Selected probability'],
      ['margin', 'Top-2 margin'],
    ] as const) {
      fixture.componentRef.setInput('panel', {...panel, trend: {...settings, metric}});
      fixture.detectChanges();
      expect(view.metricDefinition().value).toBe(label);
      expect(
        el.querySelector('section[aria-label="Analysis Info"] [aria-label="Metric"]')?.textContent,
      ).not.toMatch(/\((bits|%|pp)\)/);
    }
    fixture.destroy();
  });

  it('keeps unavailable values and the comparison structure when switching to a single position', () => {
    const fixture = details();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput(
      'points',
      points(new Map([[pairKey(2, rows[0]), record(rows[0], false)]])),
    );
    fixture.detectChanges();
    expect(view.sideStats('ref')).toMatchObject({min: null, max: null, mean: null, valid_count: 0});
    expect(view.sideStats('target')).toMatchObject({
      min: null,
      max: null,
      mean: null,
      valid_count: 0,
    });
    fixture.componentRef.setInput('range', {start: 100, end: 101});
    fixture.detectChanges();
    expect(view.samplingInfoItems()).toContainEqual(
      expect.objectContaining({
        label: 'Token Range',
        value: '100–100',
      }),
    );
    expect(
      fixture.nativeElement.querySelectorAll('section[aria-label="Analysis Info"]'),
    ).toHaveLength(1);
    expect(
      fixture.nativeElement.querySelector('table[aria-label="Sampling distribution statistics"]'),
    ).not.toBeNull();
    expect(view.samplingMetricMetadata()).toContainEqual(
      expect.objectContaining({label: 'Shape', value: '[2, 1]'}),
    );
    expect(
      fixture.nativeElement.querySelector('section[aria-label="Metric Metadata"]'),
    ).not.toBeNull();
    fixture.destroy();
  });

  it('keeps single-position runtime statistics and metadata scoped to Steps coordinates', () => {
    const fixture = details();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('range', {start: 100, end: 101});
    fixture.componentRef.setInput('alignment', 'steps');
    fixture.componentRef.setInput('points', [
      {...points()[0], targetPosition: 101},
      {...points()[1], targetPosition: 100},
    ]);
    fixture.detectChanges();
    expect(view.sideStats('ref')).toMatchObject({min: 0, max: 0, mean: 0, valid_count: 1});
    expect(view.sideStats('target')).toMatchObject({min: 0, max: 0, mean: 0, valid_count: 1});
    expect(view.samplingMetricMetadata()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({label: 'Shape', value: '[2, 1]'}),
        expect.objectContaining({label: 'Dimensions', value: '[Runtime, Token position]'}),
      ]),
    );
    fixture.destroy();
  });

  it('keeps legacy saved inspection settings on the comparison chart', () => {
    const fixture = trends();
    fixture.componentRef.setInput('settings', {
      ...settings,
      samplingView: 'inspection',
      samplingPosition: 101,
      samplingTopK: 10,
    });
    fixture.detectChanges();
    expect(fixture.componentInstance.finalChart()).toBeDefined();
    expect(fixture.nativeElement.querySelector('sampling-distribution-inspection')).toBeNull();
    fixture.destroy();
  });

  it('moves inspection into details and follows single-position and multi-position selection', () => {
    const fixture = details();
    const source = record(rows[0]);
    source.sampling_distribution!.rows = [
      {
        id: 7,
        label: 'token',
        ref: {probability: 1, rank: 1, selected: true},
        target: {probability: 1, rank: 1, selected: true},
      },
    ];
    fixture.componentRef.setInput('points', points(new Map([[pairKey(2, rows[0]), source]])));
    const table = () =>
      fixture.nativeElement.querySelector(
        'table[aria-label="Sampling distribution probabilities"]',
      );
    fixture.detectChanges();
    expect(table().querySelectorAll('tbody tr')).toHaveLength(1);
    expect(table().textContent).not.toContain('100.00%');
    fixture.componentRef.setInput('range', {start: 100, end: 101});
    fixture.detectChanges();
    expect(table().textContent).toContain('100.00%');
    expect(table().querySelectorAll('tbody td')).toHaveLength(4);
    fixture.componentRef.setInput('range', {start: 100, end: 103});
    fixture.detectChanges();
    expect(table().textContent).not.toContain('100.00%');
    expect(
      Array.from(table().querySelectorAll('tbody td'), (td: any) => td.textContent.trim()),
    ).toEqual(['-', '-', '-', '-']);
    fixture.destroy();
  });

  it('shows the same mode and saved range in the parent analysis properties', () => {
    const parent = Object.assign(Object.create(ConversationPanel.prototype), {
      analysisPanels: signal([panel]),
      overviewKv: () => undefined,
      overviewGraph: () => undefined,
      analysisViewportRange: () => settings.positionRange,
    }) as ConversationPanel;
    expect(parent.analysisLabel(panel)).toBe('Sampling Distribution');
    expect(parent.analysisViewProperties(panel)).toEqual([
      {label: 'Mode', value: 'Distribution Comparison'},
      {label: 'Visualization', value: 'Line chart'},
      {label: 'Token Range', value: '100–103'},
      {label: 'Metric', value: 'Entropy', expandable: true},
    ]);
    parent.analysisPanels.set([
      {
        ...panel,
        trend: {...settings, samplingView: 'inspection', samplingPosition: 101, samplingTopK: 20},
      },
    ]);
    expect(parent.analysisViewProperties(panel)).toEqual([
      {label: 'Mode', value: 'Distribution Comparison'},
      {label: 'Visualization', value: 'Line chart'},
      {label: 'Token Range', value: '100–103'},
      {label: 'Metric', value: 'Entropy', expandable: true},
    ]);
  });
});
