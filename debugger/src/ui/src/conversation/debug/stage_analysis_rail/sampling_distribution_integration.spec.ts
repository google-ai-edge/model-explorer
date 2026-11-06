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

import {ANALYSES, type LogitPanelState, type StagePoint} from '../../../analyses/registry';
import {CaptureStore} from '../../../server/capture_store';
import type {Reason} from '../../../server/contracts/metrics';
import type {
  SamplingDistributionSide,
  TokenAnalysisPair,
  TokenRangeRequest,
  TokenRangeResponse,
} from '../../../server/contracts/token_analysis';
import {FinalLogitsApi} from '../../../server/final_logits_api';
import {KvApi} from '../../../server/kv_api';
import {MetricCatalogStore} from '../../../server/metric_catalog';
import {TokenAnalysisApi} from '../../../server/token_analysis_api';
import type {StageAnalysisPanel} from '../../stage_analysis';
import type {TokenPair} from '../../token_alignment';
import {pairKey} from '../../token_analysis';
import {debugParts} from '../testing/debug_parts';
import {LogitPanel} from './logit_panel';

const rows: TokenPair[] = [
  {index: 0, ref: {step: 0, text: 'a'}, target: {step: 0, text: 'a'}, match: true},
  {index: 1, target: {step: 1, text: 'extra'}, match: false},
  {index: 2, ref: {step: 2, text: 'b'}, target: {step: 2, text: 'b'}, match: true},
  {index: 3, ref: {step: 3, text: 'c'}, target: {step: 3, text: 'c'}, match: true},
];
const settings: LogitPanelState = {
  open: true,
  metric: 'entropy',
  positionRange: {start: 100, end: 104},
};
const panel: StageAnalysisPanel = {
  id: 'analysis-1',
  kind: 'distribution',
  turn: 2,
  stage: 'decode',
  open: true,
  logit: settings,
};
function side(): SamplingDistributionSide {
  return {
    entropy: 0,
    selected_probability: 1,
    margin: 1,
    size: 262144,
    candidate_count: 262144,
    source_forward_id: 0,
    basis: 'logits_softmax',
    temperature: 1,
    sampler: {temperature: 0, top_k: 1, top_p: 1, seed: 0},
  };
}
function record(row: TokenPair, sampling = true): TokenAnalysisPair {
  return {
    stage: 'decode',
    alignment: 'content',
    row: row.index,
    first_divergence: false,
    ref_step: row.ref?.step ?? null,
    target_step: row.target?.step ?? null,
    metrics: Object.fromEntries(
      [
        'kl',
        'js',
        'relative_l2',
        'cosine_similarity',
        'cosine_distance',
        'norm_ratio',
        'max_abs',
      ].map((key) => [key, {value: 0.5, reasons: [] as Reason[]}]),
    ) as TokenAnalysisPair['metrics'],
    distribution: {
      compatible: true,
      reasons: [],
      ref: {entropy: 5, selected_probability: 0.2, margin: 0.3},
      target: {entropy: 6, selected_probability: 0.4, margin: 0.5},
      rows: [],
    },
    ...(sampling
      ? {
          sampling_distribution: {
            context: 'different' as const,
            vocabulary_compatible: true,
            rows_complete: false,
            reasons: [],
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

/** What the fake server answers token-range from: each row's pair, by its key. */
let served = data();
/** The server's token-range of `served`, one bin per request, as token_analysis.token_range bins
 *  each side's own value at the rows with its token, row k at column 100 + k. */
function tokenRange(request: TokenRangeRequest): TokenRangeResponse {
  const cell = (side: 'ref' | 'target') => {
    const values = rows.flatMap((row) => {
      const column = 100 + row.index;
      if (!row[side] || column < request.start || column >= request.end) {
        return [];
      }
      const value = served.get(pairKey(2, row))?.sampling_distribution?.[side][
        request.metric as 'entropy'
      ];
      return typeof value === 'number' && Number.isFinite(value) ? [{column, value}] : [];
    });
    const extreme = (sign: number) =>
      values.reduce<(typeof values)[number] | null>(
        (best, next) => (!best || sign * (next.value - best.value) > 0 ? next : best),
        null,
      );
    return {
      start: request.start,
      end: request.end,
      min: extreme(-1)?.value ?? null,
      max: extreme(1)?.value ?? null,
      mean: values.length ? values.reduce((sum, {value}) => sum + value, 0) / values.length : null,
      min_position: extreme(-1)?.column ?? null,
      max_position: extreme(1)?.column ?? null,
      valid_count: values.length,
    };
  };
  const metadata = {
    vocabulary_sizes: [262144],
    sampler: {temperature: [0], top_k: [1], top_p: [1], seed: [0]},
  };
  return {
    ...request,
    bins: [{start: request.start, end: request.end}],
    rows: [
      {side: 'ref', bins: [cell('ref')]},
      {side: 'target', bins: [cell('target')]},
    ],
    all: [],
    metadata: {
      ref: metadata,
      target: metadata,
      output: {
        dimensions: ['Runtime', 'Position'],
        shape: [2, request.end - request.start],
        dtype: 'float64',
      },
    },
    peak_kl: null,
  };
}

describe('Sampling Distribution panel and details', () => {
  beforeEach(() => {
    served = data();
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({
      matches: true,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
    }));
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: 0}),
    });
    TestBed.configureTestingModule({
      providers: [
        {provide: FinalLogitsApi, useValue: {}},
        {provide: KvApi, useValue: {}},
        {provide: CaptureStore, useValue: {captureId: signal('capture')}},
        {
          provide: TokenAnalysisApi,
          useValue: {
            tokenRange: async (_capture: string, request: TokenRangeRequest) => tokenRange(request),
          },
        },
      ],
    });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });

  function logitPanel() {
    const fixture = TestBed.createComponent(LogitPanel);
    for (const [name, value] of Object.entries({
      turn: 2,
      rows,
      analysis: data(),
      settings,
      globalStart: 100,
      logitMode: 'distribution',
    })) {
      fixture.componentRef.setInput(name, value);
    }
    return fixture;
  }
  function details() {
    const fixture = TestBed.createComponent(ANALYSES.sampling.details);
    fixture.componentRef.setInput('panel', panel);
    fixture.componentRef.setInput('label', 'Sampling Distribution Analysis 0');
    fixture.componentRef.setInput('range', {start: 100, end: 104});
    fixture.componentRef.setInput('points', points());
    fixture.detectChanges();
    return fixture;
  }
  /** Runs the details' effects and waits for the server's answer of their range. */
  async function answered(fixture: ReturnType<typeof details>) {
    fixture.detectChanges();
    await vi.waitFor(() => expect(fixture.componentInstance.rangeData.data()).not.toBeNull());
    fixture.detectChanges();
  }

  it.each(['entropy', 'selected_probability', 'margin'] as const)(
    'draws final sampling %s as one line per side, at the rows of its tokens',
    (metric) => {
      const fixture = logitPanel();
      fixture.componentRef.setInput('settings', {...settings, metric});
      const view = fixture.componentInstance;
      expect(view.logitMetricLabel()).toBe(
        {
          entropy: 'Entropy (bits)',
          selected_probability: 'Selected probability (%)',
          margin: 'Top-2 margin (pp)',
        }[metric],
      );
      expect(view.logitSeries().map((series) => series.points.map((p) => p.position))).toEqual([
        [100, 102, 103],
        [100, 101, 102, 103],
      ]);
      fixture.destroy();
    },
  );

  it('shows no sampling notice for partial, absent, complete, or loading data', () => {
    const fixture = logitPanel();
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

  it('uses unified Analysis Info and per-side statistics while exposing sampling source metadata', async () => {
    const fixture = details();
    await answered(fixture);
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
    // The help says what the server does: the softmax of the logits at the sampler temperature.
    const inputs = view
      .metricDefinition()
      .segments?.[0].help?.rows.find((row) => row.label === 'Inputs')?.value;
    expect(inputs).toContain('softmax of its full captured vocabulary logits');
    expect(inputs).not.toContain('after configured sampling rules');
    for (const [metric, label] of [
      ['entropy', 'Entropy'],
      ['selected_probability', 'Selected probability'],
      ['margin', 'Top-2 margin'],
    ] as const) {
      fixture.componentRef.setInput('panel', {...panel, logit: {...settings, metric}});
      fixture.detectChanges();
      expect(view.metricDefinition().value).toBe(label);
      await vi.waitFor(() => expect(view.rangeData.data()?.metric).toBe(metric));
      expect(
        el.querySelector('section[aria-label="Analysis Info"] [aria-label="Metric"]')?.textContent,
      ).not.toMatch(/\((bits|%|pp)\)/);
    }
    fixture.destroy();
  });

  it('keeps unavailable values and the comparison structure when switching to a single position', async () => {
    served = new Map([[pairKey(2, rows[0]), record(rows[0], false)]]);
    const fixture = details();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('points', points(served));
    await answered(fixture);
    expect(view.sideStats('ref')).toMatchObject({min: null, max: null, mean: null, valid_count: 0});
    expect(view.sideStats('target')).toMatchObject({
      min: null,
      max: null,
      mean: null,
      valid_count: 0,
    });
    fixture.componentRef.setInput('range', {start: 100, end: 101});
    fixture.detectChanges();
    await vi.waitFor(() => expect(view.rangeData.data()?.end).toBe(101));
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

  it('keeps single-position runtime statistics and metadata scoped to Steps coordinates', async () => {
    const fixture = details();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('range', {start: 100, end: 101});
    fixture.componentRef.setInput('alignment', 'steps');
    fixture.detectChanges();
    await vi.waitFor(() => expect(view.rangeData.data()?.alignment).toBe('steps'));
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
    const fixture = logitPanel();
    // Sampling settings saved before they were removed.
    const saved = {samplingView: 'inspection', samplingPosition: 101, samplingTopK: 10};
    fixture.componentRef.setInput('settings', {...settings, ...saved});
    fixture.detectChanges();
    expect(fixture.componentInstance.finalChart()).toBeDefined();
    expect(fixture.nativeElement.querySelector('sampling-distribution-inspection')).toBeNull();
    fixture.destroy();
  });

  it('moves inspection into details: the distributions of one position, at the one position of the range, else at the selected row, else at the first answered row', () => {
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
    const heading = () =>
      fixture.nativeElement.querySelector('.sampling-inspection-item .item-name').textContent;
    const cells = () =>
      Array.from(
        table().querySelectorAll('tbody td'),
        (td: HTMLTableCellElement) => (td.textContent ?? '').trim(),
      );
    // A range of several positions: its first answered row (a greedy point mass is drawn too).
    fixture.detectChanges();
    expect(table().textContent).toContain('100.00%');
    expect(heading()).toContain('Position 100');
    // The selected row, while it is in the range; not answered yet, nothing to list.
    fixture.componentRef.setInput('inspectedRow', 2);
    fixture.detectChanges();
    expect(heading()).toContain('Position 102');
    expect(cells()).toEqual(['-', '-', '-', '-']);
    // The range's one position wins over the selected row.
    fixture.componentRef.setInput('range', {start: 100, end: 101});
    fixture.detectChanges();
    expect(table().textContent).toContain('100.00%');
    expect(table().querySelectorAll('tbody td')).toHaveLength(4);
    // A range of no answered row, the selected one outside it: empty.
    fixture.componentRef.setInput('inspectedRow', 0);
    fixture.componentRef.setInput('range', {start: 101, end: 103});
    fixture.detectChanges();
    expect(table().textContent).not.toContain('100.00%');
    expect(cells()).toEqual(['-', '-', '-', '-']);
    fixture.destroy();
  });

  it('shows the same mode and saved range in the parent analysis properties', () => {
    const {panels, analysisInfo: parent} = debugParts({
      panels: {
        analysisPanels: signal([panel]),
        overviewKv: () => undefined,
        overviewGraph: () => undefined,
        analysisViewportRange: () => settings.positionRange,
      },
      analysisInfo: {metricCatalog: TestBed.inject(MetricCatalogStore)},
    });
    expect(parent.analysisLabel(panel)).toBe('Sampling Distribution');
    expect(parent.analysisViewProperties(panel)).toEqual([
      {label: 'Mode', value: 'Distribution Comparison'},
      {label: 'Visualization', value: 'Line chart'},
      {label: 'Token Range', value: '100–103'},
      {label: 'Metric', value: 'Entropy', expandable: true},
    ]);
    const saved = {samplingView: 'inspection', samplingPosition: 101, samplingTopK: 20};
    panels.analysisPanels.set([{...panel, logit: {...settings, ...saved}}]);
    expect(parent.analysisViewProperties(panel)).toEqual([
      {label: 'Mode', value: 'Distribution Comparison'},
      {label: 'Visualization', value: 'Line chart'},
      {label: 'Token Range', value: '100–103'},
      {label: 'Metric', value: 'Entropy', expandable: true},
    ]);
  });
});
