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

import {FinalLogitsInspection, type LogitPanelState} from '../../../analyses/registry';
import {CaptureStore} from '../../../server/capture_store';
import type {
  TokenAnalysisPair,
  TokenRangeCell,
  TokenRangeRequest,
  TokenRangeResponse,
} from '../../../server/contracts/token_analysis';
import {TokenAnalysisApi} from '../../../server/token_analysis_api';
import type {StageAnalysisPanel} from '../../stage_analysis';
import type {TokenPair} from '../../token_alignment';
import {pairKey} from '../../token_analysis';
import {debugParts} from '../testing/debug_parts';
import {LogitPanel} from './logit_panel';

describe('Final logits wrapper and parent state', () => {
  const rows: TokenPair[] = [
    {
      index: 0,
      ref: {step: 4, text: 'a', produced_text: 'Decode step 104 · reads "," · produces "a"'},
      target: {step: 4, text: 'a', produced_text: 'Decode step 104 · reads "," · produces "a"'},
      match: true,
    },
    {
      index: 1,
      target: {
        step: 5,
        text: 'extra',
        produced_text: 'Decode step 105 · reads "a" · produces "extra"',
      },
      match: false,
    },
    {index: 2, ref: {step: 6, text: 'b'}, target: {step: 6, text: 'b'}, match: true},
    {index: 3, ref: {step: 7, text: 'c'}, target: {step: 7, text: 'c'}, match: true},
  ];
  const initial: LogitPanelState = {
    open: true,
    metric: 'logit_rmse',
    positionRange: {start: 101, end: 104},
    logitPosition: 100,
    logitTransform: 'raw',
    logitOffset: 128,
    logitOrder: 'absolute_delta',
  };
  const missingRef = [
    {code: 'token_not_captured', side: 'ref' as const, text: 'Missing reference'},
  ];
  const notCaptured = [{code: 'activations_not_captured', side: null, text: 'Not captured'}];
  function record(row: TokenPair, raw: number | null, centered: number | null): TokenAnalysisPair {
    const nonFinite = (value: number | null) =>
      value === null ? [{code: 'non_finite_tensor', side: null, text: 'Non-finite logits'}] : [];
    const result = (value: number | null) => ({
      rmse: {value, reasons: nonFinite(value)},
      max_abs: {value: value === null ? null : value * 2, reasons: nonFinite(value)},
      max_abs_index: value === null ? null : 3,
    });
    return {
      stage: 'decode',
      alignment: 'content',
      row: row.index,
      first_divergence: false,
      ref_step: row.ref?.step ?? null,
      target_step: row.target?.step ?? null,
      final_logits: {raw: result(raw), centered: result(centered)},
      metrics: {
        kl: {value: row.ref ? 0.125 : null, reasons: row.ref ? [] : missingRef},
        js: {value: row.ref ? 0.0625 : null, reasons: row.ref ? [] : missingRef},
        relative_l2: {value: null, reasons: notCaptured},
        cosine_distance: {value: null, reasons: notCaptured},
        norm_ratio: {value: null, reasons: notCaptured},
        cosine_similarity: {value: null, reasons: notCaptured},
        max_abs: {value: null, reasons: notCaptured},
      },
      distribution: {compatible: !!row.ref, reasons: [], ref: {}, target: {}, rows: []},
    };
  }
  function setup() {
    // Keep Angular's real signal inputs and outputs; no DOM render or network request is needed.
    const fixture = TestBed.createComponent(LogitPanel);
    const data = new Map([
      [pairKey(2, rows[0]), record(rows[0], 2, 0.25)],
      [pairKey(2, rows[1]), record(rows[1], null, null)],
      [pairKey(2, rows[2]), record(rows[2], 0, 0)],
    ]);
    for (const [name, value] of Object.entries({
      turn: 2,
      rows,
      analysis: data,
      settings: initial,
      globalStart: 100,
      logitMode: 'comparison',
    })) {
      fixture.componentRef.setInput(name, value);
    }
    const changes: LogitPanelState[] = [];
    fixture.componentInstance.settingsChange.subscribe((value) => changes.push(value));
    return {fixture, wrapper: fixture.componentInstance, changes, data};
  }

  it("waits for its first data while the turn's token analysis loads", () => {
    const {fixture, wrapper} = setup();
    fixture.componentRef.setInput('loading', true);
    expect(wrapper.waiting()).toBe(true);
    fixture.componentRef.setInput('loading', false);
    // Not drawn: no chart asks for its bins.
    expect(wrapper.waiting()).toBe(false);
  });

  it('shows only the columns the server gives the analysis', () => {
    const {fixture, wrapper} = setup();
    expect(wrapper.fullPositionRange()).toEqual({start: 100, end: 104});
    // Columns the server does not give the analysis are not shown.
    fixture.componentRef.setInput('columns', {start: 100, end: 102, omitted: [], notes: []});
    expect(wrapper.rows().map((row) => row.index)).toEqual([0, 1]);
    expect(wrapper.fullPositionRange()).toEqual({start: 100, end: 102});
    expect(wrapper.logitSeries()[0].points).toHaveLength(2);
  });

  it('shows the columns after the divergence too, each two-sided one saying why it compares', () => {
    const {fixture, wrapper} = setup();
    const reason = {
      code: 'context_different',
      side: null,
      text: 'After the divergence at position 100 the two sides read different tokens; the logits still compare.',
    };
    fixture.componentRef.setInput('columns', {
      start: 100,
      end: 104,
      omitted: [],
      notes: [],
      context: {start: 101, reason},
    });
    expect(wrapper.rows().map((row) => row.index)).toEqual([0, 1, 2, 3]);
    expect(wrapper.columnSteps(100)?.context).toBeUndefined();
    // Column 101 holds the Target's token alone: nothing compares there.
    expect(wrapper.columnSteps(101)?.context).toBeUndefined();
    expect(wrapper.columnSteps(102)?.context).toBe(reason.text);
  });

  it("says which step produced each column's tokens, in the server's words", () => {
    const {wrapper} = setup();
    expect(wrapper.columnSteps(100)).toEqual({
      ref: 'Decode step 104 · reads "," · produces "a"',
      target: 'Decode step 104 · reads "," · produces "a"',
    });
    expect(wrapper.columnSteps(101)).toEqual({
      ref: undefined,
      target: 'Decode step 105 · reads "a" · produces "extra"',
    });
    expect(wrapper.columnSteps(99)).toBeNull();
  });

  it('draws a point at every row with a token and keeps four comparison metrics', () => {
    const {fixture, wrapper} = setup();
    expect(wrapper.logitMetrics().map(({id}) => id)).toEqual([
      'logit_rmse',
      'logit_max_abs',
      'kl',
      'js',
    ]);
    expect(wrapper.logitSeries()[0].points.map(({position}) => position)).toEqual([
      100, 101, 102, 103,
    ]);
    fixture.destroy();
  });

  it('opens the current selected range or row before a previous tensor position, including a one-sided row', () => {
    const {fixture, wrapper, changes, data} = setup();
    fixture.componentRef.setInput('selected', 1);
    wrapper.openInspection();
    expect(changes[0]).toEqual({
      ...initial,
      logitView: 'inspection',
      logitPlot: 'index',
      logitPosition: 101,
      logitOffset: 0,
    });
    fixture.componentRef.setInput('settings', changes[0]);
    expect(wrapper.inspectionActive()).toBe(true);
    expect(wrapper.inspectionAnalysis()).toBe(data.get(pairKey(2, rows[1])));
    expect(wrapper.inspectionAnalysis()?.ref_step).toBeNull();
    fixture.componentRef.setInput('settings', {...initial, logitPosition: 100});
    fixture.componentRef.setInput('selectedRange', {start: 102, end: 104, layer: null});
    wrapper.openInspection();
    expect(changes[1].logitPosition).toBe(102);
    expect(changes[1].positionRange).toEqual(initial.positionRange);
    fixture.destroy();
  });

  /** The server's token-range of the pairs of `data`: the raw RMSE of row k at column 100 + k,
   *  in the bins the server splits the range into. */
  function tokenRange(
    request: TokenRangeRequest,
    data: Map<string, TokenAnalysisPair>,
  ): TokenRangeResponse {
    const count = Math.max(1, Math.min(request.bins, request.end - request.start));
    const bins = Array.from({length: count}, (_, i) => ({
      start: request.start + Math.floor((i * (request.end - request.start)) / count),
      end: request.start + Math.floor(((i + 1) * (request.end - request.start)) / count),
    }));
    const cell = (bin: {start: number; end: number}): TokenRangeCell => {
      const values = rows.flatMap((row) => {
        const column = 100 + row.index;
        const value = data.get(pairKey(2, row))?.final_logits?.raw.rmse.value;
        return column >= bin.start && column < bin.end && typeof value === 'number'
          ? [{column, value}]
          : [];
      });
      const extreme = (sign: number) =>
        values.reduce<(typeof values)[number] | null>(
          (best, next) => (!best || sign * (next.value - best.value) > 0 ? next : best),
          null,
        );
      return {
        ...bin,
        min: extreme(-1)?.value ?? null,
        max: extreme(1)?.value ?? null,
        mean: values.length
          ? values.reduce((sum, {value}) => sum + value, 0) / values.length
          : null,
        min_position: extreme(-1)?.column ?? null,
        max_position: extreme(1)?.column ?? null,
        valid_count: values.length,
      };
    };
    return {
      ...request,
      bins,
      rows: [{side: null, bins: bins.map(cell)}],
      all: bins.map(cell),
      metadata: {
        ref: {vocabulary_sizes: []},
        target: {vocabulary_sizes: []},
        output: {dimensions: ['Position'], shape: [request.end - request.start], dtype: 'float64'},
      },
      peak_kl: null,
    };
  }

  it('carries the chart bin representative through the parent stage selection into Inspection', async () => {
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
    let served = new Map<string, TokenAnalysisPair>();
    TestBed.configureTestingModule({
      providers: [
        {provide: CaptureStore, useValue: {captureId: signal('capture')}},
        {
          provide: TokenAnalysisApi,
          useValue: {
            tokenRange: async (_capture: string, request: TokenRangeRequest) =>
              tokenRange(request, served),
          },
        },
      ],
    });
    const {fixture, wrapper, changes, data} = setup();
    data.set(pairKey(2, rows[2]), record(rows[2], 4, 0));
    data.set(pairKey(2, rows[3]), record(rows[3], 1, 0));
    served = data;
    const analysis: StageAnalysisPanel = {
      id: 'analysis-1',
      kind: 'tokens',
      turn: 2,
      stage: 'decode',
      open: true,
      logit: initial,
    };
    const selection = signal<
      | {kind: 'token'; turn: number; index: number}
      | {kind: 'stage'; turn: number; stage: string; range: {start: number; end: number}}
      | null
    >(null);
    const {tokens, panels, analyses} = debugParts({
      tokens: {
        analysisOverview: signal(null),
        tokenMetricInspector: signal(null),
        selection,
        select: (turn: number, index: number) => selection.set({kind: 'token', turn, index}),
        tokenSelection: () => {
          const value = selection();
          return value?.kind === 'token' ? value : null;
        },
      },
      panels: {
        tokenMetricRange: signal(null),
        overviewGraph: () => undefined,
        overviewFinalLogits: () => undefined,
      },
    });
    wrapper.selectRange.subscribe(({index, range}) => {
      analyses.inspectTokenMetric(analysis, index, range);
      // The real parent template feeds both the representative and complete range back.
      fixture.componentRef.setInput('selected', panels.tokenPanelSelectedIndex(analysis));
      fixture.componentRef.setInput('selectedRange', range);
    });
    for (const [positionReduction, expectedPosition] of [
      ['max', 102],
      ['min', 103],
      ['mean', 100],
    ] as const) {
      fixture.componentRef.setInput('settings', {
        ...initial,
        positionRange: {start: 100, end: 104},
        positionReduction,
      });
      fixture.detectChanges();
      const chart = wrapper.finalChart()!;
      chart.capacity.set(1);
      fixture.detectChanges();
      await vi.waitFor(() => expect(chart.bins()).toEqual([{start: 100, end: 104}]));
      const range = {...chart.bins()[0], layer: 0};
      chart.select(range);
      expect(tokens.tokenSelection()).toBeNull();
      expect(tokens.selection()).toMatchObject({kind: 'stage', range});
      expect(panels.tokenMetricRange()).toMatchObject({
        index: expectedPosition - 100,
        start: 100,
        end: 104,
      });
      expect(panels.analysisRange(analysis)).toMatchObject({start: 100, end: 104});
      expect(wrapper.selected()).toBe(expectedPosition - 100);
      wrapper.openInspection();
      expect(changes.at(-1)?.logitPosition).toBe(expectedPosition);
      expect(changes.at(-1)?.positionRange).toEqual({start: 100, end: 104});
      fixture.componentRef.setInput('settings', changes.at(-1));
      expect(wrapper.inspectionPosition()).toBe(expectedPosition);
      expect(tokens.selection()).toMatchObject({kind: 'stage', range});
      if (positionReduction === 'max') {
        for (const [nextReduction, nextPosition] of [
          ['mean', 100],
          ['min', 103],
        ] as const) {
          fixture.componentRef.setInput('settings', {
            ...initial,
            positionRange: {start: 100, end: 104},
            positionReduction: nextReduction,
          });
          fixture.detectChanges();
          // Resizing changes the chart bins, but the selection still describes the old bin,
          // which the chart asks the server for on its own.
          chart.capacity.set(2);
          fixture.detectChanges();
          await vi.waitFor(() => expect(chart.bins()).toHaveLength(2));
          expect(chart.bins()).not.toContainEqual({start: 100, end: 104});
          fixture.detectChanges();
          await vi.waitFor(() => expect(chart.selectionData.data()).not.toBeNull());
          expect(panels.tokenMetricRange()?.index).toBe(2);
          wrapper.openInspection();
          expect(changes.at(-1)?.logitPosition).toBe(nextPosition);
          expect(panels.analysisRange(analysis)).toMatchObject({start: 100, end: 104});
          expect(tokens.selection()).toMatchObject({kind: 'stage', range});
        }
      }
    }
    fixture.destroy();
    vi.unstubAllGlobals();
  });

  it('passes Scatter plot through the wrapper while preserving element and value viewports', () => {
    const {fixture, wrapper, changes} = setup();
    const settings: LogitPanelState = {
      ...initial,
      logitElementRange: {start: 128, end: 256},
      logitValueRange: {start: -8.5, end: 3.25},
      logitElementReduction: 'max',
    };
    fixture.componentRef.setInput('settings', settings);
    fixture.componentRef.setInput('selected', 2);
    wrapper.openInspection('scatter');
    expect(changes[0]).toEqual({
      ...settings,
      logitView: 'inspection',
      logitPlot: 'scatter',
      logitPosition: 102,
      logitOffset: 0,
    });
    fixture.componentRef.setInput('settings', changes[0]);
    expect(wrapper.inspectionActive()).toBe(true);
    wrapper.openInspection('index');
    expect(changes[1]).toEqual({...changes[0], logitPlot: 'index'});
    fixture.destroy();
  });

  it('defaults legacy inspection state to the line plot and honors an explicit saved scatter plot', () => {
    TestBed.configureTestingModule({
      providers: [{provide: CaptureStore, useValue: {captureId: signal(null)}}],
    });
    const fixture = TestBed.createComponent(FinalLogitsInspection);
    for (const [name, value] of Object.entries({
      turn: 2,
      position: 100,
      fullRange: {start: 100, end: 104},
      settings: {...initial, logitView: 'inspection'},
    })) {
      fixture.componentRef.setInput(name, value);
    }
    expect(fixture.componentInstance.plot()).toBe('index');
    expect(fixture.componentInstance.reduction()).toBe('mean');
    fixture.componentRef.setInput('settings', {
      ...initial,
      logitView: 'inspection',
      logitPlot: 'scatter',
      logitTransform: 'centered',
      logitElementReduction: 'min',
    });
    expect(fixture.componentInstance.plot()).toBe('scatter');
    expect(fixture.componentInstance.reduction()).toBe('min');
    fixture.destroy();
  });

  it('ignores legacy centered settings while preserving token positions and comparison scope', () => {
    const {fixture, wrapper, changes, data} = setup();
    fixture.componentRef.setInput('settings', {...initial, logitTransform: 'centered'});
    fixture.componentRef.setInput('alignment', 'steps');
    fixture.componentRef.setInput('selected', 1);
    expect(wrapper.fullPositionRange()).toEqual({start: 104, end: 108});
    wrapper.openInspection();
    fixture.componentRef.setInput('settings', changes[0]);
    expect(wrapper.settings().logitTransform).toBe('centered');
    expect(wrapper.inspectionPosition()).toBe(105);
    expect(wrapper.inspectionAnalysis()).toBe(data.get(pairKey(2, rows[1])));
    const selections: unknown[] = [];
    wrapper.selectRange.subscribe((value) => selections.push(value));
    wrapper.inspectPosition(106);
    expect(changes[1]).toEqual({...changes[0], logitPosition: 106, logitOffset: 0});
    expect(selections).toEqual([{index: 2, range: {start: 106, end: 107, layer: null}}]);
    fixture.componentRef.setInput('settings', {...changes[1], logitView: 'comparison'});
    expect(wrapper.inspectionActive()).toBe(false);
    expect(wrapper.settings().logitTransform).toBe('centered');
    expect(wrapper.settings().positionRange).toEqual(initial.positionRange);
    fixture.componentRef.setInput('settings', {...changes[1], logitPosition: 999});
    expect(wrapper.inspectionPosition()).toBe(107);
    fixture.componentRef.setInput('logitMode', 'distribution');
    expect(wrapper.inspectionActive()).toBe(false);
    fixture.destroy();
  });

  it('lets the parent use the live tensor position and restore the saved comparison viewport', () => {
    const analysis: StageAnalysisPanel = {
      id: 'analysis-1',
      kind: 'tokens',
      turn: 2,
      stage: 'decode',
      open: true,
      logit: initial,
    };
    const position = signal(102);
    const inspection = signal<{position: () => number} | undefined>({position});
    const {panels: panel} = debugParts({
      // The server's comparison gives the analysis no columns of its own: every row's.
      conversation: {alignment: () => 'content', comparison: () => undefined},
      tokens: {
        tokenMetricInspector: signal(null),
        rows: () => rows,
        stageGlobalStart: () => 100,
      },
      panels: {
        overviewGraph: () => undefined,
        tokenPanels: () => [{panelId: () => analysis.id, inspection}],
      },
    });
    expect(panel.analysisRange(analysis)).toEqual({start: 102, end: 103});
    position.set(101);
    expect(panel.analysisRange(analysis)).toEqual({start: 101, end: 102});
    inspection.set(undefined);
    expect(panel.analysisRange(analysis)).toEqual({start: 101, end: 104});
    expect(analysis.logit?.positionRange).toEqual(initial.positionRange);
  });

  it('omits deprecated Transform from nested Inspection properties in an old saved view', () => {
    const analysis: StageAnalysisPanel = {
      id: 'analysis-1',
      kind: 'tokens',
      turn: 2,
      stage: 'decode',
      open: true,
      logit: {
        ...initial,
        logitView: 'inspection',
        logitPlot: 'scatter',
        logitTransform: 'centered',
      },
    };
    const {analysisInfo} = debugParts({
      panels: {
        analysisPanels: () => [analysis],
        overviewKv: () => undefined,
        overviewGraph: () => undefined,
        overviewFinalLogits: () => undefined,
      },
    });
    expect(analysisInfo.analysisViewProperties(analysis)).toEqual([
      {label: 'Mode', value: 'Tensor Inspection'},
      {label: 'Visualization', value: 'Scatter plot'},
      {label: 'Token Position', value: '100'},
    ]);
  });
});
