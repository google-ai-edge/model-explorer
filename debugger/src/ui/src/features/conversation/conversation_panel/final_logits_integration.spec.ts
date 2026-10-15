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
import type {TokenAnalysisPair} from '../../../data/contracts/token_analysis';
import type {TokenPair} from '../token_alignment';
import type {StageAnalysisPanel} from '../stage_analysis';
import type {TokenTrendState} from '../token_trends';
import {pairKey} from '../token_analysis';
import {TokenTrends} from './token_trends';
import {ConversationPanel} from './conversation_panel';
import {FinalLogitsInspection} from '../../analysis/final_logits_inspection';
import {ReportStateService} from '../../../data/report_state_service';

describe('Final logits wrapper and parent state', () => {
  const rows: TokenPair[] = [
    {index: 0, ref: {step: 4, text: 'a'}, target: {step: 4, text: 'a'}, match: true},
    {index: 1, target: {step: 5, text: 'extra'}, match: false},
    {index: 2, ref: {step: 6, text: 'b'}, target: {step: 6, text: 'b'}, match: true},
    {index: 3, ref: {step: 7, text: 'c'}, target: {step: 7, text: 'c'}, match: true},
  ];
  const initial: TokenTrendState = {
    open: true,
    metric: 'logit_rmse',
    start: 0,
    end: 3,
    positionRange: {start: 101, end: 104},
    logitPosition: 100,
    logitTransform: 'raw',
    logitOffset: 128,
    logitOrder: 'absolute_delta',
  };
  function record(row: TokenPair, raw: number | null, centered: number | null): TokenAnalysisPair {
    const result = (value: number | null) => ({
      rmse: {value, reason: value === null ? 'Non-finite logits' : null},
      max_abs: {
        value: value === null ? null : value * 2,
        reason: value === null ? 'Non-finite logits' : null,
      },
      max_abs_index: value === null ? null : 3,
    });
    return {
      ref_step: row.ref?.step ?? null,
      target_step: row.target?.step ?? null,
      final_logits: {raw: result(raw), centered: result(centered)},
      metrics: {
        kl: {value: row.ref ? 0.125 : null, reason: row.ref ? null : 'Missing reference'},
        js: {value: row.ref ? 0.0625 : null, reason: row.ref ? null : 'Missing reference'},
        relative_l2: {value: null, reason: 'Not captured'},
        cosine_distance: {value: null, reason: 'Not captured'},
        norm_ratio: {value: null, reason: 'Not captured'},
        max_abs_error: {value: null, reason: 'Not captured'},
      },
      distribution: {compatible: !!row.ref, reason: null, ref: {}, target: {}, rows: []},
    };
  }
  function setup() {
    // Keep Angular's real signal inputs and outputs; no DOM render or network request is needed.
    const fixture = TestBed.createComponent(TokenTrends);
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
      logitsOnly: true,
      logitMode: 'comparison',
    }))
      fixture.componentRef.setInput(name, value);
    const changes: TokenTrendState[] = [];
    fixture.componentInstance.settingsChange.subscribe((value) => changes.push(value));
    return {fixture, wrapper: fixture.componentInstance, changes, data};
  }

  it('uses raw numeric series for all saved transforms and keeps four comparison metrics', () => {
    const {fixture, wrapper} = setup();
    expect(wrapper.logitMetrics().map(({id}) => id)).toEqual([
      'logit_rmse',
      'logit_max_abs',
      'kl',
      'js',
    ]);
    expect(wrapper.series()).toEqual([2, null, 0, null]);
    expect(wrapper.logitSeries()[0].points.map(({position, value}) => [position, value])).toEqual([
      [100, 2],
      [101, null],
      [102, 0],
      [103, null],
    ]);
    fixture.componentRef.setInput('settings', {
      ...initial,
      logitTransform: 'centered',
    });
    expect(wrapper.series()).toEqual([2, null, 0, null]);
    fixture.componentRef.setInput('settings', {
      ...initial,
      logitTransform: 'centered',
      metric: 'logit_max_abs',
    });
    expect(wrapper.series()).toEqual([4, null, 0, null]);
    expect(wrapper.sampled().available).toBe(2);
    fixture.componentRef.setInput('settings', {
      ...initial,
      logitTransform: 'centered',
      metric: 'kl',
    });
    expect(wrapper.series()).toEqual([0.125, null, 0.125, null]);
    fixture.componentRef.setInput('settings', {
      ...initial,
      logitTransform: 'centered',
      metric: 'js',
    });
    expect(wrapper.series()).toEqual([0.0625, null, 0.0625, null]);
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

  it('carries the chart bin representative through the parent stage selection into Inspection', () => {
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
    const {fixture, wrapper, changes, data} = setup();
    data.set(pairKey(2, rows[2]), record(rows[2], 4, 0));
    data.set(pairKey(2, rows[3]), record(rows[3], 1, 0));
    const analysis: StageAnalysisPanel = {
      id: 'analysis-1',
      kind: 'tokens',
      turn: 2,
      stage: 'decode',
      open: true,
      trend: initial,
    };
    const selection = signal<
      | {kind: 'token'; turn: number; index: number}
      | {kind: 'stage'; turn: number; stage: string; range: {start: number; end: number}}
      | null
    >(null);
    const parent = Object.assign(Object.create(ConversationPanel.prototype), {
      analysisOverview: signal(null),
      tokenMetricInspector: signal(null),
      tokenMetricRange: signal(null),
      selection,
      select: (turn: number, index: number) => selection.set({kind: 'token', turn, index}),
      tokenSelection: () => {
        const value = selection();
        return value?.kind === 'token' ? value : null;
      },
      overviewGraph: () => undefined,
      overviewFinalLogits: () => undefined,
    }) as ConversationPanel;
    wrapper.selectRange.subscribe(({index, range}) => {
      parent.inspectTokenMetric(analysis, index, range);
      // The real parent template feeds both the representative and complete range back.
      fixture.componentRef.setInput('selected', parent.tokenPanelSelectedIndex(analysis));
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
      const range = {...chart.bins()[0], layer: 0};
      chart.select(range);
      expect(parent.tokenSelection()).toBeNull();
      expect(parent.selection()).toMatchObject({kind: 'stage', range});
      expect(parent.tokenMetricRange()).toMatchObject({
        index: expectedPosition - 100,
        start: 100,
        end: 104,
      });
      expect(parent.analysisRange(analysis)).toMatchObject({start: 100, end: 104});
      expect(wrapper.selected()).toBe(expectedPosition - 100);
      wrapper.openInspection();
      expect(changes.at(-1)?.logitPosition).toBe(expectedPosition);
      expect(changes.at(-1)?.positionRange).toEqual({start: 100, end: 104});
      fixture.componentRef.setInput('settings', changes.at(-1));
      expect(wrapper.inspectionPosition()).toBe(expectedPosition);
      expect(parent.selection()).toMatchObject({kind: 'stage', range});
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
          // Resizing changes the chart bins, but the selection still describes the old bin.
          chart.capacity.set(2);
          expect(chart.bins()).not.toContainEqual({start: 100, end: 104});
          expect(parent.tokenMetricRange()?.index).toBe(2);
          wrapper.openInspection();
          expect(changes.at(-1)?.logitPosition).toBe(nextPosition);
          expect(parent.analysisRange(analysis)).toMatchObject({start: 100, end: 104});
          expect(parent.selection()).toMatchObject({kind: 'stage', range});
        }
      }
    }
    fixture.destroy();
    vi.unstubAllGlobals();
  });

  it('passes Scatter plot through the wrapper while preserving element and value viewports', () => {
    const {fixture, wrapper, changes} = setup();
    const settings: TokenTrendState = {
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
      providers: [{provide: ReportStateService, useValue: {captureId: signal(null)}}],
    });
    const fixture = TestBed.createComponent(FinalLogitsInspection);
    for (const [name, value] of Object.entries({
      turn: 2,
      position: 100,
      fullRange: {start: 100, end: 104},
      settings: {...initial, logitView: 'inspection'},
    }))
      fixture.componentRef.setInput(name, value);
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
    expect(wrapper.series()).toEqual([2, null, 0, null]);
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
      trend: initial,
    };
    const position = signal(102);
    const inspection = signal<{position: () => number} | undefined>({position});
    const panel = Object.assign(Object.create(ConversationPanel.prototype), {
      overviewGraph: () => undefined,
      tokenPanels: () => [{panelId: () => analysis.id, inspection}],
      tokenMetricInspector: signal(null),
      alignment: () => 'content',
      rows: () => rows,
      stageGlobalStart: () => 100,
    }) as ConversationPanel;
    expect(panel.analysisRange(analysis)).toEqual({start: 102, end: 103});
    position.set(101);
    expect(panel.analysisRange(analysis)).toEqual({start: 101, end: 102});
    inspection.set(undefined);
    expect(panel.analysisRange(analysis)).toEqual({start: 101, end: 104});
    expect(analysis.trend?.positionRange).toEqual(initial.positionRange);
  });

  it('omits deprecated Transform from nested Inspection properties in an old saved view', () => {
    const analysis: StageAnalysisPanel = {
      id: 'analysis-1',
      kind: 'tokens',
      turn: 2,
      stage: 'decode',
      open: true,
      trend: {
        ...initial,
        logitView: 'inspection',
        logitPlot: 'scatter',
        logitTransform: 'centered',
      },
    };
    const panel = Object.assign(Object.create(ConversationPanel.prototype), {
      analysisPanels: () => [analysis],
      overviewKv: () => undefined,
      overviewGraph: () => undefined,
      overviewFinalLogits: () => undefined,
    }) as ConversationPanel;
    expect(panel.analysisViewProperties(analysis)).toEqual([
      {label: 'Mode', value: 'Tensor Inspection'},
      {label: 'Visualization', value: 'Scatter plot'},
      {label: 'Token Position', value: '100'},
    ]);
  });
});
