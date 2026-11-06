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

import {GraphMetricPanel, KvPanel} from '../../../analyses/registry';
import type {CapturedTurn, TurnStage} from '../../../server/contracts/capture';
import type {StageAnalysisPanel} from '../../stage_analysis';
import {debugParts} from '../testing/debug_parts';

/** A read turn whose stages start where the server says, the same on both sides. */
function turnAt(n: number, prefill: number, decode: number) {
  const stage = (start: number) => ({
    positions: {ref: {start, end: start + 1}, target: {start, end: start + 1}},
  });
  return {n, stages: {prefill: stage(prefill), decode: stage(decode)}};
}

describe('leaving a focused analysis', () => {
  function setup(kind: StageAnalysisPanel['kind'] = 'kv') {
    const analysis: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind,
      open: true,
    };
    const range = signal({start: 112, end: 113, layer: 0});
    const inspector = Object.assign(kind === 'kv' ? Object.create(KvPanel.prototype) : {}, {
      panelId: () => analysis.id,
      tensorView: () => false,
      inspectedRange: range,
      selected: range,
    });
    const pair = {index: 12, ref: {step: 12}, target: {step: 12}};
    const parts = debugParts({
      conversation: {alignment: () => 'content', turns: () => [turnAt(2, 90, 100)]},
      tokens: {
        selection: signal({kind: 'stage', turn: 2, stage: 'decode', side: 'target'}),
        analysisOverview: signal(null),
        metricInspector: signal(inspector),
        tokenMetricInspector: signal(null),
        rows: () => [pair],
        selectedInputPair: () => null,
        selectedPair: () => pair,
      },
      panels: {
        analysisPanels: signal([analysis]),
        tokenMetricRange: signal(null),
        analysisHover: signal({panelId: analysis.id, range: range()}),
        expandedAnalysis: signal(null),
      },
    });
    return {...parts, analysis, inspector};
  }

  for (const action of ['toggleAnalysis', 'removeAnalysis'] as const) {
    it.each(['kv', 'activations', 'tokens', 'distribution'] as const)(
      `${action} returns a focused %s position to Stage Info, not Token Info`,
      (kind) => {
        const {tokens, panels, analyses, analysis} = setup(kind);
        if (kind === 'tokens' || kind === 'distribution') {
          tokens.metricInspector.set(null);
          tokens.tokenMetricInspector.set(analysis);
          tokens.selection.set({kind: 'token', turn: 2, index: 12});
        }
        analyses[action](analysis);
        expect(tokens.selection()).toEqual({
          kind: 'stage',
          turn: 2,
          stage: 'decode',
          side: 'target',
          range: {start: 112, end: 113},
        });
        expect(tokens.metricInspector()).toBeNull();
        expect(tokens.tokenMetricInspector()).toBeNull();
        expect(tokens.analysisOverview()).toBeNull();
        expect(
          action === 'removeAnalysis'
            ? panels.analysisPanels().length === 0
            : !panels.analysisPanels()[0].open,
        ).toBe(true);
      },
    );
  }

  it('keeps another panel inspection when removing an inactive analysis', () => {
    const {tokens, panels, analyses, analysis, inspector} = setup();
    const other = {...analysis, id: 'analysis-2'};
    panels.analysisPanels.update((items) => [...items, other]);
    analyses.removeAnalysis(other);
    expect(tokens.metricInspector()).toBe(inspector);
    expect(panels.analysisPanels()).toEqual([analysis]);
  });
});

describe('analysis scope and inspector ownership', () => {
  function setup(kind: StageAnalysisPanel['kind'] = 'kv', tensorView = false) {
    const analysis: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind,
      open: true,
    };
    const viewport = signal({start: 40, end: 51});
    const picked = signal({start: 44, end: 47, layer: 0});
    const tensor = {
      position: signal(44),
      layer: signal(3),
      selectedIndex: signal<number | null>(2),
      channelSelection: signal<{start: number; end: number; plane: null} | null>({
        start: 2,
        end: 4,
        plane: null,
      }),
      elementRange: signal({start: 2, end: 8}),
      clearElementSelection() {
        this.selectedIndex.set(null);
        this.channelSelection.set(null);
      },
    };
    const chart = Object.assign(
      kind === 'kv'
        ? Object.create(KvPanel.prototype)
        : kind === 'activations'
          ? Object.create(GraphMetricPanel.prototype)
          : {},
      {
        panelId: () => analysis.id,
        visibleRange: viewport,
        range: viewport,
        tensorView: () => tensorView,
        tensorPosition: tensor.position,
        tensorLayer: tensor.layer,
        tensorChannels: signal({key: 2, value: 2}),
        tensorChannelSelection: signal({start: 2, end: 4, plane: null}),
        inspectedRange: picked,
        selected: picked,
        inspectionMode: () => kind === 'activations' && tensorView,
        inspection: () => (kind === 'activations' && tensorView ? tensor : undefined),
        fullRange: () => ({start: 32, end: 67}),
        layers: () => [0, 1, 2, 3],
        selectedLayers: () => [0],
        view: () => 'lineChart',
        request: {loading: () => false},
      },
    );
    const parts = debugParts({
      tokens: {
        metricInspector: signal(null),
        tokenMetricInspector: signal(null),
        selection: signal(null),
        analysisOverview: signal(null),
        details: signal(false),
      },
      panels: {
        analysisPanels: signal([analysis]),
        kvPanels: () => (kind === 'kv' ? [chart] : []),
        graphPanels: () => (kind === 'activations' ? [chart] : []),
        tokenPanels: () => [
          {panelId: () => analysis.id, finalChart: () => chart, inspection: () => undefined},
        ],
        tokenMetricRange: signal(null),
        analysisHover: signal(null),
        expandedAnalysis: signal(null),
      },
      analyses: {analysisViewports: new Map()},
    });
    const sync = () => parts.analyses.syncAnalysisViewport();
    return {...parts, analysis, chart, tensor, viewport, picked, sync};
  }
  it.each(['kv', 'activations', 'tokens', 'distribution'] as const)(
    '%s activates Analysis Info and updates scope immediately when range changes',
    (kind) => {
      const {tokens, panels, analyses, analysis, viewport, sync} = setup(kind);
      analyses.inspectAnalysisStage(analysis, new MouseEvent('click'));
      expect(panels.overviewPanel()).toEqual(analysis);
      sync();
      expect(tokens.selection()).toMatchObject({range: {start: 40, end: 51}});
      viewport.set({start: 42, end: 49});
      sync();
      expect(tokens.selection()).toMatchObject({range: {start: 42, end: 49}});
      expect(panels.overviewPanel()).toEqual(analysis);
      analyses.toggleAnalysis(analysis);
      expect(panels.overviewPanel()).toBeNull();
      expect(tokens.selection()).toMatchObject({range: {start: 42, end: 49}});
    },
  );
  it('clears a selected bin on background click and returns the viewport when closing', () => {
    const {tokens, panels, analyses, analysis, chart} = setup();
    tokens.metricInspector.set(chart);
    analyses.inspectAnalysisStage(analysis, new MouseEvent('click'));
    expect(tokens.metricInspector()).toBeNull();
    expect(tokens.analysisOverview()).toEqual(analysis);
    analyses.removeAnalysis(analysis);
    expect(panels.overviewPanel()).toBeNull();
    expect(tokens.selection()).toMatchObject({range: {start: 40, end: 51}});
  });
  it.each(['activations', 'tokens', 'distribution'] as const)(
    '%s clears the plotted selection and keeps the current analysis open',
    (kind) => {
      const {tokens, panels, analyses, analysis, chart} = setup(kind);
      if (kind === 'activations') {
        tokens.metricInspector.set(chart);
      } else {
        tokens.tokenMetricInspector.set(analysis);
      }
      analyses.inspectAnalysisStage(analysis, new MouseEvent('click'));
      expect(tokens.metricInspector()).toBeNull();
      expect(tokens.tokenMetricInspector()).toBeNull();
      expect(tokens.analysisOverview()).toEqual(analysis);
      expect(tokens.selection()).toMatchObject({range: {start: 40, end: 51}});
      if (kind === 'activations') {
        expect(chart.selected()).toBeNull();
      }
    },
  );
  it.each(['kv', 'activations'] as const)(
    '%s uses inspection coordinates for token focus instead of the stale comparison cursor',
    (kind) => {
      const {tokens, panels, analyses, analysis, chart} = setup(kind, true);
      chart.tensorPosition.set(46);
      tokens.metricInspector.set(chart);
      const range = panels.selectionRangeFor(analysis);
      expect(range).toEqual({start: 46, end: 47, layer: 3});
    },
  );
  it.each(['activations'] as const)(
    'clears %s elements and keeps tensor scope and element zoom on background click',
    (kind) => {
      const {tokens, panels, analyses, analysis, chart, tensor, viewport} = setup(kind, true);
      tokens.metricInspector.set(chart);
      analyses.inspectAnalysisStage(analysis, new MouseEvent('click'));
      expect(tensor.selectedIndex()).toBeNull();
      expect(tensor.channelSelection()).toBeNull();
      expect(tensor.position()).toBe(44);
      expect(tensor.layer()).toBe(3);
      expect(tensor.elementRange()).toEqual({start: 2, end: 8});
      expect(viewport()).toEqual({start: 40, end: 51});
      expect(tokens.metricInspector()).toBeNull();
      expect(tokens.analysisOverview()).toEqual(analysis);
      expect(tokens.selection()).toMatchObject({range: {start: 44, end: 45}});
    },
  );
  it.each(['activations'] as const)(
    'keeps %s inspection focus outside comparison zoom and hidden comparison layers',
    (kind) => {
      const {tokens, panels, analyses, analysis, chart, tensor, viewport, picked} = setup(
        kind,
        true,
      );
      viewport.set({start: 40, end: 43});
      tensor.position.set(60);
      picked.set({start: 44, end: 47, layer: 3});
      tokens.metricInspector.set(chart);
      const syncVisibility = () => analyses.syncMetricInspectorVisibility();
      syncVisibility();
      expect(tokens.metricInspector()).toBe(chart);
      expect(tokens.analysisOverview()).toBeNull();
      chart.inspectionMode = () => false;
      syncVisibility();
      expect(tokens.metricInspector()).toBeNull();
      expect(tokens.analysisOverview()).toEqual(analysis);
    },
  );
  it('clears inspection elements without resetting tensor coordinates or zoom', () => {
    const {tokens, panels, analyses, analysis, chart, viewport} = setup('kv', true);
    tokens.metricInspector.set(chart);
    analyses.inspectAnalysisStage(analysis, new MouseEvent('click'));
    expect(chart.tensorChannels()).toEqual({key: null, value: null});
    expect(chart.tensorChannelSelection()).toBeNull();
    expect(chart.tensorPosition()).toBe(44);
    expect(viewport()).toEqual({start: 40, end: 51});
    expect(tokens.analysisOverview()).toEqual(analysis);
    expect(tokens.selection()).toMatchObject({range: {start: 44, end: 45}});
  });
  it('range changes replace a bin even when the bin still fits in the new viewport', () => {
    const {tokens, panels, analyses, analysis, chart, viewport, sync} = setup();
    tokens.metricInspector.set(chart);
    sync();
    viewport.set({start: 42, end: 49});
    sync();
    expect(tokens.metricInspector()).toBeNull();
    expect(tokens.analysisOverview()).toEqual(analysis);
    expect(tokens.selection()).toMatchObject({range: {start: 42, end: 49}});
  });
  it.each(['kv', 'activations'] as const)(
    '%s tensor scope follows its position, independently of metric or element zoom',
    (kind) => {
      const {tokens, panels, analyses, analysis, chart, viewport, sync} = setup(kind, true);
      analyses.inspectAnalysisStage(analysis, new MouseEvent('click'));
      sync();
      viewport.set({start: 45, end: 49});
      sync();
      expect(tokens.selection()).toMatchObject({range: {start: 44, end: 45}});
      chart.tensorPosition.set(46);
      sync();
      analyses.toggleAnalysis(analysis);
      expect(panels.overviewPanel()).toBeNull();
      expect(tokens.selection()).toMatchObject({range: {start: 46, end: 47}});
    },
  );
});

describe('the analysis picker of a rejected turn', () => {
  const TEXT =
    'Target: Turn 2 could not be read: a.safetensors holds 9 bytes; the capture recorded 10.';
  /** A rejected turn's stage, as the server sends it: every analysis unavailable, for `text`. */
  function rejectedStage(moment: string, text: string): TurnStage {
    const why = {available: false, reasons: [{code: 'turn_unreadable', side: null, text}]};
    return {
      positions: {ref: null, target: null},
      kv: {moment, context_id: null},
      analyses: {final_logits: why, sampling: why, kv: why, intermediate_tensors: why},
    };
  }
  const read: CapturedTurn = {
    n: 1,
    step_start: 12,
  };
  const rejected: CapturedTurn = {
    n: 2,
    prompt: 'And more?',
    step_start: null,
    rejected: {code: 'turn_unreadable', side: 'target', text: TEXT},
    stages: {
      prefill: rejectedStage('prefill_post', TEXT),
      decode: rejectedStage('terminal', TEXT),
    },
  };

  it('answers with the server text for a rejected turn', () => {
    const session = signal({turns: [read, rejected]});
    const {panels} = debugParts({
      conversation: {state: {session}},
      tokens: {
        rows: () => {
          throw new Error('A rejected turn has no rows to look at.');
        },
      },
    });
    for (const stage of ['prefill', 'decode']) {
      for (const kind of ['tokens', 'distribution', 'activations', 'kv'] as const) {
        expect(panels.analysisUnavailable(2, stage, kind)).toBe(TEXT);
      }
    }
  });
});
