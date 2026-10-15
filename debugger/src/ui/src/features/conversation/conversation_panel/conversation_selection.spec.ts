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

import {Component, signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {ConversationPanel} from './conversation_panel';
import {KvPanel} from '../../kv/kv_panel/kv_panel';
import {GraphMetricPanel} from '../graph_metric_panel';
import type {StageAnalysisPanel} from '../stage_analysis';

@Component({template: '<div></div>'})
class RenderHost {}

describe('token selection and explicit location', () => {
  function setup() {
    vi.stubGlobal('matchMedia', () => ({matches: true}));
    const fixture = TestBed.createComponent(RenderHost);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;
    root.innerHTML = `<div class="reading"><div class="column-headings"></div>
      <div data-turn="2">
        <button data-turn="2" data-row="12" data-kind="input" data-side="target">input</button>
        <button data-turn="2" data-row="12" data-side="ref">reference</button>
        <button data-turn="2" data-row="12" data-side="target">target</button>
      </div></div><div class="composer"></div>`;
    const reading = root.querySelector('.reading') as HTMLElement;
    const token = root.querySelector('button[data-side="target"]:not([data-kind])') as HTMLElement;
    const scroll = vi.fn();
    for (const e of Array.from(root.querySelectorAll('*'))) {
      e.scrollIntoView = scroll;
      e.animate = vi.fn(() => ({cancel: vi.fn()}) as unknown as Animation);
    }
    reading.scrollTo = vi.fn();
    const panel = Object.assign(Object.create(ConversationPanel.prototype), {
      element: {nativeElement: root},
      injector: fixture.debugElement.injector,
      selection: signal(null),
      selectionSide: signal('target'),
      analysisOverview: signal(null),
      metricInspector: signal(null),
      tokenMetricInspector: signal(null),
      details: signal(true),
      collapsed: signal(new Set<string>()),
      reading: {visibleBounds: () => ({top: 110, bottom: 700})},
      state: {selectTurn: vi.fn()},
      virtualOutput: () => false,
      zone: {runOutsideAngular: (fn: () => void) => fn()},
    }) as ConversationPanel;
    return {fixture, root, panel, reading, token, scroll};
  }
  afterEach(() => vi.unstubAllGlobals());

  it('opens a layer graph using recorded forward links and the current tensor context', () => {
    const {fixture, panel} = setup();
    const item: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'activations',
      open: true,
    };
    const graph = vi.fn();
    Object.assign(panel, {
      turns: () => [{n: 2, step_start: 32}],
      rows: () => [
        {
          index: 1,
          ref: {step: 3, text: 'a', batch: 9, source_forward_id: 80},
          target: {step: 4, text: 'a', batch: 9, source_forward_id: 90},
        },
      ],
      alignment: () => 'content',
      state: {
        session: () => ({
          batches: [
            {batch: 8, turn: 2, phase: 'decode', forward_ids: {ref: 79, target: 89}},
            {batch: 9, turn: 2, phase: 'decode', forward_ids: {ref: 80, target: 90}},
          ],
        }),
      },
      overviewGraph: () => ({
        layerGraphContext: () => ({layer: 7, position: 33, anchor: 'pre_attention_norm'}),
      }),
      workspace: {graph},
      captureView: () => ({}),
    });
    panel.openAnalysisLayerGraph(item);
    expect(graph).toHaveBeenCalledWith(2, 'decode', {
      batch: 9,
      layer: 7,
      anchor: 'pre_attention_norm',
    });
    // A native-forward mismatch must ask for an observation, not select a coincidental batch ID.
    Object.assign(panel, {
      rows: () => [{index: 1, target: {step: 4, text: 'a', batch: 9, source_forward_id: 999}}],
    });
    panel.openAnalysisLayerGraph(item);
    expect(graph).toHaveBeenLastCalledWith(2, 'decode', {
      batch: undefined,
      layer: 7,
      anchor: 'pre_attention_norm',
    });
    fixture.destroy();
  });

  it('does not guess a graph forward from positions or mismatched runtime pairs', () => {
    const {fixture, panel} = setup();
    const item: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'activations',
      open: true,
    };
    Object.assign(panel, {
      turns: () => [{n: 2, step_start: 32}],
      alignment: () => 'steps',
      state: {
        session: () => ({
          batches: [{batch: 9, turn: 2, phase: 'decode', forward_ids: {ref: 80, target: 90}}],
        }),
      },
      rows: () => [{index: 1, target: {step: 4, text: 'a'}}],
    });
    expect(panel.recordedPositionBatch(item, 36)).toBeUndefined();
    Object.assign(panel, {
      rows: () => [
        {
          index: 1,
          ref: {step: 3, text: 'a', batch: 8},
          target: {step: 4, text: 'a', batch: 9, source_forward_id: 90},
        },
      ],
    });
    expect(panel.recordedPositionBatch(item, 36)).toBeUndefined();
    fixture.destroy();
  });

  it('retains analysis panels on a layer graph round trip without changing ordinary exits', () => {
    const {fixture, panel} = setup();
    const item: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'activations',
      open: true,
      graph: {batchId: null, metric: 'relative_l2', anchor: 'pre_attention_norm'},
    };
    const view = {analysisPanels: [item], inspectedPanel: item.id, analysisFindScope: 'analysis'};
    const saved = new Map();
    Object.assign(panel, {
      captureView: () => view,
      contextKey: 'capture',
      state: {
        loading: () => false,
        batchId: () => 9,
        selectBatch: vi.fn(),
        session: () => ({
          batches: [
            {batch: 0, turn: 2, phase: 'decode'},
            {batch: 9, turn: 2, phase: 'decode'},
          ],
        }),
      },
      workspace: {
        graphTarget: signal({capture: 'stale', batch: 0, layer: 0}),
        view: signal('Token Diff'),
        conversationViews: saved,
      },
    });
    (panel as unknown as {saveView(): void}).saveView();
    expect(saved.get('capture').analysisPanels).toEqual([]);
    panel.openLayerGraph();
    expect(panel.state.selectBatch).not.toHaveBeenCalled();
    expect(panel.workspace.graphTarget()).toBeNull();
    expect(panel.workspace.view()).toBe('Graph Diff');
    // A destroyed child no longer appears in viewChildren; preserve the pre-navigation snapshot.
    Object.assign(panel, {captureView: () => ({analysisPanels: [{...item, graph: undefined}]})});
    (panel as unknown as {saveView(): void}).saveView();
    expect(saved.get('capture')).toEqual(view);
    fixture.destroy();
  });

  it('opens a captured batch when the previous graph observation is no longer available', () => {
    const {fixture, panel} = setup();
    Object.assign(panel, {
      state: {
        loading: () => false,
        batchId: () => 999,
        selectBatch: vi.fn(),
        session: () => ({batches: [{batch: 8, turn: 2, phase: 'decode'}]}),
      },
      workspace: {graphTarget: signal(null), view: signal('Token Diff')},
      captureView: vi.fn(() => ({})),
    });
    panel.openLayerGraph();
    expect(panel.state.selectBatch).toHaveBeenCalledWith(8);
    expect(panel.workspace.graphTarget()).toBeNull();
    expect(panel.workspace.view()).toBe('Graph Diff');
    fixture.destroy();
  });

  it('does not leave analysis while the session is loading or has no observations', () => {
    const {fixture, panel} = setup();
    const loading = signal(false);
    const session = signal<{batches: {batch: number}[]} | null>(null);
    const captureView = vi.fn();
    Object.assign(panel, {
      state: {loading, session},
      workspace: {graphTarget: signal(null), view: signal('Token Diff')},
      captureView,
    });
    panel.openLayerGraph();
    session.set({batches: []});
    panel.openLayerGraph();
    loading.set(true);
    session.set({batches: [{batch: 0}]});
    panel.openLayerGraph();
    expect(captureView).not.toHaveBeenCalled();
    expect(panel.workspace.view()).toBe('Token Diff');
    fixture.destroy();
  });

  it('shows live analysis view properties and falls back to saved inspection settings', () => {
    const {fixture, panel} = setup();
    const item: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'kv',
      open: false,
    };
    const tensorView = signal(false),
      view = signal('heatmap'),
      tensorPlot = signal('index');
    const metric = signal('cosine_similarity'),
      position = signal(42);
    const live = {
      tensorView,
      view,
      tensorPlot,
      metric,
      tensorPosition: position,
      tensorLayer: () => 3,
      tensorHead: () => 1,
      visibleRange: () => ({start: 32, end: 67}),
    };
    Object.assign(panel, {
      analysisPanels: () => [item],
      overviewKv: () => live,
      overviewGraph: () => undefined,
    });
    const properties = () =>
      Object.fromEntries(panel.analysisViewProperties(item).map((row) => [row.label, row.value]));
    expect(properties()).toEqual({
      Mode: 'Tensor Comparison',
      Visualization: 'Heatmap',
      'Token Range': '32–66',
      Metric: 'Cosine similarity',
    });
    view.set('trends');
    metric.set('max_abs');
    expect(properties()).toMatchObject({Visualization: 'Line chart', Metric: 'Max abs error'});
    tensorView.set(true);
    tensorPlot.set('scatter');
    position.set(50);
    expect(properties()).toEqual({
      Mode: 'Tensor Inspection',
      Visualization: 'Scatter plot',
      'Token Position': '50',
      'Tensor Index': '[Layer: 3, Head: 1]',
    });
    Object.assign(item, {
      kv: {tensor: {active: true, plot: 'index', position: 44, layer: 2, head: 0}},
    });
    Object.assign(panel, {overviewKv: () => undefined});
    expect(properties()).toEqual({
      Mode: 'Tensor Inspection',
      Visualization: 'Line chart',
      'Token Position': '44',
      'Tensor Index': '[Layer: 2, Head: 0]',
    });
    fixture.destroy();
  });
  it.each([
    ['activations', undefined],
    ['activations', 'attention_output'],
    ['graph', 'mlp_output'],
  ] as const)(
    'keeps %s inspection properties for %s available after the panel is collapsed',
    (kind, anchor) => {
      const {fixture, panel} = setup();
      const inspection = {
        position: 44,
        anchor,
        layer: 2,
        plot: 'scatter' as const,
        elementRange: null,
        valueRange: null,
        selection: null,
        reduction: 'mean' as const,
      };
      const item: StageAnalysisPanel = {
        id: 'analysis-1',
        turn: 2,
        stage: 'decode',
        kind,
        open: false,
        instanceNumber: 1,
        graph: {
          batchId: null,
          metric: 'cosine_similarity',
          anchor: 'output',
          mode: 'inspection',
          inspection,
        },
      };
      Object.assign(panel, {
        analysisPanels: () => [item],
        overviewKv: () => undefined,
        overviewGraph: () => undefined,
      });
      const properties = () =>
        Object.fromEntries(panel.analysisViewProperties(item).map((row) => [row.label, row.value]));
      expect(properties()).toEqual({
        Mode: 'Tensor Inspection',
        Visualization: 'Scatter plot',
        'Token Position': '44',
        'Tensor Index': `[Layer: 2, Sync point: ${anchor ?? 'Layer output'}]`,
      });
      expect(panel.analysisLabel(item)).toBe('Intermediate Tensors');
      expect(panel.analysisInstanceLabel(item)).toBe('Intermediate Tensor Analysis 0');
      Object.assign(panel, {
        overviewGraph: () => ({
          inspectionMode: () => true,
          inspection: () => ({
            captureView: () => ({
              ...inspection,
              position: 45,
              anchor: 'pre_attention_norm',
              plot: 'index',
            }),
          }),
          view: () => 'heatmap',
        }),
      });
      expect(properties()).toMatchObject({
        Visualization: 'Line chart',
        'Token Position': '45',
        'Tensor Index': '[Layer: 2, Sync point: pre_attention_norm]',
      });
      fixture.destroy();
    },
  );
  it('adds the unified kind after existing legacy panels with the next shared number', () => {
    const {fixture, panel} = setup();
    const legacy: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'graph',
      open: true,
      instanceNumber: 2,
      graph: {
        batchId: null,
        metric: 'cosine_similarity',
        anchor: 'max',
        anchorSelection: ['mlp_output'],
      },
    };
    Object.assign(panel, {
      analysisPanels: signal([legacy]),
      analysisSerial: 1,
      tokenMetricInspector: signal(null),
      setGroup: vi.fn(),
      isExpandedStage: () => false,
      locateAnalysis: vi.fn(),
    });
    panel.addAnalysis(2, 'decode', 'graph');
    const added = panel.analysisPanels().at(-1)!;
    expect(added).toMatchObject({id: 'analysis-2', kind: 'activations', instanceNumber: 3});
    expect(panel.analysisPanels()[0].graph).toEqual(legacy.graph);
    expect(panel.analysisInstanceLabel(added)).toBe('Intermediate Tensor Analysis 2');
    expect(panel.analysisInstanceLabel(legacy)).toBe('Intermediate Tensor Analysis 1');
    fixture.destroy();
  });

  it('enables Intermediate Tensors for internal tensors when output was not captured', () => {
    const {fixture, panel} = setup();
    const rows = signal([{anchor: 'pre_attention_norm', shape: [16]}]);
    Object.assign(panel, {
      analysisGraph: {loading: () => false, error: () => null, data: () => ({rows: rows()})},
    });
    expect(panel.analysisUnavailable(2, 'decode', 'activations')).toBe('');
    expect(panel.analysisUnavailable(2, 'decode', 'graph')).toBe('');
    rows.set([]);
    expect(panel.analysisUnavailable(2, 'decode', 'activations')).toBe(
      'Comparable layer tensors not captured',
    );
    fixture.destroy();
  });

  it('uses the same sidebar scope for direct tokens and analysis ranges, including gaps', () => {
    const {fixture, panel} = setup();
    const pairs = [
      {index: 0, ref: {step: 0, text: 'a'}, target: {step: 0, text: 'a'}, match: true},
      {index: 1, target: {step: 1, text: 'b'}, match: false},
    ];
    Object.assign(panel, {
      turns: () => [{n: 2, step_start: 32, prefill_tokens: 4}],
      alignment: () => 'content',
      compactPairs: () => pairs,
      captured: (_turn: number, side: string) => ({
        tokens: side === 'ref' ? [pairs[0].ref] : pairs.map((p) => p.target),
      }),
    });
    const summary = () => [
      panel.stagePositions(2, 'decode'),
      panel.stageInfoCount(2, 'ref', 'decode'),
      panel.stageInfoCount(2, 'target', 'decode'),
      panel.stageTokenMatchRate(2, 'decode'),
    ];
    panel.selection.set({kind: 'token', turn: 2, index: 1});
    expect(summary()).toEqual(['33', 0, 1, '0%']);
    expect(panel.stagePositions(2, 'decode', true)).toBe('32–33');
    expect(panel.infoTokenPosition(pairs[1], 'ref')).toBe('∅');
    panel.selection.set({
      kind: 'stage',
      turn: 2,
      stage: 'decode',
      side: 'target',
      range: {start: 33, end: 34},
    });
    expect(summary()).toEqual(['33', 0, 1, '0%']);
    panel.selection.set({kind: 'stage', turn: 2, stage: 'decode', side: 'target'});
    expect(summary()).toEqual(['32–33', 1, 2, '50%']);
    panel.selection.set({kind: 'input', turn: 2, index: 0});
    expect(panel.selectionInfo()).toMatchObject({stage: 'prefill', range: {start: 28, end: 29}});
    fixture.destroy();
  });

  it('focuses an analysis range separately from direct token selection and clears it', () => {
    const {fixture, root, panel} = setup();
    const interval = signal<{start: number; end: number} | null>({start: 112, end: 113});
    Object.assign(panel, {
      analysisPanels: () => [{id: 'analysis', turn: 2, stage: 'decode', open: true}],
      analysisHover: () => null,
      metricInspector: () => ({panelId: () => 'analysis', selected: interval}),
      alignment: () => 'content',
      inputPairRows: () => [],
      rows: () => Array.from({length: 13}, () => ({ref: {step: 9}, target: {step: 12}})),
      state: {session: () => ({turns: [{n: 2, step_start: 100, prefill_tokens: 10}]})},
    });
    const refresh = () => (panel as unknown as {refreshTokenLinks(): void}).refreshTokenLinks();
    refresh();
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(2);
    expect(root.querySelectorAll('.linked-token-selected,.selection-origin')).toHaveLength(0);
    root.querySelector('.reading')!.insertAdjacentHTML(
      'beforeend',
      `
      <button data-turn="2" data-row="11" data-side="ref">outside reference</button>
      <button data-turn="2" data-row="11" data-side="target">outside target</button>`,
    );
    panel.selection.set({kind: 'stage', turn: 2, stage: 'decode', side: 'target'});
    refresh();
    // A panel's point/range takes precedence over its Stage identity.
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(2);
    Object.assign(panel, {metricInspector: () => null});
    refresh();
    // Selecting the entire Stage focuses every token in that Stage, not Prefill.
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(4);
    expect(
      root.querySelector('[data-kind="input"]')!.classList.contains('linked-token-focus'),
    ).toBe(false);
    panel.selection.set({
      kind: 'stage',
      turn: 2,
      stage: 'decode',
      side: 'target',
      range: {start: 112, end: 113},
    });
    refresh();
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(2);
    panel.selection.set({kind: 'token', turn: 2, index: 12});
    refresh();
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(0);
    expect(root.querySelectorAll('.linked-token-selected')).toHaveLength(1);
    expect(root.querySelectorAll('.selection-origin')).toHaveLength(1);
    panel.selection.set(null);
    interval.set(null);
    refresh();
    expect(
      root.querySelectorAll('.linked-token-focus,.linked-token-selected,.selection-origin'),
    ).toHaveLength(0);
    fixture.destroy();
  });

  it('locates the whole analysis area without changing token or chart selection', () => {
    const {fixture, root, panel} = setup();
    const view: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'kv',
      open: false,
    };
    root.insertAdjacentHTML(
      'beforeend',
      '<section class="stage-analysis"><div class="analysis-location-frame"></div><div id="analysis-1"><div class="chart"></div></div></section>',
    );
    const area = root.querySelector('.stage-analysis') as HTMLElement;
    const frame = root.querySelector('.analysis-location-frame') as HTMLElement;
    area.scrollIntoView = vi.fn();
    frame.animate = vi.fn(() => ({cancel: vi.fn()}) as unknown as Animation);
    Object.assign(panel, {analysisPanels: signal([view])});
    panel.selection.set({kind: 'token', turn: 2, index: 12});
    panel.locateAnalysisArea(view);
    fixture.detectChanges();
    expect(panel.analysisPanels()[0].open).toBe(true);
    expect(area.scrollIntoView).toHaveBeenCalledOnce();
    expect(frame.animate).toHaveBeenCalledOnce();
    const frames = vi.mocked(frame.animate).mock.calls[0][0] as Keyframe[];
    expect(frames.every((keyframe) => keyframe.transform === undefined)).toBe(true);
    expect(frames[0].boxShadow).toContain('var(--me-primary-color)');
    expect(panel.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    expect(panel.metricInspector()).toBeNull();
    fixture.destroy();
  });

  it('locates Activation range and groups without replacing the current selection', () => {
    const {fixture, panel} = setup();
    const analysis: StageAnalysisPanel = {
      id: 'activation-1',
      turn: 2,
      stage: 'decode',
      kind: 'activations',
      open: false,
    };
    const chart = {
      panelId: () => analysis.id,
      locateRange: vi.fn(),
      locateLayers: vi.fn(),
      locateView: vi.fn(),
    };
    Object.assign(panel, {
      analysisPanels: signal([analysis]),
      kvPanels: () => [],
      graphPanels: () => [chart],
    });
    panel.selection.set({kind: 'token', turn: 2, index: 12});
    panel.locateScopeControl(analysis, 'positions');
    fixture.detectChanges();
    panel.locateScopeControl(analysis, 'layers');
    fixture.detectChanges();
    panel.locateTokenView(analysis);
    fixture.detectChanges();
    expect(chart.locateRange).toHaveBeenCalledOnce();
    expect(chart.locateLayers).toHaveBeenCalledOnce();
    expect(chart.locateView).toHaveBeenCalledOnce();
    expect(panel.analysisPanels()[0].open).toBe(true);
    expect(panel.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    fixture.destroy();
  });

  it('updates a clicked token without scrolling any ancestor', () => {
    const {fixture, panel, reading, token, scroll} = setup();
    panel.select(2, 12, {currentTarget: token} as unknown as Event);
    fixture.detectChanges();
    expect(panel.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    expect(scroll).not.toHaveBeenCalled();
    expect(reading.scrollTo).not.toHaveBeenCalled();
    fixture.destroy();
  });

  it('re-inspecting the selected token via inspectTokenMetric keeps the selection and details open', () => {
    const {fixture, panel, token} = setup();
    Object.assign(panel, {tokenMetricRange: signal(null)});
    const analysis: StageAnalysisPanel = {
      id: 'tokens-1',
      turn: 2,
      stage: 'decode',
      kind: 'tokens',
      open: true,
    };
    panel.select(2, 12, {currentTarget: token} as unknown as Event);
    expect(panel.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    expect(panel.details()).toBe(true);
    panel.inspectTokenMetric(analysis, 12);
    expect(panel.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    expect(panel.details()).toBe(true);
    expect(panel.tokenMetricInspector()).toBe(analysis);
    fixture.destroy();
  });

  it('locates the selected runtime and stage inside the unobscured reading viewport', () => {
    const {fixture, root, panel, reading, token, scroll} = setup();
    panel.selection.set({kind: 'token', turn: 2, index: 12});
    const box = (top: number, bottom: number) => ({top, bottom, height: bottom - top}) as DOMRect;
    reading.scrollTop = 100;
    reading.getBoundingClientRect = () => box(50, 850);
    root.querySelector('.column-headings')!.getBoundingClientRect = () => box(50, 110);
    root.querySelector('.composer')!.getBoundingClientRect = () => box(700, 840);
    token.getBoundingClientRect = () => box(600, 620);
    panel.locateSelectedToken();
    fixture.detectChanges();
    expect(scroll).not.toHaveBeenCalled();
    // Visible area is 110..700; center the actual target token there.
    expect(reading.scrollTo).toHaveBeenCalledWith({top: 305, behavior: 'auto'});
    expect(token.animate).toHaveBeenCalledOnce();
    expect(root.querySelector('button[data-side="ref"]')!.animate).toHaveBeenCalledOnce();
    fixture.destroy();
  });
});

describe('Stage Info token scope', () => {
  function setup() {
    const reference = [
      {step: 0, text: '<bos>', kind: 'special'},
      {step: 1, text: '<turn>', kind: 'template'},
      {step: 2, text: 'hello'},
    ];
    const target = [...reference.slice(0, 2), {step: 2, text: 'extra'}, {step: 3, text: 'hello'}];
    const pairs = [
      {index: 0, ref: reference[0], target: target[0], match: true},
      {index: 1, ref: reference[1], target: target[1], match: true},
      {index: 2, target: target[2], match: false},
      {index: 3, ref: reference[2], target: target[3], match: true},
    ];
    const selection = signal<{turn: number; stage: string; range?: {start: number; end: number}}>({
      turn: 1,
      stage: 'prefill',
    });
    const captures = new Map([
      ['ref', {input_tokens: reference, tokens: reference}],
      ['target', {input_tokens: target, tokens: target}],
    ]);
    const panel = Object.assign(Object.create(ConversationPanel.prototype), {
      selectedStage: selection,
      selection: () => ({kind: 'stage', ...selection()}),
      alignment: () => 'content',
      turns: () => [{n: 1, step_start: 4, prefill_tokens: 4}],
      captured: (_turn: number, side: string) => captures.get(side),
      inputPairRows: () => [{leading: [], text: pairs, trailing: []}],
      rows: () => pairs,
    }) as ConversationPanel;
    return {panel, selection, captures, pairs};
  }

  it('counts special and template tokens but excludes aligned gaps', () => {
    const {panel, selection} = setup();
    expect(panel.stagePositions(1, 'prefill')).toBe('0–3');
    expect(panel.stageInfoCount(1, 'ref', 'prefill')).toBe(3);
    expect(panel.stageInfoCount(1, 'target', 'prefill')).toBe(4);
    expect(panel.stageTokenMatchRate(1, 'prefill')).toBe('75%');
    selection.set({turn: 1, stage: 'prefill', range: {start: 0, end: 2}});
    expect(panel.stageInfoCount(1, 'ref', 'prefill')).toBe(2);
    expect(panel.stageTokenMatchRate(1, 'prefill')).toBe('100%');
    selection.set({turn: 1, stage: 'prefill', range: {start: 2, end: 3}});
    expect(panel.stagePositions(1, 'prefill')).toBe('2');
    expect(panel.stageInfoCount(1, 'ref', 'prefill')).toBe(0);
    expect(panel.stageInfoCount(1, 'target', 'prefill')).toBe(1);
    expect(panel.stageTokenMatchRate(1, 'prefill')).toBe('0%');
  });

  it('uses aligned range positions instead of each runtime token step', () => {
    const {panel, selection} = setup();
    selection.set({turn: 1, stage: 'prefill', range: {start: 3, end: 4}});
    expect(panel.stageInfoCount(1, 'ref', 'prefill')).toBe(1);
    expect(panel.stageInfoCount(1, 'target', 'prefill')).toBe(1);
    expect(panel.stageTokenMatchRate(1, 'prefill')).toBe('100%');
    selection.set({turn: 1, stage: 'decode', range: {start: 7, end: 8}});
    expect(panel.stageInfoCount(1, 'ref', 'decode')).toBe(1);
    expect(panel.stageTokenMatchRate(1, 'decode')).toBe('100%');
  });

  it('does not claim a match for missing captures or an empty range', () => {
    const {panel, selection, captures} = setup();
    selection.set({turn: 1, stage: 'prefill', range: {start: 10, end: 11}});
    expect(panel.stageTokenMatchRate(1, 'prefill')).toBe('-');
    captures.delete('target');
    expect(panel.stageInfoCount(1, 'target', 'prefill')).toBe('Not captured');
    expect(panel.stageTokenMatchRate(1, 'prefill')).toBe('-');
  });
});

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
    const panel = Object.assign(Object.create(ConversationPanel.prototype), {
      selection: signal({kind: 'stage', turn: 2, stage: 'decode', side: 'target'}),
      analysisPanels: signal([analysis]),
      analysisOverview: signal(null),
      metricInspector: signal(inspector),
      tokenMetricInspector: signal(null),
      tokenMetricRange: signal(null),
      analysisHover: signal({panelId: analysis.id, range: range()}),
      expandedAnalysis: signal(null),
      alignment: () => 'content',
      turns: () => [{n: 2, step_start: 100, prefill_tokens: 10}],
      rows: () => [pair],
      selectedInputPair: () => null,
      selectedPair: () => pair,
    }) as ConversationPanel;
    return {panel, analysis, inspector};
  }

  for (const action of ['toggleAnalysis', 'removeAnalysis'] as const) {
    it.each(['kv', 'graph', 'tokens', 'distribution'] as const)(
      `${action} returns a focused %s position to Stage Info, not Token Info`,
      (kind) => {
        const {panel, analysis} = setup(kind);
        if (kind === 'tokens' || kind === 'distribution') {
          panel.metricInspector.set(null);
          panel.tokenMetricInspector.set(analysis);
          panel.selection.set({kind: 'token', turn: 2, index: 12});
        }
        panel[action](analysis);
        expect(panel.selection()).toEqual({
          kind: 'stage',
          turn: 2,
          stage: 'decode',
          side: 'target',
          range: {start: 112, end: 113},
        });
        expect(panel.metricInspector()).toBeNull();
        expect(panel.tokenMetricInspector()).toBeNull();
        expect(panel.analysisOverview()).toBeNull();
        expect(
          action === 'removeAnalysis'
            ? panel.analysisPanels().length === 0
            : !panel.analysisPanels()[0].open,
        ).toBe(true);
      },
    );
  }

  it('keeps another panel inspection when removing an inactive analysis', () => {
    const {panel, analysis, inspector} = setup();
    const other = {...analysis, id: 'analysis-2'};
    panel.analysisPanels.update((panels) => [...panels, other]);
    panel.removeAnalysis(other);
    expect(panel.metricInspector()).toBe(inspector);
    expect(panel.analysisPanels()).toEqual([analysis]);
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
        : ['graph', 'activations'].includes(kind)
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
        inspectionMode: () => ['graph', 'activations'].includes(kind) && tensorView,
        inspection: () =>
          ['graph', 'activations'].includes(kind) && tensorView ? tensor : undefined,
        fullRange: () => ({start: 32, end: 67}),
        layers: () => [0, 1, 2, 3],
        selectedLayers: () => [0],
        view: () => 'trends',
        request: {loading: () => false},
      },
    );
    const panel = Object.assign(Object.create(ConversationPanel.prototype), {
      analysisPanels: signal([analysis]),
      analysisViewports: new Map(),
      kvPanels: () => (kind === 'kv' ? [chart] : []),
      graphPanels: () => (['graph', 'activations'].includes(kind) ? [chart] : []),
      tokenPanels: () => [
        {panelId: () => analysis.id, finalChart: () => chart, inspection: () => undefined},
      ],
      metricInspector: signal(null),
      tokenMetricInspector: signal(null),
      tokenMetricRange: signal(null),
      analysisHover: signal(null),
      selection: signal(null),
      analysisOverview: signal(null),
      details: signal(false),
      expandedAnalysis: signal(null),
    }) as ConversationPanel;
    const sync = () => (panel as unknown as {syncAnalysisViewport(): void}).syncAnalysisViewport();
    return {analysis, panel, chart, tensor, viewport, picked, sync};
  }
  it.each(['kv', 'graph', 'activations', 'tokens', 'distribution'] as const)(
    '%s activates Analysis Info and updates scope immediately when range changes',
    (kind) => {
      const {panel, analysis, viewport, sync} = setup(kind);
      panel.inspectAnalysisStage(analysis, new MouseEvent('click'));
      expect(panel.overviewPanel()).toEqual(analysis);
      sync();
      expect(panel.selection()).toMatchObject({range: {start: 40, end: 51}});
      viewport.set({start: 42, end: 49});
      sync();
      expect(panel.selection()).toMatchObject({range: {start: 42, end: 49}});
      expect(panel.overviewPanel()).toEqual(analysis);
      panel.toggleAnalysis(analysis);
      expect(panel.overviewPanel()).toBeNull();
      expect(panel.selection()).toMatchObject({range: {start: 42, end: 49}});
    },
  );
  it('clears a selected bin on background click and returns the viewport when closing', () => {
    const {panel, analysis, chart} = setup();
    panel.metricInspector.set(chart);
    panel.inspectAnalysisStage(analysis, new MouseEvent('click'));
    expect(panel.metricInspector()).toBeNull();
    expect(panel.analysisOverview()).toEqual(analysis);
    panel.removeAnalysis(analysis);
    expect(panel.overviewPanel()).toBeNull();
    expect(panel.selection()).toMatchObject({range: {start: 40, end: 51}});
  });
  it.each(['graph', 'activations', 'tokens', 'distribution'] as const)(
    '%s clears the plotted selection and keeps the current analysis open',
    (kind) => {
      const {panel, analysis, chart} = setup(kind);
      if (kind === 'graph' || kind === 'activations') panel.metricInspector.set(chart);
      else panel.tokenMetricInspector.set(analysis);
      panel.inspectAnalysisStage(analysis, new MouseEvent('click'));
      expect(panel.metricInspector()).toBeNull();
      expect(panel.tokenMetricInspector()).toBeNull();
      expect(panel.analysisOverview()).toEqual(analysis);
      expect(panel.selection()).toMatchObject({range: {start: 40, end: 51}});
      if (kind === 'graph' || kind === 'activations') expect(chart.selected()).toBeNull();
    },
  );
  it.each(['kv', 'activations', 'graph'] as const)(
    '%s uses inspection coordinates for token focus instead of the stale comparison cursor',
    (kind) => {
      const {panel, analysis, chart} = setup(kind, true);
      chart.tensorPosition.set(46);
      panel.metricInspector.set(chart);
      const range = (
        panel as unknown as {
          selectionRangeFor(panel: StageAnalysisPanel): unknown;
        }
      ).selectionRangeFor(analysis);
      expect(range).toEqual({start: 46, end: 47, layer: 3});
    },
  );
  it.each(['activations', 'graph'] as const)(
    'clears %s elements and keeps tensor scope and element zoom on background click',
    (kind) => {
      const {panel, analysis, chart, tensor, viewport} = setup(kind, true);
      panel.metricInspector.set(chart);
      panel.inspectAnalysisStage(analysis, new MouseEvent('click'));
      expect(tensor.selectedIndex()).toBeNull();
      expect(tensor.channelSelection()).toBeNull();
      expect(tensor.position()).toBe(44);
      expect(tensor.layer()).toBe(3);
      expect(tensor.elementRange()).toEqual({start: 2, end: 8});
      expect(viewport()).toEqual({start: 40, end: 51});
      expect(panel.metricInspector()).toBeNull();
      expect(panel.analysisOverview()).toEqual(analysis);
      expect(panel.selection()).toMatchObject({range: {start: 44, end: 45}});
    },
  );
  it.each(['activations', 'graph'] as const)(
    'keeps %s inspection focus outside comparison zoom and hidden comparison layers',
    (kind) => {
      const {panel, analysis, chart, tensor, viewport, picked} = setup(kind, true);
      viewport.set({start: 40, end: 43});
      tensor.position.set(60);
      picked.set({start: 44, end: 47, layer: 3});
      panel.metricInspector.set(chart);
      const syncVisibility = () =>
        (
          panel as unknown as {syncMetricInspectorVisibility(): void}
        ).syncMetricInspectorVisibility();
      syncVisibility();
      expect(panel.metricInspector()).toBe(chart);
      expect(panel.analysisOverview()).toBeNull();
      chart.inspectionMode = () => false;
      syncVisibility();
      expect(panel.metricInspector()).toBeNull();
      expect(panel.analysisOverview()).toEqual(analysis);
    },
  );
  it('clears inspection elements without resetting tensor coordinates or zoom', () => {
    const {panel, analysis, chart, viewport} = setup('kv', true);
    panel.metricInspector.set(chart);
    panel.inspectAnalysisStage(analysis, new MouseEvent('click'));
    expect(chart.tensorChannels()).toEqual({key: null, value: null});
    expect(chart.tensorChannelSelection()).toBeNull();
    expect(chart.tensorPosition()).toBe(44);
    expect(viewport()).toEqual({start: 40, end: 51});
    expect(panel.analysisOverview()).toEqual(analysis);
    expect(panel.selection()).toMatchObject({range: {start: 44, end: 45}});
  });
  it('range changes replace a bin even when the bin still fits in the new viewport', () => {
    const {panel, analysis, chart, viewport, sync} = setup();
    panel.metricInspector.set(chart);
    sync();
    viewport.set({start: 42, end: 49});
    sync();
    expect(panel.metricInspector()).toBeNull();
    expect(panel.analysisOverview()).toEqual(analysis);
    expect(panel.selection()).toMatchObject({range: {start: 42, end: 49}});
  });
  it.each(['kv', 'activations', 'graph'] as const)(
    '%s tensor scope follows its position, independently of metric or element zoom',
    (kind) => {
      const {panel, analysis, chart, viewport, sync} = setup(kind, true);
      panel.inspectAnalysisStage(analysis, new MouseEvent('click'));
      sync();
      viewport.set({start: 45, end: 49});
      sync();
      expect(panel.selection()).toMatchObject({range: {start: 44, end: 45}});
      chart.tensorPosition.set(46);
      sync();
      panel.toggleAnalysis(analysis);
      expect(panel.overviewPanel()).toBeNull();
      expect(panel.selection()).toMatchObject({range: {start: 46, end: 47}});
    },
  );
});
