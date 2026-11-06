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

import {MetricCatalogStore} from '../../server/metric_catalog';
import type {StageAnalysisPanel} from '../stage_analysis';
import {debugParts, type PartFields} from './testing/debug_parts';

function pointerOverEvent(
  target: EventTarget,
  relatedTarget: EventTarget | null = null,
): PointerEvent {
  const event = new PointerEvent('pointerover', {bubbles: true, relatedTarget});
  Object.defineProperty(event, 'target', {value: target});
  return event;
}

function clickEventOn(currentTarget: EventTarget): MouseEvent {
  const event = new MouseEvent('click', {bubbles: true});
  Object.defineProperty(event, 'currentTarget', {value: currentTarget});
  return event;
}

/** A read turn whose stages start where the server says, the same on both sides. */
function turnAt(n: number, prefill: number, decode: number) {
  const stage = (start: number) => ({
    positions: {ref: {start, end: start + 1}, target: {start, end: start + 1}},
  });
  return {n, stages: {prefill: stage(prefill), decode: stage(decode)}};
}

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
    const reading = root.querySelector<HTMLElement>('.reading');
    const token = root.querySelector<HTMLElement>('button[data-side="target"]:not([data-kind])');
    if (!reading || !token) {
      throw new Error('Expected reading and token elements');
    }
    const scroll = vi.fn();
    for (const e of Array.from(root.querySelectorAll('*'))) {
      e.scrollIntoView = scroll;
      e.animate = vi.fn().mockReturnValue({cancel: vi.fn()});
    }
    reading.scrollTo = vi.fn();
    /** The parts over this page's DOM, with `fields` added to them. */
    const parts = (fields: PartFields = {}) => {
      const metricCatalog = TestBed.inject(MetricCatalogStore);
      return debugParts({
        conversation: {
          element: {nativeElement: root},
          injector: fixture.debugElement.injector,
          reading: {visibleBounds: () => ({top: 110, bottom: 700})},
          state: {selectTurn: vi.fn(), session: () => null},
          virtualOutput: () => false,
          zone: {runOutsideAngular: (fn: () => void) => fn()},
          ...fields.conversation,
        },
        tokens: {
          metricCatalog,
          selection: signal(null),
          selectionSide: signal('target'),
          analysisOverview: signal(null),
          metricInspector: signal(null),
          tokenMetricInspector: signal(null),
          details: signal(true),
          collapsed: signal(new Set<string>()),
          ...fields.tokens,
        },
        panels: fields.panels,
        analysisInfo: {metricCatalog, ...fields.analysisInfo},
        detailsInfo: {metricCatalog, ...fields.detailsInfo},
        dom: fields.dom,
        views: {metricCatalog, pendingView: signal(null), ...fields.views},
      });
    };
    return {fixture, root, parts, reading, token, scroll};
  }
  afterEach(() => vi.unstubAllGlobals());

  it('drops analysis panels from the saved view when leaving the conversation', () => {
    const {fixture, parts} = setup();
    const item: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'activations',
      open: true,
      graph: {metric: 'relative_l2', anchor: 'pre_attention_norm'},
    };
    const saved = new Map();
    const {views} = parts({
      conversation: {contextKey: 'capture', memory: {conversationViews: saved}},
      views: {
        captureView: () => ({
          analysisPanels: [item],
          inspectedPanel: item.id,
          analysisFindScope: 'analysis',
        }),
      },
    });
    views.saveView();
    expect(saved.get('capture')).toEqual({
      analysisPanels: [],
      inspectedPanel: null,
      analysisFindScope: '',
    });
    fixture.destroy();
  });

  it('shows live analysis view properties and falls back to saved inspection settings', () => {
    const {fixture, parts} = setup();
    const item: StageAnalysisPanel = {
      id: 'analysis-1',
      turn: 2,
      stage: 'decode',
      kind: 'kv',
      open: false,
    };
    const tensorView = signal(false);
    const view = signal('heatmap');
    const tensorPlot = signal('index');
    const metric = signal('cosine_similarity');
    const position = signal(42);
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
    const {panels, analysisInfo} = parts({
      panels: {
        analysisPanels: () => [item],
        overviewKv: () => live,
        overviewGraph: () => undefined,
      },
    });
    const properties = () =>
      Object.fromEntries(
        analysisInfo.analysisViewProperties(item).map((row) => [row.label, row.value]),
      );
    expect(properties()).toEqual({
      Mode: 'Tensor Comparison',
      Visualization: 'Heatmap',
      'Token Range': '32–66',
      Metric: 'Cosine similarity',
    });
    view.set('lineChart');
    metric.set('max_abs');
    expect(properties()).toMatchObject({Visualization: 'Line chart', Metric: 'Max absolute error'});
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
    Object.assign(panels, {overviewKv: () => undefined});
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
    ['activations', 'mlp_output'],
  ] as const)(
    'keeps %s inspection properties for %s available after the panel is collapsed',
    (kind, anchor) => {
      const {fixture, parts} = setup();
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
          metric: 'cosine_similarity',
          anchor: 'output',
          mode: 'inspection',
          inspection,
        },
      };
      const {panels, analysisInfo} = parts({
        panels: {
          analysisPanels: () => [item],
          overviewKv: () => undefined,
          overviewGraph: () => undefined,
        },
      });
      const properties = () =>
        Object.fromEntries(
          analysisInfo.analysisViewProperties(item).map((row) => [row.label, row.value]),
        );
      expect(properties()).toEqual({
        Mode: 'Tensor Inspection',
        Visualization: 'Scatter plot',
        'Token Position': '44',
        'Tensor Index': `[Layer: 2, Sync point: ${anchor ?? 'Layer output'}]`,
      });
      expect(analysisInfo.analysisLabel(item)).toBe('Intermediate Tensors');
      expect(analysisInfo.analysisInstanceLabel(item)).toBe('Intermediate Tensor Analysis 0');
      Object.assign(panels, {
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
          // The server's label of the sync point.
          anchorLabel: (id: string) => `attn/norm · ${id}`,
        }),
      });
      expect(properties()).toMatchObject({
        Visualization: 'Line chart',
        'Token Position': '45',
        'Tensor Index': '[Layer: 2, Sync point: attn/norm · pre_attention_norm]',
      });
      fixture.destroy();
    },
  );
  it('offers the analyses the server says a stage offers, with its reasons', () => {
    const {fixture, parts} = setup();
    const yes = {available: true, reasons: []};
    const no = (...reasons: [string, 'ref' | 'target' | null, string][]) => ({
      available: false,
      reasons: reasons.map(([code, side, text]) => ({code, side, text})),
    });
    const stage = (moment: string, analyses: object) => ({
      positions: {ref: {start: 0, end: 4}, target: {start: 0, end: 4}},
      kv: {moment, context_id: 'snapshot-a'},
      analyses,
    });
    const tensors = no(['no_intermediate_tensors', null, 'No intermediate tensors captured']);
    const notDecode =
      'Only Decode rows name generated tokens, whose logits produced them; a Prefill samples none';
    const {panels} = parts({
      conversation: {
        state: {
          session: () => ({
            turns: [
              {
                n: 2,
                stages: {
                  prefill: stage('prefill_post', {
                    final_logits: no(['not_decode', null, notDecode]),
                    sampling: no(['not_decode', null, notDecode]),
                    kv: yes,
                    intermediate_tensors: tensors,
                  }),
                  decode: stage('terminal', {
                    final_logits: yes,
                    sampling: no(
                      ['logits_not_captured', 'ref', 'A'],
                      ['logits_invalid', 'target', 'B'],
                    ),
                    kv: yes,
                    intermediate_tensors: tensors,
                  }),
                },
              },
            ],
          }),
        },
      },
    });
    expect(panels.analysisUnavailable(2, 'decode', 'tokens')).toBe('');
    expect(panels.analysisUnavailable(2, 'decode', 'kv')).toBe('');
    expect(panels.analysisUnavailable(2, 'decode', 'distribution')).toBe('ref: A; target: B');
    expect(panels.analysisUnavailable(2, 'decode', 'activations')).toBe(
      'No intermediate tensors captured',
    );
    expect(panels.analysisUnavailable(2, 'prefill', 'tokens')).toBe(notDecode);
    expect(panels.analysisUnavailable(2, 'prefill', 'distribution')).toBe(notDecode);
    fixture.destroy();
  });

  it('uses the same sidebar scope for direct tokens and analysis ranges, including gaps', () => {
    const {fixture, parts} = setup();
    const pairs = [
      {index: 0, ref: {step: 0, text: 'a'}, target: {step: 0, text: 'a'}, match: true},
      {index: 1, target: {step: 1, text: 'b'}, match: false},
    ];
    const {tokens, detailsInfo: info} = parts({
      conversation: {
        turns: () => [turnAt(2, 28, 32)],
        alignment: () => 'content',
        captured: (_turn: number, side: string) => ({
          tokens: side === 'ref' ? [pairs[0].ref] : pairs.map((p) => p.target),
        }),
      },
      tokens: {compactPairs: () => pairs},
    });
    const summary = () => [
      info.stagePositions(2, 'decode'),
      info.stageInfoCount(2, 'ref', 'decode'),
      info.stageInfoCount(2, 'target', 'decode'),
      info.matchRateRequest()?.rows,
    ];
    tokens.selection.set({kind: 'token', turn: 2, index: 1});
    expect(summary()).toEqual(['33', 0, 1, [1]]);
    expect(info.stagePositions(2, 'decode', true)).toBe('32–33');
    tokens.selection.set({
      kind: 'stage',
      turn: 2,
      stage: 'decode',
      side: 'target',
      range: {start: 33, end: 34},
    });
    expect(summary()).toEqual(['33', 0, 1, [1]]);
    tokens.selection.set({kind: 'stage', turn: 2, stage: 'decode', side: 'target'});
    // A whole stage names no rows: the server counts all of them.
    expect(summary()).toEqual(['32–33', 1, 2, undefined]);
    tokens.selection.set({kind: 'input', turn: 2, index: 0});
    expect(info.selectionInfo()).toMatchObject({stage: 'prefill', range: {start: 28, end: 29}});
    fixture.destroy();
  });

  it('focuses an analysis range separately from direct token selection and clears it', () => {
    const {fixture, root, reading, parts} = setup();
    const interval = signal<{start: number; end: number} | null>({start: 112, end: 113});
    const {tokens, dom} = parts({
      conversation: {
        alignment: () => 'content',
        state: {session: () => ({turns: [turnAt(2, 90, 100)]})},
      },
      tokens: {
        metricInspector: () => ({panelId: () => 'analysis', selected: interval}),
        inputPairRows: () => [],
        rows: () => Array.from({length: 13}, () => ({ref: {step: 9}, target: {step: 12}})),
      },
      panels: {
        analysisPanels: () => [{id: 'analysis', turn: 2, stage: 'decode', open: true}],
        analysisHover: () => null,
      },
    });
    const refresh = () => dom.refreshTokenLinks();
    refresh();
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(2);
    expect(root.querySelectorAll('.linked-token-selected,.selection-origin')).toHaveLength(0);
    reading.insertAdjacentHTML(
      'beforeend',
      `
      <button data-turn="2" data-row="11" data-side="ref">outside reference</button>
      <button data-turn="2" data-row="11" data-side="target">outside target</button>`,
    );
    tokens.selection.set({kind: 'stage', turn: 2, stage: 'decode', side: 'target'});
    refresh();
    // A panel's point/range takes precedence over its Stage identity.
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(2);
    Object.assign(tokens, {metricInspector: () => null});
    refresh();
    // Selecting the entire Stage focuses every token in that Stage, not Prefill.
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(4);
    const inputBtn = root.querySelector<HTMLElement>('[data-kind="input"]');
    expect(inputBtn?.classList.contains('linked-token-focus')).toBe(false);
    tokens.selection.set({
      kind: 'stage',
      turn: 2,
      stage: 'decode',
      side: 'target',
      range: {start: 112, end: 113},
    });
    refresh();
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(2);
    tokens.selection.set({kind: 'token', turn: 2, index: 12});
    refresh();
    expect(root.querySelectorAll('.linked-token-focus')).toHaveLength(0);
    expect(root.querySelectorAll('.linked-token-selected')).toHaveLength(1);
    expect(root.querySelectorAll('.selection-origin')).toHaveLength(1);
    tokens.selection.set(null);
    interval.set(null);
    refresh();
    expect(
      root.querySelectorAll('.linked-token-focus,.linked-token-selected,.selection-origin'),
    ).toHaveLength(0);
    fixture.destroy();
  });

  it('locates the whole analysis area without changing token or chart selection', () => {
    const {fixture, root, parts} = setup();
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
    const area = root.querySelector<HTMLElement>('.stage-analysis');
    const frame = root.querySelector<HTMLElement>('.analysis-location-frame');
    if (!area || !frame) {
      throw new Error('Expected stage-analysis and analysis-location-frame elements');
    }
    area.scrollIntoView = vi.fn();
    frame.animate = vi.fn().mockReturnValue({cancel: vi.fn()});
    const {tokens, panels, analysisInfo} = parts({panels: {analysisPanels: signal([view])}});
    tokens.selection.set({kind: 'token', turn: 2, index: 12});
    analysisInfo.locateAnalysisArea(view);
    fixture.detectChanges();
    expect(panels.analysisPanels()[0].open).toBe(true);
    expect(area.scrollIntoView).toHaveBeenCalledOnce();
    expect(frame.animate).toHaveBeenCalledOnce();
    const frames = vi.mocked(frame.animate).mock.calls[0][0] as Keyframe[];
    expect(frames.every((keyframe) => keyframe.transform === undefined)).toBe(true);
    expect(frames[0].boxShadow).toContain('var(--me-primary-color)');
    expect(tokens.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    expect(tokens.metricInspector()).toBeNull();
    fixture.destroy();
  });

  it('locates Activation range and groups without replacing the current selection', () => {
    const {fixture, parts} = setup();
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
    const {tokens, panels, analysisInfo} = parts({
      panels: {
        analysisPanels: signal([analysis]),
        kvPanels: () => [],
        graphPanels: () => [chart],
      },
    });
    tokens.selection.set({kind: 'token', turn: 2, index: 12});
    analysisInfo.locateScopeControl(analysis, 'positions');
    fixture.detectChanges();
    analysisInfo.locateScopeControl(analysis, 'layers');
    fixture.detectChanges();
    analysisInfo.locateTokenView(analysis);
    fixture.detectChanges();
    expect(chart.locateRange).toHaveBeenCalledOnce();
    expect(chart.locateLayers).toHaveBeenCalledOnce();
    expect(chart.locateView).toHaveBeenCalledOnce();
    expect(panels.analysisPanels()[0].open).toBe(true);
    expect(tokens.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    fixture.destroy();
  });

  it('clears the pair-hover frame on a downward wheel or key even when nothing scrolls', () => {
    // jsdom has no PointerEvent; the reading asks of each event whether it is one.
    vi.stubGlobal('PointerEvent', class extends MouseEvent {});
    const {fixture, parts, root, token} = setup();
    const {dom} = parts({
      conversation: {workspace: {mode: () => 'Debug'}},
      tokens: {forks: () => new Map()},
      dom: {tokenFeedback: []},
    });
    const framed = () =>
      Array.from(root.querySelectorAll('.pair-hover')).map((e) => e.getAttribute('data-side'));
    const intents = [
      new WheelEvent('wheel', {deltaY: 40}),
      new KeyboardEvent('keydown', {key: 'ArrowDown'}),
      new KeyboardEvent('keydown', {key: 'PageDown'}),
      new KeyboardEvent('keydown', {key: 'End'}),
    ];
    for (const intent of intents) {
      dom.tokenPointerOver(pointerOverEvent(token));
      expect(framed()).toEqual(['ref', 'target']);
      dom.trackReadingIntent(intent);
      expect(framed()).toEqual([]);
      expect(root.querySelector('.hover-origin')).toBeNull();
    }
    // Reading up, or a key that scrolls nothing, keeps the frame until the reading scrolls.
    for (const intent of [
      new WheelEvent('wheel', {deltaY: -40}),
      new KeyboardEvent('keydown', {key: 'ArrowUp'}),
      new KeyboardEvent('keydown', {key: 'a'}),
    ]) {
      dom.tokenPointerOver(pointerOverEvent(token));
      dom.trackReadingIntent(intent);
      expect(framed()).toEqual(['ref', 'target']);
    }
    dom.readingScrolled();
    expect(framed()).toEqual([]);
    fixture.destroy();
  });

  it('updates a clicked token without scrolling any ancestor', () => {
    const {fixture, parts, reading, token, scroll} = setup();
    const {tokens} = parts();
    tokens.select(2, 12, clickEventOn(token));
    fixture.detectChanges();
    expect(tokens.selection()).toEqual({kind: 'token', turn: 2, index: 12});
    expect(scroll).not.toHaveBeenCalled();
    expect(reading.scrollTo).not.toHaveBeenCalled();
    fixture.destroy();
  });

  it('locates the selected runtime and stage inside the unobscured reading viewport', () => {
    const {fixture, root, parts, reading, token, scroll} = setup();
    const {tokens, dom} = parts();
    tokens.selection.set({kind: 'token', turn: 2, index: 12});
    const box = (top: number, bottom: number): DOMRect => ({
      x: 0,
      y: top,
      width: 0,
      height: bottom - top,
      top,
      right: 0,
      bottom,
      left: 0,
      toJSON: () => ({}),
    });
    reading.scrollTop = 100;
    reading.getBoundingClientRect = () => box(50, 850);
    const columnHeadings = root.querySelector<HTMLElement>('.column-headings');
    const composer = root.querySelector<HTMLElement>('.composer');
    const refBtn = root.querySelector<HTMLElement>('button[data-side="ref"]');
    if (!columnHeadings || !composer || !refBtn) {
      throw new Error('Expected column-headings, composer, and ref button');
    }
    columnHeadings.getBoundingClientRect = () => box(50, 110);
    composer.getBoundingClientRect = () => box(700, 840);
    token.getBoundingClientRect = () => box(600, 620);
    dom.locateSelectedToken();
    fixture.detectChanges();
    expect(scroll).not.toHaveBeenCalled();
    // Visible area is 110..700; center the actual target token there.
    expect(reading.scrollTo).toHaveBeenCalledWith({top: 305, behavior: 'auto'});
    expect(token.animate).toHaveBeenCalledOnce();
    expect(refBtn.animate).toHaveBeenCalledOnce();
    fixture.destroy();
  });
});
