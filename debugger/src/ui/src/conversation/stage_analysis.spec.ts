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

import {restoreLogitPanel, validStagePanels, type StageAnalysisPanel} from './stage_analysis';

describe('stage analysis panels', () => {
  /** Restores a panel as a page reads one back: saved in any shape, which the code checks. */
  const restore = (saved: object) => restoreLogitPanel(saved as StageAnalysisPanel);

  const panel = {
    id: 'analysis-1',
    turn: 1,
    stage: 'decode',
    kind: 'tokens',
    open: true,
    logit: {open: true, metric: 'kl'},
  };

  it('reject invalid saved panels before partially restoring a panel list', () => {
    expect(validStagePanels([panel])).toBe(true);
    // A panel saved while KV panels recorded their entry observation still restores.
    expect(validStagePanels([{...panel, entry: {turn: 1, forward_id: 3}}])).toBe(true);
    // A panel saved while the logit state recorded a position range still restores.
    expect(validStagePanels([{...panel, logit: {...panel.logit, start: 0, end: 10}}])).toBe(true);
    for (const list of [
      [panel, panel],
      [{...panel, stage: 'prefill'}],
      [{...panel, logit: {...panel.logit, metric: 'invented'}}],
      [{...panel, kv: {version: 1}}],
      [{...panel, id: '<script>'}],
      [{...panel, kind: 'graph', logit: undefined}],
    ])
      expect(validStagePanels(list)).toBe(false);
  });

  it('graph panels retain captured observation settings independently', () => {
    expect(
      validStagePanels([
        {
          ...panel,
          kind: 'activations',
          logit: undefined,
          graph: {metric: 'relative_l2', anchor: 'output'},
        },
      ]),
    ).toBe(true);
    expect(
      validStagePanels([
        {
          ...panel,
          kind: 'activations',
          graph: {metric: 4, anchor: 'output'},
        },
      ]),
    ).toBe(false);
  });

  it('accepts graph panels saved with the batch id they no longer keep', () => {
    for (const batchId of [null, 4, '4'])
      expect(
        validStagePanels([
          {
            ...panel,
            kind: 'activations',
            logit: undefined,
            graph: {batchId, metric: 'relative_l2', anchor: 'output'},
          },
        ]),
      ).toBe(true);
  });

  it('retains saved Layer Activation inspections and rejects invalid tensor coordinates', () => {
    const inspection = {
      plot: 'scatter',
      position: 44,
      layer: 2,
      elementRange: {start: 0, end: 16},
      valueRange: {start: -2.5, end: 3.5},
      selection: {channel: 4},
      reduction: 'mean',
    };
    const saved = {
      ...panel,
      kind: 'activations',
      logit: undefined,
      graph: {
        metric: 'cosine_similarity',
        anchor: 'output',
        mode: 'inspection',
        inspection,
      },
    };
    expect(validStagePanels([saved])).toBe(true);
    for (const patch of [
      {position: -1},
      {position: 0.5},
      {layer: NaN},
      {plot: 'heatmap'},
      {selection: {channel: -1}},
      {elementRange: {start: -1, end: 2}},
      {valueRange: {start: 2, end: 2}},
      {reduction: 'sum'},
    ])
      expect(
        validStagePanels([
          {...saved, graph: {...saved.graph, inspection: {...inspection, ...patch}}},
        ]),
      ).toBe(false);
    expect(validStagePanels([{...saved, graph: {...saved.graph, mode: 'unknown'}}])).toBe(false);
  });

  it('retains Layer Activation runtime visibility across saved-panel serialization', () => {
    const saved = {
      ...panel,
      kind: 'activations',
      logit: undefined,
      graph: {
        metric: 'cosine_similarity',
        anchor: 'output',
        mode: 'inspection',
        inspection: {plot: 'index', position: 44, layer: 2},
      },
    };
    for (const visibleRuntimes of [
      undefined,
      Object.freeze([]),
      Object.freeze(['ref']),
      Object.freeze(['target']),
      Object.freeze(['ref', 'target']),
      Object.freeze(['target', 'ref']),
    ]) {
      const source = {
        ...saved,
        graph: {...saved.graph, inspection: {...saved.graph.inspection, visibleRuntimes}},
      };
      expect(validStagePanels([source])).toBe(true);
      const restored = JSON.parse(JSON.stringify([source]));
      expect(validStagePanels(restored)).toBe(true);
      expect(restored[0].graph.inspection.visibleRuntimes).toStrictEqual(visibleRuntimes);
      expect(restore(restored[0])).toStrictEqual(restored[0]);
    }
    for (const visibleRuntimes of [
      null,
      false,
      'ref',
      {},
      ['Reference'],
      ['ref', 'unknown'],
      ['ref', 'ref'],
      ['target', 'target'],
      ['ref', 'target', 'ref'],
      [0],
      [null],
      [undefined],
    ])
      expect(
        validStagePanels([
          {
            ...saved,
            graph: {...saved.graph, inspection: {...saved.graph.inspection, visibleRuntimes}},
          },
        ]),
        `visibleRuntimes=${JSON.stringify(visibleRuntimes)}`,
      ).toBe(false);
  });

  it('restore old saved logits panels into the matching independent analysis', () => {
    for (const metric of ['entropy', 'margin', 'selected_probability']) {
      const restored = restore({...panel, logit: {...panel.logit, metric}});
      expect(restored.kind).toBe('distribution');
      expect(restored.logit!.metric).toBe(metric);
      expect(validStagePanels([restored])).toBe(true);
      expect(validStagePanels([{...restored, stage: 'prefill'}])).toBe(false);
    }
    expect(restore(panel).logit!.metric).toBe('kl');
    expect(restore({...panel, logit: {...panel.logit, metric: 'js'}}).logit!.metric).toBe('js');
    expect(restore({...panel, logit: {...panel.logit, metric: 'relative_l2'}}).logit!.metric).toBe(
      'logit_rmse',
    );
  });

  it('restores logits inspection settings without moving KL/JS or losing the comparison range', () => {
    const settings = {
      ...panel.logit,
      logitView: 'inspection',
      logitPlot: 'scatter',
      logitElementRange: {start: 128, end: 256},
      logitValueRange: {start: -9.5, end: 2.25},
      logitElementReduction: 'max',
      logitTransform: 'centered',
      logitPosition: 42,
      logitOffset: 128,
      logitOrder: 'absolute_delta',
      logitSortDirection: 'ascending',
      positionRange: {start: 32, end: 64},
    };
    for (const metric of ['logit_rmse', 'logit_max_abs', 'kl', 'js']) {
      const saved = {...panel, logit: {...settings, metric}};
      const restored = restore(saved);
      expect(restored.kind).toBe('tokens');
      expect(restored.logit).toStrictEqual(saved.logit);
      expect(validStagePanels([restored])).toBe(true);
    }
    expect(restore({...panel, logit: undefined}).logit!.metric).toBe('logit_rmse');
  });

  it('moves legacy saved distribution divergence panels back to Final Logits', () => {
    for (const metric of ['kl', 'js']) {
      const saved = {
        ...panel,
        kind: 'distribution',
        instanceNumber: 3,
        logit: {...panel.logit, metric},
      };
      const restored = restore(saved);
      expect(restored.kind).toBe('tokens');
      expect(restored.logit).toStrictEqual(saved.logit);
      expect(restored.instanceNumber).toBe(undefined);
      expect(validStagePanels([restored])).toBe(true);
    }
    const saved = {
      ...panel,
      kind: 'distribution',
      instanceNumber: 3,
      logit: {...panel.logit, metric: 'entropy'},
    };
    expect(restore(saved)).toStrictEqual(saved);
  });

  it('validates persisted logits view, plot, reduction, transform, position, page offset, and column sorting', () => {
    const validValues = {
      logitView: ['comparison', 'inspection'],
      logitPlot: ['index', 'scatter'],
      logitElementReduction: ['min', 'max', 'mean'],
      logitTransform: ['raw', 'centered'],
      logitPosition: [0, 42],
      logitOffset: [0, 256],
      logitOrder: ['index', 'ref', 'target', 'delta', 'absolute_delta'],
      logitSortDirection: ['ascending', 'descending'],
    };
    const invalidValues = {
      logitView: [null, 'table', true],
      logitPlot: [null, 'line', 'table', true],
      logitElementReduction: [null, 'sum', 1],
      logitTransform: [null, 'softmax', 1],
      logitPosition: [null, -1, 0.5, '42', NaN, Infinity],
      logitOffset: [null, -1, 0.5, '256', NaN, Infinity],
      logitOrder: [null, 'value', false],
      logitSortDirection: [null, 'asc', 'desc', false, 1],
    };
    for (const [field, values] of Object.entries(validValues))
      for (const value of values)
        expect(validStagePanels([{...panel, logit: {...panel.logit, [field]: value}}])).toBe(true);
    for (const [field, values] of Object.entries(invalidValues))
      for (const value of values)
        expect(
          validStagePanels([{...panel, logit: {...panel.logit, [field]: value}}]),
          `${field}=${String(value)}`,
        ).toBe(false);
  });

  it('keeps legacy saved inspection panels valid without a saved plot or element viewport', () => {
    const saved = {...panel, logit: {...panel.logit, logitView: 'inspection'}};
    const restored = restore(saved);
    expect(validStagePanels([restored])).toBe(true);
    expect(restored.logit).toStrictEqual(saved.logit);
    expect(restored.logit!.logitPlot).toBe(undefined);
    expect(restored.logit!.logitElementRange).toBe(undefined);
    expect(restored.logit!.logitValueRange).toBe(undefined);
  });

  it('accepts logit panels saved with the Sampling settings they no longer keep', () => {
    for (const saved of [
      {samplingView: 'inspection', samplingPosition: 42, samplingTopK: 10},
      {samplingView: 'table', samplingPosition: -1, samplingTopK: 1001},
    ])
      expect(validStagePanels([{...panel, logit: {...panel.logit, ...saved}}])).toBe(true);
  });

  it('retains legacy absolute-difference ordering without inventing a saved direction', () => {
    const saved = {...panel, logit: {...panel.logit, logitOrder: 'absolute_delta'}};
    expect(validStagePanels([saved])).toBe(true);
    const restored = restore(saved);
    expect(restored.logit!.logitOrder).toBe('absolute_delta');
    expect(restored.logit!.logitSortDirection).toBe(undefined);
  });

  it('validates element index ranges separately from signed floating-point value ranges', () => {
    const validValues = {
      logitElementRange: [
        {start: 0, end: 1},
        {start: 200000, end: 262144},
      ],
      logitValueRange: [
        {start: -9.5, end: -2.25},
        {start: -1.25, end: 3.5},
        {start: 0, end: 1},
      ],
    };
    const invalidValues = {
      logitElementRange: [
        null,
        {},
        [],
        '0–10',
        {start: -1, end: 2},
        {start: 0.5, end: 2},
        {start: 0, end: 2.5},
        {start: 2, end: 2},
        {start: 3, end: 2},
        {start: NaN, end: 2},
        {start: 0, end: Infinity},
        {start: '0', end: 2},
      ],
      logitValueRange: [
        null,
        {},
        [],
        '-1–2',
        {start: 2, end: 2},
        {start: 3, end: -2},
        {start: -Infinity, end: 2},
        {start: 0, end: Infinity},
        {start: NaN, end: 2},
        {start: -1, end: NaN},
        {start: '-1', end: 2},
      ],
    };
    for (const [field, values] of Object.entries(validValues))
      for (const value of values)
        expect(validStagePanels([{...panel, logit: {...panel.logit, [field]: value}}])).toBe(true);
    for (const [field, values] of Object.entries(invalidValues))
      for (const value of values)
        expect(
          validStagePanels([{...panel, logit: {...panel.logit, [field]: value}}]),
          `${field}=${JSON.stringify(value)}`,
        ).toBe(false);
  });

  it('sync point subsets survive saved-panel validation and malformed selections are rejected', () => {
    const graphPanel = {
      ...panel,
      kind: 'activations',
      logit: undefined,
      graph: {metric: 'cosine', anchor: 'mean'},
    };
    for (const anchorSelection of [undefined, null, ['pre_attention_norm', 'mlp_output']]) {
      expect(
        validStagePanels([{...graphPanel, graph: {...graphPanel.graph, anchorSelection}}]),
      ).toBe(true);
    }
    for (const anchorSelection of [[], ['mlp_output', 'mlp_output'], [0], 'mlp_output']) {
      expect(
        validStagePanels([{...graphPanel, graph: {...graphPanel.graph, anchorSelection}}]),
      ).toBe(false);
    }
  });

  it('saved panels keep a valid sync point scope and inspection anchor and refuse invalid ones', () => {
    const saved = {
      ...panel,
      kind: 'activations',
      logit: undefined,
      graph: {
        metric: 'cosine_similarity',
        anchor: 'min',
        anchorSelection: ['attention_output'],
        mode: 'inspection',
        inspection: {plot: 'index', position: 44, layer: 2},
      },
    };
    for (const anchorScope of [undefined, 'output', 'internal', 'all'])
      for (const anchor of [undefined, 'output', 'attention_output', 'x'.repeat(128)]) {
        const item = {
          ...saved,
          graph: {
            ...saved.graph,
            anchorScope,
            inspection: {...saved.graph.inspection, anchor},
          },
        };
        const restored = JSON.parse(JSON.stringify([item]));
        expect(validStagePanels(restored)).toBe(true);
        expect(restored[0].graph.inspection.anchor).toStrictEqual(anchor);
        expect(restored[0].graph.anchorScope).toStrictEqual(anchorScope);
      }
    for (const anchor of [null, '', '   ', 3, false, 'x'.repeat(129)])
      expect(
        validStagePanels([
          {
            ...saved,
            graph: {...saved.graph, inspection: {...saved.graph.inspection, anchor}},
          },
        ]),
      ).toBe(false);
    for (const anchorScope of [null, '', 'legacy', 3, false])
      expect(
        validStagePanels([
          {
            ...saved,
            graph: {...saved.graph, anchorScope},
          },
        ]),
      ).toBe(false);
  });

  it('retains Graph Diff as an Intermediate Tensors comparison view', () => {
    const graph = {metric: 'cosine_similarity', anchor: 'output', view: 'graph'};
    expect(validStagePanels([{...panel, kind: 'activations', logit: undefined, graph}])).toBe(true);
    expect(
      validStagePanels([
        {...panel, kind: 'activations', logit: undefined, graph: {...graph, view: 'unknown'}},
      ]),
    ).toBe(false);
  });
});
