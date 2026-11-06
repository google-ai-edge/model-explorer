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

import type {KvAnalysis, KvAnalysisContext} from '../../server/contracts/kv';
import {
  defaultKvView,
  kvContextRange,
  KvViewIdentity,
  KvViewSnapshot,
  parseKvView,
  restoreKvView,
} from './kv_view_state';

describe('KV tensor view snapshots', () => {
  it('persists head dimension range and rejects malformed snapshots', () => {
    const tensor = {
      active: true,
      plot: 'index' as const,
      position: 42,
      layer: 3,
      head: 1,
      channelRange: {start: 100, end: 500},
      channelReduction: 'mean' as const,
    };
    expect(parseKvView({...defaultKvView(), tensor})?.tensor).toEqual(tensor);
    expect(
      parseKvView({...defaultKvView(), tensor: {...tensor, channelRange: {start: 3, end: 3}}}),
    ).toBeNull();
    expect(
      parseKvView({...defaultKvView(), tensor: {...tensor, channelReduction: 'sum'}}),
    ).toBeNull();
  });
  it('persists value zoom separately from channel zoom and rejects malformed domains', () => {
    const saved = {
      ...defaultKvView(),
      tensor: {
        active: true,
        plot: 'scatter',
        position: 32,
        layer: 0,
        head: 0,
        channelRange: {start: 10, end: 20},
        valueRange: {start: -0.8, end: 1.3},
      },
    };
    expect(parseKvView(saved)?.tensor).toEqual(saved.tensor);
    for (const valueRange of [
      {start: 1, end: 1},
      {start: 2, end: 1},
      {start: NaN, end: 1},
      {start: '0', end: 1},
    ]) {
      expect(parseKvView({...saved, tensor: {...saved.tensor, valueRange}})).toBeNull();
    }
  });
});

describe('KV view restore', () => {
  const plain = (value: unknown) => JSON.parse(JSON.stringify(value));
  const identity: KvViewIdentity = {captureId: 'capture-a', turn: 2};
  const context = (id: string, turn = 2) =>
    ({
      id,
      label: id,
      turn,
      source: 'snapshot',
      phase: 'decode',
      moment: 'terminal',
      step: 9,
      runtime: 'CPU',
      layers: ['key', 'value'].map((kind) => ({
        layer: 3,
        kind,
        status: 'ok',
        head_count: 2,
        position_start: 10,
        position_count: 90,
      })),
    }) as KvAnalysisContext;
  const analysis: KvAnalysis = {
    contexts: [
      context('first'),
      {...context('second'), moment: 'prefill_post', phase: 'prefill', step: null},
    ],
  };
  const snapshot: KvViewSnapshot = {
    ...defaultKvView(),
    contextId: 'first',
    kind: 'value',
    metric: 'max_abs',
    head: '1',
    colorScale: 'thresholds',
    view: 'lineChart',
    layer: 3,
    position: 25,
    selectionEnd: 40,
    column: true,
    zoomRange: {start: 20, end: 60},
  };

  it('KV snapshot preserves view/range/head chart, accepts fields of older versions, and never restores capture or Turn as navigation', () => {
    const restored = restoreKvView(snapshot, analysis, identity);
    expect(plain(restored.state)).toStrictEqual(plain(snapshot));
    expect(restored.notice).toBe('');
    expect('turn' in restored.state).toBe(false);
    expect('phase' in restored.state).toBe(false);
    restored.state.zoomRange!.end = 13;
    expect(snapshot.zoomRange!.end, 'restoring cannot mutate the saved snapshot').toBe(60);
    // Views saved before Find, the standalone KV page, the head list, the KV details and
    // the zoom history were removed carry their fields.
    const legacy = {
      ...snapshot,
      zoomHistory: [{start: 10, end: 100}],
      infoExpanded: false,
      tensorExpanded: {key: false, value: true},
      formula: 'not a formula',
      panelWidth: 420,
      showDetails: false,
      showResults: true,
      actionsExpanded: true,
      savedExpanded: true,
      expandedHeads: {'value:1': true},
      headViews: {'value:1': {view: 'scatter', numericalExpanded: false, channelsExpanded: true}},
      detailHeadPages: {key: 0, value: 0},
    };
    expect(parseKvView(legacy)).not.toBe(null);
    const restoredLegacy = restoreKvView(legacy, analysis, identity);
    expect(restoredLegacy.notice).toBe('');
    expect(restoredLegacy.state.contextId).toBe('first');
    expect(restoredLegacy.state.position).toBe(25);
  });

  it('unsupported snapshots and outdated observation/layer/head/ranges safely reset; missing metadata stays unavailable', () => {
    for (const invalid of [
      null,
      {},
      {...snapshot, version: 8},
      {...snapshot, metric: 'fake'},
      {...snapshot, zoomRange: {start: 2, end: 1}},
      {...snapshot, position: Infinity},
    ]) {
      expect(parseKvView(invalid)).toBe(null);
    }
    const old = {
      ...snapshot,
      contextId: 'deleted',
      position: 999,
      selectionEnd: 1000,
      layer: 999,
      head: '999',
    };
    const result = restoreKvView(old, analysis, identity);
    expect(result.state.contextId).toBe('first');
    expect(result.state.layer).toBe(3);
    expect(result.state.head).toBe('max');
    expect(result.state.position).toBe(10);
    expect(result.state.selectionEnd).toBe(null);
    expect(result.state.zoomRange).toBe(null);
    expect(result.notice).toMatch(/no longer available/);
    const shortened = restoreKvView(
      snapshot,
      {
        contexts: [
          {
            ...context('first'),
            layers: context('first').layers.map((row) => ({...row, position_count: 20})),
          },
        ],
      },
      identity,
    );
    expect(shortened.state.selectionEnd).toBe(null);
    expect(shortened.state.zoomRange).toBe(null);
    const absent = restoreKvView(snapshot, {contexts: []}, identity);
    expect(absent.state.contextId).toBe('');
    expect(
      restoreKvView(snapshot, {contexts: [context('foreign', 9)]}, identity).state.contextId,
    ).toBe('');
    expect(kvContextRange(undefined)).toBe(null);
    expect(
      kvContextRange({
        ...context('missing'),
        layers: context('missing').layers.map((row) => ({
          ...row,
          position_start: null,
          position_count: null,
        })),
      }),
    ).toBe(null);
  });

  it('selected aggregation heads persist, validate and restore against captured heads', () => {
    const selected = {...snapshot, head: 'mean', headSelection: [0, 1]};
    expect(plain(parseKvView(selected)!.headSelection)).toStrictEqual([0, 1]);
    expect(plain(restoreKvView(selected, analysis, identity).state.headSelection)).toStrictEqual([
      0, 1,
    ]);
    for (const headSelection of [[], [0, 0], [-1], [1.5]]) {
      expect(parseKvView({...selected, headSelection})).toBe(null);
    }
    expect(
      plain(
        restoreKvView({...selected, headSelection: [1, 9]}, analysis, identity).state.headSelection,
      ),
    ).toStrictEqual([1]);
  });
});
