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
import {KvDetails} from './kv_details';
import {KvApi} from '../../server/kv_api';
import {CaptureStore} from '../../server/capture_store';
import type {KvRangeMetadata, ShapeAxis} from '../../server/contracts/kv';

/** The metadata of a kv-range answer of no shape. */
const NO_METADATA: KvRangeMetadata = {
  ref: {dimensions: [], shape: null, dtype: null},
  target: {dimensions: [], shape: null, dtype: null},
  output: {dimensions: [], shape: null, dtype: null},
};

describe('Tensor Info sorting', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  function create() {
    TestBed.configureTestingModule({
      providers: [
        {provide: KvApi, useValue: {}},
        {provide: CaptureStore, useValue: {captureId: signal('capture')}},
      ],
    });
    const fixture = TestBed.createComponent(KvDetails);
    fixture.componentRef.setInput('panel', {id: 'kv-1', kind: 'kv', turn: 2, stage: 'decode'});
    fixture.componentRef.setInput('label', 'KV cache');
    fixture.componentRef.setInput('range', {start: 32, end: 66});
    return fixture;
  }
  it('shows the inspected head statistics as the server gives them, none while loading', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const selected = signal<{start: number; end: number} | null>({start: 1, end: 2});
    const loading = signal(false);
    const error = signal('');
    const stat = (min: number, max: number, mean: number) => ({
      min,
      max,
      mean,
      min_position: 0,
      max_position: 17,
      valid_count: 2,
    });
    const statistics = {
      ref: stat(-2, 4, 1),
      target: stat(1, 9, 4),
      delta: stat(-2, 3, 0.5),
      abs_delta: stat(2, 3, 2.5),
    };
    fixture.componentRef.setInput('kv', {
      tensorGroups: () => ['key', 'value'].map((kind) => ({kind, request: {loading, error}})),
      tensorEvidence: () => ({evidence: {channels: [], statistics}}),
      tensorSelectedRange: selected,
      tensorChannels: () => ({key: null, value: null}),
    });
    const summaries = () => view.tensorStatistics()['key'].map(({label, stats}) => [label, stats]);
    const expected = [
      ['Reference', statistics.ref],
      ['Target', statistics.target],
      ['Δ', statistics.delta],
      ['|Δ|', statistics.abs_delta],
    ];
    expect(summaries()).toEqual(expected);
    selected.set({start: 17, end: 19});
    expect(summaries()).toEqual(expected);
    expect(view.tensorStatistics()['value']).toEqual(view.tensorStatistics()['key']);
    loading.set(true);
    expect(view.tensorStatistics()['key'].every((row) => row.stats.mean === null)).toBe(true);
    loading.set(false);
    error.set('Unavailable');
    expect(view.tensorStatistics()['key'].every((row) => row.stats.valid_count === 0)).toBe(true);
    fixture.destroy();
  });
  it('keeps every tensor element while point and bin selection only highlight rows', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const selected = signal<{start: number; end: number} | null>(null);
    const loading = signal(false);
    const error = signal('');
    const currentHead = signal(0);
    const channels = [2, 0, 1].map((channel) => ({
      channel,
      ref: channel === 2 ? null : channel,
      target: channel === 2 ? 'Infinity' : channel + 1,
      delta: channel === 2 ? null : 1,
      abs_delta: channel === 2 ? null : 1,
      status: 'ok',
    }));
    fixture.componentRef.setInput('kv', {
      tensorGroups: () => ['key', 'value'].map((kind) => ({kind, request: {loading, error}})),
      tensorEvidence: () => ({
        evidence: {channels: currentHead() === 0 ? channels : channels.slice(0, 1)},
      }),
      tensorSelectedRange: selected,
      tensorChannels: () => ({key: null, value: null}),
    });
    for (const selection of [null, {start: 1, end: 2}, {start: 0, end: 2}, null]) {
      selected.set(selection);
      for (const kind of ['key', 'value']) {
        const rows = view.tensorInspectionRows()[kind];
        expect(rows.map((row) => row.index)).toEqual([0, 1, 2]);
        expect(
          rows.filter((row) => view.tensorElementSelected(row.index)).map((row) => row.index),
        ).toEqual(selection ? (selection.start === 1 ? [1] : [0, 1]) : []);
        expect(rows[2].values).toEqual([null, 'Infinity', null, null]);
      }
    }
    expect(view.formatTensorValue('Infinity')).toBe('Infinity');
    expect(view.formatTensorValue(null)).toBe('-');
    currentHead.set(1);
    expect(view.tensorInspectionRows()['key'].map((row) => row.index)).toEqual([2]);
    loading.set(true);
    expect(view.tensorInspectionRows()['key']).toEqual([]);
    loading.set(false);
    error.set('Unavailable');
    expect(view.tensorInspectionRows()['value']).toEqual([]);
  });
  it('uses the current token and a single selection row independent of the panel range', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const position = signal(42);
    const layer = signal(10);
    const head = signal(2);
    fixture.componentRef.setInput('kv', {
      tensorPosition: position,
      tensorLayer: layer,
      tensorHead: head,
      columnSteps: (column: number) => ({
        ref: `Written at decode step ${column + 1}, when the model reads "blue"`,
        target: `Written at decode step ${column + 1}, when the model reads "blue"`,
      }),
    });
    expect(view.inspectionInfoItems().find((item) => item.label === 'Token position')?.value).toBe(
      '42',
    );
    // The step that wrote the inspected K and V, in the server's words, once for both sides.
    expect(view.inspectionInfoItems().find((item) => item.label === 'Step')?.value).toBe(
      'Written at decode step 43, when the model reads "blue"',
    );
    expect(view.inspectionInfoItems().find((item) => item.label === 'Tensor Index')?.value).toBe(
      '[Layer: 10, Head: 2]',
    );
    position.set(43);
    head.set(3);
    expect(view.inspectionInfoItems().find((item) => item.label === 'Token position')?.value).toBe(
      '43',
    );
    expect(view.inspectionInfoItems().find((item) => item.label === 'Tensor Index')?.value).toBe(
      '[Layer: 10, Head: 3]',
    );
  });
  it('pages each cache by 16 values, reveals selected bins and resets for a new slice', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const position = signal(32);
    const layer = signal(0);
    const head = signal(0);
    const selected = signal<{start: number; end: number} | null>(null);
    const loading = signal(false);
    const error = signal('');
    const size = signal(33);
    const representatives = signal<Record<string, number | null>>({key: null, value: null});
    fixture.componentRef.setInput('kv', {
      tensorPosition: position,
      tensorLayer: layer,
      tensorHead: head,
      tensorSelectedRange: selected,
      tensorChannels: representatives,
      tensorGroups: () => ['key', 'value'].map((kind) => ({kind, request: {loading, error}})),
      tensorEvidence: () => ({
        evidence: {
          channels: Array.from({length: size()}, (_, channel) => ({
            channel,
            ref: channel,
            target: channel + 1,
            delta: 1,
            abs_delta: 1,
          })),
        },
      }),
    });
    expect(view.tensorInspectionPage('key').rows.map((row) => row.index)).toEqual(
      Array.from({length: 16}, (_, i) => i),
    );
    expect(view.tensorInspectionPage('key').pages).toBe(3);
    view.changeTensorPage('key', 1);
    expect(view.tensorInspectionPage('key').rows.map((row) => row.index)).toEqual(
      Array.from({length: 16}, (_, i) => i + 16),
    );
    expect(view.tensorInspectionPage('value').index).toBe(0);
    view.changeTensorPage('key', 1);
    expect(view.tensorInspectionPage('key').rows.map((row) => row.index)).toEqual([32]);
    view.changeTensorPage('key', 1);
    expect(view.tensorInspectionPage('key').index).toBe(2);
    expect(selected()).toBeNull();
    selected.set({start: 15, end: 18});
    for (const kind of ['key', 'value']) {
      expect(view.tensorInspectionPage(kind).index).toBe(0);
      expect(
        view
          .tensorInspectionPage(kind)
          .rows.filter((row) => view.tensorElementSelected(row.index))
          .map((row) => row.index),
      ).toEqual([15]);
    }
    view.changeTensorPage('key', 1);
    expect(
      view
        .tensorInspectionPage('key')
        .rows.filter((row) => view.tensorElementSelected(row.index))
        .map((row) => row.index),
    ).toEqual([16, 17]);
    representatives.set({key: 17, value: 16});
    for (const kind of ['key', 'value']) {
      expect(view.tensorInspectionPage(kind).index).toBe(1);
      expect(
        view.tensorInspectionPage(kind).rows.some((row) => row.index === representatives()[kind]),
      ).toBe(true);
    }
    expect(selected()).toEqual({start: 15, end: 18});
    // A representative from the previous bin cannot override the current selection.
    selected.set({start: 32, end: 33});
    expect(view.tensorInspectionPage('key').index).toBe(2);
    expect(view.tensorInspectionPage('value').index).toBe(2);
    selected.set(null);
    head.set(1);
    expect(view.tensorInspectionPage('key').index).toBe(0);
    view.changeTensorPage('key', 1);
    position.set(33);
    expect(view.tensorInspectionPage('key').index).toBe(0);
    view.changeTensorPage('key', 1);
    layer.set(1);
    expect(view.tensorInspectionPage('key').index).toBe(0);
    size.set(16);
    expect(view.tensorInspectionPage('key').pages).toBe(1);
    size.set(0);
    expect(view.tensorInspectionPage('key')).toEqual({index: 0, pages: 1, rows: []});
  });
  it('sorts all KV values before pagination and locates selection within the sorted rows', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const selected = signal<{start: number; end: number} | null>(null);
    fixture.componentRef.setInput('kv', {
      tensorPosition: () => 32,
      tensorLayer: () => 0,
      tensorHead: () => 0,
      tensorSelectedRange: selected,
      tensorChannels: () => ({key: null, value: null}),
      tensorGroups: () =>
        ['key', 'value'].map((kind) => ({kind, request: {loading: () => false, error: () => ''}})),
      tensorEvidence: () => ({
        evidence: {
          channels: Array.from({length: 33}, (_, channel) => ({
            channel,
            ref: 33 - channel,
            target: channel,
            delta: 2 * channel - 33,
            abs_delta: Math.abs(2 * channel - 33),
          })),
        },
      }),
    });
    view.changeTensorPage('value', 1);
    const before = view.tensorStatistics();
    view.sortTensorValues('key', 'ref');
    expect(view.tensorInspectionPage('key').index).toBe(0);
    expect(view.tensorInspectionPage('key').rows.map((row) => row.index)).toEqual(
      Array.from({length: 16}, (_, i) => 32 - i),
    );
    expect(view.tensorInspectionPage('value').index).toBe(1);
    view.changeTensorPage('key', 1);
    expect(view.tensorInspectionPage('key').rows[0].index).toBe(16);
    view.sortTensorValues('key', 'ref');
    expect(view.tensorInspectionPage('key').rows[0].index).toBe(0);
    view.sortTensorValues('key', 'index');
    view.sortTensorValues('key', 'index');
    expect(view.tensorValueSortDirection('key', 'index')).toBe('descending');
    expect(selected()).toBeNull();
    selected.set({start: 0, end: 1});
    expect(view.tensorInspectionPage('key').index).toBe(2);
    expect(view.tensorInspectionPage('key').rows[0].index).toBe(0);
    expect(view.tensorStatistics()).toEqual(before);
    fixture.destroy();
  });
  it('describes the selected head slice rather than the parent tensor shape', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const loading = signal(false);
    fixture.componentRef.setInput('kv', {
      tensorGroups: () => [{kind: 'key', request: {loading, error: () => ''}}],
      tensorEvidence: () => ({
        error: '',
        evidence: {
          sources: {
            ref: {
              source_shape: [1, 4, 35, 128],
              source_dtype: 'float16',
              comparison_shape: [128],
              comparison_dtype: 'float32',
            },
            target: null,
          },
        },
      }),
    });
    expect(view.inspectionMetadata('key', 'ref').map((row) => row.value)).toEqual([
      '[128]',
      'float32',
    ]);
    expect(view.inspectionMetadata('key', 'target').map((row) => row.value)).toEqual(['-', '-']);
    loading.set(true);
    expect(view.inspectionMetadata('key', 'ref').map((row) => row.value)).toEqual(['-', '-']);
  });
  it('explains tensor index coordinates using the currently inspected layer and head', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const layer = signal(3);
    const head = signal(1);
    fixture.componentRef.setInput('kv', {
      tensorPosition: () => 33,
      tensorLayer: layer,
      tensorHead: head,
      columnSteps: () => null,
    });
    const index = () => view.inspectionInfoItems().find((row) => row.label === 'Tensor Index')!;
    expect(index().value).toBe('[Layer: 3, Head: 1]');
    expect(
      index()
        .segments?.filter((segment) => segment.help)
        .map((segment) => segment.text),
    ).toEqual(['Layer', 'Head']);
    layer.set(7);
    head.set(0);
    expect(index().segments?.find((segment) => segment.text === 'Head')?.help?.rows).toContainEqual(
      {label: 'Layer', value: '7'},
    );
    expect(index().value).toBe('[Layer: 7, Head: 0]');
    fixture.destroy();
  });
  it('routes token and tensor locators to their controls without locating the whole analysis', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    let positions = 0;
    let indices = 0;
    let analyses = 0;
    let views = 0;
    fixture.componentRef.setInput('kv', {
      locateView: () => views++,
      locateTensorPosition: () => positions++,
      locateTensorIndex: () => indices++,
    });
    view.locateAnalysis.subscribe(() => analyses++);
    view.locateInspection({label: 'Token position'});
    view.locateInspection({label: 'Tensor Index'});
    expect([positions, indices, analyses]).toEqual([1, 1, 0]);
    view.locateInspection({label: 'Mode'});
    view.locateDefinition({label: 'Mode'});
    expect(views).toBe(2);
    view.locateInspection({label: 'Name'});
    expect(analyses).toBe(1);
  });
  it('reads dimension counts from each shape instead of the aligned range length', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('kv', {
      positionAlignment: () => ({stage: 'decode', alignment: 'content', start: 32, pairs: []}),
    });
    const input = ['Position', 'Layer', 'Head', 'Head dimension'];
    const cases: [string[], ShapeAxis[], string][] = [
      [input, [32, 15, 4, [256, 512]], '32'],
      [input, [24, 15, 4, [256, 512]], '24'],
      [['Position', 'Layer'], [35, 15], '35'],
    ];
    for (const [dimensions, shape, count] of cases) {
      const items = view.metadataItems({dimensions, shape, dtype: 'float16'});
      const position = items[0].segments?.find((part) => part.text === 'Token position')?.help;
      expect(position?.rows.find((row) => row.label === 'Count')?.value).toBe(count);
      expect(position?.rows.find((row) => row.label === 'Alignment')?.value).toBe('Context');
      if (dimensions.length === 4) {
        expect(items[1].value).toBe(`[${shape[0]}, 15, 4, Mixed]`);
        const mixed = items[1].segments?.find((part) => part.text === 'Mixed')?.help;
        expect(mixed?.rows[0].value).toBe('Head dimension');
      }
    }
    // Under Steps the Decode's columns are its steps.
    fixture.componentRef.setInput('kv', {
      positionAlignment: () => ({stage: 'decode', alignment: 'steps', start: 32, pairs: []}),
    });
    const steps = view
      .metadataItems({dimensions: ['Position', 'Layer'], shape: [35, 15], dtype: 'float16'})[0]
      .segments?.find((part) => part.text === 'Token position')?.help;
    expect(steps?.rows.find((row) => row.label === 'Alignment')?.value).toBe('Steps');
  });
  it('lists the server metadata of each cache, with the layers it was asked for', async () => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: 0}),
    });
    const fixture = create();
    const view = fixture.componentInstance;
    const kvRange = vi.fn(async (...args: unknown[]) => {
      const [, , context_id, start, end] = args as [string, number, string, number, number];
      return {
        context_id,
        start,
        end,
        bins: [{start, end}],
        rows: [],
        metadata: {
          ref: {
            dimensions: ['Position', 'Layer', 'Head', 'Head dimension'],
            shape: [8, 1, 1, 256],
            dtype: 'I8',
            layers: [{layer: 3, shape: [8, 1, 256], dtype: 'I8'}],
          },
          target: {
            dimensions: ['Position', 'Layer', 'Head', 'Head dimension'],
            shape: null,
            dtype: ['I8', 'F16'],
            layers: [{layer: 3, shape: null, dtype: 'I8'}],
          },
          output: {
            dimensions: ['Position', 'Layer'],
            shape: [end - start, 1],
            dtype: 'float64',
            layers: [{layer: 3, shape: [end - start]}],
          },
        },
      };
    });
    (TestBed.inject(KvApi) as unknown as {kvRange: typeof kvRange}).kvRange = kvRange;
    fixture.componentRef.setInput('kv', {
      context: () => ({id: 'end', layers: []}),
      head: () => 'min',
      aggregationHeads: () => null,
      selectedHeads: () => [0],
      view: () => 'lineChart',
      positionAlignment: () => null,
      tensorView: () => false,
      visibleMetricLayers: () => [5, 3],
      loading: () => false,
      metric: () => 'cosine_similarity',
    });
    fixture.detectChanges();
    await vi.waitFor(() => expect(view.keyData.data()).not.toBeNull());
    await vi.waitFor(() => expect(view.valueData.data()).not.toBeNull());
    expect(kvRange.mock.calls.map((call) => [call[6], call[11]])).toEqual([
      ['key', [3, 5]],
      ['value', [3, 5]],
    ]);
    const [key] = view.kvMetadata();
    const items = (metadata: Parameters<typeof view.metadataItems>[0], output = false) =>
      view.metadataItems(metadata, output).map(({label, value}) => [label, value]);
    expect(items(key.sources[0].metadata)).toEqual([
      ['Dimensions', '[Token position, Layer, Head, Head dimension]'],
      ['Shape', '[8, 1, 1, 256]'],
      ['Dtype', 'I8'],
    ]);
    expect(items(key.sources[1].metadata)).toEqual([
      ['Dimensions', '[Token position, Layer, Head, Head dimension]'],
      ['Shape', '—'],
      ['Dtype', 'Mixed'],
    ]);
    expect(items(key.output, true)).toEqual([
      ['Dimensions', '[Token position, Layer]'],
      ['Shape', '[34, 1]'],
      ['Dtype', 'float64'],
      ['Display precision', expect.any(String)],
    ]);
    const layer = view
      .metadataItems(key.output)[0]
      .segments?.find((part) => part.text === 'Layer')?.help;
    expect(layer?.rows).toContainEqual({label: 'Selection', value: '[L3]'});
    // A new range: the last metadata stays shown while its answer loads, and the server's reason
    // replaces it when that answer fails.
    const failures: ((error: Error) => void)[] = [];
    kvRange.mockImplementation(
      () => new Promise((_, reject) => failures.push(reject)) as ReturnType<typeof kvRange>,
    );
    fixture.componentRef.setInput('range', {start: 40, end: 50});
    fixture.detectChanges();
    await vi.waitFor(() => expect(view.keyData.loading()).toBe(true));
    const [loading] = view.kvMetadata();
    expect(loading.loading).toBe(false);
    expect(items(loading.output, true)[1]).toEqual(['Shape', '[34, 1]']);
    for (const fail of failures) {
      fail(new Error('The KV snapshot cannot be read.'));
    }
    await vi.waitFor(() => expect(view.keyData.error()).not.toBe(''));
    const [failed] = view.kvMetadata();
    expect([failed.error, failed.loading, failed.output]).toEqual([
      'The KV snapshot cannot be read.',
      false,
      null,
    ]);
    fixture.destroy();
    Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });
  it.each(['key', 'value'])('keeps both caches when selecting a %s layer', (kind) => {
    const fixture = create();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('kv', {
      view: () => 'heatmap',
      layerIds: () => [0, 1],
      visibleMetricLayers: () => [0, 1],
      loading: () => false,
      metric: () => 'cosine_similarity',
    });
    fixture.componentRef.setInput('kvSelection', {start: 40, end: 43, layer: 1, kind});
    const data = {
      start: 40,
      end: 43,
      context_id: 'test',
      bins: [],
      rows: [
        {layer: 0, bins: []},
        {layer: 1, bins: []},
      ],
      metadata: NO_METADATA,
    };
    view.keyData.data.set(data);
    view.valueData.data.set(data);
    expect(
      view.sections().map((section) => [section.label, section.rows.map((row) => row.layer)]),
    ).toEqual([
      ['Key cache', [1]],
      ['Value cache', [1]],
    ]);
  });
  it('keeps analysis scope in sync with position and layer selection', () => {
    const fixture = create();
    fixture.componentRef.setInput('kv', {
      view: () => 'heatmap',
      tensorView: () => false,
      layerIds: () => [0, 1, 2],
      visibleMetricLayers: () => [0, 1, 2],
      metric: () => 'cosine_similarity',
      metricLabel: () => 'Cosine similarity',
      head: () => 'min',
      selectedHeads: () => [0, 1, 2, 3],
      positionAlignment: () => null,
      context: () => ({
        layers: ['key', 'value'].flatMap((kind) =>
          [0, 1, 2].map((layer) => {
            const structure = {
              status: 'ok',
              position_start: 32,
              position_count: 35,
              head_count: 4,
              channel_count: 128,
              batch_count: 1,
              dtype: 'float16',
            };
            return {kind, layer, token_structure: {ref: structure, target: structure}};
          }),
        ),
      }),
    });
    expect(fixture.componentInstance.metricInfoItems().at(-1)?.label).toBe('Aggregation');
    expect(
      fixture.componentInstance.metricInfoItems().find((item) => item.label === 'Mode')?.value,
    ).toBe('Tensor Comparison');
    const scope = () =>
      fixture.componentInstance
        .metricInfoItems()
        .filter(({label}) => ['Stage', 'Token Range', 'Groups'].includes(label))
        .map(({label, value}) => [label, value]);
    expect(scope()).toEqual([
      ['Stage', 'Turn 2 · Decode'],
      ['Token Range', '32–65'],
      ['Groups', 'Transformer Layers [L0, L1, L2]'],
    ]);
    fixture.componentRef.setInput('range', {start: 40, end: 43});
    fixture.componentRef.setInput('kvSelection', {start: 40, end: 43, layer: 1, kind: 'key'});
    expect(scope()).toEqual([
      ['Stage', 'Turn 2 · Decode'],
      ['Token Range', '40–42'],
      ['Groups', 'Transformer Layers [L1]'],
    ]);
    fixture.componentRef.setInput('kvSelection', null);
    expect(scope()[2]).toEqual(['Groups', 'Transformer Layers [L0, L1, L2]']);
  });
  it('resets sorting when the analysis instance changes', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    view.sortBy('Key cache heads', 'head', 'head');
    expect(view.sortDirection('Key cache heads', 'head', 'head')).toBe('descending');
    fixture.componentRef.setInput('panel', {id: 'kv-2', kind: 'kv', turn: 2, stage: 'decode'});
    expect(view.sortDirection('Key cache heads', 'head', 'head')).toBe('ascending');
  });
});
