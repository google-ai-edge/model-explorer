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
import {StageStateInfo} from './stage_state_info';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';
import type {KvTensorHead} from '../kv/kv_panel/kv_tensor_data';
import type {KvHeadEvidence} from '../../data/contracts/kv';
import {TENSOR_METRICS} from '../analysis/position_tensor_model';

const head = (index: number, value: number | null): KvTensorHead => ({
  label: `Head ${index}`,
  head: index,
  evidence: {metrics: {cosine_similarity: value}} as KvHeadEvidence,
  error: '',
});

describe('Tensor Info sorting', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  function create() {
    TestBed.configureTestingModule({
      providers: [
        {provide: ReportApiService, useValue: {}},
        {provide: ReportStateService, useValue: {captureId: signal('capture')}},
      ],
    });
    const fixture = TestBed.createComponent(StageStateInfo);
    fixture.componentRef.setInput('panel', {id: 'kv-1', kind: 'kv', turn: 2, stage: 'decode'});
    fixture.componentRef.setInput('label', 'KV cache');
    fixture.componentRef.setInput('range', {start: 32, end: 66});
    return fixture;
  }
  it('uses the shared analysis context for Arch instead of treating the subgraph as the analysis', () => {
    const fixture = create(), view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {id: 'arch-1', kind: 'activations', turn: 2, stage: 'decode'});
    fixture.componentRef.setInput('label', 'Intermediate Tensor Analysis 0');
    const position = signal(32), summary = signal('max');
    const mapping = signal({active: false, title: 'Layer 0'});
    const locateRange = vi.fn(), locateSummary = vi.fn();
    fixture.componentRef.setInput('graph', {
      metric: () => 'cosine_similarity', metrics: TENSOR_METRICS,
      graphPosition: position, moduleSummary: summary, locateRange, locateSummary,
      archView: () => ({mapping}),
      archDetails: () => [{label: 'Analysis Info', items: [{label: 'Name', value: 'Per-Layer Embedding'}]}],
    });
    const info = () => Object.fromEntries(view.graphArchInfoItems().map(item => [item.label, item.value]));
    expect(info()).toMatchObject({Name: 'Intermediate Tensor Analysis 0', Mode: 'Tensor Comparison',
      Stage: 'Turn 2 · Decode', 'Token Position': '32', Subgraph: 'Per-Layer Embedding', Summary: 'Max'});
    position.set(33); summary.set('min');
    expect(info()).toMatchObject({'Token Position': '33', Summary: 'Min'});
    view.locateGraphArchDefinition({label: 'Token Position'});
    view.locateGraphArchDefinition({label: 'Summary'});
    expect(locateRange).toHaveBeenCalledOnce();
    expect(locateSummary).toHaveBeenCalledOnce();
    mapping.set({active: true, title: 'Layer 0'});
    expect(info()).toMatchObject({Name: 'Intermediate Tensor Analysis 0', Subgraph: 'Layer 0', 'Token Position': '33'});
    expect(info()).not.toHaveProperty('Summary');
  });
  it('summarizes every tensor element independently of selection and pagination', () => {
    const fixture = create(),
      view = fixture.componentInstance;
    const selected = signal<{start: number; end: number} | null>({start: 1, end: 2});
    const loading = signal(false),
      error = signal('');
    const channels = [
      {channel: 0, ref: -2, target: 1, delta: 3, abs_delta: 3},
      {channel: 1, ref: 4, target: 2, delta: -2, abs_delta: 2},
      {channel: 17, ref: null, target: 9, delta: null, abs_delta: null},
      {channel: 18, ref: 'NaN', target: Infinity, delta: null, abs_delta: null},
    ];
    fixture.componentRef.setInput('kv', {
      tensorGroups: () => ['key', 'value'].map((kind) => ({kind, request: {loading, error}})),
      tensorEvidence: () => ({evidence: {channels}}),
      tensorSelectedRange: selected,
      tensorChannels: () => ({key: null, value: null}),
    });
    const summaries = () =>
      view
        .tensorStatistics()
        [
          'key'
        ].map(({label, stats}) => [label, stats.min, stats.max, stats.mean, stats.valid_count]);
    const expected = [
      ['Reference', -2, 4, 1, 2],
      ['Target', 1, 9, 4, 3],
      ['Δ', -2, 3, 0.5, 2],
      ['|Δ|', 2, 3, 2.5, 2],
    ];
    expect(summaries()).toEqual(expected);
    expect(
      view.tensorStatistics()['key'].map(({stats}) => [stats.min_position, stats.max_position]),
    ).toEqual([
      [0, 1],
      [0, 17],
      [1, 0],
      [1, 0],
    ]);
    selected.set({start: 17, end: 19});
    expect(summaries()).toEqual(expected);
    selected.set(null);
    expect(summaries()).toEqual(expected);
    expect(view.tensorStatistics()['value']).toEqual(view.tensorStatistics()['key']);
    loading.set(true);
    expect(view.tensorStatistics()['key'].every((row) => row.stats.mean === null)).toBe(true);
    loading.set(false);
    error.set('Unavailable');
    expect(view.tensorStatistics()['key'].every((row) => row.stats.mean === null)).toBe(true);
    fixture.destroy();
  });
  it('keeps every tensor element while point and bin selection only highlight rows', () => {
    const fixture = create(),
      view = fixture.componentInstance;
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
    const fixture = create(),
      view = fixture.componentInstance;
    const position = signal(42),
      layer = signal(10),
      head = signal(2);
    fixture.componentRef.setInput('kv', {
      tensorPosition: position,
      tensorLayer: layer,
      tensorHead: head,
    });
    expect(view.inspectionInfoItems().find((item) => item.label === 'Token position')?.value).toBe(
      '42',
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
    const fixture = create(),
      view = fixture.componentInstance;
    const position = signal(32),
      layer = signal(0),
      head = signal(0);
    const selected = signal<{start: number; end: number} | null>(null);
    const loading = signal(false),
      error = signal('');
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
    const fixture = create(),
      view = fixture.componentInstance;
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
    const fixture = create(),
      view = fixture.componentInstance;
    const loading = signal(false);
    fixture.componentRef.setInput('kv', {
      tensorGroups: () => [{kind: 'key', request: {loading, error: () => ''}}],
      tensorSourceStatus: (_kind: string, side: string) =>
        side === 'ref' ? 'Captured' : 'Not aligned',
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
    const fixture = create(),
      view = fixture.componentInstance;
    const layer = signal(3),
      head = signal(1);
    fixture.componentRef.setInput('kv', {
      tensorPosition: () => 33,
      tensorLayer: layer,
      tensorHead: head,
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
    const fixture = create(),
      view = fixture.componentInstance;
    let positions = 0,
      indices = 0,
      analyses = 0,
      views = 0;
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
    const fixture = create(),
      view = fixture.componentInstance;
    fixture.componentRef.setInput('kv', {positionAlignment: () => ({start: 32, pairs: []})});
    for (const [dimensions, shape, count] of [
      ['Position × Layer × Head × Head dimension', '[32, 15, 4, Mixed]', '32'],
      ['Position × Layer × Head × Head dimension', '[24, 15, 4, Mixed]', '24'],
      ['Position × Layer', '[35, 15]', '35'],
    ]) {
      const items = view.metadataItems({dimensions, shape, dtype: 'float16', layers: []});
      const position = items[0].segments?.find((part) => part.text === 'Token position')?.help;
      expect(position?.rows.find((row) => row.label === 'Count')?.value).toBe(count);
      expect(position?.rows.find((row) => row.label === 'Alignment')?.value).toBe('Context');
      if (shape.includes('Mixed')) {
        const mixed = items[1].segments?.find((part) => part.text === 'Mixed')?.help;
        expect(mixed?.rows[0].value).toBe('Head dimension');
      }
    }
  });
  it.each(['key', 'value'])('keeps both caches when selecting a %s layer', (kind) => {
    const fixture = create(),
      view = fixture.componentInstance;
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
    for (const group of fixture.componentInstance.kvMetadata()) {
      expect(group.sources[0].shape).toBe('[3, 1, 4, 128]');
      expect(group.output.shape).toBe('[3, 1]');
    }
    fixture.componentRef.setInput('kvSelection', null);
    expect(scope()[2]).toEqual(['Groups', 'Transformer Layers [L0, L1, L2]']);
  });
  it('keeps Intermediate Tensors statistics and metadata on the inspected position and layer scope', () => {
    const createRange = document.createRange.bind(document);
    vi.spyOn(document, 'createRange').mockImplementation(() => {
      const range = createRange();
      range.getBoundingClientRect = () => new DOMRect();
      return range;
    });
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    const fixture = create(),
      view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {
      id: 'activation-1',
      kind: 'activations',
      turn: 2,
      stage: 'decode',
    });
    fixture.componentRef.setInput('range', {start: 32, end: 35});
    const captured = signal(true);
    const chartView = signal('heatmap'),
      visible = signal([0, 2]),
      error = signal('');
    fixture.componentRef.setInput('graph', {
      view: chartView,
      layers: () => [0, 1, 2],
      selectedLayers: visible,
      metric: () => 'relative_l2',
      metrics: TENSOR_METRICS,
      anchor: () => 'output',
      selectedAnchors: () => ['output'],
      label: () => 'Intermediate Tensors',
      color: () => null,
      source: () => (captured() ? [{anchor: 'output'}] : []),
      request: {loading: () => false, error},
      metricSource: () =>
        [0, 1, 2].flatMap((layer) =>
          [32, 33, 34].map((position) => ({
            layer,
            position,
            anchor: 'output',
            shape: [4],
            status: position === 33 ? 'missing_ref' : 'ok',
            metrics: position === 33 ? {} : {'Relative L2': {value: 0.01 * (layer + 1)}},
          })),
        ),
    });
    const info = () =>
      Object.fromEntries(view.activationInfoItems().map((item) => [item.label, item.value]));
    const metadata = () =>
      Object.fromEntries(view.activationMetricMetadata().map((item) => [item.label, item.value]));
    expect(info()).toMatchObject({
      Mode: 'Tensor Comparison',
      Groups: 'Transformer Layers [L0, L1, L2]',
      Metric: 'Relative L2',
      Aggregation: 'Sync points [Layer output]',
    });
    expect(metadata()).toMatchObject({
      Dimensions: '[Token position, Layer]',
      Shape: '[3, 3]',
      Dtype: 'float64',
    });
    expect(view.sections()[0].rows.map((row) => row.stats.valid_count)).toEqual([2, 2, 2]);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;
    expect(
      Array.from(root.querySelectorAll(':scope > explorer-info-section > section')).map((section) =>
        section.getAttribute('aria-label'),
      ),
    ).toEqual(['Analysis Info', 'Statistics', 'Metric Metadata']);
    expect(
      root.querySelector('table[aria-label="Intermediate Tensors statistics"]'),
    ).not.toBeNull();
    expect(root.textContent).not.toContain('Value Definition');
    expect(root.textContent).not.toContain('Tensor Metadata');
    chartView.set('trends');
    expect(metadata().Shape).toBe('[3, 2]');
    expect(view.sections()[0].rows.map((row) => row.layer)).toEqual([0, 2]);
    fixture.componentRef.setInput('range', {start: 33, end: 34});
    fixture.componentRef.setInput('graphSelection', {start: 33, end: 34, layer: 2});
    expect(info()).toMatchObject({'Token Range': '33–33', Groups: 'Transformer Layers [L2]'});
    expect(metadata().Shape).toBe('[1, 1]');
    expect(view.sections()[0].rows.map((row) => view.fmtCompact(row.stats.mean))).toEqual(['-']);
    fixture.detectChanges();
    expect(root.querySelector('section[aria-label="Metric Values"]')).toBeNull();
    error.set('Request failed');
    expect(metadata()).toMatchObject({Shape: '-', Dtype: '-'});
    expect(view.sections()[0].rows.every((row) => row.stats.valid_count === 0)).toBe(true);
    error.set('');
    captured.set(false);
    expect(metadata()).toMatchObject({Shape: '-', Dtype: '-'});
    fixture.destroy();
  });
  it('uses unified comparison details and selected sync points for a legacy Intermediate panel', () => {
    const createRange = document.createRange.bind(document);
    vi.spyOn(document, 'createRange').mockImplementation(() => {
      const range = createRange();
      range.getBoundingClientRect = () => new DOMRect();
      return range;
    });
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    const fixture = create(),
      view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {
      id: 'legacy-intermediate',
      kind: 'graph',
      turn: 2,
      stage: 'decode',
    });
    fixture.componentRef.setInput('range', {start: 32, end: 34});
    fixture.componentRef.setInput('label', 'Intermediate Tensor Analysis 0');
    const anchor = signal('attention_output');
    fixture.componentRef.setInput('graph', {
      view: () => 'heatmap',
      layers: () => [2],
      selectedLayers: () => [2],
      metric: () => 'max_abs',
      metrics: TENSOR_METRICS,
      anchor,
      selectedAnchors: () =>
        anchor() === 'attention_output' ? ['attention_output'] : ['output', 'attention_output'],
      label: () => 'Intermediate Tensors',
      color: () => null,
      source: () => [{anchor: 'attention_output'}],
      request: {loading: () => false, error: () => ''},
      metricSource: () =>
        [32, 33].flatMap((position, index) =>
          ['output', 'attention_output'].map((point) => ({
            layer: 2,
            position,
            anchor: point,
            shape: [4],
            status: 'ok',
            metrics: {
              'Max abs error': {value: point === 'output' ? 20 + index * 10 : 2 + index * 2},
            },
          })),
        ),
    });
    fixture.detectChanges();
    expect(view.activationInfo()).toBe(true);
    expect(view.activationInfoItems().map((row) => row.label)).toEqual([
      'Name',
      'Mode',
      'Stage',
      'Token Range',
      'Groups',
      'Metric',
      'Aggregation',
    ]);
    expect(view.aggregationDefinition().value).toBe('Sync points [attention_output]');
    expect(view.sections()[0].rows[0].stats).toMatchObject({
      min: 2,
      max: 4,
      mean: 3,
      min_position: 32,
      max_position: 33,
      valid_count: 2,
    });
    const root = fixture.nativeElement as HTMLElement;
    expect(
      Array.from(root.querySelectorAll(':scope > explorer-info-section > section')).map((section) =>
        section.getAttribute('aria-label'),
      ),
    ).toEqual(['Analysis Info', 'Statistics', 'Metric Metadata']);
    expect(root.querySelector('section[aria-label="Metric Values"]')).toBeNull();
    anchor.set('max');
    expect(view.aggregationDefinition().value).toBe(
      'Maximum · Sync points [Layer output, attention_output]',
    );
    expect(view.sections()[0].rows[0].stats).toMatchObject({min: 20, max: 30, mean: 25});
    anchor.set('mean');
    expect(view.sections()[0].rows[0].stats).toMatchObject({min: 11, max: 17, mean: 14});
    fixture.destroy();
  });
  it.each(['activations', 'graph'])(
    'inspects %s elements without inferring dtype or replacing missing sides',
    (kind) => {
      const createRange = document.createRange.bind(document);
      vi.spyOn(document, 'createRange').mockImplementation(() => {
        const range = createRange();
        range.getBoundingClientRect = () => new DOMRect();
        return range;
      });
      vi.stubGlobal(
        'ResizeObserver',
        class {
          observe() {}
          disconnect() {}
        },
      );
      const fixture = create(),
        view = fixture.componentInstance;
      fixture.componentRef.setInput('panel', {
        id: 'activation-inspection',
        kind,
        turn: 2,
        stage: 'decode',
      });
      fixture.componentRef.setInput('label', 'Intermediate Tensor Analysis 0');
      const loading = signal(false),
        error = signal(''),
        compatible = signal(true);
      const selection = signal<{start: number; end: number} | null>(null);
      const selectedIndex = signal<number | null>(null);
      const rows = signal(
        Array.from({length: 18}, (_, channel) => ({
          channel,
          ref: channel,
          target: channel + 2,
          delta: 2,
          abs_delta: 2,
        })),
      );
      const data = signal<{
        reference: {shape: number[]; dtype: string | null} | null;
        target: {shape: number[]; dtype: string | null} | null;
      }>({reference: {shape: [18], dtype: null}, target: {shape: [18], dtype: 'float16'}});
      const source = {
        loading,
        error,
        compatible,
        data,
        channels: rows,
        position: signal(33),
        layer: signal(2),
        anchor: signal('attention_output'),
        elementRange: () => ({start: 0, end: 18}),
        valueRange: () => ({start: 0, end: 19}),
        channelSelection: selection,
        selectedIndex,
        notices: () => [],
        locatePosition: vi.fn(),
        locateLayer: vi.fn(),
        locateElements: vi.fn(),
        locateValues: vi.fn(),
        locateTensor: vi.fn(),
        locateElement: vi.fn(),
        showElement: vi.fn((index: number) => {
          selectedIndex.set(index);
          selection.set(null);
        }),
        retry: vi.fn(),
      };
      const child = signal<typeof source | undefined>(undefined);
      fixture.componentRef.setInput('graph', {inspectionMode: () => true, inspection: child, view: () => 'heatmap'});
      fixture.detectChanges();
      const root = fixture.nativeElement as HTMLElement;
      expect(
        Array.from(root.querySelectorAll(':scope > explorer-info-section > section')).map(
          (section) => section.getAttribute('aria-label'),
        ),
      ).toEqual(['Analysis Info', 'Statistics', 'Tensor Metadata', 'Tensor Values']);
      expect(root.textContent).toContain('Tensor Inspection');
      expect(root.textContent).not.toContain('Metric Values');
      child.set(source);
      fixture.detectChanges();
      expect(view.activationInspectionInfoItems().map((item) => item.label)).toEqual(
        view.inspectionInfoItems().map((item) => item.label),
      );
      expect(
        view.activationInspectionInfoItems().find((item) => item.label === 'Tensor Index')?.value,
      ).toBe('[Layer: 2, Sync point: attention_output]');
      expect(view.activationInspectionMetadata('ref')).toContainEqual({label: 'Dtype', value: '-'});
      expect(view.activationInspectionMetadata('target')).toContainEqual({
        label: 'Dtype',
        value: 'float16',
      });
      const means = () => view.activationInspectionStatistics().map((row) => row.stats.mean);
      means().forEach((mean, index) => expect(mean).toBeCloseTo([8.5, 10.5, 2, 2][index]));
      expect(view.activationInspectionPage().pages).toBe(2);
      view.changeActivationPage(1);
      expect(view.activationInspectionPage().rows.map((row) => row.index)).toEqual([16, 17]);
      source.anchor.set('pre_attention_norm');
      expect(view.activationInspectionPage().index).toBe(0);
      expect(
        view.activationInspectionInfoItems().find((item) => item.label === 'Tensor Index')?.value,
      ).toBe('[Layer: 2, Sync point: pre_attention_norm]');
      selection.set({start: 2, end: 4});
      expect(view.activationInspectionPage().index).toBe(0);
      expect(view.activationElementSelected(2)).toBe(true);
      expect(view.activationElementSelected(3)).toBe(true);
      expect(view.activationElementSelected(4)).toBe(false);
      means().forEach((mean, index) => expect(mean).toBeCloseTo([8.5, 10.5, 2, 2][index]));
      view.sortActivationValues('ref');
      view.sortActivationValues('ref');
      expect(view.activationInspectionPage().rows[0].index).toBe(17);
      fixture.detectChanges();
      const valueTable = root.querySelector(
        'table[aria-label="Intermediate Tensors tensor values"]',
      )!;
      const valueRow = valueTable.querySelector('tr[aria-label="Index 17"]') as HTMLElement;
      valueRow.click();
      valueRow.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
      expect(source.showElement).not.toHaveBeenCalled();
      expect(valueRow.hasAttribute('tabindex')).toBe(false);
      expect(valueTable.querySelector('button[explorer-locator]')).toBeNull();
      const chartLocators = root.querySelectorAll<HTMLButtonElement>(
        'button[aria-label="Locate Intermediate Tensors tensor chart"]',
      );
      expect(chartLocators.length).toBe(2);
      chartLocators.forEach((button) => button.click());
      expect(source.locateTensor).toHaveBeenCalledTimes(2);
      expect(selection()).toEqual({start: 2, end: 4});
      source.showElement(17);
      expect(view.activationElementSelected(17)).toBe(true);
      expect(view.activationElementSelected(2)).toBe(false);
      view.sortActivationValues('index');
      selectedIndex.set(17);
      selection.set({start: 0, end: 18});
      expect(view.activationInspectionPage().index).toBe(1);
      expect(view.activationElementSelected(0)).toBe(true);
      expect(view.activationElementSelected(17)).toBe(true);
      // A representative from a previous bin must not override the current selection.
      selection.set({start: 0, end: 2});
      expect(view.activationInspectionPage().index).toBe(0);
      for (const label of ['Token position', 'Tensor Index'])
        view.locateActivationInspection({label});
      expect(source.locatePosition).toHaveBeenCalledOnce();
      expect(source.locateLayer).toHaveBeenCalledOnce();
      compatible.set(false);
      data.set({reference: null, target: {shape: [1], dtype: null}});
      rows.set(rows().slice(0, 1));
      expect(view.activationInspectionRows()[0].values.slice(2)).toEqual([null, null]);
      expect(
        view.activationInspectionMetadata('ref').find((item) => item.label === 'Shape')?.value,
      ).toBe('-');
      expect(view.activationInspectionPage().pages).toBe(1);
      fixture.detectChanges();
      const pagination = root.querySelector('explorer-pagination');
      expect(pagination?.textContent).toContain('1 / 1');
      expect(
        Array.from(pagination?.querySelectorAll('button') ?? []).every((button) => button.disabled),
      ).toBe(true);
      loading.set(true);
      expect(view.activationInspectionRows()).toEqual([]);
      expect(
        view.activationInspectionMetadata('target').find((item) => item.label === 'Shape')?.value,
      ).toBe('-');
      loading.set(false);
      error.set('Tensor unavailable');
      fixture.detectChanges();
      expect(view.activationInspectionRows()).toEqual([]);
      expect(
        root.querySelector('table[aria-label="Intermediate Tensors tensor values"]'),
      ).toBeNull();
      fixture.destroy();
    },
  );
  it('routes Activation view and scope locators without changing the inspected selection', () => {
    const fixture = create(),
      view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {
      id: 'activation-1',
      kind: 'activations',
      turn: 2,
      stage: 'decode',
    });
    const locateView = vi.fn(),
      scope = vi.fn();
    view.locateView.subscribe(locateView);
    const selection = {start: 32, end: 34, layer: 1};
    fixture.componentRef.setInput('graphSelection', selection);
    view.locateScope.subscribe(scope);
    view.locateDefinition({label: 'Mode'});
    view.locateDefinition({label: 'Token Range'});
    view.locateDefinition({label: 'Groups'});
    expect(locateView).toHaveBeenCalledOnce();
    expect(scope.mock.calls).toEqual([['positions'], ['layers']]);
    expect(view.graphSelection()).toEqual(selection);
    fixture.destroy();
  });
  it('sorts each cache independently without reordering the source used by head locators', () => {
    const view = create().componentInstance;
    const rows = [head(2, 0.3), head(0, 0.9), head(1, 0.1)];
    expect(view.tensorRows(rows, 'Key cache').map((row) => row.head)).toEqual([0, 1, 2]);
    view.sortBy('Key cache heads', 'cosine_similarity', 'head');
    expect(view.tensorRows(rows, 'Key cache').map((row) => row.head)).toEqual([1, 2, 0]);
    expect(view.tensorRows(rows, 'Value cache').map((row) => row.head)).toEqual([0, 1, 2]);
    expect(rows.map((row) => row.head)).toEqual([2, 0, 1]);
    expect(view.sortDirection('Key cache', 'layer')).toBe('ascending');
  });
  it('keeps unavailable metrics last in both directions and ties in head order', () => {
    const view = create().componentInstance;
    const rows = [head(2, null), head(1, 0.5), head(3, NaN), head(0, 0.5), head(4, 0.9)];
    view.sortBy('Key cache heads', 'cosine_similarity', 'head');
    expect(view.tensorRows(rows, 'Key cache').map((row) => row.head)).toEqual([0, 1, 4, 2, 3]);
    view.sortBy('Key cache heads', 'cosine_similarity', 'head');
    expect(view.tensorRows(rows, 'Key cache').map((row) => row.head)).toEqual([4, 0, 1, 2, 3]);
  });
  it('resets sorting when the analysis instance changes', () => {
    const fixture = create(),
      view = fixture.componentInstance;
    view.sortBy('Key cache heads', 'head', 'head');
    expect(view.sortDirection('Key cache heads', 'head', 'head')).toBe('descending');
    fixture.componentRef.setInput('panel', {id: 'kv-2', kind: 'kv', turn: 2, stage: 'decode'});
    expect(view.sortDirection('Key cache heads', 'head', 'head')).toBe('ascending');
  });
});
