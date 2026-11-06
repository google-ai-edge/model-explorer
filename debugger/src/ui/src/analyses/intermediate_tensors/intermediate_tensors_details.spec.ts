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
import {IntermediateTensorsDetails} from './intermediate_tensors_details';
import {KvApi} from '../../server/kv_api';
import {CaptureStore} from '../../server/capture_store';
import type {KvRangeMetricSummary} from '../../server/contracts/kv';

/** A statistic the server sends, of `values` at positions from `start`. */
function statistic(values: (number | null)[], start: number): KvRangeMetricSummary {
  const valid = values
    .map((value, i) => ({value, position: start + i}))
    .filter((v): v is {value: number; position: number} => v.value !== null);
  const min = valid.reduce<(typeof valid)[number] | null>(
    (a, b) => (!a || b.value < a.value ? b : a),
    null,
  );
  const max = valid.reduce<(typeof valid)[number] | null>(
    (a, b) => (!a || b.value > a.value ? b : a),
    null,
  );
  return {
    min: min?.value ?? null,
    max: max?.value ?? null,
    mean: valid.length ? valid.reduce((sum, v) => sum + v.value, 0) / valid.length : null,
    min_position: min?.position ?? null,
    max_position: max?.position ?? null,
    valid_count: valid.length,
  };
}
const totalsRow = (layer: number, metric: string, stats: KvRangeMetricSummary) => ({
  layer,
  kind: 'Intermediate Tensors',
  status: 'ok',
  reasons: [],
  bins: [{start: 0, end: 1, metrics: {[metric]: stats}}],
});
const anchorLabel = (anchor: string | null | undefined) =>
  anchor === 'output' ? 'Layer output' : (anchor ?? '-');

describe('Intermediate Tensors details', () => {
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
    const fixture = TestBed.createComponent(IntermediateTensorsDetails);
    fixture.componentRef.setInput('panel', {id: 'kv-1', kind: 'kv', turn: 2, stage: 'decode'});
    fixture.componentRef.setInput('label', 'KV cache');
    fixture.componentRef.setInput('range', {start: 32, end: 66});
    return fixture;
  }
  it('uses the shared analysis context for Arch instead of treating the subgraph as the analysis', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {
      id: 'arch-1',
      kind: 'activations',
      turn: 2,
      stage: 'decode',
    });
    fixture.componentRef.setInput('label', 'Intermediate Tensor Analysis 0');
    const position = signal(32);
    const summary = signal('max');
    const mapping = signal({active: false, title: 'Layer 0'});
    const locateRange = vi.fn();
    const locateSummary = vi.fn();
    fixture.componentRef.setInput('graph', {
      metric: () => 'cosine_similarity',
      graphPosition: position,
      columnSteps: (column: number) => ({
        ref: `Decode step ${column} · reads "," · produces "blue"`,
        target: `Decode step ${column} · reads "," · produces "blue"`,
      }),
      moduleSummary: summary,
      locateRange,
      locateSummary,
      archView: () => ({mapping}),
      archDetails: () => [
        {
          label: 'Analysis Info',
          items: [
            {label: 'Name', value: 'Per-Layer Embedding'},
            {label: 'Attention', value: 'Sliding window · 512'},
            {label: 'K / V source', value: 'Layer 13'},
            {label: 'Signature', value: 'decode'},
          ],
        },
      ],
    });
    const info = () =>
      Object.fromEntries(view.graphArchInfoItems().map((item) => [item.label, item.value]));
    expect(info()).toMatchObject({
      Name: 'Intermediate Tensor Analysis 0',
      Mode: 'Tensor Comparison',
      Stage: 'Turn 2 · Decode',
      'Token Position': '32',
      Step: 'Decode step 32 · reads "," · produces "blue"',
      Subgraph: 'Per-Layer Embedding',
      // The layer's facts the server's skeleton gives, and the column's signature.
      Attention: 'Sliding window · 512',
      'K / V source': 'Layer 13',
      Signature: 'decode',
      Summary: 'Max',
    });
    position.set(33);
    summary.set('min');
    expect(info()).toMatchObject({'Token Position': '33', Summary: 'Min'});
    view.locateGraphArchDefinition({label: 'Token Position'});
    view.locateGraphArchDefinition({label: 'Summary'});
    expect(locateRange).toHaveBeenCalledOnce();
    expect(locateSummary).toHaveBeenCalledOnce();
    mapping.set({active: true, title: 'Layer 0'});
    expect(info()).toMatchObject({
      Name: 'Intermediate Tensor Analysis 0',
      Subgraph: 'Layer 0',
      'Token Position': '33',
    });
    expect(info()).not.toHaveProperty('Summary');
    expect(info()).not.toHaveProperty('Attention');
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
    const fixture = create();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {
      id: 'activation-1',
      kind: 'activations',
      turn: 2,
      stage: 'decode',
    });
    fixture.componentRef.setInput('range', {start: 32, end: 35});
    const captured = signal(true);
    const chartView = signal('heatmap');
    const visible = signal([0, 2]);
    const error = signal('');
    // The server's totals over the visible range: positions 32 and 34 hold a value.
    const totals = signal(
      [0, 1, 2].map((layer) =>
        totalsRow(
          layer,
          'relative_l2',
          statistic([0.01 * (layer + 1), null, 0.01 * (layer + 1)], 32),
        ),
      ),
    );
    fixture.componentRef.setInput('graph', {
      view: chartView,
      layers: () => [0, 1, 2],
      selectedLayers: visible,
      metric: () => 'relative_l2',
      anchor: () => 'output',
      selectedAnchors: () => ['output'],
      label: () => 'Intermediate Tensors',
      color: () => null,
      source: () => (captured() ? [{anchor: 'output'}] : []),
      request: {loading: () => false, error},
      rangeRequest: {
        loading: () => false,
        error: () => '',
        data: () => ({metadata: {output: {dimensions: [], shape: null, dtype: 'float64'}}}),
      },
      totals,
      chartPending: () => false,
      anchorLabel,
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
    chartView.set('lineChart');
    expect(metadata().Shape).toBe('[3, 2]');
    expect(view.sections()[0].rows.map((row) => row.layer)).toEqual([0, 2]);
    fixture.componentRef.setInput('range', {start: 33, end: 34});
    fixture.componentRef.setInput('graphSelection', {start: 33, end: 34, layer: 2});
    totals.set([0, 1, 2].map((layer) => totalsRow(layer, 'relative_l2', statistic([null], 33))));
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
  it('uses unified comparison details and selected sync points for an Intermediate Tensors panel', () => {
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
    const fixture = create();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {
      id: 'intermediate',
      kind: 'activations',
      turn: 2,
      stage: 'decode',
    });
    fixture.componentRef.setInput('range', {start: 32, end: 34});
    fixture.componentRef.setInput('label', 'Intermediate Tensor Analysis 0');
    const anchor = signal('attention_output');
    // What the server answers of layer 2 over positions 32 and 33 for each aggregation.
    const answered: Record<string, (number | null)[]> = {
      attention_output: [2, 4],
      max: [20, 30],
      mean: [11, 17],
    };
    fixture.componentRef.setInput('graph', {
      view: () => 'heatmap',
      layers: () => [2],
      selectedLayers: () => [2],
      metric: () => 'max_abs',
      anchor,
      selectedAnchors: () =>
        anchor() === 'attention_output' ? ['attention_output'] : ['output', 'attention_output'],
      label: () => 'Intermediate Tensors',
      color: () => null,
      source: () => [{anchor: 'attention_output'}],
      request: {loading: () => false, error: () => ''},
      rangeRequest: {loading: () => false, error: () => '', data: () => null},
      totals: () => [totalsRow(2, 'max_abs', statistic(answered[anchor()], 32))],
      chartPending: () => false,
      anchorLabel,
    });
    fixture.detectChanges();
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
  it.each(['activations'])(
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
      const fixture = create();
      const view = fixture.componentInstance;
      fixture.componentRef.setInput('panel', {
        id: 'activation-inspection',
        kind,
        turn: 2,
        stage: 'decode',
      });
      fixture.componentRef.setInput('label', 'Intermediate Tensor Analysis 0');
      const loading = signal(false);
      const error = signal('');
      const compatible = signal(true);
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
      // The server's statistics of the elements: Ref 0..17, Target 2..19, Δ 2.
      const statistics = {
        ref: statistic(
          Array.from({length: 18}, (_, i) => i),
          0,
        ),
        target: statistic(
          Array.from({length: 18}, (_, i) => i + 2),
          0,
        ),
        delta: statistic(Array(18).fill(2), 0),
        abs_delta: statistic(Array(18).fill(2), 0),
      };
      const data = signal<{
        reference: {shape: number[]; dtype: string | null} | null;
        target: {shape: number[]; dtype: string | null} | null;
        statistics: typeof statistics;
      }>({
        reference: {shape: [18], dtype: null},
        target: {shape: [18], dtype: 'float16'},
        statistics,
      });
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
        locateTensor: vi.fn(),
        locateElement: vi.fn(),
        showElement: vi.fn((index: number) => {
          selectedIndex.set(index);
          selection.set(null);
        }),
        retry: vi.fn(),
      };
      const child = signal<typeof source | undefined>(undefined);
      fixture.componentRef.setInput('graph', {
        inspectionMode: () => true,
        inspection: child,
        view: () => 'heatmap',
        anchorLabel,
        // The sides' tokens were produced at different steps.
        columnSteps: () => ({
          ref: 'Decode step 33 · reads "," · produces "blue"',
          target: 'Decode step 34 · reads "," · produces "blue"',
        }),
      });
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
      // The labels of the KV inspection's Analysis Info.
      expect(view.activationInspectionInfoItems().map((item) => item.label)).toEqual([
        'Name',
        'Mode',
        'Stage',
        'Token position',
        'Reference step',
        'Target step',
        'Tensor Index',
      ]);
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
      for (const label of ['Token position', 'Tensor Index']) {
        view.locateActivationInspection({label});
      }
      expect(source.locatePosition).toHaveBeenCalledOnce();
      expect(source.locateLayer).toHaveBeenCalledOnce();
      compatible.set(false);
      data.set({reference: null, target: {shape: [1], dtype: null}, statistics});
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
    const fixture = create();
    const view = fixture.componentInstance;
    fixture.componentRef.setInput('panel', {
      id: 'activation-1',
      kind: 'activations',
      turn: 2,
      stage: 'decode',
    });
    const locateView = vi.fn();
    const scope = vi.fn();
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
});
