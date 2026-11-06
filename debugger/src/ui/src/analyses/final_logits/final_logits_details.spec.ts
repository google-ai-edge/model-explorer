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

import {computed, signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {afterEach, beforeEach, vi} from 'vitest';
import {KvApi} from '../../server/kv_api';
import {CaptureStore} from '../../server/capture_store';
import {reasonText} from '../../server/reasons';
import type {
  FinalLogitsResponse,
  FinalLogitsOrder,
  FinalLogitsDirection,
} from '../../server/contracts/final_logits';
import type {TokenRangeRequest, TokenRangeResponse} from '../../server/contracts/token_analysis';
import type {ElementStatistics} from '../../server/contracts/kv';
import {TokenAnalysisApi} from '../../server/token_analysis_api';
import {ExplorerInfoValue} from '../../widgets/explorer_info_value/explorer_info_value';
import {FinalLogitsDetails} from './final_logits_details';

/** The server's value of each metric at the range's one paired position (32). */
const VALUES: Record<string, number> = {rmse: 2, max_abs: 4, kl: 0.5, js: 0.2};
/** The server's token-range answer of a request over the range 32..33: one bin, the metric's
 *  value at position 32, and one ref and two target vectors of four logits. */
function answer(request: TokenRangeRequest): TokenRangeResponse {
  const value = VALUES[request.metric];
  const cell = {
    start: request.start,
    end: request.end,
    min: value,
    max: value,
    mean: value,
    min_position: 32,
    max_position: 32,
    valid_count: 1,
  };
  const side = (count: number) => ({
    count,
    vocabulary_sizes: [4],
    dtypes: [null],
    vocabulary_identities: [null],
  });
  return {
    ...request,
    bins: [{start: request.start, end: request.end}],
    rows: [{side: null, bins: [cell]}],
    all: [cell],
    metadata: {
      ref: side(1),
      target: side(1),
      output: {dimensions: ['Position'], shape: [request.end - request.start], dtype: 'float64'},
    },
    peak_kl: null,
  };
}

describe('Logits comparison inspector', () => {
  beforeEach(() => {
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
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });
  let requests: TokenRangeRequest[];
  function create(metric: 'kl' | 'js' | 'logit_rmse' | 'logit_max_abs' = 'logit_rmse') {
    requests = [];
    TestBed.configureTestingModule({
      providers: [
        {provide: KvApi, useValue: {}},
        {
          provide: CaptureStore,
          useValue: {
            captureId: signal('capture'),
            // Each side's first generated token of turn 2, and the step that produced it.
            session: signal({
              model: '',
              runs: [],
              turns: [],
              batches: [],
              conversation: ['ref', 'target'].map((run) => ({
                turn: 2,
                run,
                tokens: [
                  {
                    text: 'blue',
                    step: 0,
                    position: 32,
                    produced_text: 'Decode step 32 · reads "," · produces "blue"',
                  },
                ],
              })),
            }),
          },
        },
        {
          provide: TokenAnalysisApi,
          useValue: {
            tokenRange: async (_capture: string, request: TokenRangeRequest) => {
              requests.push(request);
              return answer(request);
            },
          },
        },
      ],
    });
    const fixture = TestBed.createComponent(FinalLogitsDetails);
    fixture.componentRef.setInput('panel', {
      id: 'analysis-0',
      kind: 'tokens',
      turn: 2,
      stage: 'decode',
      open: true,
      logit: {metric, open: true},
    });
    fixture.componentRef.setInput('label', 'Logits Analysis 0');
    fixture.componentRef.setInput('range', {start: 32, end: 34});
    fixture.detectChanges();
    return fixture;
  }
  /** The details with the server's answer of their range. */
  async function created(metric?: Parameters<typeof create>[0]) {
    const fixture = create(metric);
    await vi.waitFor(() => expect(fixture.componentInstance.rangeData.data()).not.toBeNull());
    fixture.detectChanges();
    return fixture;
  }
  it('asks the server for the metric over the range, by the catalog id', async () => {
    const fixture = await created();
    expect(requests).toEqual([
      {
        turn: 2,
        stage: 'decode',
        alignment: 'content',
        analysis: 'final_logits',
        metric: 'rmse',
        start: 32,
        end: 34,
        bins: 1,
      },
    ]);
    fixture.destroy();
  });
  it('uses the KV four-column statistics layout without coverage or missing-reason rows', async () => {
    const fixture = await created();
    const el: HTMLElement = fixture.nativeElement;
    const sections = Array.from(el.querySelectorAll(':scope > explorer-info-section'));
    expect(
      sections.map((section) => section.querySelector('section')?.getAttribute('aria-label')),
    ).toEqual(['Analysis Info', 'Statistics', 'Tensor Metadata', 'Metric Metadata']);
    const table = el.querySelector('table[aria-label="Final logits statistics"]')!;
    expect(
      Array.from(table.querySelectorAll('th .sort-label')).map((label) =>
        label.textContent?.trim(),
      ),
    ).toEqual(['Grouping', 'Min', 'Max', 'Mean']);
    expect(table.querySelectorAll('col').length).toBe(5);
    const cells = Array.from(table.querySelectorAll('tbody td'));
    expect(cells).toHaveLength(4);
    expect(cells[0].textContent?.trim()).toBe('Final logits');
    expect(table.querySelector('thead th:last-child')?.getAttribute('colspan')).toBe('2');
    expect(cells[3].getAttribute('colspan')).toBe('2');
    expect(sections[1].textContent).not.toContain('1 / 2');
    expect(sections[1].textContent).not.toContain('Valid positions');
    expect(sections[1].textContent).not.toContain('Alignment gap');
    expect(sections[1].querySelector('explorer-info-list')).toBeNull();
    expect(sections[2].textContent).not.toContain('float64');
    expect(sections[3].textContent).not.toContain('Softmax (T=1)');
    expect(sections[3].textContent).not.toContain('Reference → Target');
    fixture.destroy();
  });
  it('matches KV tensor metadata with a compact Logits subsection and three properties per runtime', async () => {
    const fixture = await created();
    const view = fixture.componentInstance;
    const el: HTMLElement = fixture.nativeElement;
    const metadata = el.querySelector('section[aria-label="Tensor Metadata"]')!;
    const runtimes = Array.from(metadata.querySelectorAll('.analysis-item'));
    expect(runtimes.map((runtime) => runtime.querySelector('.item-name')?.textContent)).toEqual([
      'Reference',
      'Target',
    ]);
    for (const runtime of runtimes) {
      const group = runtime.querySelector('explorer-info-section.source-section.compact')!;
      expect(group.querySelector('section')?.getAttribute('aria-label')).toBe('Final logits');
      expect(
        Array.from(group.querySelectorAll('.flat-item')).map((item) =>
          item.getAttribute('aria-label'),
        ),
      ).toEqual(['Dimensions', 'Shape', 'Dtype']);
      expect(group.querySelectorAll('.bullet')).toHaveLength(3);
    }
    for (const side of ['ref', 'target'] as const) {
      expect(view.logitSourceItems(side).map(({label, value}) => [label, value])).toEqual([
        ['Dimensions', '[Token position, Vocabulary]'],
        ['Shape', '[1, 4]'],
        ['Dtype', '—'],
      ]);
    }
    fixture.destroy();
  });
  it.each([
    ['kl', 'Kullback–Leibler divergence', 'nats'],
    ['js', 'Jensen–Shannon divergence', 'bits'],
  ] as const)(
    'keeps the complete %s metric name visible and puts units in help',
    (metric, name, unit) => {
      const fixture = create(metric);
      const view = fixture.componentInstance;
      const el: HTMLElement = fixture.nativeElement;
      const property = el.querySelector('[role="listitem"][aria-label="Metric"]')!;
      const value = property.querySelector('explorer-info-value')!;
      expect(value.classList.contains('wrap')).toBe(true);
      expect(value.querySelector('.value')?.getAttribute('aria-label')).toBe(`Metric: ${name}`);
      expect(value.querySelector('.read-value')).toBeNull();
      expect(
        el.querySelector(`.analysis-item[aria-label="${name} output"] .item-name`)?.textContent,
      ).toBe(name);
      const help = view.metricDefinition().segments?.[0].help?.rows;
      expect(help).toContainEqual({label: 'Unit', value: unit});
      const inputs = help?.find((row) => row.label === 'Inputs')?.value;
      expect(inputs).toContain('softmax (T=1), independent of the generation settings');
      expect(inputs).toContain('no top-k, top-p or sampling constraints');
      fixture.destroy();
    },
  );
  it('matches KV output metadata fields and reports the live global alignment', async () => {
    const fixture = await created();
    const view = fixture.componentInstance;
    const el: HTMLElement = fixture.nativeElement;
    const metadata = el.querySelector('section[aria-label="Metric Metadata"]')!;
    const group = metadata.querySelector('explorer-info-section.source-section.compact')!;
    expect(group.querySelector('section')?.getAttribute('aria-label')).toBe('Final logits');
    expect(
      Array.from(group.querySelectorAll('.flat-item')).map((item) =>
        item.getAttribute('aria-label'),
      ),
    ).toEqual(['Dimensions', 'Shape', 'Dtype', 'Display precision']);
    expect(group.querySelectorAll('.bullet')).toHaveLength(4);
    expect(view.logitMetricMetadata().map(({label, value}) => [label, value])).toEqual([
      ['Dimensions', '[Token position]'],
      ['Shape', '[2]'],
      ['Dtype', 'float64'],
      ['Display precision', '6 significant digits'],
    ]);
    const positionHelp = () =>
      view.logitMetricMetadata()[0].segments?.find((segment) => segment.text === 'Token position')
        ?.help?.rows;
    expect(positionHelp()).toContainEqual({label: 'Alignment', value: 'Context'});
    expect(positionHelp()).toContainEqual({
      label: 'Counting',
      value: 'All selected aligned positions',
    });
    fixture.componentRef.setInput('alignment', 'steps');
    fixture.detectChanges();
    await vi.waitFor(() => expect(view.rangeData.data()?.alignment).toBe('steps'));
    expect(positionHelp()).toContainEqual({label: 'Alignment', value: 'Steps'});
    expect(view.logitMetricMetadata().find((item) => item.label === 'Shape')?.value).toBe('[2]');
    fixture.destroy();
  });
  it('routes view, range and extrema locators without changing the inspected range', async () => {
    const fixture = await created();
    const view = fixture.componentInstance;
    const events: unknown[] = [];
    view.locateView.subscribe(() => events.push('view'));
    view.locateScope.subscribe((scope) => events.push(scope));
    view.location.subscribe((location) => events.push(location));
    view.locateDefinition(view.logitInfoItems().find((item) => item.label === 'Mode')!);
    view.locateDefinition(view.logitInfoItems().find((item) => item.label === 'Token Range')!);
    const locator: HTMLButtonElement = fixture.nativeElement.querySelector(
      'button[aria-label="Locate max position in Final"]',
    );
    locator.click();
    expect(events).toEqual([
      'view',
      'positions',
      expect.objectContaining({position: 32, inspect: false}),
    ]);
    expect(view.range()).toEqual({start: 32, end: 34});
    expect(fixture.nativeElement.querySelector('button[aria-label*="Locate mean"]')).toBeNull();
    fixture.destroy();
  });
  it('asks nothing anew, and shows no loading, for a panel change that keeps the request', async () => {
    const fixture = await created();
    const view = fixture.componentInstance;
    // A new panel object, a reduction, an equal range: the answer cannot change.
    fixture.componentRef.setInput('panel', {
      ...view.panel(),
      logit: {...view.panel().logit!, positionReduction: 'mean'},
    });
    fixture.componentRef.setInput('range', {start: 32, end: 34});
    fixture.detectChanges();
    expect(view.rangeData.loading()).toBe(false);
    expect(view.sections()[0].loading).toBe(false);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(requests).toHaveLength(1);
    fixture.destroy();
  });
  it('keeps metrics and inspector information raw when an old view saved centered', async () => {
    const fixture = await created();
    const view = fixture.componentInstance;
    expect(view.sections()[0].rows[0].stats).toMatchObject({min: 2, max: 2, mean: 2});
    fixture.componentRef.setInput('panel', {
      ...view.panel(),
      logit: {
        ...view.panel().logit!,
        logitTransform: 'centered',
        metric: 'logit_max_abs',
      },
    });
    fixture.detectChanges();
    await vi.waitFor(() => expect(view.rangeData.data()?.metric).toBe('max_abs'));
    expect(view.logitInfoItems().some((item) => item.label === 'Transform')).toBe(false);
    expect(view.sections()[0].rows[0].stats).toMatchObject({min: 4, max: 4, mean: 4});
    expect(requests.map((request) => request.metric)).toEqual(['rmse', 'max_abs']);
    expect(view.metricDefinition().segments?.[0].help?.rows).toContainEqual({
      label: 'Inputs',
      value: 'Raw captured final logits, before softmax; no centering',
    });
    fixture.componentRef.setInput('panel', {
      ...view.panel(),
      logit: {...view.panel().logit!, logitView: 'inspection'},
    });
    fixture.detectChanges();
    expect(view.logitInspectionInfoItems().map((item) => item.label)).toEqual([
      'Name',
      'Mode',
      'Stage',
      'Token position',
    ]);
    expect(fixture.nativeElement.textContent).not.toContain('Centered');
    expect(fixture.nativeElement.querySelector('[aria-label="Transform"]')).toBeNull();
    // Inspection asks no range of the server.
    expect(view.rangeData.data()).toBeNull();
    expect(requests).toHaveLength(2);
    fixture.destroy();
  });
  it.each(['kl', 'js'] as const)('keeps one paired %s group in Final Logits', async (metric) => {
    const fixture = await created(metric);
    const view = fixture.componentInstance;
    expect(view.sections()).toHaveLength(1);
    expect(view.sections()[0].label).toBe('Final logits');
    expect(view.sections()[0].rows[0].stats.mean).toBe(metric === 'kl' ? 0.5 : 0.2);
    expect(view.logitMetricInfo()).toBe(true);
    expect(view).toBeInstanceOf(FinalLogitsDetails);
    expect(view.logitInfoItems().some((item) => item.label === 'Transform')).toBe(false);
    expect(
      fixture.nativeElement.querySelectorAll('table[aria-label="Final logits statistics"]'),
    ).toHaveLength(1);
    fixture.destroy();
  });
  it('locates the View control without providing another inspection entry or changing scope', () => {
    const fixture = create();
    const view = fixture.componentInstance;
    const locate = vi.fn();
    view.locateView.subscribe(locate);
    fixture.componentRef.setInput('range', {start: 32, end: 33});
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).not.toContain('Inspect tensor');
    const locator: HTMLButtonElement = fixture.nativeElement.querySelector(
      'button[aria-label="Locate View control"]',
    );
    locator.click();
    expect(locate).toHaveBeenCalledOnce();
    expect(view.panel().logit?.logitView).toBeUndefined();
    expect(view.range()).toEqual({start: 32, end: 33});
    fixture.destroy();
  });
  function inspect() {
    const fixture = create();
    const view = fixture.componentInstance;
    const metadata = {
      source_shape: [1, 1, 4],
      comparison_shape: [4],
      source_dtype: 'float16',
      vocab_identity: 'vocab',
    };
    const data = signal<FinalLogitsResponse | null>({
      turn: 2,
      stage: 'decode',
      alignment: 'content',
      row: 0,
      first_divergence: false,
      ref_step: 0,
      target_step: 0,
      transform: 'raw',
      compatible: true,
      reasons: [],
      total: 4,
      offset: 0,
      limit: 1,
      rows: [{index: 0, label: 'zero', ref: 1, target: 1, delta: 0, absolute_delta: 0}],
      ref: {size: 4, logits_metadata: metadata, source_forward_id: 12},
      target: {
        size: 4,
        logits_metadata: {...metadata, source_dtype: 'float32'},
        source_forward_id: 13,
      },
      metrics: {
        rmse: {value: 2, reasons: []},
        max_abs: {value: 4, reasons: []},
        max_abs_index: 3,
      },
    });
    // The server's statistics of ref [1, -2, 0, 3] and target [2, -4, 2, 7].
    const stat = (min: number, max: number, mean: number, minPosition: number) => ({
      min,
      max,
      mean,
      min_position: minPosition,
      max_position: 3,
      valid_count: 4,
    });
    const statistics = signal<ElementStatistics | null>({
      ref: stat(-2, 3, 0.5, 1),
      target: stat(-4, 7, 1.75, 1),
      delta: stat(-2, 4, 1.25, 1),
      abs_delta: stat(1, 4, 2.25, 0),
    });
    const loading = signal(false);
    const error = signal('');
    const valuesLoading = signal(false);
    const valuesError = signal('');
    const pageReason = signal<string | null>(null);
    const selectedIndex = signal<number | null>(null);
    const order = signal<FinalLogitsOrder>('index');
    const direction = signal<FinalLogitsDirection>('ascending');
    const analysis = signal<{ref_step: number | null; target_step: number | null} | undefined>(
      undefined,
    );
    const source = {
      analysis,
      data,
      statistics,
      loading,
      error,
      valuesLoading,
      valuesError,
      selectedIndex,
      order,
      total: computed(() => data()?.total ?? 0),
      columnSortDirection: (column: FinalLogitsOrder) =>
        order() === column ? direction() : 'none',
      position: signal(32),
      pageIndex: computed(() => Math.floor((data()?.offset ?? 0) / 16)),
      pages: computed(() => Math.max(1, Math.ceil((data()?.total ?? 0) / 16))),
      rows: computed(() =>
        (data()?.rows ?? []).map((row) => ({
          ...row,
          label: String(row.index),
          tokenLabel: row.label === null ? '—' : JSON.stringify(row.label),
        })),
      ),
      notices: computed(() => (data()?.reasons.length ? [reasonText(data()!.reasons)] : [])),
      pageValuesNotices: computed(() =>
        [reasonText(data()?.reasons), pageReason()].filter((reason): reason is string => !!reason),
      ),
      formatValue: view.formatTensorValue,
      changePage: vi.fn(),
      sortBy: vi.fn((column: FinalLogitsOrder) => {
        direction.set(
          order() === column && direction() === 'ascending' ? 'descending' : 'ascending',
        );
        order.set(column);
      }),
      locateElement: vi.fn(),
      selectElement: vi.fn((index: number) => selectedIndex.set(index)),
    };
    fixture.componentRef.setInput('panel', {
      ...view.panel(),
      logit: {...view.panel().logit!, logitView: 'inspection', logitPosition: 32},
    });
    fixture.componentRef.setInput('finalLogits', source);
    fixture.detectChanges();
    return {
      fixture,
      view,
      data,
      loading,
      error,
      source,
      statistics,
      valuesLoading,
      valuesError,
      pageReason,
    };
  }
  it.each(['Comparison Statistics', 'Inspection Statistics', 'Inspection Values'] as const)(
    'provides explicit expand and collapse actions for clipped %s numbers without selecting a row',
    async (tableKind) => {
      const inspection = tableKind === 'Comparison Statistics' ? null : inspect();
      const fixture = inspection?.fixture ?? (await created());
      const tableLabel =
        tableKind === 'Comparison Statistics'
          ? 'Final logits statistics'
          : tableKind === 'Inspection Statistics'
            ? 'Final logits element statistics'
            : 'Final logits tensor values';
      const valueDebug = fixture.debugElement.query(
        By.css(`table[aria-label="${tableLabel}"] tbody td:nth-child(2) explorer-info-value`),
      );
      const value = valueDebug.componentInstance as ExplorerInfoValue;
      const valueElement = valueDebug.nativeElement as HTMLElement;
      const cell = valueElement.closest('td')!;
      const locatorSlot = cell.querySelector('.stat-locator')!;
      const locator = cell.querySelector('button[explorer-locator]');
      const locate = vi.fn();
      fixture.componentInstance.locateLogitElement.subscribe(locate);
      fixture.componentInstance.location.subscribe(locate);
      expect(value.truncated()).toBe(false);
      expect(valueElement.querySelector('.read-value')).toBeNull();
      if (tableKind !== 'Inspection Values') {
        expect(locator).not.toBeNull();
      }
      expect(Array.from(locatorSlot.querySelectorAll('button'))).toEqual(locator ? [locator] : []);

      // Simulate a narrow numeric slot; the component's collapse measurement
      // still sees clipped content, so its explicit action must remain available.
      Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
        configurable: true,
        value: () => ({width: 120}),
      });
      value.truncated.set(true);
      fixture.detectChanges();
      const toggle = valueElement.querySelector<HTMLButtonElement>('.read-value')!;
      expect(toggle.closest('explorer-info-value')).toBe(valueElement);
      expect(toggle.closest('.stat-locator')).toBeNull();
      expect(toggle.getAttribute('aria-label')).toBe('Expand ' + value.label());
      expect(toggle.getAttribute('aria-expanded')).toBe('false');
      expect(toggle.textContent).toContain('unfold_more');
      toggle.focus();
      toggle.click();
      fixture.detectChanges();
      expect(value.expanded()).toBe(true);
      const expandedValue = valueElement.querySelector<HTMLElement>('.expanded-value')!;
      expect(expandedValue.textContent).toBe(value.text());
      expect(getComputedStyle(expandedValue).visibility).toBe('visible');
      expect(toggle.getAttribute('aria-label')).toBe('Collapse ' + value.label());
      expect(toggle.getAttribute('aria-expanded')).toBe('true');
      expect(toggle.textContent).toContain('unfold_less');
      expect(document.activeElement).toBe(toggle);
      expect(Array.from(locatorSlot.querySelectorAll('button'))).toEqual(locator ? [locator] : []);
      toggle.click();
      fixture.detectChanges();
      expect(value.expanded()).toBe(false);
      expect(valueElement.querySelector('.expanded-value')).toBeNull();
      expect(valueElement.querySelector('.read-value')).toBe(toggle);
      expect(toggle.getAttribute('aria-label')).toBe('Expand ' + value.label());
      expect(toggle.getAttribute('aria-expanded')).toBe('false');
      expect(document.activeElement).toBe(toggle);
      expect(cell.querySelector('button[explorer-locator]')).toBe(locator);
      expect(Array.from(locatorSlot.querySelectorAll('button'))).toEqual(locator ? [locator] : []);
      expect(locate).not.toHaveBeenCalled();
      if (inspection) {
        expect(inspection.source.selectElement).not.toHaveBeenCalled();
      }
      fixture.destroy();
    },
  );
  it('fits the visible indices and retains the column width while the next page is pending', () => {
    const {fixture, view, data, source, valuesLoading, valuesError} = inspect();
    expect(view.tensorIdentityWidth()).toBe(48);
    const showIndex = (index: number) => {
      data.update((value) => ({
        ...value!,
        total: 262144,
        offset: Math.floor(index / 16) * 16,
        rows: [{index, label: 'token', ref: 1, target: 2, delta: 1, absolute_delta: 1}],
      }));
    };
    showIndex(262143);
    expect(view.tensorIdentityWidth()).toBe(65);
    valuesLoading.set(true);
    data.update((value) => ({...value!, rows: []}));
    expect(view.tensorIdentityWidth()).toBe(65);
    valuesLoading.set(false);
    valuesError.set('Values page unavailable');
    expect(view.tensorIdentityWidth()).toBe(65);
    valuesError.set('');
    showIndex(15);
    expect(view.tensorIdentityWidth()).toBe(48);
    showIndex(262143);
    expect(view.tensorIdentityWidth()).toBe(65);
    valuesLoading.set(true);
    data.update((value) => ({...value!, rows: []}));
    expect(view.tensorIdentityWidth()).toBe(65);
    source.position.set(33);
    expect(view.tensorIdentityWidth()).toBe(48);
    fixture.destroy();
  });
  it('shows page ordering fallbacks only in Tensor Values while keeping full-vector statistics', () => {
    const {fixture, view, data, pageReason} = inspect();
    const fallback = 'Absolute delta ordering unavailable; showing vocabulary index order';
    pageReason.set(fallback);
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelector('section[aria-label="Tensor Values"]').textContent,
    ).toContain(fallback);
    expect(
      fixture.nativeElement.querySelector('section[aria-label="Statistics"]').textContent,
    ).not.toContain(fallback);
    expect(data()?.reasons).toEqual([]);
    expect(view.logitInspectionStatistics()[0].stats.mean).toBe(0.5);
    fixture.destroy();
  });
  it('shows full-vector element statistics with the same columns as KV and locates extrema', () => {
    const {fixture, view, data, source} = inspect();
    expect(
      Array.from(
        fixture.nativeElement.querySelectorAll(
          ':scope > explorer-info-section',
        ) as NodeListOf<HTMLElement>,
      ).map((section) => section.querySelector('section')?.getAttribute('aria-label')),
    ).toEqual(['Analysis Info', 'Statistics', 'Tensor Metadata', 'Tensor Values']);
    expect(
      view
        .logitInspectionStatistics()
        .map(({label, stats}) => [
          label,
          stats.min,
          stats.max,
          stats.mean,
          stats.min_position,
          stats.max_position,
        ]),
    ).toEqual([
      ['Reference', -2, 3, 0.5, 1, 3],
      ['Target', -4, 7, 1.75, 1, 3],
      ['Δ', -2, 4, 1.25, 1, 3],
      ['|Δ|', 1, 4, 2.25, 0, 3],
    ]);
    const statistics: HTMLTableElement = fixture.nativeElement.querySelector(
      'table[aria-label="Final logits element statistics"]',
    );
    expect(
      Array.from(statistics.querySelectorAll('th')).map((cell) => cell.textContent?.trim()),
    ).toEqual(['Group', 'Min', 'Max', 'Mean']);
    expect(statistics.querySelectorAll('tbody tr')).toHaveLength(4);
    expect(statistics.textContent).not.toContain('Root mean square error');
    expect(statistics.querySelector('button[aria-label*="Mean"]')).toBeNull();
    expect(view.logitInspectionInfoItems().map((item) => item.label)).toEqual([
      'Name',
      'Mode',
      'Stage',
      'Token position',
    ]);
    // Once the inspected row is known, the step that produced its tokens, once for both sides.
    source.analysis.set({ref_step: 0, target_step: 0});
    expect(view.logitInspectionInfoItems().at(-1)).toEqual({
      label: 'Step',
      value: 'Decode step 32 · reads "," · produces "blue"',
    });
    source.analysis.set(undefined);
    expect(view.logitInspectionMetadata('ref').map(({label, value}) => [label, value])).toEqual([
      ['Dimensions', '[Vocabulary]'],
      ['Shape', '[4]'],
      ['Dtype', 'float16'],
    ]);
    expect(view.logitInspectionMetadata('target')[2].value).toBe('float32');
    for (const side of ['ref', 'target'] as const) {
      for (const item of view.logitInspectionMetadata(side).slice(1)) {
        expect(item.segments).toBeUndefined();
      }
    }
    const events: unknown[] = [];
    view.locateLogitElement.subscribe((index) => events.push(index));
    view.locateTensorPosition.subscribe(() => events.push('position'));
    (
      fixture.nativeElement.querySelector(
        'button[aria-label="Locate Final logits |Δ| Max element 3"]',
      ) as HTMLButtonElement
    ).click();
    view.locateDefinition(
      view.logitInspectionInfoItems().find((item) => item.label === 'Token position')!,
    );
    expect(events).toEqual([3, 'position']);
    expect(view.range()).toEqual({start: 32, end: 34});
    data.update((value) => ({
      ...value!,
      offset: 3,
      rows: [{index: 3, label: 'three', ref: 0, target: 4, delta: 4, absolute_delta: 4}],
    }));
    expect(view.logitInspectionStatistics()[0].stats.mean).toBe(0.5);
    fixture.destroy();
  });
  it('uses five sortable value headers with token help, selection, location, and pagination', () => {
    const {fixture, view, data, source} = inspect();
    data.update((value) => ({
      ...value!,
      total: 32,
      rows: [
        {index: 0, label: 'zero', ref: 1, target: 1, delta: 0, absolute_delta: 0},
        {index: 1, label: '\n', ref: 1, target: 2, delta: 1, absolute_delta: 1},
      ],
    }));
    fixture.detectChanges();
    const table: HTMLTableElement = fixture.nativeElement.querySelector(
      'table[aria-label="Final logits tensor values"]',
    );
    expect(
      Array.from(table.querySelectorAll('thead .sort-label')).map((cell) =>
        cell.textContent?.trim(),
      ),
    ).toEqual(['Index', 'Ref', 'Target', 'Δ', '|Δ|']);
    expect(fixture.nativeElement.querySelector('config-picker')).toBeNull();
    expect(table.querySelector('th')?.getAttribute('aria-sort')).toBe('ascending');
    expect(table.querySelectorAll('col')).toHaveLength(5);
    expect(view.logitIndexSegments(source.rows()[1])[0].help).toEqual({
      title: 'Token',
      text: '"\\n"',
      rows: [],
    });
    const row = table.querySelector<HTMLTableRowElement>('[data-vocabulary-index="1"]')!;
    row.click();
    fixture.detectChanges();
    expect(source.selectElement).toHaveBeenCalledWith(1);
    expect(source.locateElement).not.toHaveBeenCalled();
    expect(row.classList.contains('tensor-element-selected')).toBe(true);
    expect(table.querySelectorAll('tbody tr')).toHaveLength(2);
    source.selectElement.mockClear();
    table
      .querySelector<HTMLButtonElement>('button[aria-label="Locate vocabulary index 0"]')!
      .click();
    expect(source.locateElement).toHaveBeenCalledWith(0);
    expect(source.selectElement).not.toHaveBeenCalled();
    table
      .querySelector<HTMLTableRowElement>('[data-vocabulary-index="0"]')!
      .dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
    expect(source.selectElement).toHaveBeenCalledWith(0);
    for (const [i, key] of ['index', 'ref', 'target', 'delta', 'absolute_delta'].entries()) {
      const header = table.querySelectorAll('th')[i];
      header.querySelector('button')!.click();
      fixture.detectChanges();
      expect(source.sortBy).toHaveBeenLastCalledWith(key);
      const firstDirection = i === 0 ? 'descending' : 'ascending';
      expect(header.getAttribute('aria-sort')).toBe(firstDirection);
      header.querySelector('button')!.click();
      fixture.detectChanges();
      expect(header.getAttribute('aria-sort')).toBe(
        firstDirection === 'ascending' ? 'descending' : 'ascending',
      );
    }
    fixture.nativeElement
      .querySelector('button[aria-label="Next Final logits values page"]')
      .click();
    expect(source.changePage).toHaveBeenCalledWith(1);
    expect(view.range()).toEqual({start: 32, end: 34});
    expect(view.logitInspectionStatistics()[0].stats.mean).toBe(0.5);
    data.update((value) => ({
      ...value!,
      rows: [{index: 1, label: '\n', ref: 'NaN', target: 2, delta: null, absolute_delta: null}],
      metrics: {
        rmse: {
          value: null,
          reasons: [{code: 'non_finite_tensor', side: null, text: 'Non-finite logits'}],
        },
        max_abs: {
          value: null,
          reasons: [{code: 'non_finite_tensor', side: null, text: 'Non-finite logits'}],
        },
        max_abs_index: null,
      },
    }));
    fixture.detectChanges();
    expect(table.textContent).toContain('NaN');
    fixture.destroy();
  });
  it('keeps full-vector statistics available while the values page loads or fails', () => {
    const {fixture, view, valuesLoading, valuesError} = inspect();
    valuesLoading.set(true);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Loading Final logits values');
    expect(
      fixture.nativeElement.querySelector('table[aria-label="Final logits tensor values"]'),
    ).toBeNull();
    expect(view.logitInspectionStatistics()[0].stats.mean).toBe(0.5);
    expect(view.logitInspectionMetadata('ref')[2].value).toBe('float16');
    valuesLoading.set(false);
    valuesError.set('Values page unavailable');
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Values page unavailable');
    expect(view.logitInspectionStatistics()[0].stats.mean).toBe(0.5);
    fixture.destroy();
  });
  it('shows the server statistics across page changes, and new ones for a new vector', () => {
    const {fixture, view, data, statistics} = inspect();
    const shown = view.logitInspectionStatistics();
    data.update((value) => ({
      ...value!,
      offset: 3,
      rows: [{index: 3, label: 'three', ref: 3, target: 7, delta: 4, absolute_delta: 4}],
    }));
    expect(view.logitInspectionStatistics()).toBe(shown);
    statistics.update((value) => ({...value!, ref: {...value!.ref, max: 5}}));
    expect(view.logitInspectionStatistics()[0].stats.max).toBe(5);
    fixture.destroy();
  });
  it('shows no difference statistics the server leaves empty, and no locator for them', () => {
    const {fixture, view, data, statistics} = inspect();
    const empty = {
      min: null,
      max: null,
      mean: null,
      min_position: null,
      max_position: null,
      valid_count: 0,
    };
    statistics.update((value) => ({...value!, delta: empty, abs_delta: empty}));
    data.update((value) => ({
      ...value!,
      compatible: false,
      reasons: [{code: 'vocabulary_mismatch', side: null, text: 'Matching vocabulary unavailable'}],
    }));
    expect(view.logitInspectionStatistics().map(({stats}) => stats.valid_count)).toEqual([
      4, 4, 0, 0,
    ]);
    fixture.detectChanges();
    const table: HTMLTableElement = fixture.nativeElement.querySelector(
      'table[aria-label="Final logits element statistics"]',
    );
    expect(table.querySelector('button[aria-label*="|Δ|"]')).toBeNull();
    expect(table.querySelector('button[aria-label*="Reference"]')).not.toBeNull();
    statistics.set(null);
    expect(
      view
        .logitInspectionStatistics()
        .every(({stats}) => stats.valid_count === 0 && stats.mean === null),
    ).toBe(true);
    fixture.destroy();
  });
  it('does not expose stale inspection metrics or metadata while loading or after errors', () => {
    const {fixture, view, loading, error, data, statistics} = inspect();
    loading.set(true);
    fixture.detectChanges();
    expect(view.logitInspectionData()).toBeNull();
    expect(fixture.nativeElement.textContent).toContain('Loading statistics');
    expect(
      fixture.nativeElement.querySelector(
        'button[aria-label="Locate Final logits |Δ| Max element 3"]',
      ),
    ).toBeNull();
    loading.set(false);
    error.set('Tensor request failed');
    fixture.detectChanges();
    expect(view.logitInspectionMetadata('ref')[2].value).toBe('—');
    expect(fixture.nativeElement.textContent).toContain('Tensor request failed');
    error.set('');
    data.update((value) => ({
      ...value!,
      compatible: false,
      reasons: [{code: 'vocabulary_mismatch', side: null, text: 'Matching vocabulary unavailable'}],
      metrics: {
        rmse: {
          value: null,
          reasons: [
            {code: 'vocabulary_mismatch', side: null, text: 'Matching vocabulary unavailable'},
          ],
        },
        max_abs: {
          value: null,
          reasons: [
            {code: 'vocabulary_mismatch', side: null, text: 'Matching vocabulary unavailable'},
          ],
        },
        max_abs_index: null,
      },
    }));
    // The server compares no vectors of two vocabularies: no difference statistics.
    const empty = {
      min: null,
      max: null,
      mean: null,
      min_position: null,
      max_position: null,
      valid_count: 0,
    };
    statistics.update((value) => ({...value!, delta: empty, abs_delta: empty}));
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Matching vocabulary unavailable');
    expect(view.logitInspectionMetadata('ref')[2].value).toBe('float16');
    expect(
      fixture.nativeElement.querySelector(
        'button[aria-label="Locate Final logits |Δ| Max element 3"]',
      ),
    ).toBeNull();
    fixture.destroy();
  });
});
