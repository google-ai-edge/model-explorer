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

import {effect, signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {debugParts} from '../testing/debug_parts';
import {
  METRIC_CATALOG,
  METRIC_CATALOG_RETRIES,
  MetricCatalogStore,
  MetricLookup,
} from '../../../server/metric_catalog';
import {MetricsApi} from '../../../server/metrics_api';
import type {MetricCatalog} from '../../../server/contracts/metrics';
import {METRIC_CATALOG_FIXTURE} from '../../../server/testing/metric_catalog_fixture';
import {prepareViewRestore} from '../../conversation_view_state';
import {CaptureStore} from '../../../server/capture_store';
import {SessionsApi} from '../../../server/sessions_api';
import type {CapturedToken, ComparisonRow, Session} from '../../../server/contracts/capture';
import {comparisonPairs} from '../../token_alignment';
import type {StageAnalysisPanel} from '../../stage_analysis';
import type {PanelStorage} from './stored_panels';

/** A saved view whose formula has a `%`, which only the catalog says how to read. */
const PERCENT_FORMULA = 'relative_l2 > 0.4%';
const view = {
  version: 3,
  mode: 'Debug',
  observation: null,
  formula: PERCENT_FORMULA,
  alignment: 'content',
  selection: null,
  details: true,
  boundaries: false,
  whitespace: false,
  color: 'none',
  collapsed: [],
  thinking: [],
  infoExpanded: false,
  groups: {},
  panelWidth: 320,
  scrollTop: 0,
  anchor: null,
};
/** The views' private restore path, which a return takes. */
function restorer(views: unknown) {
  return views as {applyView(value: unknown): boolean};
}

/** A catalog store whose one request is answered by `answer`; null keeps the preset. */
function provide(answer: (() => Promise<MetricCatalog>) | null) {
  TestBed.configureTestingModule({
    providers: answer
      ? [
          {provide: METRIC_CATALOG, useValue: null},
          {provide: MetricsApi, useValue: {catalog: answer}},
        ]
      : [],
  });
}

/** The views' restore path, on the store TestBed gives, with followPendingView in an effect. */
function setup() {
  const parts = debugParts({
    conversation: {
      turns: () => [{n: 1}],
      state: {session: () => null, selectTurn: vi.fn()},
      reading: {apply: vi.fn()},
      alignment: signal('content'),
      thinkingExpanded: signal(new Set<string>()),
    },
    tokens: {
      formula: signal('NOT token_match'),
      selection: signal(null),
      details: signal(false),
      boundaries: signal(true),
      color: signal('none'),
      collapsed: signal(new Set<string>()),
      infoExpanded: signal(false),
      groups: signal({}),
      panelWidth: signal(400),
      tokenMetricInspector: signal(null),
      analysisOverview: signal(null),
      metricInspector: signal(null),
    },
    panels: {
      analysisPanels: signal([]),
      pendingInspector: signal(null),
      tokenMetricRange: signal(null),
    },
    views: {
      metricCatalog: TestBed.inject(MetricCatalogStore),
      pendingView: signal(null),
      viewNotice: signal(''),
    },
  });
  TestBed.runInInjectionContext(() => effect(() => parts.views.followPendingView()));
  TestBed.tick();
  return parts;
}

describe('restoring a saved view whose formula needs the metric catalog', () => {
  afterEach(() => vi.useRealTimers());

  it('is held until the catalog arrives, then restored', async () => {
    let arrive!: (catalog: MetricCatalog) => void;
    provide(() => new Promise((resolve) => (arrive = resolve)));
    const {views, tokens} = setup();
    expect(restorer(views).applyView(view)).toBe(false);
    TestBed.tick();
    expect(tokens.formula()).toBe('NOT token_match');
    expect(views.viewNotice()).toBe('');
    // Nothing else was applied either.
    expect(tokens.panelWidth()).toBe(400);
    arrive(METRIC_CATALOG_FIXTURE);
    await Promise.resolve();
    TestBed.tick();
    expect(tokens.formula()).toBe(PERCENT_FORMULA);
    expect(tokens.panelWidth()).toBe(320);
    expect(views.viewNotice()).toBe('');
  });

  it('returns to a view saved on leaving once the catalog arrives, and keeps it until then', async () => {
    let arrive!: (catalog: MetricCatalog) => void;
    provide(() => new Promise((resolve) => (arrive = resolve)));
    const {conversation, views, tokens} = setup();
    const saved = new Map([['capture', view]]);
    Object.assign(conversation, {contextKey: 'capture', memory: {conversationViews: saved}});
    expect(restorer(views).applyView(view)).toBe(false);
    // Leaving before the catalog arrives keeps the view as it was saved.
    views.saveView();
    expect(saved.get('capture')).toBe(view);
    arrive(METRIC_CATALOG_FIXTURE);
    await Promise.resolve();
    TestBed.tick();
    expect(tokens.formula()).toBe(PERCENT_FORMULA);
    expect(views.viewNotice()).toBe('');
  });

  it('is restored at once when the catalog has arrived', () => {
    provide(null);
    const {views, tokens} = setup();
    expect(restorer(views).applyView(view)).toBe(true);
    expect(tokens.formula()).toBe(PERCENT_FORMULA);
    expect(views.viewNotice()).toBe('');
  });

  it('keeps the defaults and says why when the catalog could not be read', async () => {
    vi.useFakeTimers();
    provide(() => Promise.reject(new Error('offline')));
    const {views, tokens} = setup();
    restorer(views).applyView(view);
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_RETRIES.reduce((a, b) => a + b, 0) + 1);
    TestBed.tick();
    expect(tokens.formula()).toBe('NOT token_match');
    expect(views.viewNotice()).toBe(
      'Metric descriptions could not be read; conversation defaults were retained.',
    );
  });

  it('tells a formula that needs the catalog from an invalid one', () => {
    const context = {hasTurn: () => true, rows: () => []};
    const missing = new MetricLookup(null);
    expect(prepareViewRestore(view, context, undefined, missing).status).toBe('needs-catalog');
    expect(prepareViewRestore({...view, formula: 'kl >'}, context, undefined, missing).status).toBe(
      'invalid',
    );
    expect(prepareViewRestore({...view, panelWidth: -1}, context, undefined, missing).status).toBe(
      'invalid',
    );
    const loaded = new MetricLookup(METRIC_CATALOG_FIXTURE);
    expect(prepareViewRestore(view, context, undefined, loaded).status).toBe('ready');
    expect(
      prepareViewRestore({...view, formula: 'kl > 1%'}, context, undefined, loaded).status,
    ).toBe('invalid');
  });
});

describe('returning to the conversation', () => {
  /** Three turns; the Session's batches are of turns 1 and 3, as the server sends them when the
   *  Session observes tensors. */
  const SESSION = {
    model: 'm',
    runs: [],
    turns: [{n: 1}, {n: 2}, {n: 3}],
    batches: [
      {batch: 0, turn: 1, phase: 'prefill', index: 0, step: 13},
      {batch: 1, turn: 3, phase: 'decode', index: 0, step: 40},
    ],
  } as unknown as Session;
  const TOKEN = {kind: 'text', text: 'a', step: 0, position: 40, id: 5} as CapturedToken;
  const ROWS: ComparisonRow[] = [{ref: 0, target: 0, same: true}];
  /** A browser's storage that outlives the page: what a reload keeps. */
  function browserStorage(): PanelStorage {
    const kept = new Map<string, string>();
    return {read: (key) => kept.get(key) ?? null, save: (key, value) => kept.set(key, value)};
  }

  function setup(preferences: PanelStorage = browserStorage()) {
    TestBed.configureTestingModule({
      providers: [{provide: SessionsApi, useValue: {session: () => Promise.resolve(SESSION)}}],
    });
    const store = TestBed.inject(CaptureStore);
    const reading = {
      capture: () => ({mode: 'Debug', scrollTop: 240, anchor: null}),
      apply: vi.fn(),
    };
    const parts = debugParts({
      conversation: {
        state: store,
        turns: () => SESSION.turns,
        comparison: () => ({alignments: {content: {rows: ROWS}}}),
        captured: () => ({tokens: [TOKEN]}),
        reading,
        alignment: signal('content'),
        thinkingExpanded: signal(new Set<string>()),
        contextKey: 's',
        memory: {conversationViews: new Map()},
      },
      tokens: {
        formula: signal('NOT token_match'),
        selection: signal(null),
        selectedPair: () => comparisonPairs(ROWS, [TOKEN], [TOKEN])[0],
        details: signal(false),
        boundaries: signal(true),
        color: signal('none'),
        collapsed: signal(new Set<string>()),
        infoExpanded: signal(false),
        groups: signal({}),
        panelWidth: signal(400),
        tokenMetricInspector: signal(null),
        analysisOverview: signal(null),
        metricInspector: signal(null),
      },
      panels: {
        analysisPanels: signal([]),
        kvPanels: () => [],
        graphPanels: () => [],
        pendingInspector: signal(null),
        tokenMetricRange: signal(null),
      },
      views: {
        metricCatalog: TestBed.inject(MetricCatalogStore),
        pendingView: signal(null),
        viewNotice: signal(''),
        preferences,
      },
    });
    return {...parts, store, reading};
  }

  it('restores the analysis panels added to its stages, after a reload too', () => {
    const storage = browserStorage();
    const added: StageAnalysisPanel[] = [
      {id: 'analysis-1', turn: 1, stage: 'prefill', kind: 'kv', open: true},
      {id: 'analysis-3', turn: 3, stage: 'decode', kind: 'kv', open: false},
    ];
    const before = setup(storage);
    before.panels.analysisPanels.set(added);
    before.views.keepPanels();
    // A reload: nothing is kept in memory, only in the browser's storage.
    TestBed.resetTestingModule();
    const after = setup(storage);
    after.views.restoreReturnedView();
    expect(after.panels.analysisPanels()).toEqual(added);
    // The next one added follows the last restored.
    expect(after.panels.analysisSerial).toBe(3);
  });

  it('keeps a token selected on turn 3, and the reading, after going Home and back', async () => {
    const {store, tokens, views, reading} = setup();
    store.attach('s');
    await store.load();
    // Debug: a token of turn 3 is selected, which observes turn 3.
    tokens.selection.set({kind: 'token', turn: 3, index: 0});
    store.selectTurn(3);
    expect(store.batch()?.turn).toBe(3);
    views.saveView();
    // Home, then back: the capture is attached anew and loaded again.
    store.attach(null);
    store.attach('s');
    await store.load();
    tokens.selection.set(null);
    views.restoreReturnedView();
    expect(tokens.selection()).toEqual({kind: 'token', turn: 3, index: 0});
    expect(reading.apply).toHaveBeenCalledOnce();
    const [snapshot, replace, observation] = reading.apply.mock.calls[0];
    expect(snapshot).toEqual(expect.objectContaining({mode: 'Debug', scrollTop: 240}));
    expect(replace).toBe(false);
    expect(observation).toBeNull();
    expect(views.viewNotice()).toBe('');
    // No batch is observed again until the reader chooses one.
    expect(store.batchId()).toBeNull();
  });
});
