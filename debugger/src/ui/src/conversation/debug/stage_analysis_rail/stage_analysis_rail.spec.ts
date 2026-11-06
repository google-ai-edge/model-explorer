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
import {afterEach, beforeEach, vi} from 'vitest';

import {KvPanel} from '../../../analyses/registry';
import {CaptureStore} from '../../../server/capture_store';
import {TokenAnalysisApi} from '../../../server/token_analysis_api';
import {ConversationState} from '../../conversation_state';
import type {StageAnalysisPanel} from '../../stage_analysis';
import {TokenDiff} from '../token_diff/token_diff';
import {AnalysisInfo} from './analysis_info';
import {AnalysisPanels} from './analysis_panels';
import {ExpandedAnalysis} from './expanded_analysis';
import {StageAnalyses} from './stage_analyses';
import {StageAnalysisRail} from './stage_analysis_rail';

const kvPanel: StageAnalysisPanel = {
  id: 'analysis-1',
  turn: 2,
  stage: 'decode',
  kind: 'kv',
  open: true,
};

describe('a panel loading its first data', () => {
  function setup() {
    const captureId = signal<string | null>('capture-a');
    TestBed.configureTestingModule({
      providers: [
        AnalysisPanels,
        {provide: ConversationState, useValue: {state: {captureId}}},
        {provide: TokenDiff, useValue: {}},
      ],
    });
    const panels = TestBed.inject(AnalysisPanels);
    panels.analysisPanels.set([kvPanel]);
    const waiting = signal(true);
    const live: KvPanel = Object.assign(Object.create(KvPanel.prototype), {
      panelId: () => kvPanel.id,
      waiting,
    });
    const kvPanels = signal<readonly KvPanel[]>([]);
    panels.addRail({kvPanels, graphPanels: signal([]), tokenPanels: signal([])});
    const show = () => kvPanels.set([live]);
    return {panels, waiting, show, captureId};
  }

  it('loads until its live panel has its first data, and stays loaded after', () => {
    const {panels, waiting, show} = setup();
    TestBed.tick();
    // Not on the page yet: it waits.
    expect(panels.panelLoading(kvPanel)).toBe(true);
    show();
    TestBed.tick();
    expect(panels.panelLoading(kvPanel)).toBe(true);
    // Answered, or failed: either ends the wait.
    waiting.set(false);
    TestBed.tick();
    expect(panels.panelLoading(kvPanel)).toBe(false);
    // A later request of its view loads inside the open panel.
    waiting.set(true);
    TestBed.tick();
    expect(panels.panelLoading(kvPanel)).toBe(false);
  });

  it('loads again for another capture, as a restored panel does', () => {
    const {panels, waiting, show, captureId} = setup();
    show();
    waiting.set(false);
    TestBed.tick();
    expect(panels.panelLoading(kvPanel)).toBe(false);
    waiting.set(true);
    captureId.set('capture-b');
    TestBed.tick();
    expect(panels.panelLoading(kvPanel)).toBe(true);
  });
});

describe('the rail entry of a loading panel', () => {
  // A Final Logits panel: its chart asks nothing of a turn without rows.
  const logitPanel: StageAnalysisPanel = {
    id: 'analysis-2',
    turn: 2,
    stage: 'decode',
    kind: 'tokens',
    open: true,
    logit: {
      open: true,
      metric: 'logit_rmse',
      logitTransform: 'raw',
      logitOffset: 128,
      logitOrder: 'absolute_delta',
    },
  };
  beforeEach(() => {
    // The browser APIs its chart reads, as the chart's own tests stub them.
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({
      matches: true,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
    }));
  });
  afterEach(() => vi.unstubAllGlobals());
  function setup(panel = logitPanel) {
    const loading = signal(true);
    const analyses = {toggleAnalysis: vi.fn(), removeAnalysis: vi.fn()};
    const panels = {
      panelsFor: () => [panel],
      tokenPanelSelectedIndex: () => null,
      isExpandedStage: () => false,
      expandedAnalysis: () => null,
      panelLoading: () => loading(),
      analysisChartSelections: () => new Map(),
      addRail: () => () => undefined,
    };
    const tokens = {
      details: () => null,
      metricInspector: () => null,
      tokenMetricInspector: () => null,
      analysisOverview: () => null,
      analysisNotes: () => [],
      stageGlobalStart: () => 0,
      rows: () => [],
      analysisColumns: () => null,
      tokenAnalysis: () => new Map(),
      analysisLoading: () => false,
    };
    TestBed.configureTestingModule({
      providers: [
        {provide: ConversationState, useValue: {alignment: () => 'content'}},
        {provide: TokenDiff, useValue: tokens},
        {provide: AnalysisPanels, useValue: panels},
        {provide: StageAnalyses, useValue: analyses},
        {provide: ExpandedAnalysis, useValue: {expandedTop: () => 0}},
        {provide: AnalysisInfo, useValue: {analysisInstanceLabel: () => 'Final Logits'}},
        {provide: CaptureStore, useValue: {captureId: signal(null), session: signal(null)}},
        {provide: TokenAnalysisApi, useValue: {}},
      ],
    });
    const fixture = TestBed.createComponent(StageAnalysisRail);
    fixture.componentRef.setInput('turn', 2);
    fixture.componentRef.setInput('stage', 'decode');
    fixture.detectChanges();
    const host = fixture.nativeElement as HTMLElement;
    const entry = () => host.querySelector<HTMLButtonElement>('.analysis-entry')!;
    const content = () => host.querySelector<HTMLElement>('.analysis-content')!;
    return {fixture, host, loading, analyses, entry, content};
  }

  it('cannot open, shows nothing, and holds a spinner where its remove button goes', () => {
    const {host, analyses, entry, content} = setup();
    expect(entry().getAttribute('aria-disabled')).toBe('true');
    expect(entry().getAttribute('aria-expanded')).toBe('false');
    expect(entry().getAttribute('aria-label')).toBe('Final Logits, Loading…');
    expect(entry().querySelector('mat-icon')!.textContent).toBe('chevron_right');
    expect(host.querySelector('.stage-analysis')!.classList).not.toContain('open');
    expect(content().classList).toContain('unseen');
    const spinner = host.querySelector('.analysis-loading')!;
    expect(spinner.getAttribute('aria-label')).toBe('Loading…');
    expect(spinner.querySelector('mat-spinner')).not.toBeNull();
    expect(host.querySelector('button.analysis-delete')).toBeNull();
    entry().click();
    expect(analyses.toggleAnalysis).not.toHaveBeenCalled();
  });

  it('turns the spinner back into the remove button and opens once loaded', () => {
    const {fixture, host, loading, analyses, entry, content} = setup();
    loading.set(false);
    fixture.detectChanges();
    expect(entry().hasAttribute('aria-disabled')).toBe(false);
    expect(entry().hasAttribute('aria-label')).toBe(false);
    expect(entry().getAttribute('aria-expanded')).toBe('true');
    expect(host.querySelector('.stage-analysis')!.classList).toContain('open');
    expect(content().classList).not.toContain('unseen');
    expect(content().hidden).toBe(false);
    expect(host.querySelector('.analysis-loading')).toBeNull();
    const remove = host.querySelector<HTMLButtonElement>('button.analysis-delete')!;
    expect(remove.getAttribute('aria-label')).toBe('Remove Final Logits');
    remove.click();
    expect(analyses.removeAnalysis).toHaveBeenCalledWith(logitPanel);
    entry().click();
    expect(analyses.toggleAnalysis).toHaveBeenCalledWith(logitPanel);
  });

  it('lays a closed panel out unseen, at its width, so that it opens drawn', () => {
    const {fixture, host, loading, content} = setup({...logitPanel, open: false});
    loading.set(false);
    fixture.detectChanges();
    expect(host.querySelector('.stage-analysis')!.classList).not.toContain('open');
    expect(content().hidden).toBe(false);
    expect(content().classList).toContain('unseen');
  });
});
