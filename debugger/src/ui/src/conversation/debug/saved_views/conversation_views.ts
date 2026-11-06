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

import {Injectable, computed, inject, signal, untracked} from '@angular/core';
import {MetricCatalogStore} from '../../../server/metric_catalog';
import {ConversationState} from '../../conversation_state';
import {
  captureViewState,
  pairIdentity,
  prepareViewRestore,
  type ConversationObservation,
  type TokenDiffViewState,
} from '../../conversation_view_state';
import {PreferencesService} from '../../../state/preferences_service';
import {restoreLogitPanel, type StageAnalysisPanel} from '../../stage_analysis';
import {comparisonPairs, type AlignmentMode, type TokenPair} from '../../token_alignment';
import {AnalysisPanels} from '../stage_analysis_rail/analysis_panels';
import {TokenDiff} from '../token_diff/token_diff';
import {readStoredPanels, storePanels} from './stored_panels';

/**
 * The Token Diff's saved view: the one kept when the reader leaves the conversation and returned
 * to on coming back, restored through a validated path. It holds what the page's address does not:
 * display settings, selection and scroll position. The analysis panels added to the stages are
 * this reader's own: this browser keeps them per Session (stored_panels), as it keeps drafts and
 * preferences, so a reload or a return restores them.
 */
@Injectable()
export class ConversationViews {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);
  private readonly panels = inject(AnalysisPanels);
  private readonly metricCatalog = inject(MetricCatalogStore);
  private readonly preferences = inject(PreferencesService);

  /** Why a saved view was not restored, if one was not. */
  readonly viewNotice = signal('');
  /** A saved view whose formula needs the metric catalog, held until the catalog has loaded. */
  private readonly pendingView = signal<unknown>(null);
  /** Whether a saved view is held until the metric catalog has loaded. */
  readonly restoring = computed(() => !!this.pendingView());

  private captureView(): TokenDiffViewState {
    return captureViewState({
      observation: this.currentObservation(),
      formula: this.tokens.formula(),
      alignment: this.conversation.alignment(),
      selection: this.tokens.selection(),
      identity: pairIdentity(this.tokens.selectedPair()),
      details: this.tokens.details(),
      boundaries: this.tokens.boundaries(),
      // Invisible characters are always drawn; the field keeps saved views in one shape.
      whitespace: true,
      color: this.tokens.color(),
      collapsed: [...this.tokens.collapsed()],
      thinking: [...this.conversation.thinkingExpanded()],
      infoExpanded: this.tokens.infoExpanded(),
      groups: {...this.tokens.groups()},
      panelWidth: this.tokens.panelWidth(),
      ...this.conversation.reading.capture(),
      analysisPanels: this.panels.capturePanels(),
      inspectedPanel:
        this.tokens.metricInspector()?.panelId() ?? this.tokens.tokenMetricInspector()?.id ?? null,
    });
  }
  private currentObservation(): ConversationObservation | null {
    const batch = this.conversation.state
      .session()
      ?.batches.find((item) => item.batch === this.conversation.state.batchId());
    return batch ? {turn: batch.turn, phase: batch.phase} : null;
  }
  saveView(): void {
    // A saved view still waiting for the metric catalog stays as it was saved.
    if (this.pendingView()) {
      return;
    }
    this.conversation.memory.conversationViews.set(this.conversation.contextKey, {
      ...this.captureView(),
      analysisPanels: [],
      inspectedPanel: null,
      analysisFindScope: '',
    });
  }
  private rowsForRestore(turn: number, alignment: AlignmentMode): readonly TokenPair[] {
    return comparisonPairs(
      this.conversation.comparison(turn)?.alignments[alignment].rows,
      this.conversation.captured(turn, 'ref')?.tokens,
      this.conversation.captured(turn, 'target')?.tokens,
    );
  }
  /**
   * Applies the held view once the metric catalog has loaded; drops it, and says so, once asking
   * for the catalog has failed. Run in an effect.
   */
  followPendingView(): void {
    const pending = this.pendingView();
    const lookup = this.metricCatalog.lookup();
    if (!pending || (!lookup.catalog && !lookup.error)) {
      return;
    }
    untracked(() => {
      this.pendingView.set(null);
      if (lookup.catalog) {
        this.applyView(pending);
      } else {
        this.viewNotice.set(
          'Metric descriptions could not be read; conversation defaults were retained.',
        );
      }
    });
  }
  /**
   * Restores a saved view through the validated path. A view that needs the metric catalog
   * before it has loaded is held, and followPendingView() applies it.
   */
  private applyView(value: unknown): boolean {
    this.pendingView.set(null);
    const prepared = prepareViewRestore(
      value,
      {
        hasTurn: (turn) => this.conversation.turns().some((item) => item.n === turn),
        rows: (turn, alignment) => this.rowsForRestore(turn, alignment),
      },
      this.currentObservation(),
      this.metricCatalog.lookup(),
    );
    if (prepared.status === 'needs-catalog') {
      this.pendingView.set(value);
      return false;
    }
    if (prepared.status !== 'ready') {
      const message = {
        invalid: 'Saved settings are invalid or from an unsupported version.',
        'missing-turn': 'This saved turn is no longer available.',
        'missing-token': 'This saved token is no longer available.',
      }[prepared.status];
      this.viewNotice.set(message + ' Conversation defaults were retained.');
      return false;
    }
    const view = prepared.state;
    // The panels are the ones this browser restored (restoreReturnedView).
    this.tokens.formula.set(view.formula);
    this.conversation.alignment.set(view.alignment);
    this.tokens.selection.set(view.selection);
    this.tokens.details.set(view.details);
    this.tokens.boundaries.set(view.boundaries);
    this.tokens.color.set(view.color);
    this.tokens.collapsed.set(new Set(view.collapsed));
    this.conversation.thinkingExpanded.set(new Set(view.thinking));
    this.tokens.infoExpanded.set(view.infoExpanded);
    this.tokens.groups.set(view.groups);
    this.tokens.panelWidth.set(view.panelWidth);
    this.panels.pendingInspector.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.panels.tokenMetricRange.set(null);
    this.tokens.analysisOverview.set(null);
    this.tokens.metricInspector.set(null);
    this.conversation.reading.apply(view, false, prepared.observation);
    return true;
  }
  private setPanels(panels: StageAnalysisPanel[]): void {
    this.panels.analysisPanels.set(panels.map(restoreLogitPanel));
    this.panels.analysisSerial = Math.max(0, ...panels.map((p) => Number(p.id.slice(9))));
  }
  /**
   * Returns to the view this conversation was left with, if any, and to the analysis panels this
   * browser kept of its chat, after a reload too.
   */
  restoreReturnedView(): void {
    const stored = readStoredPanels(this.preferences, this.conversation.contextKey);
    if (stored) {
      this.setPanels(stored);
    }
    const saved = this.conversation.memory.conversationViews.get(this.conversation.contextKey);
    if (saved) {
      this.applyView(saved);
    }
  }
  /** Keeps the conversation's analysis panels in this browser as they change. Run in an effect. */
  keepPanels(): void {
    const panels = this.panels.capturePanels();
    const chat = this.conversation.contextKey;
    untracked(() => storePanels(this.preferences, chat, panels));
  }
}
