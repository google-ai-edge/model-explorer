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
import {ANALYSES, analysisOf} from '../../analyses/registry';
import {NO_PLACE, samePlace, type DebugPlace} from '../../state/page_address/page_address';
import type {StageAnalysisId} from '../../server/contracts/capture';
import {ConversationState} from '../conversation_state';
import type {StageAnalysisPanel} from '../stage_analysis';
import {ConversationViews} from './saved_views/conversation_views';
import {AnalysisInfo} from './stage_analysis_rail/analysis_info';
import {AnalysisPanels} from './stage_analysis_rail/analysis_panels';
import {StageAnalyses} from './stage_analysis_rail/stage_analyses';
import {TokenDiff} from './token_diff/token_diff';
import {TokenDiffDom} from './token_diff/token_diff_dom';

/**
 * Where a Debug reading is: the analysis panel the details describe, else the turn of what
 * they describe, else nowhere (the conversation's overview).
 */
export function readingPlace(
  panel: Pick<StageAnalysisPanel, 'turn' | 'stage' | 'kind'> | null,
  selection: {turn: number} | null,
): DebugPlace {
  if (panel) {
    return {turn: panel.turn, stage: panel.stage, analysis: analysisOf(panel.kind)};
  }
  return selection ? {turn: selection.turn, stage: null, analysis: null} : NO_PLACE;
}

/**
 * The Debug reading and the page's address, kept in step: an address the page arrives at (opened,
 * reloaded, back or forward, or a link) moves the reading there, and where the reader goes is
 * written in the address. A turn opens its Prefill as the stage index does; an analysis opens the
 * stage's panel of it, added as "Add analysis" adds one when the stage has none. Where the address
 * cannot go, the page says why and the address follows the reading.
 */
@Injectable()
export class DebugAddress {
  private readonly conversation = inject(ConversationState);
  private readonly workspace = this.conversation.workspace;
  private readonly tokens = inject(TokenDiff);
  private readonly panels = inject(AnalysisPanels);
  private readonly analyses = inject(StageAnalyses);
  private readonly analysisInfo = inject(AnalysisInfo);
  private readonly views = inject(ConversationViews);
  private readonly dom = inject(TokenDiffDom);

  /** Why the page is not where its address was; '' when it is. */
  readonly notice = signal('');
  readonly place = computed(() =>
    readingPlace(this.panels.overviewPanel(), this.tokens.selection()),
  );
  /** The address place last followed; null until the reading is in Debug with its capture. */
  private arrived: DebugPlace | null = null;
  /**
   * Places this reading asked the address for, oldest first, which it need not follow when they
   * arrive. The router finishes one navigation at a time and a newer one cancels it, so an address
   * that arrives ends every request made before it: the ones before an own request arriving, and
   * all of them when an address arrives that the reading did not ask for (Back, Forward, a link).
   */
  private readonly requested: DebugPlace[] = [];
  /** Whether the next address written replaces the one the reading could not go to. */
  private replace = false;

  /** Run in an effect. */
  follow(): void {
    const address = this.workspace.address();
    const place = this.place();
    const loaded = !!this.conversation.state.session();
    const settling = this.views.restoring() || !!this.panels.pendingInspector();
    untracked(() => {
      this.workspace.debugPlace.set(place);
      if (address.mode !== 'Debug' || !loaded) {
        this.arrived = null;
        this.requested.length = 0;
        return;
      }
      const wanted: DebugPlace = {
        turn: address.turn,
        stage: address.stage,
        analysis: address.analysis,
      };
      if (!this.arrived || !samePlace(this.arrived, wanted)) {
        this.arrived = wanted;
        const asked = this.requested.findIndex((item) => samePlace(item, wanted));
        this.requested.splice(0, asked >= 0 ? asked + 1 : this.requested.length);
        if (asked >= 0) {
          // Its own address arrived. If the reading has moved since and asked for no other
          // place, the address takes where it is now, in place of where it was.
          if (!this.requested.length && !settling && !samePlace(wanted, place)) {
            this.replace = true;
            this.write(place);
          }
        } else if (!samePlace(wanted, place)) {
          this.go(wanted);
          // Where it could not go, the address says at once where the reading stayed.
          const stayed = this.place();
          if (this.replace && !samePlace(wanted, stayed)) {
            this.write(stayed);
          }
        }
        return;
      }
      if (settling) {
        return;
      }
      if (samePlace(wanted, place)) {
        this.replace = false;
        return;
      }
      // A place the reader went to; one the address could not go to keeps saying why.
      if (!this.replace) {
        this.notice.set('');
      }
      this.write(place);
    });
  }
  private write(place: DebugPlace): void {
    this.requested.push(place);
    this.workspace.showDebugPlace(place, this.replace);
    this.replace = false;
  }
  /** Moves the reading to an address's place, or says why it cannot. */
  private go(place: DebugPlace): void {
    this.notice.set('');
    const {turn, stage, analysis} = place;
    if (turn === null) {
      this.leaveAnalysis();
      this.tokens.selection.set(null);
      return;
    }
    if (!this.conversation.turns().some((item) => item.n === turn)) {
      return this.refuse(`This Session has no turn ${turn}.`);
    }
    if (stage === null || analysis === null) {
      this.leaveAnalysis();
      if (this.tokens.selection()?.turn !== turn) {
        this.dom.openStage(turn, 'prefill');
      }
      return;
    }
    const kind = (ANALYSES as Partial<Record<string, (typeof ANALYSES)[StageAnalysisId]>>)[analysis]
      ?.kind;
    if (!kind) {
      return this.refuse(`There is no analysis ${analysis}.`);
    }
    const shown = this.panels.overviewPanel();
    const panels = this.panels.panelsFor(turn, stage).filter((p) => p.kind === kind);
    const panel = panels.find((p) => p.id === shown?.id) ?? panels.find((p) => p.open) ?? panels[0];
    if (panel) {
      if (shown?.id !== panel.id) {
        this.analyses.showAnalysis(panel);
      }
      return;
    }
    const where = `Turn ${turn} · ${stage === 'prefill' ? 'Prefill' : 'Decode'}`;
    const why = this.panels.analysisUnavailable(turn, stage, kind);
    if (why) {
      const unavailablePanel: StageAnalysisPanel = {id: '', turn, stage, kind, open: false};
      return this.refuse(
        `${this.analysisInfo.analysisLabel(unavailablePanel)} cannot be added to ${where}: ${why}`,
      );
    }
    this.analyses.addAnalysis(turn, stage, kind);
  }
  /** The details stop describing an analysis view. */
  private leaveAnalysis(): void {
    this.tokens.analysisOverview.set(null);
    this.tokens.metricInspector.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.panels.tokenMetricRange.set(null);
  }
  private refuse(text: string): void {
    this.notice.set(text);
    this.replace = true;
  }
}
