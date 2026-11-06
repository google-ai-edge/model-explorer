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

import {Injectable, afterNextRender, inject, signal} from '@angular/core';

import {KvPanel} from '../../../analyses/registry';
import type {CapturedToken} from '../../../server/contracts/capture';
import type {RangeSelection} from '../../../widgets/charts/range_selection';
import {animateLocationFeedback} from '../../../widgets/locate_feedback';
import {ConversationState} from '../../conversation_state';
import type {StageAnalysisPanel} from '../../stage_analysis';
import type {TokenPair} from '../../token_alignment';
import {AnalysisPanels} from '../stage_analysis_rail/analysis_panels';
import {syncPairWidths} from './conversation_line_sync';
import {TokenDiff, stageStart} from './token_diff';

/** Aligned tokens on either runtime: static and virtualized Decode rows share these attributes. */
const HOVER_TARGET = '[data-turn][data-side][data-row]';

/**
 * The rendered Token Diff's own DOM work: the shared line widths of aligned pairs, the marks that
 * link tokens to the selection and to the analysis views, the pointer feedback on a token and its
 * partner, and moving the reading to a token, a stage or a node.
 */
@Injectable()
export class TokenDiffDom {
  private readonly conversation = inject(ConversationState);
  private readonly tokens = inject(TokenDiff);
  private readonly panels = inject(AnalysisPanels);

  readonly layoutTick = signal(0);
  /** Shared line breaks: every aligned pair takes the width of its wider side on both runtimes.
   *  Run after rendering. */
  alignPairWidths(): void {
    // Everything that changes token widths or the column width re-runs the sync.
    this.tokens.aligned();
    this.tokens.boundaries();
    this.tokens.color();
    this.conversation.alignment();
    this.tokens.panelWidth();
    this.tokens.collapsed();
    this.conversation.turns();
    this.layoutTick();
    if (this.conversation.workspace.mode() === 'Debug') {
      syncPairWidths(this.conversation.element.nativeElement);
    }
  }
  /** Refreshes the token marks whenever the page's nodes change; returns what stops it. */
  observeTokenLinks(): () => void {
    const observer = new MutationObserver(() => this.refreshTokenLinks());
    observer.observe(this.conversation.element.nativeElement, {childList: true, subtree: true});
    return () => observer.disconnect();
  }
  refreshTokenLinks(): void {
    const panels = this.panels.analysisPanels();
    const hovered = this.panels.analysisHover();
    const inspector = this.tokens.metricInspector();
    const hoveredPanel = panels.find((p) => p.id === hovered?.panelId && p.open);
    const selectedPanel = panels.find((p) => p.id === inspector?.panelId() && p.open);
    const range = inspector instanceof KvPanel ? inspector.inspectedRange() : inspector?.selected();
    const selection = this.tokens.selection();
    const selectedSide = this.tokens.selectionSide();
    const inputPairs = new Map<number, Map<number, TokenPair>>();
    for (const node of Array.from(
      (this.conversation.element.nativeElement as HTMLElement).querySelectorAll<HTMLElement>(
        HOVER_TARGET,
      ),
    )) {
      const turn = Number(node.dataset['turn']);
      const row = Number(node.dataset['row']);
      const side = node.dataset['side'] ?? 'target';
      const input = node.dataset['kind'] === 'input';
      const stage = input ? 'prefill' : 'decode';
      let token: CapturedToken | undefined;
      if (input) {
        if (!inputPairs.has(turn)) {
          inputPairs.set(
            turn,
            new Map(
              (this.tokens.inputPairRows(turn) ?? [])
                .flatMap((r) => [...r.leading, ...r.text, ...r.trailing])
                .map((p) => [p.index, p]),
            ),
          );
        }
        const pair = inputPairs.get(turn)?.get(row);
        token = pair ? this.tokens.pairToken(pair, side) : undefined;
      } else {
        const pair = this.tokens.rows(turn)[row];
        token = pair ? this.tokens.pairToken(pair, side) : undefined;
      }
      const info = this.conversation.state.session()?.turns.find((t) => t.n === turn);
      const start = stageStart(info, input ? 'prefill' : 'decode');
      const position = token && start !== null ? start + token.step : null;
      const coordinate =
        this.conversation.alignment() === 'content' && start !== null ? start + row : position;
      const matches = (
        panel: StageAnalysisPanel | undefined,
        interval: RangeSelection | null | undefined,
      ) => {
        return (
          !!panel &&
          panel.turn === turn &&
          panel.stage === stage &&
          !!interval &&
          coordinate !== null &&
          coordinate >= interval.start &&
          coordinate < interval.end
        );
      };
      const selectedPair =
        !!selection &&
        selection.turn === turn &&
        selection.kind === (input ? 'input' : 'token') &&
        'index' in selection &&
        selection.index === row;
      const origin = selectedPair && side === selectedSide;
      node.classList.toggle('selection-origin', origin);
      node.classList.toggle('linked-token-selected', selectedPair && !origin);
      const inspectedStage = selectedPanel?.turn === turn && selectedPanel.stage === stage;
      const selectedStage =
        selection?.kind === 'stage' && selection.turn === turn && selection.stage === stage;
      const focusRange = inspectedStage && range ? range : selectedStage ? selection.range : null;
      node.classList.toggle(
        'linked-token-focus',
        !selectedPair &&
          (inspectedStage || selectedStage) &&
          (!focusRange ||
            (coordinate !== null && coordinate >= focusRange.start && coordinate < focusRange.end)),
      );
      node.classList.toggle('linked-token-hover', matches(hoveredPanel, hovered?.range));
    }
  }
  // Pointer feedback frames the hovered token and its aligned partner on the other runtime,
  // as in the design study. It never changes the selection or opens a detail surface.
  private tokenFeedback: Element[] = [];
  private hoverTarget(target: EventTarget | null): HTMLElement | null {
    return target instanceof Element ? target.closest<HTMLElement>(HOVER_TARGET) : null;
  }
  private clearTokenFeedback(): void {
    for (const token of this.tokenFeedback) {
      token.classList.remove('pair-hover', 'hover-origin');
    }
    this.tokenFeedback = [];
  }
  private showTokenFeedback(token: HTMLElement): void {
    this.clearTokenFeedback();
    if (this.conversation.workspace.mode() !== 'Debug') {
      return;
    }
    const {turn, row, kind} = token.dataset;
    const partners = this.conversation.element.nativeElement.querySelectorAll(
      `[data-turn="${turn}"][data-row="${row}"]` +
        (kind ? `[data-kind="${kind}"]` : ':not([data-kind])'),
    );
    const fork = kind ? null : this.tokens.forks().get(Number(turn));
    const stepPartner =
      fork?.filled && fork.partner !== null && String(fork.index) === row
        ? this.conversation.element.nativeElement.querySelectorAll(
            `[data-turn="${turn}"][data-row="${fork.partner}"][data-side="${fork.filled}"]:not([data-kind])`,
          )
        : [];
    this.tokenFeedback = Array.from(new Set([token, ...partners, ...stepPartner]));
    for (const el of this.tokenFeedback) {
      el.classList.add('pair-hover');
    }
    token.classList.add('hover-origin');
  }
  tokenLocationAnimations: Animation[] = [];
  locateSelectedToken(selected = this.tokens.selection()): void {
    if (!selected || !('index' in selected)) {
      return;
    }
    const side = this.tokens.selectionSide() === 'ref' ? 'ref' : 'target';
    const stage = selected.kind === 'input' ? 'prefill' : 'decode';
    this.tokens.collapsed.update(
      (set) => new Set([...set].filter((k) => !k.startsWith(selected.turn + ':'))),
    );
    afterNextRender(
      () => {
        if (selected.kind === 'token' && this.conversation.virtualOutput(selected.turn)) {
          const row = this.conversation
            .virtualModels()
            .get(selected.turn)
            ?.rows.find((r) => r.index === selected.index);
          const runtime = row?.[side] != null ? side : side === 'ref' ? 'target' : 'ref';
          const step = row?.[runtime];
          if (step != null) {
            this.conversation.zone.runOutsideAngular(() => {
              this.conversation.virtualizer?.mount();
              this.conversation.virtualizer?.jump(selected.turn, step, runtime, false);
            });
          }
        }
        const root = this.conversation.element.nativeElement;
        const selector =
          '[data-turn="' +
          selected.turn +
          '"][data-row="' +
          selected.index +
          '"]' +
          (stage === 'prefill' ? '[data-kind="input"]' : ':not([data-kind])');
        const node =
          root.querySelector(selector + '[data-side="' + side + '"]') ??
          root.querySelector(selector);
        const reading = root.querySelector('.reading') as HTMLElement | null;
        const bounds = this.conversation.reading.visibleBounds();
        if (!node || !reading || !bounds) {
          return;
        }
        const rect = node.getBoundingClientRect();
        const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
        reading.scrollTo({
          top: reading.scrollTop + (rect.top + rect.bottom - bounds.top - bounds.bottom) / 2,
          behavior: reduced ? 'auto' : 'smooth',
        });
        this.tokenLocationAnimations?.forEach((animation) => animation.cancel());
        this.tokenLocationAnimations = Array.from(
          (root as HTMLElement).querySelectorAll<HTMLElement>(selector),
        ).map((token) => animateLocationFeedback(token, reduced));
      },
      {injector: this.conversation.injector},
    );
  }
  /** Open a stage from the index or from the page's address: select it on the Target side and
   *  bring its card into view. */
  openStage(turn: number, stage: 'prefill' | 'decode', event?: Event): void {
    const card = this.conversation.element.nativeElement.querySelector(
      `[data-node-turn="${turn}"][data-node-stage="${stage}"][data-node-side="target"]`,
    ) as HTMLElement | null;
    this.tokens.selectStage(
      turn,
      'target',
      stage,
      card ?? (event?.currentTarget as HTMLElement | undefined) ?? null,
    );
    // After the selection's own render work, which restores the previous scroll anchor.
    afterNextRender(
      () =>
        card?.scrollIntoView({
          block: 'center',
          behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        }),
      {injector: this.conversation.injector},
    );
  }
  togglePairedNode(turn: number, stage: string): void {
    const reading = this.conversation.element.nativeElement.querySelector('.reading');
    const node = reading?.querySelector(`[data-node-turn="${turn}"][data-node-stage="${stage}"]`);
    const viewport = reading?.getBoundingClientRect();
    const before = node?.getBoundingClientRect();
    const collapsed = this.tokens.nodeCollapsed(turn, 'target', stage);
    const anchor =
      before && viewport
        ? Math.max(viewport.top + 72, Math.min(before.top, viewport.bottom - 112))
        : null;
    this.tokens.collapsed.update((set) => {
      const next = new Set(set);
      for (const side of ['ref', 'target']) {
        const key = `${turn}:${side}:${stage}`;
        if (collapsed) {
          next.delete(key);
        } else {
          next.add(key);
        }
      }
      return next;
    });
    afterNextRender(
      () => {
        this.conversation.nodeNavigator?.mount();
        if (reading && node && anchor !== null) {
          reading.scrollTop += node.getBoundingClientRect().top - anchor;
        }
        this.conversation.virtualizer?.mount();
      },
      {injector: this.conversation.injector},
    );
  }
  readingBackgroundClick(event: MouseEvent): void {
    if (this.conversation.workspace.mode() !== 'Debug' || window.getSelection()?.toString()) {
      return;
    }
    const target = event.target as Element;
    if (
      target.closest(
        '[data-node-stage], .analysis-content, .expanded-analysis-header, button, a, input, select, textarea, label, [role="button"], [role="tab"], [role="dialog"], [contenteditable="true"]',
      )
    ) {
      return;
    }
    this.tokens.analysisOverview.set(null);
    this.tokens.metricInspector.set(null);
    this.tokens.tokenMetricInspector.set(null);
    this.tokens.selection.set(null);
    this.tokens.returnFocus = null;
    this.tokens.details.set(true);
  }
  locateConversationStage(turn: number, stage: 'prefill' | 'decode'): void {
    const reading = this.conversation.element.nativeElement.querySelector(
      '.reading',
    ) as HTMLElement | null;
    const cards = reading?.querySelectorAll<HTMLElement>(
      `[data-node-turn="${turn}"][data-node-stage="${stage}"]`,
    );
    const node = cards?.[0];
    if (!reading || !node) {
      return;
    }
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const heading = reading.querySelector('.column-headings')?.getBoundingClientRect().height ?? 0;
    reading.scrollTo({
      top:
        reading.scrollTop +
        node.getBoundingClientRect().top -
        reading.getBoundingClientRect().top -
        heading -
        8,
      behavior: reduced ? 'auto' : 'smooth',
    });
    if (!reduced) {
      for (const card of Array.from(cards ?? [])) {
        card.getAnimations().forEach((animation) => animation.cancel());
        card.animate(
          [
            {outline: '2px solid transparent', outlineOffset: '2px'},
            {
              outline: '2px solid var(--me-color-primary, #1a73e8)',
              outlineOffset: '2px',
              offset: 0.4,
            },
            {outline: '2px solid transparent', outlineOffset: '2px'},
          ],
          {duration: 1000, easing: 'ease-out'},
        );
      }
    }
  }
  find(direction: number): void {
    const results = this.tokens.mismatches();
    const i = direction === 0 ? 0 : this.tokens.resultIndex() + direction;
    if (i >= 0 && i < results.length) {
      this.tokens.select(results[i].turn, results[i].index);
      this.locateSelectedToken();
    }
  }
  /** The Debug reading scrolled: the pointer left whatever token it framed. */
  readingScrolled(): void {
    this.clearTokenFeedback();
    this.conversation.readingScrolled();
  }
  /** A wheel or key that scrolls the Debug reading down leaves the framed token, as a scroll
   *  does, even at the end of the reading, where nothing scrolls. */
  trackReadingIntent(event: WheelEvent | KeyboardEvent | PointerEvent): void {
    if (this.conversation.trackReadingIntent(event)) {
      this.clearTokenFeedback();
    }
  }
  tokenPointerOver(event: PointerEvent): void {
    const token = this.hoverTarget(event.target);
    if (token && !token.contains(event.relatedTarget as Node | null)) {
      this.showTokenFeedback(token);
    }
  }
  tokenPointerOut(event: PointerEvent): void {
    const token = this.hoverTarget(event.target);
    if (token && !token.contains(event.relatedTarget as Node | null)) {
      this.clearTokenFeedback();
    }
  }
  tokenFocusIn(event: FocusEvent): void {
    const token = this.hoverTarget(event.target);
    if (token) {
      this.showTokenFeedback(token);
    }
  }
  tokenFocusOut(event: FocusEvent): void {
    if (this.hoverTarget(event.target)) {
      this.clearTokenFeedback();
    }
  }
}
