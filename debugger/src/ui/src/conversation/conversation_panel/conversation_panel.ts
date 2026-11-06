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

import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  OnDestroy,
  afterNextRender,
  afterRenderEffect,
  effect,
  inject,
  untracked,
  viewChild,
} from '@angular/core';

import {NO_PLACE} from '../../state/page_address/page_address';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {ChatReading} from '../chat/chat_reading/chat_reading';
import {ConversationComposerController} from '../composer/conversation_composer';
import {MessageComposer} from '../composer/message_composer/message_composer';
import {isTemplateToken, tokenPhase} from '../conversation_content';
import {createConversationScrollbar} from '../conversation_scrollbar';
import {ConversationState} from '../conversation_state';
import {
  renderInputToken as renderInput,
  renderOutputToken as renderOutput,
} from '../conversation_virtualization';
import {DebugAddress} from '../debug/debug_address';
import {DetailsInfo} from '../debug/details_panel/details_info';
import {DetailsPanel} from '../debug/details_panel/details_panel';
import {ConversationViews} from '../debug/saved_views/conversation_views';
import {AnalysisInfo} from '../debug/stage_analysis_rail/analysis_info';
import {AnalysisPanels} from '../debug/stage_analysis_rail/analysis_panels';
import {ExpandedAnalysis} from '../debug/stage_analysis_rail/expanded_analysis';
import {StageAnalyses} from '../debug/stage_analysis_rail/stage_analyses';
import {CompactPreviews} from '../debug/token_diff/compact_previews';
import {TokenDiff} from '../debug/token_diff/token_diff';
import {TokenDiffDisplay} from '../debug/token_diff/token_diff_display/token_diff_display';
import {TokenDiffDom} from '../debug/token_diff/token_diff_dom';
import {TokenDiffFind} from '../debug/token_diff/token_diff_find/token_diff_find';
import {TokenDiffReading} from '../debug/token_diff/token_diff_reading/token_diff_reading';
import {createFlowNodeNavigator} from '../flow_node_navigation';
import {createFlowVirtualizer, type FlowTurn} from '../flow_virtualization';

/**
 * The conversation page: the Chat reading, or in Debug the Token Diff reading with its details
 * panel, over the message composer, which holds the Token Diff's tools in Debug.
 *
 * The page provides what its parts share (ConversationState for both readings, the Debug
 * reading's state for its parts) and wires it: the effects and render hooks that keep the state,
 * the long-text layout and the rendered tokens in step live here, in the order they run.
 */
@Component({
  selector: 'conversation-panel',
  imports: [
    ChatReading,
    DetailsPanel,
    MessageComposer,
    TokenDiffDisplay,
    TokenDiffFind,
    TokenDiffReading,
  ],
  providers: [
    ConversationState,
    TokenDiff,
    AnalysisPanels,
    StageAnalyses,
    ExpandedAnalysis,
    AnalysisInfo,
    DetailsInfo,
    TokenDiffDom,
    CompactPreviews,
    ConversationViews,
    DebugAddress,
    {
      provide: ConversationComposerController,
      useFactory: () => new ConversationComposerController(inject(WorkspaceStateService)),
    },
  ],
  templateUrl: './conversation_panel.ng.html',
  styleUrl: './conversation_panel.scss',
  host: {
    '(document:pointermove)': 'analyses.analysisPointerMove($event)',
    '(document:pointercancel)': 'analyses.analysisPointerCancel()',
    '(pointerover)': 'dom.tokenPointerOver($event)',
    '(pointerout)': 'dom.tokenPointerOut($event)',
    '(focusin)': 'dom.tokenFocusIn($event)',
    '(focusout)': 'dom.tokenFocusOut($event)',
    '(keydown.escape)': 'escape($event)',
  },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConversationPanel implements OnDestroy {
  // Provided by this page, so its element, injector and zone, which the other parts use, are this
  // page's in whatever order the parts are created.
  private readonly conversation = inject(ConversationState);
  protected readonly workspace = this.conversation.workspace;
  protected readonly tokens = inject(TokenDiff);
  private readonly panels = inject(AnalysisPanels);
  protected readonly analyses = inject(StageAnalyses);
  private readonly analysisInfo = inject(AnalysisInfo);
  private readonly detailsInfo = inject(DetailsInfo);
  protected readonly dom = inject(TokenDiffDom);
  private readonly compact = inject(CompactPreviews);
  protected readonly views = inject(ConversationViews);
  protected readonly address = inject(DebugAddress);
  protected readonly composer = inject(ConversationComposerController);
  private readonly destroyRef = inject(DestroyRef);
  private readonly composerHost = viewChild(MessageComposer, {read: ElementRef});
  private composerObserver: ResizeObserver | null = null;

  constructor() {
    const {conversation, tokens} = this;
    const emptyTurn: FlowTurn = {
      steps: {ref: [], target: []},
      inputTokens: {ref: [], target: []},
      alignments: {},
    };
    afterRenderEffect(() => this.dom.alignPairWidths());
    afterRenderEffect(() => this.dom.refreshTokenLinks());
    afterNextRender(() => this.destroyRef.onDestroy(this.dom.observeTokenLinks()));
    afterRenderEffect(() => this.compact.syncLayout());
    effect((onCleanup) => this.detailsInfo.loadMatchRate(onCleanup));
    effect(() => this.analyses.syncAnalysisViewport());
    effect(() => this.analyses.syncMetricInspectorVisibility());
    effect(() => this.analyses.syncTokenMetricInspector());
    effect(() => this.analyses.followSelection());
    effect((onCleanup) => tokens.loadTokenAnalysis(onCleanup));
    afterNextRender(() =>
      conversation.zone.runOutsideAngular(() => {
        const host = conversation.element.nativeElement;
        const composerElement = this.composerHost()?.nativeElement;
        const composer =
          composerElement?.firstElementChild instanceof HTMLElement
            ? composerElement.firstElementChild
            : composerElement;
        if (!(composer instanceof HTMLElement)) {
          return;
        }
        conversation.scrollbar = createConversationScrollbar(
          host,
          () => this.workspace.mode() === 'Debug',
        );
        conversation.scrollbar.mount();
        const measure = () =>
          host.style.setProperty(
            '--composer-clearance',
            `${composer.getBoundingClientRect().height + (parseFloat(getComputedStyle(composer).bottom) || 0) + 16}px`,
          );
        this.composerObserver = new ResizeObserver(measure);
        this.composerObserver.observe(composer);
        measure();
      }),
    );
    conversation.virtualizer = conversation.zone.runOutsideAngular(() =>
      createFlowVirtualizer({
        threshold: 2048,
        getTurn: (turn) => conversation.virtualModels().get(turn) ?? emptyTurn,
        getRows: (turn) => conversation.virtualModels().get(turn)?.rows ?? [],
        getAlignment: () => conversation.alignment(),
        isDebug: () => this.workspace.mode() === 'Debug',
        getScroller: () => conversation.element.nativeElement.querySelector<HTMLElement>('.reading'),
        phase: (turn, side, index) =>
          tokenPhase(conversation.captured(turn, side)?.tokens?.[index]) === 'thinking'
            ? 'Thinking'
            : 'Output',
        getVisibleBounds: () => conversation.reading.visibleBounds(),
        findRow: (turn, side, index) => {
          const model = conversation.virtualModels().get(turn);
          return (
            model?.bySide[side]?.get(index) ??
            (!model?.hasCapturedTokens ? model?.rows[index] : undefined)
          );
        },
        getSelectedRow: (turn) => {
          const model = conversation.virtualModels().get(turn);
          const selection = tokens.tokenSelection();
          return selection?.turn === turn
            ? (model?.rows.find((r) => r.index === selection.index) ?? null)
            : null;
        },
        renderInputToken: (word, index, side) => renderInput(word, index, side),
        renderOutputToken: (word, index, side, turn) => {
          const model = conversation.virtualModels().get(turn);
          const row = model?.bySide[side]?.get(index);
          const pair = model && row ? model.pairs[row.index] : null;
          const debug = this.workspace.mode() === 'Debug';
          const heat = tokens.tokenHeat(turn, pair, side);
          return renderOutput(
            word,
            {turn, side, step: index, row: row?.index},
            {
              interactive: debug,
              debug,
              side,
              numeric:
                debug && side === 'target' && tokens.numericColor()
                  ? heat === null
                    ? 'missing'
                    : 'token'
                  : null,
              first: debug && tokens.isFirstDivergence(turn, row?.index, side),
              partner: debug && tokens.isStepPartner(turn, row?.index, side),
              template:
                debug && isTemplateToken(conversation.captured(turn, side)?.tokens?.[index]),
              mismatch: pair?.match === false && (debug ? tokens.color() === 'token_match' : true),
              active:
                tokens.selection()?.turn === turn && tokens.tokenSelection()?.index === row?.index,
              boundaries: tokens.boundaries() && debug,
              palette: tokens.heatPalette(tokens.color()),
              heat,
              empty: tokens.zeroWidthText(word),
            },
          );
        },
        displayKey: () => String(tokens.boundaries()),
        styleKey: () =>
          [
            this.workspace.mode(),
            tokens.color(),
            tokens.boundaries(),
            tokens.tokenAnalysis().size,
            tokens.selection()?.turn,
            tokens.tokenSelection()?.index,
          ].join(':'),
        widthAdjustment: () => (tokens.boundaries() && this.workspace.mode() === 'Debug' ? 10 : 0),
        lineHeight: () => (tokens.boundaries() && this.workspace.mode() === 'Debug' ? 30 : 26),
        onSelect: (turn, index, side) => {
          const model = conversation.virtualModels().get(turn);
          const row = model?.bySide[side]?.get(index);
          if (row && this.workspace.mode() === 'Debug') {
            conversation.zone.run(() => tokens.select(turn, row.index));
          }
        },
      }),
    );
    conversation.nodeNavigator = conversation.zone.runOutsideAngular(() =>
      createFlowNodeNavigator({
        getRoot: () => conversation.element.nativeElement,
        getScroller: () => conversation.element.nativeElement.querySelector<HTMLElement>('.reading'),
        isEnabled: () => this.workspace.mode() === 'Debug',
        isCollapsed: (turn, stage) => tokens.nodeCollapsed(turn, 'target', stage),
        toggleCollapsed: (turn, stage) =>
          conversation.zone.run(() => this.dom.togglePairedNode(turn, stage)),
        afterJump: () => conversation.virtualizer?.paint(),
      }),
    );
    let lastSession: unknown = null;
    // Structure: rows, the session, collapsed stages and the details pane change the layout.
    afterRenderEffect(() => {
      conversation.virtualModels();
      const session = conversation.state.session();
      tokens.details();
      tokens.collapsed();
      conversation.thinkingExpanded();
      untracked(() =>
        conversation.zone.runOutsideAngular(() => {
          if (lastSession !== session) {
            conversation.virtualizer?.clearLayout();
            lastSession = session;
          }
          conversation.virtualizer?.mount();
          conversation.nodeNavigator?.mount();
          conversation.scrollbar?.mount();
        }),
      );
    });
    // Style: selection, colors and metrics only repaint the rendered tokens.
    afterRenderEffect(() => {
      tokens.selection();
      tokens.color();
      tokens.tokenAnalysis();
      tokens.boundaries();
      untracked(() =>
        conversation.zone.runOutsideAngular(() => conversation.virtualizer?.paint()),
      );
    });
    effect(() => conversation.followLiveGeneration());
    effect(() => conversation.openLiveThinking());
    this.views.restoreReturnedView();
    effect(() => this.views.keepPanels());
    effect(() => this.views.followPendingView());
    effect(() => {
      this.workspace.mode();
      untracked(() => conversation.reading.syncMode());
    });
    effect(() => this.analyses.followPendingInspector());
    effect(() => this.address.follow());
  }

  escape(event: KeyboardEvent): void {
    if (this.panels.expandedAnalysis()) {
      event.stopPropagation();
      this.panels.exitExpandedAnalysis();
      return;
    }
    if (this.tokens.selection()) {
      event.stopPropagation();
      this.tokens.closeDetails();
    }
  }
  ngOnDestroy(): void {
    // Capture while the rendered virtual layout and semantic anchor still exist.
    this.views.saveView();
    this.workspace.debugPlace.set(NO_PLACE);
    this.dom.tokenLocationAnimations.forEach((animation) => animation.cancel());
    this.analysisInfo.analysisLocationAnimation?.cancel();
    this.conversation.reading.destroy();
    this.conversation.scrollbar?.destroy();
    this.composerObserver?.disconnect();
    this.conversation.nodeNavigator?.destroy();
  }
}
