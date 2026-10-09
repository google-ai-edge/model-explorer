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
  afterNextRender,
  afterRenderEffect,
  ChangeDetectionStrategy,
  Component,
  computed,
  DestroyRef,
  effect,
  ElementRef,
  HostListener,
  inject,
  Injector,
  NgZone,
  OnDestroy,
  signal,
  untracked,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  ConversationBookmark,
  WorkspaceStateService,
} from '../../../app/workspace_state_service';
import type {
  CapturedConversation,
  CapturedToken,
} from '../../../data/contracts/capture';
import {KvSnapshot} from '../../../data/contracts/telemetry';
import {ReportApiService} from '../../../data/report_api_service';
import {ReportStateService} from '../../../data/report_state_service';
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {ExplorerInfoSection} from '../../../shared/explorer_info_section/explorer_info_section';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {FindNavigator} from '../../../shared/find_navigator/find_navigator';
import {formatCount, formatSignificant} from '../../../shared/format/format';
import {ResizeHandle} from '../../../shared/resize_handle/resize_handle';
import {batchForwardId, tokenMatchesBatch} from '../../graph/batch_forward';
import {RunnerStatus} from '../../sessions/runner_status/runner_status';
import {
  capturedInputPairRows,
  capturedInputRows,
  capturedInputSegments,
  capturedPhaseText,
  capturedTokenContent,
  capturedTokenLabel,
  ConversationSelection,
  hasPhaseTokens,
  historyStatus,
  isTemplateToken,
  pairLineBreaks,
  splitPhasePairs,
  spokenTokenText,
  tokenContent,
  tokenMarkup,
  tokenPhase,
  TokenPhase,
} from '../conversation_content';
import {syncPairWidths} from '../conversation_line_sync';
import {ConversationReadingController} from '../conversation_reading_controller';
import {createConversationScrollbar} from '../conversation_scrollbar';
import {
  captureViewState,
  pairIdentity,
  PANEL_WIDTH,
  prepareViewRestore,
  readViewState,
  TokenDiffViewState,
  type ConversationObservation,
} from '../conversation_view_state';
import {
  renderInputToken as renderInput,
  renderOutputToken as renderOutput,
  streamingConversation,
  virtualConversation,
} from '../conversation_virtualization';
import {EmptyChatDemo} from '../empty_chat_demo/empty_chat_demo';
import {
  createFlowNodeNavigator,
  FlowNodeNavigator,
} from '../flow_node_navigation';
import {createFlowVirtualizer, FlowVirtualizer} from '../flow_virtualization';
import {
  AlignmentMode,
  alignTokens,
  decodeOutcome,
  forkPair,
  TokenPair,
} from '../token_alignment';
import {
  colorLabel,
  heatLevel,
  heatPalette,
  pairKey,
  queryRecord,
  TOKEN_COLOR_MAX,
  TokenColor,
} from '../token_analysis';
import {TokenDisplay} from '../token_display/token_display';
import {TOKEN_METRICS} from '../token_metric_metadata';
import {compileQuery} from '../token_query';
import {TokenQueryControl} from '../token_query_control/token_query_control';
import {TokenTrendState} from '../token_trends';
import {ConversationComposerController} from './conversation_composer';
import {TokenAnalysisController} from './token_analysis_controller';
import {TokenDistribution} from './token_distribution';
import {TokenNumericalInfo} from './token_numerical_info';
import {TokenTrends} from './token_trends';
/** Aligned tokens on either runtime: static and virtualized Decode rows share these attributes. */
const HOVER_TARGET = '[data-turn][data-side][data-row]';
const NO_BREAKS: readonly number[] = [];

@Component({
  selector: 'conversation-panel',
  standalone: true,
  imports: [
    FindNavigator,
    ResizeHandle,
    RunnerStatus,
    TokenTrends,
    TokenDistribution,
    TokenNumericalInfo,
    ExplorerInfoValue,
    ExplorerInfoSection,
    EmptyChatDemo,
    ConfigPicker,
    TokenQueryControl,
    TokenDisplay,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatTooltipModule,
  ],
  templateUrl: './conversation_panel.ng.html',
  styleUrl: './conversation_panel.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConversationPanel implements OnDestroy {
  private readonly injector = inject(Injector);
  private readonly dialog = inject(MatDialog);
  async readFullText(
    turn: number,
    side: string,
    kind: 'input' | 'output' | 'thinking',
  ) {
    const {FullTextDialog} = await import(
      '../../../shared/full_text_dialog/full_text_dialog'
    );
    const capture = this.captured(turn, side),
      serialized =
        kind === 'input' && this.workspace.mode() === 'Debug'
          ? capture?.serialized_input
          : undefined;
    this.dialog.open(FullTextDialog, {
      width: '760px',
      maxWidth: 'calc(100vw - 24px)',
      data: {
        title:
          (side === 'ref' ? 'Reference' : 'Target') +
          ' · ' +
          (kind === 'input'
            ? serialized !== undefined
              ? 'Serialized input'
              : 'User message'
            : kind === 'thinking'
              ? 'Thinking'
              : 'Model response'),
        text:
          kind === 'thinking'
            ? capturedPhaseText(capture, 'thinking')
            : (serialized ??
              capture?.[kind] ??
              (kind === 'input'
                ? this.liveGeneration()?.prompt
                : this.liveGeneration()?.output[side]) ??
              ''),
      },
      ariaLabel: 'Read full captured text',
    });
  }
  private readonly contextKey =
    inject(WorkspaceStateService).selectedId() +
    ':' +
    inject(WorkspaceStateService).activeChat();
  private readonly element = inject(ElementRef<HTMLElement>);
  readonly state = inject(ReportStateService);
  private readonly api = inject(ReportApiService);
  /** Keyed by `${captureId}:${turn}` so a capture change cannot serve another capture's snapshots. */
  readonly kvSnapshots = signal(new Map<string, KvSnapshot[]>());
  readonly kvLoadErrors = signal(new Map<string, string>());
  private snapshotCapture: string | null = null;
  private snapshotKey(turn: number, capture = this.state.captureId()) {
    return (capture ?? '') + ':' + turn;
  }
  private readonly analysis = new TokenAnalysisController(this.api);
  readonly tokenAnalysis = this.analysis.results;
  readonly analysisLoading = this.analysis.loading;
  readonly analysisError = this.analysis.error;
  readonly analysisRetry = signal(0);
  readonly numericFindPending = computed(
    () =>
      this.predicate().requiresMetrics &&
      (this.analysisLoading() || !!this.analysisError()),
  );
  readonly findResultLabel = computed(() =>
    this.numericFindPending()
      ? this.analysisError()
        ? 'Find results unavailable; retry token metrics'
        : 'Find results loading'
      : 'Find results: ' + this.mismatches().length + ' tokens',
  );
  readonly selectedAnalysis = computed(() => {
    const pair = this.detailPair();
    return pair
      ? this.tokenAnalysis().get(pairKey(pair.turn, pair))
      : undefined;
  });
  readonly workspace = inject(WorkspaceStateService);
  private readonly zone = inject(NgZone);
  private virtualizer: FlowVirtualizer | null = null;
  private composerObserver: ResizeObserver | null = null;
  private followLive = true;
  private resumeAtTail = true;
  trackReadingIntent(event: WheelEvent | KeyboardEvent | PointerEvent) {
    if (
      (event instanceof WheelEvent && event.deltaY < 0) ||
      (event instanceof KeyboardEvent &&
        ['ArrowUp', 'PageUp', 'Home'].includes(event.key))
    ) {
      this.followLive = false;
      this.resumeAtTail = false;
    }
    if (
      (event instanceof WheelEvent && event.deltaY > 0) ||
      (event instanceof KeyboardEvent &&
        ['ArrowDown', 'PageDown', 'End'].includes(event.key))
    ) {
      this.resumeAtTail = true;
      this.readingScrolled();
    }
    if (event instanceof PointerEvent) {
      const r = this.element.nativeElement
        .querySelector('.reading')
        ?.getBoundingClientRect();
      if (r && event.clientX >= r.right - 16) {
        this.followLive = false;
        this.resumeAtTail = true;
      }
    }
  }
  readingScrolled() {
    this.clearTokenFeedback();
    const r = this.element.nativeElement.querySelector('.reading');
    if (
      this.resumeAtTail &&
      r &&
      r.scrollHeight - r.clientHeight - r.scrollTop < 80
    )
      this.followLive = true;
  }
  // Shared line breaks: every aligned pair takes the width of its wider side on both runtimes.
  private readonly destroyRef = inject(DestroyRef);
  private readonly layoutTick = signal(0);
  private readonly pairWidthSync = afterRenderEffect(() => {
    // Everything that changes token widths or the column width re-runs the sync.
    this.aligned();
    this.boundaries();
    this.color();
    this.alignment();
    this.panelWidth();
    this.collapsed();
    this.turns();
    this.layoutTick();
    if (this.workspace.mode() === 'Debug')
      syncPairWidths(this.element.nativeElement);
  });
  private readonly readingResize = afterNextRender(() => {
    const reading = this.element.nativeElement.querySelector('.reading');
    if (!reading || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() =>
      this.layoutTick.update((n) => n + 1),
    );
    observer.observe(reading);
    this.destroyRef.onDestroy(() => observer.disconnect());
    document.fonts?.ready.then(() => this.layoutTick.update((n) => n + 1));
  });
  infoTokenHtml(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token ? tokenMarkup(capturedTokenLabel(token), false) : '∅';
  }
  infoTokenTitle(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token
      ? JSON.stringify(capturedTokenLabel(token)) +
          (token.id != null ? ` · ID ${token.id}` : '')
      : 'No token at this aligned position';
  }
  /** Prefill rows over aligned pairs; null when either side lacks token kinds. */
  inputPairRows(turn: number) {
    return capturedInputPairRows(
      this.captured(turn, 'ref'),
      this.captured(turn, 'target'),
    );
  }
  pairToken(pair: TokenPair, side: string) {
    return side === 'ref' ? pair.ref : pair.target;
  }
  inputTokenText(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token ? capturedTokenLabel(token) : '∅';
  }
  /** Prefill text follows the same whitespace rule as Decode tokens. */
  inputTokenHtml(pair: TokenPair, side: string) {
    const token = this.pairToken(pair, side);
    return token ? tokenMarkup(capturedTokenLabel(token)) : '∅';
  }
  /** A token's text for labels: an invisible-only token is named ("line feed × 2"), a gap is ∅. */
  spokenToken(token: CapturedToken | undefined) {
    return token ? spokenTokenText(capturedTokenLabel(token)) : '∅';
  }
  // Pointer feedback frames the hovered token and its aligned partner on the other runtime,
  // as in the design study. It never changes the selection or opens a detail surface.
  private tokenFeedback: Element[] = [];
  private hoverTarget(target: EventTarget | null) {
    return target instanceof Element
      ? target.closest<HTMLElement>(HOVER_TARGET)
      : null;
  }
  @HostListener('pointerover', ['$event']) tokenPointerOver(
    event: PointerEvent,
  ) {
    const token = this.hoverTarget(event.target);
    if (token && !token.contains(event.relatedTarget as Node | null))
      this.showTokenFeedback(token);
  }
  @HostListener('pointerout', ['$event']) tokenPointerOut(event: PointerEvent) {
    const token = this.hoverTarget(event.target);
    if (token && !token.contains(event.relatedTarget as Node | null))
      this.clearTokenFeedback();
  }
  @HostListener('focusin', ['$event']) tokenFocusIn(event: FocusEvent) {
    const token = this.hoverTarget(event.target);
    if (token) this.showTokenFeedback(token);
  }
  @HostListener('focusout', ['$event']) tokenFocusOut(event: FocusEvent) {
    if (this.hoverTarget(event.target)) this.clearTokenFeedback();
  }
  private clearTokenFeedback() {
    for (const token of this.tokenFeedback)
      token.classList.remove('pair-hover', 'hover-origin');
    this.tokenFeedback = [];
  }
  private showTokenFeedback(token: HTMLElement) {
    this.clearTokenFeedback();
    if (this.workspace.mode() !== 'Debug') return;
    const {turn, row, kind} = token.dataset;
    const partners = this.element.nativeElement.querySelectorAll(
      `[data-turn="${turn}"][data-row="${row}"]` +
        (kind ? `[data-kind="${kind}"]` : ':not([data-kind])'),
    );
    const fork = kind ? null : this.forks().get(Number(turn));
    const stepPartner =
      fork?.filled && fork.partner !== null && String(fork.index) === row
        ? this.element.nativeElement.querySelectorAll(
            `[data-turn="${turn}"][data-row="${fork.partner}"][data-side="${fork.filled}"]:not([data-kind])`,
          )
        : [];
    this.tokenFeedback = [token, ...partners, ...stepPartner].filter(
      (el, i, all) => all.indexOf(el) === i,
    );
    for (const el of this.tokenFeedback) el.classList.add('pair-hover');
    token.classList.add('hover-origin');
  }
  private nodeNavigator: FlowNodeNavigator | null = null;
  nodeIndex(turn: number, stage: string) {
    return (
      this.turns().findIndex((t) => t.n === turn) * 2 +
      (stage === 'decode' ? 1 : 0)
    );
  }
  readonly virtualModels = computed(() => {
    const models = new Map(
      this.turns().map((t) => [
        t.n,
        virtualConversation(
          this.captured(t.n, 'ref'),
          this.captured(t.n, 'target'),
          this.alignment(),
          this.workspace.mode() === 'Debug',
        ),
      ]),
    );
    const job = this.liveGeneration();
    if (
      job &&
      Math.max(
        job.prompt?.length ?? 0,
        ...Object.values(job.output).map((t) => t.length),
      ) > 16000
    )
      models.set(job.turn, streamingConversation(job.prompt ?? '', job.output));
    return models;
  });
  virtualOutput(turn: number) {
    const model = this.virtualModels().get(turn);
    return (
      (model?.rows.length ?? 0) > 2048 &&
      (this.workspace.mode() === 'Debug' || !!model?.responseTextComplete)
    );
  }
  virtualPhase(turn: number, phase: TokenPhase) {
    return (
      this.virtualOutput(turn) &&
      ['ref', 'target'].some((side) =>
        hasPhaseTokens(this.captured(turn, side), phase),
      )
    );
  }
  virtualInput(turn: number) {
    const t = this.virtualModels().get(turn);
    return (
      Math.max(
        t?.inputTokens.ref.length ?? 0,
        t?.inputTokens.target.length ?? 0,
      ) > 2048
    );
  }
  readonly usesVirtual = computed(() =>
    [...this.virtualModels().keys()].some(
      (n) => this.virtualInput(n) || this.virtualOutput(n),
    ),
  );
  readonly longAlignmentUnavailable = computed(
    () =>
      this.workspace.mode() === 'Chat' &&
      [...this.virtualModels().values()].some((t) => t.alignmentUnavailable),
  );
  readonly runs = computed(
    () =>
      this.state.session()?.runs ??
      this.workspace.runtime.selected()?.runs ??
      this.workspace.sessions
        .items()
        .find((s) => s.id === this.workspace.selectedId())?.runs ??
      [],
  );
  readonly turns = computed(() => this.state.session()?.turns ?? []);
  readonly thinkingExpanded = signal(new Set<string>());
  readonly liveGeneration = computed(() => {
    const job = this.workspace.runtime.job();
    if (
      this.workspace.mode() !== 'Chat' ||
      job?.operation !== 'generate' ||
      job.session_id !== this.workspace.context().captureRecordId
    )
      return null;
    const captured = this.turns().some((t) => t.n === job.turn);
    return job.status === 'completed' && captured ? null : job;
  });
  readonly composer = new ConversationComposerController(this.workspace);
  thinkingOpen(turn: number, run: string) {
    return this.thinkingExpanded().has(`${turn}:${run}`);
  }
  toggleThinking(turn: number, run: string) {
    const key = `${turn}:${run}`;
    this.thinkingExpanded.update((s) => {
      const n = new Set(s);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });
  }
  private readonly chatText = computed(() => {
    const result: Record<string, {text: string; mismatch: boolean}[]> = {};
    for (const turn of this.state.session()?.turns ?? []) {
      if (this.virtualOutput(turn.n)) continue;
      let rows: TokenPair[] = [];
      try {
        rows = alignTokens(
          this.captured(turn.n, 'ref')?.tokens,
          this.captured(turn.n, 'target')?.tokens,
          'content',
          this.captured(turn.n, 'ref')?.alignments?.content,
        );
      } catch {
        continue;
      }
      for (const run of ['ref', 'target']) {
        const parts = rows.flatMap((r) => {
          const t = run === 'ref' ? r.ref : r.target;
          return t?.text != null && (t.phase ?? 'response') === 'response'
            ? [{text: t.text, mismatch: r.match === false}]
            : [];
        });
        if (
          parts.map((t) => t.text).join('') ===
          this.captured(turn.n, run)?.output
        )
          result[`${turn.n}:${run}`] = parts;
      }
    }
    return result;
  });
  chatTokens(turn: number, run: string) {
    return this.chatText()[`${turn}:${run}`] ?? [];
  }

  readonly formula = signal('NOT token_match');
  readonly predicate = computed(() => compileQuery(this.formula()));

  readonly alignment = signal<AlignmentMode>('content');
  readonly selection = signal<ConversationSelection | null>(null);
  readonly tokenSelection = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'token' ? selected : null;
  });
  readonly selectedStage = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'stage' ? selected : null;
  });
  readonly inputSelection = computed(() => {
    const selected = this.selection();
    return selected?.kind === 'input' ? selected : null;
  });
  /** The selected admitted Prefill pair, found across its message rows. */
  readonly selectedInputPair = computed(() => {
    const s = this.inputSelection();
    if (!s) return null;
    for (const row of this.inputPairRows(s.turn) ?? [])
      for (const pair of [...row.leading, ...row.text, ...row.trailing])
        if (pair.index === s.index) return {...pair, turn: s.turn};
    return null;
  });
  /** A generated token's content kind, read against its own run's admitted template tokens. */
  tokenContentFor(
    turn: number,
    side: string,
    token: CapturedToken | undefined,
  ) {
    return capturedTokenContent(this.captured(turn, side), token);
  }
  /** What an admitted token is: message text, a chat-template marker or a special token. */
  inputTokenContent(token: CapturedToken | undefined) {
    return !token
      ? '∅'
      : token.kind === 'template'
        ? 'Chat template'
        : token.kind === 'special'
          ? 'Special token'
          : 'Input context';
  }
  isInputSelected(turn: number, index: number) {
    const s = this.inputSelection();
    return !!s && s.turn === turn && s.index === index;
  }
  selectInput(turn: number, index: number, event?: Event) {
    if (this.isInputSelected(turn, index)) {
      this.closeDetails();
      return;
    }
    const trigger = event?.currentTarget ?? document.activeElement;
    if (trigger instanceof HTMLElement) this.returnFocus = trigger;
    this.selection.set({kind: 'input', turn, index});
  }
  readonly details = signal(true);
  readonly boundaries = signal(false);
  readonly color = signal<TokenColor>('token_match');
  readonly colorLabel = colorLabel;
  readonly heatPalette = heatPalette;
  readonly colorMax = TOKEN_COLOR_MAX;
  readonly metricAvailability = computed(() =>
    Object.fromEntries(
      TOKEN_METRICS.map((m) => [
        m.key,
        [...this.tokenAnalysis().values()].some(
          (p) => p.metrics[m.key]?.value != null,
        ),
      ]),
    ),
  );
  readonly numericColor = computed(
    () => this.color() !== 'none' && this.color() !== 'token_match',
  );
  readonly colorMaximum = computed(() =>
    this.numericColor()
      ? this.colorMax[this.color() as keyof typeof this.colorMax]
      : null,
  );
  readonly colorUnit = computed(() =>
    this.color() === 'relative_l2'
      ? '%'
      : this.color() === 'js'
        ? 'bits'
        : this.color() === 'kl'
          ? 'nats'
          : this.color() === 'norm_ratio'
            ? '×'
            : '',
  );
  tokenHeat(turn: number, pair: TokenPair | undefined | null, side: string) {
    return side === 'target' && pair && this.numericColor()
      ? heatLevel(this.color(), this.tokenAnalysis().get(pairKey(turn, pair)))
      : null;
  }
  readonly collapsed = signal(new Set<string>());
  readonly bookmarks = this.workspace.bookmarks;
  readonly bookmarkNotice = signal('');
  readonly groups = signal<Record<string, boolean>>({});
  readonly trends = signal<TokenDiffViewState['trends']>({});
  trendState(turn: number): TokenTrendState {
    return (
      this.trends()[turn + ':' + this.alignment()] ?? {
        open: false,
        metric: 'kl',
        start: 0,
        end: Math.max(1, this.rows(turn).length - 1),
      }
    );
  }
  setTrend(turn: number, setting: TokenTrendState) {
    this.trends.update((s) => ({
      ...s,
      [turn + ':' + this.alignment()]: setting,
    }));
  }
  toggleTrend(turn: number) {
    const state = this.trendState(turn);
    this.setTrend(turn, {...state, open: !state.open});
    // The chart opens below its turn, often under the floating toolbar: bring all of it into
    // view. Its scroll margin keeps the axis and the range slider clear of the toolbar.
    if (!state.open)
      afterNextRender(
        () =>
          this.element.nativeElement
            .querySelector('#turnTrend-' + turn)
            ?.scrollIntoView({
              block: 'nearest',
              behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
                ? 'auto'
                : 'smooth',
            }),
        {injector: this.injector},
      );
  }
  groupOpen(key: string) {
    return this.groups()[key] ?? true;
  }
  setGroup(key: string, value: boolean) {
    this.groups.update((groups) => ({...groups, [key]: value}));
  }
  readonly alignmentChoices = ['Context', 'Steps'];
  readonly alignmentDescriptions = {
    Context: 'Match output content',
    Steps: 'Match generation-step index',
  };
  readonly infoExpanded = signal(true);
  readonly panelBounds = PANEL_WIDTH;
  readonly panelWidth = signal(350);
  private scrollbar: ReturnType<typeof createConversationScrollbar> | null =
    null;
  private returnFocus: HTMLElement | null = null;
  private readonly reading = new ConversationReadingController({
    context: this.contextKey,
    positions: this.workspace.readingPositions,
    root: () => this.element.nativeElement,
    mode: () => this.workspace.mode(),
    virtualizer: () => this.virtualizer,
    afterRender: (work) =>
      afterNextRender(() => this.zone.runOutsideAngular(work), {
        injector: this.injector,
      }),
    afterRestore: () => {
      this.nodeNavigator?.mount();
      this.scrollbar?.mount();
    },
  });
  private captureView(): TokenDiffViewState {
    return captureViewState({
      observation: this.currentObservation(),
      formula: this.formula(),
      alignment: this.alignment(),
      selection: this.selection(),
      identity: pairIdentity(this.selectedPair()),
      details: this.details(),
      boundaries: this.boundaries(),
      // Invisible characters are always drawn; the field keeps saved views in one shape.
      whitespace: true,
      color: this.color(),
      collapsed: [...this.collapsed()],
      thinking: [...this.thinkingExpanded()],
      infoExpanded: this.infoExpanded(),
      groups: {...this.groups()},
      panelWidth: this.panelWidth(),
      ...this.reading.capture(),
      trends: this.trends(),
    });
  }
  private currentObservation(): ConversationObservation | null {
    const batch = this.state
      .session()
      ?.batches.find((item) => item.batch === this.state.batchId());
    return batch ? {turn: batch.turn, phase: batch.phase} : null;
  }
  private saveView() {
    this.workspace.conversationViews.set(this.contextKey, this.captureView());
  }
  private rowsForRestore(
    turn: number,
    alignment: AlignmentMode,
  ): readonly TokenPair[] {
    const ref = this.captured(turn, 'ref'),
      target = this.captured(turn, 'target');
    if (
      Math.max(ref?.tokens?.length ?? 0, target?.tokens?.length ?? 0) > 2048
    ) {
      const model = virtualConversation(ref, target, alignment, true);
      return model.alignmentUnavailable ? [] : model.pairs;
    }
    return alignTokens(
      ref?.tokens,
      target?.tokens,
      alignment,
      ref?.alignments?.content,
    );
  }
  /** Both ordinary return and bookmarks cross the same validated restore path. */
  private applyView(
    value: unknown,
    source: 'return' | 'bookmark' | 'legacy',
    expected?: {turn: number; alignment: AlignmentMode},
  ): boolean {
    const prepared = prepareViewRestore(
      value,
      {
        hasTurn: (turn) => this.turns().some((item) => item.n === turn),
        rows: (turn, alignment) => this.rowsForRestore(turn, alignment),
      },
      source === 'return'
        ? {
            source,
            observation: this.currentObservation(),
          }
        : {source, expected: expected!},
    );
    if (prepared.status !== 'ready') {
      const message = {
        invalid: 'Saved settings are invalid or from an unsupported version.',
        'missing-turn': 'This saved turn is no longer available.',
        'missing-token': 'This saved token is no longer available.',
        'inconsistent-turn': 'Saved bookmark has inconsistent turn metadata.',
      }[prepared.status];
      this.bookmarkNotice.set(
        message +
          (source === 'return'
            ? ' Conversation defaults were retained.'
            : ' The bookmark was kept.'),
      );
      return false;
    }
    const view = prepared.state;
    this.formula.set(view.formula);
    this.alignment.set(view.alignment);
    this.selection.set(view.selection);
    this.details.set(view.details);
    this.boundaries.set(view.boundaries);
    this.color.set(view.color);
    this.collapsed.set(new Set(view.collapsed));
    this.thinkingExpanded.set(new Set(view.thinking));
    this.infoExpanded.set(view.infoExpanded);
    this.groups.set(view.groups);
    this.panelWidth.set(view.panelWidth);
    this.trends.set(view.trends);
    if (source !== 'return' && view.selection)
      this.state.selectTurn(view.selection.turn);
    if (source !== 'legacy')
      this.reading.apply(view, source !== 'return', prepared.observation);
    return true;
  }
  ngOnDestroy() {
    // Capture while the rendered virtual layout and semantic anchor still exist.
    this.saveView();
    this.reading.destroy();
    this.scrollbar?.destroy();
    this.composerObserver?.disconnect();
    this.nodeNavigator?.destroy();
  }
  constructor() {
    effect((onCleanup) => {
      this.analysisRetry();
      const captureId = this.state.captureId(),
        aligned = this.aligned(),
        debug = this.workspace.mode() === 'Debug';
      const controller = new AbortController();
      onCleanup(() => controller.abort());
      // Skip uncaptured links entirely, including synthetic 128k text-only sessions.
      const pending = debug
        ? aligned
            .map((t) => ({
              turn: t.turn,
              pairs: [
                ...t.rows,
                ...(this.forks().get(t.turn)?.filled
                  ? [this.forks().get(t.turn)!.pair]
                  : []),
              ]
                .filter(
                  (p) =>
                    p.ref?.source_forward_id !== undefined ||
                    p.target?.source_forward_id !== undefined ||
                    p.ref?.batch !== undefined ||
                    p.target?.batch !== undefined,
                )
                .map((p) => ({
                  ref: p.ref?.step ?? null,
                  target: p.target?.step ?? null,
                })),
            }))
            .filter((t) => t.pairs.length)
        : [];
      const preview = debug ? this.previewStepPair() : null;
      if (preview)
        pending.push({
          turn: preview.turn,
          pairs: [
            {
              ref: preview.pair.ref?.step ?? null,
              target: preview.pair.target?.step ?? null,
            },
          ],
        });
      untracked(() =>
        this.analysis.load(captureId, pending, controller.signal),
      );
    });
    effect((onCleanup) => {
      const turns = this.turns();
      const debug = this.workspace.mode() === 'Debug',
        session = this.state.captureId();
      const controller = new AbortController();
      onCleanup(() => controller.abort());
      if (session !== this.snapshotCapture) {
        this.snapshotCapture = session;
        this.kvSnapshots.set(new Map());
        this.kvLoadErrors.set(new Map());
      }
      if (!debug || !session) return;
      for (const turn of turns) {
        const key = this.snapshotKey(turn.n, session);
        this.api
          .telemetry(session, turn.n, controller.signal)
          .then((data) => {
            if (!controller.signal.aborted)
              this.kvSnapshots.update((value) =>
                new Map(value).set(key, data.kv_snapshots ?? []),
              );
          })
          .catch(() => {
            if (!controller.signal.aborted)
              this.kvLoadErrors.update((value) =>
                new Map(value).set(
                  key,
                  'Snapshot metadata could not be loaded',
                ),
              );
          });
      }
    });
    afterNextRender(() =>
      this.zone.runOutsideAngular(() => {
        const host = this.element.nativeElement,
          composer = host.querySelector('.composer') as HTMLElement | null;
        if (!composer) return;
        this.scrollbar = createConversationScrollbar(
          host,
          () => this.workspace.mode() === 'Debug',
        );
        this.scrollbar.mount();
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
    this.virtualizer = this.zone.runOutsideAngular(() =>
      createFlowVirtualizer({
        threshold: 2048,
        getTurn: (turn) => this.virtualModels().get(turn)!,
        getRows: (turn) => this.virtualModels().get(turn)?.rows ?? [],
        getAlignment: () => this.alignment(),
        isDebug: () => this.workspace.mode() === 'Debug',
        getScroller: () => this.element.nativeElement.querySelector('.reading'),
        phase: (turn, side, index) =>
          tokenPhase(this.captured(turn, side)?.tokens?.[index]) === 'thinking'
            ? 'Thinking'
            : 'Output',
        getVisibleBounds: () => this.reading.visibleBounds(),
        findRow: (turn, side, index) => {
          const model = this.virtualModels().get(turn);
          return (
            model?.bySide[side].get(index) ??
            (!model?.hasCapturedTokens ? model?.rows[index] : undefined)
          );
        },
        getSelectedRow: (turn) => {
          const model = this.virtualModels().get(turn),
            selection = this.tokenSelection();
          return selection?.turn === turn
            ? (model?.rows.find((r) => r.index === selection.index) ?? null)
            : null;
        },
        renderInputToken: (word, index, side) => renderInput(word, index, side),
        renderOutputToken: (word, index, side, turn) => {
          const model = this.virtualModels().get(turn)!,
            row = model.bySide[side].get(index),
            pair = row ? model.pairs[row.index] : null,
            debug = this.workspace.mode() === 'Debug',
            heat = this.tokenHeat(turn, pair, side);
          return renderOutput(
            word,
            {turn, side, step: index, row: row?.index},
            {
              interactive: debug && !model.alignmentUnavailable,
              debug,
              side,
              numeric:
                debug && side === 'target' && this.numericColor()
                  ? heat === null
                    ? 'missing'
                    : 'token'
                  : null,
              first: debug && this.isFirstDivergence(turn, row?.index, side),
              partner: debug && this.isStepPartner(turn, row?.index, side),
              template:
                debug &&
                isTemplateToken(this.captured(turn, side)?.tokens?.[index]),
              mismatch:
                !model.alignmentUnavailable &&
                pair?.match === false &&
                (debug ? this.color() === 'token_match' : true),
              active:
                this.selection()?.turn === turn &&
                this.tokenSelection()?.index === row?.index,
              boundaries: this.boundaries() && debug,
              palette: heatPalette(this.color()),
              heat,
              empty: this.zeroWidthText(word),
            },
          );
        },
        displayKey: () => String(this.boundaries()),
        styleKey: () =>
          [
            this.workspace.mode(),
            this.color(),
            this.boundaries(),
            this.tokenAnalysis().size,
            this.selection()?.turn,
            this.tokenSelection()?.index,
          ].join(':'),
        widthAdjustment: () =>
          this.boundaries() && this.workspace.mode() === 'Debug' ? 8 : 0,
        lineHeight: () =>
          this.boundaries() && this.workspace.mode() === 'Debug' ? 28 : 26,
        onSelect: (turn, index, side) => {
          const model = this.virtualModels().get(turn),
            row = model?.bySide[side].get(index);
          if (
            row &&
            !model?.alignmentUnavailable &&
            this.workspace.mode() === 'Debug'
          )
            this.zone.run(() => this.select(turn, row.index));
        },
      }),
    );
    this.nodeNavigator = this.zone.runOutsideAngular(() =>
      createFlowNodeNavigator({
        getRoot: () => this.element.nativeElement,
        getScroller: () => this.element.nativeElement.querySelector('.reading'),
        isEnabled: () => this.workspace.mode() === 'Debug',
        isCollapsed: (turn, stage) => this.nodeCollapsed(turn, 'target', stage),
        toggleCollapsed: (turn, stage) =>
          this.zone.run(() => this.togglePairedNode(turn, stage)),
        afterJump: () => this.virtualizer?.paint(true),
      }),
    );
    let lastSession: unknown = null;
    // Structure: rows, the session, collapsed stages and the details pane change the layout.
    effect(() => {
      this.virtualModels();
      const session = this.state.session();
      this.details();
      this.collapsed();
      this.thinkingExpanded();
      afterNextRender(
        () =>
          this.zone.runOutsideAngular(() => {
            if (lastSession !== session) {
              this.virtualizer?.clearLayout();
              lastSession = session;
            }
            this.virtualizer?.mount();
            this.nodeNavigator?.mount();
            this.scrollbar?.mount();
          }),
        {injector: this.injector},
      );
    });
    // Style: selection, colors and metrics only repaint the rendered tokens.
    effect(() => {
      this.selection();
      this.color();
      this.tokenAnalysis();
      this.boundaries();
      afterNextRender(
        () => this.zone.runOutsideAngular(() => this.virtualizer?.paint(true)),
        {
          injector: this.injector,
        },
      );
    });
    let lastJob = '';
    effect(() => {
      const job = this.liveGeneration();
      if (!job) return;
      if (job.id !== lastJob) this.followLive = true;
      lastJob = job.id;
      // Lay out virtual text before following its tail. Native scroll intent is
      // separate from content growth, which can change scrollHeight by many pages.
      afterNextRender(
        () =>
          this.zone.runOutsideAngular(() => {
            const current =
              this.element.nativeElement.querySelector('.reading');
            if (current) {
              this.virtualizer?.mount();
              if (this.followLive) {
                current.scrollTop = current.scrollHeight;
                this.virtualizer?.paint(true);
              }
            }
          }),
        {injector: this.injector},
      );
    });
    const saved = this.workspace.conversationViews.get(this.contextKey);
    if (saved) this.applyView(saved, 'return');
    effect(() => {
      this.workspace.mode();
      untracked(() => this.reading.syncMode());
    });
    effect(() => {
      const target = this.workspace.bookmarkTarget();
      if (target) {
        this.openBookmark(target);
        this.workspace.bookmarkTarget.set(null);
      }
    });
  }

  turnEnding(turn: number) {
    const job = this.workspace.runtime.job();
    if (
      job?.operation === 'generate' &&
      job.session_id === this.workspace.context().captureRecordId &&
      job.turn === turn
    ) {
      if (!['completed', 'failed', 'cancelled'].includes(job.status))
        return 'Generation in progress';
      if (job.status === 'cancelled') return 'Generation stopped';
      if (job.status === 'failed') return 'Generation failed';
    }
    const reasons = this.runs().map(
      (run) => this.captured(turn, run.id)?.stop_reason,
    );
    if (reasons.some((r) => r === 'cancelled')) return 'Generation stopped';
    if (reasons.some((r) => r === 'error')) return 'Generation failed';
    return reasons.length &&
      reasons.every((r) => r === 'eos' || r === 'max_output_tokens')
      ? 'Conversation complete'
      : 'End of captured conversation';
  }
  /** Captured conversation items by `turn:run`; the template asks for them on every check. */
  private readonly capturedByKey = computed(() => {
    const items = new Map<string, CapturedConversation>();
    for (const item of this.state.session()?.conversation ?? [])
      items.set(`${item.turn}:${item.run}`, item);
    return items;
  });
  captured(turn: number, run: string) {
    return this.capturedByKey().get(`${turn}:${run}`);
  }
  hasTokenCapture(turn: number, run: string) {
    return this.captured(turn, run)?.tokens !== undefined;
  }
  inputSegments(turn: number, run: string) {
    return capturedInputSegments(this.captured(turn, run));
  }
  readonly historyByTurn = computed(() => {
    const conversation = this.state.session()?.conversation;
    return new Map(
      (this.state.session()?.turns ?? []).map((turn) => [
        turn.n,
        historyStatus(conversation, turn.n),
      ]),
    );
  });
  historyStatus(turn: number) {
    return this.historyByTurn().get(turn) ?? null;
  }
  historyReason(status: string) {
    return status === 'same'
      ? 'Both runtimes entered this turn with identical history: same inputs and same earlier outputs.'
      : 'The runtimes entered this turn with different history: an earlier output differed.';
  }
  /** Admitted input as the runtime tokenized it, grouped by message; null without token kinds. */
  inputRows(turn: number, run: string) {
    return capturedInputRows(this.captured(turn, run));
  }
  roleIcon(role: string) {
    return role === 'system'
      ? 'settings'
      : role === 'user'
        ? 'person'
        : role === 'model'
          ? 'smart_toy'
          : 'subject';
  }
  roleLabel(role: string) {
    return role === 'system'
      ? 'System'
      : role === 'user'
        ? 'User'
        : role === 'model'
          ? 'Model'
          : 'Input';
  }
  readonly aligned = computed(() =>
    (this.state.session()?.turns ?? []).map((turn) => {
      const virtual = this.virtualModels().get(turn.n);
      if (this.virtualOutput(turn.n) && this.workspace.mode() === 'Debug')
        return {
          turn: turn.n,
          rows: virtual?.alignmentUnavailable ? [] : (virtual?.pairs ?? []),
          error: virtual?.alignmentUnavailable
            ? 'Context alignment exceeds the supported size. Select Steps alignment.'
            : '',
        };
      try {
        return {
          turn: turn.n,
          rows: alignTokens(
            this.captured(turn.n, 'ref')?.tokens,
            this.captured(turn.n, 'target')?.tokens,
            this.alignment(),
            this.captured(turn.n, 'ref')?.alignments?.content,
          ),
          error: '',
        };
      } catch (error) {
        return {
          turn: turn.n,
          rows: [] as TokenPair[],
          error: String((error as Error).message),
        };
      }
    }),
  );
  rows(turn: number) {
    return this.aligned().find((t) => t.turn === turn)?.rows ?? [];
  }
  readonly tokenPhases: TokenPhase[] = ['thinking', 'response'];
  readonly phasePairs = computed(
    () =>
      new Map(
        this.aligned()
          .filter((t) => !this.virtualOutput(t.turn))
          .map((t) => [t.turn, splitPhasePairs(t.rows)]),
      ),
  );
  rowsForPhase(turn: number, side: string, phase: TokenPhase) {
    return (
      this.phasePairs().get(turn)?.[side === 'ref' ? 'ref' : 'target'][phase] ??
      []
    );
  }
  phaseText(turn: number, side: string, phase: TokenPhase) {
    return capturedPhaseText(this.captured(turn, side), phase);
  }
  alignmentError(turn: number) {
    return this.aligned().find((t) => t.turn === turn)?.error;
  }
  readonly selectedPair = computed(() => {
    const s = this.tokenSelection();
    const row = s ? this.rows(s.turn)[s.index] : null;
    return s && row ? {...row, turn: s.turn} : null;
  });
  /** Each turn's first differing generation step, paired by step (see `forkPair`). */
  readonly forks = computed(
    () =>
      new Map(
        this.aligned().map((t) => [
          t.turn,
          forkPair(
            t.rows,
            this.captured(t.turn, 'ref')?.tokens,
            this.captured(t.turn, 'target')?.tokens,
          ),
        ]),
      ),
  );
  /** Each turn's Decode outcome: identical, diverged at a step, unknown or not captured. */
  readonly outcomes = computed(
    () =>
      new Map(
        this.aligned().map((t) => [
          t.turn,
          decodeOutcome(
            t.rows,
            this.forks().get(t.turn) ?? null,
            this.captured(t.turn, 'ref')?.tokens,
            this.captured(t.turn, 'target')?.tokens,
          ),
        ]),
      ),
  );
  outcomeText(turn: number) {
    const outcome = this.outcomes().get(turn);
    return !outcome || outcome.kind === 'uncaptured'
      ? 'Not captured'
      : outcome.kind === 'identical'
        ? 'Identical'
        : outcome.kind === 'diverged'
          ? 'Diverged at step ' + outcome.step
          : 'Not comparable';
  }
  /** The selected Decode stage's summary. Every value comes from the captures or from analysis
   *  the Server returned; a value that is not loaded or not provable reads as a dash. */
  readonly stageDecode = computed(() => {
    const stage = this.selectedStage();
    if (!stage || stage.stage !== 'decode') return null;
    const turn = stage.turn,
      outcome = this.outcomes().get(turn),
      fork = this.forks().get(turn) ?? null,
      rows = this.rows(turn),
      data = this.tokenAnalysis();
    // The Server gives paired metrics only where both runtimes read the same context, so the
    // largest loaded value is the peak of the comparable steps.
    let peak: {value: number; step: number} | null = null;
    for (const row of rows) {
      const pair = fork?.filled && fork.index === row.index ? fork.pair : row,
        value = data.get(pairKey(turn, pair))?.metrics['kl']?.value;
      if (value != null && (!peak || value > peak.value))
        peak = {value, step: (pair.target ?? pair.ref)!.step};
    }
    const distribution = fork
        ? data.get(pairKey(turn, fork.pair))?.distribution
        : undefined,
      margin = (side: 'ref' | 'target') => {
        const value = distribution?.[side]?.margin;
        return value == null || !Number.isFinite(value)
          ? '—'
          : (value * 100).toFixed(1) + ' pp';
      };
    return {
      turn,
      outcome: this.outcomeText(turn),
      prefix:
        !outcome || outcome.kind === 'uncaptured'
          ? 'Not captured'
          : formatCount(outcome.prefix) +
            (outcome.prefix === 1 ? ' token' : ' tokens'),
      peak: peak
        ? `${formatSignificant(peak.value, 5)} nats · step ${peak.step}`
        : '—',
      fork,
      margin: {ref: margin('ref'), target: margin('target')},
    };
  });
  /** One turn's Prefill: whether both runtimes tokenized the input alike and entered the turn
   *  with the same history. */
  private prefillOutcome(turn: number) {
    const rows = this.inputPairRows(turn),
      pairs =
        rows?.flatMap((row) => [
          ...row.leading,
          ...row.text,
          ...row.trailing,
        ]) ?? [],
      first = pairs.find((pair) => pair.match !== true),
      history = this.historyStatus(turn);
    const tokenized = !rows
      ? 'Not captured'
      : !first
        ? 'Same'
        : first.match === null
          ? 'Unknown'
          : 'Different · first at token ' + first.index;
    return {
      tokenized,
      history:
        history === 'same'
          ? 'Same'
          : history === 'differs'
            ? 'Differs'
            : 'Not proven',
      historyReason:
        history === 'same' || history === 'differs'
          ? this.historyReason(history)
          : 'The captures do not prove whether both runtimes entered this turn with the same history.',
      // The index states what differs first: the input's tokens, then the history behind them.
      index: tokenized.startsWith('Different')
        ? 'Input differs'
        : tokenized !== 'Same'
          ? tokenized
          : history === 'differs'
            ? 'History differs'
            : 'Same',
    };
  }
  readonly stagePrefill = computed(() => {
    const stage = this.selectedStage();
    return stage?.stage === 'prefill' ? this.prefillOutcome(stage.turn) : null;
  });
  /** Conversation Info's index: one line per turn, each cell opening its stage. */
  readonly turnIndex = computed(() =>
    this.turns().map((turn) => {
      const outcome = this.outcomes().get(turn.n);
      return {
        turn: turn.n,
        prefill: this.prefillOutcome(turn.n).index,
        decode:
          outcome?.kind === 'diverged'
            ? 'Differs · step ' + outcome.step
            : this.outcomeText(turn.n),
      };
    }),
  );
  /** Open a stage from the index: select it on the Target side and bring its card into view. */
  openStage(turn: number, stage: 'prefill' | 'decode', event: Event) {
    const card = this.element.nativeElement.querySelector(
      `[data-node-turn="${turn}"][data-node-stage="${stage}"][data-node-side="target"]`,
    ) as HTMLElement | null;
    this.selectStage(
      turn,
      'target',
      stage,
      card ?? (event.currentTarget as HTMLElement),
    );
    // After the selection's own render work, which restores the previous scroll anchor.
    afterNextRender(
      () =>
        card?.scrollIntoView({
          block: 'center',
          behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
            ? 'auto'
            : 'smooth',
        }),
      {injector: this.injector},
    );
  }
  /** Why one runtime stopped: its recorded stop reason, else the stop token it sampled last. */
  stageEnding(turn: number, side: string) {
    const capture = this.captured(turn, side),
      reason = capture?.stop_reason;
    return reason === 'eos' ||
      (!reason && capture?.tokens?.at(-1)?.stop === true)
      ? 'Stop token'
      : reason === 'max_output_tokens'
        ? 'Output limit'
        : reason === 'cancelled'
          ? 'Stopped'
          : reason === 'error'
            ? 'Failed'
            : 'Not captured';
  }
  /** Select a turn's first differing row from its stage summary. */
  selectFork(turn: number) {
    const fork = this.forks().get(turn);
    if (fork) this.select(turn, fork.index);
  }
  /** What the details describe: the selected row, except that the first differing row shows the
   *  two tokens of its generation step when the text alignment left one side empty. */
  readonly detailPair = computed(() => {
    const pair = this.selectedPair();
    if (!pair) return null;
    const fork = this.forks().get(pair.turn);
    return fork?.filled && fork.index === pair.index
      ? {...fork.pair, turn: pair.turn, filled: fork.filled}
      : {...pair, filled: null};
  });
  pairingTitle(filled: 'ref' | 'target') {
    return (
      'The first generation step where the two runtimes produced different tokens. Context ' +
      `alignment matches text and leaves the ${filled === 'ref' ? 'Reference' : 'Target'} side of ` +
      'this row empty, so the details pair the two tokens of this step instead. Metrics appear ' +
      'only where the captures prove that both runtimes read the same context.'
    );
  }
  /** The token drawn elsewhere that the details pair with the selected first differing row. */
  isStepPartner(turn: number, index: number | undefined, side: string) {
    const fork = this.forks().get(turn),
      selected = this.tokenSelection();
    return (
      !!fork?.filled &&
      fork.filled === side &&
      fork.partner === index &&
      this.selection()?.turn === turn &&
      selected?.index === fork.index
    );
  }
  /** The first turn whose two runtimes sampled different released tokens at one step: the one
   *  pair that both differs and can share its context, so every metric may be defined for it. */
  readonly previewStepPair = computed(() => {
    for (const [turn, fork] of this.forks())
      if (
        fork &&
        fork.pair.ref?.released !== false &&
        fork.pair.target?.released !== false
      )
        return {turn, pair: fork.pair};
    return null;
  });
  // The preview is a fixed pair: the first sampled difference, whose metrics exist; identical
  // runs fall back to the first short differing pair, then the selection or the first row.
  readonly displayPreview = computed(() => {
    const fixed = this.previewStepPair();
    if (fixed)
      return {
        ref: fixed.pair.ref?.text ?? 'Not captured',
        target: fixed.pair.target?.text ?? 'Not captured',
        match: false,
        analysis: this.tokenAnalysis().get(pairKey(fixed.turn, fixed.pair)),
      };
    const short = (token: CapturedToken | undefined) =>
      /^\s*\S{1,10}$/.test(token?.text ?? '');
    let changed: {turn: number; pair: TokenPair} | undefined;
    for (const turn of this.aligned()) {
      const row = turn.rows.find(
        (r) => r.match === false && short(r.ref) && short(r.target),
      );
      if (row) {
        changed = {turn: turn.turn, pair: row};
        break;
      }
    }
    const turn =
      changed?.turn ??
      this.selection()?.turn ??
      this.state.batch()?.turn ??
      this.turns()[0]?.n;
    const rows = this.rows(turn);
    const pair =
      changed?.pair ??
      this.selectedPair() ??
      rows.find((r) => r.match === false) ??
      rows[0];
    return pair
      ? {
          ref: pair.ref?.text ?? 'Not captured',
          target: pair.target?.text ?? 'Not captured',
          match: pair.match,
          analysis: this.tokenAnalysis().get(pairKey(turn, pair)),
        }
      : null;
  });
  // The demo's first-divergence glyph is hidden; preserve its accessible identity.
  readonly firstDivergence = computed(() => {
    for (const turn of this.aligned()) {
      const row = turn.rows.find((r) => r.match === false);
      if (row) return {turn: turn.turn, index: row.index};
    }
    return null;
  });
  isFirstDivergence(turn: number, index: number | undefined, side: string) {
    const first = this.firstDivergence();
    return side === 'target' && first?.turn === turn && first.index === index;
  }
  readonly mismatches = computed(() => {
    const predicate = this.predicate(),
      records = predicate.requiresMetrics ? this.tokenAnalysis() : null;
    return this.aligned().flatMap((turn) =>
      turn.rows
        .filter(
          (row) =>
            predicate(
              records
                ? queryRecord(row, records.get(pairKey(turn.turn, row)))
                : {token_match: row.match},
            ) === true,
        )
        .map((row) => ({turn: turn.turn, index: row.index})),
    );
  });
  readonly resultIndex = computed(() =>
    this.mismatches().findIndex(
      (r) =>
        r.turn === this.selection()?.turn &&
        r.index === this.tokenSelection()?.index,
    ),
  );
  private bookmarkMatches(b: ConversationBookmark) {
    const selected = this.selection(),
      saved = readViewState(b.view)?.selection;
    if (selected?.kind === 'input')
      return (
        saved?.kind === 'input' &&
        saved.turn === selected.turn &&
        saved.index === selected.index
      );
    return selected?.kind === 'stage'
      ? saved?.kind === 'stage' &&
          saved.turn === selected.turn &&
          saved.stage === selected.stage
      : selected?.kind === 'token' &&
          b.turn === selected.turn &&
          b.index === selected.index &&
          b.alignment === this.alignment();
  }
  readonly isBookmarked = computed(() =>
    this.bookmarks().some((b) => this.bookmarkMatches(b)),
  );
  setAlignment(mode: AlignmentMode) {
    this.alignment.set(mode);
    this.selection.set(null);
  }
  readonly compactTokens = computed(() => {
    const result = new Map<
      string,
      {
        text: string;
        index: number;
        step: number;
        ellipsis: boolean;
        match: boolean | null;
      }[]
    >();
    for (const turn of this.turns())
      for (const side of ['ref', 'target'] as const) {
        const tokens = this.captured(turn.n, side)?.tokens ?? [],
          length = tokens.length;
        const indices =
          length > 12
            ? [
                0,
                1,
                2,
                3,
                4,
                5,
                length - 6,
                length - 5,
                length - 4,
                length - 3,
                length - 2,
                length - 1,
              ]
            : tokens.map((_, i) => i);
        const virtual = this.virtualModels().get(turn.n);
        result.set(
          `${turn.n}:${side}`,
          indices.flatMap((step, offset) => {
            const row = virtual?.bySide[side].get(step),
              pair = row
                ? virtual?.pairs[row.index]
                : this.rows(turn.n).find(
                    (p) => (side === 'ref' ? p.ref : p.target) === tokens[step],
                  );
            const text = capturedTokenLabel(tokens[step]);
            return pair
              ? [
                  {
                    text: text.length > 80 ? text.slice(0, 80) + '…' : text,
                    index: pair.index,
                    step: tokens[step].step,
                    ellipsis: length > 12 && offset === 6,
                    match: pair.match,
                  },
                ]
              : [];
          }),
        );
      }
    return result;
  });
  compactInput(turn: number, side: string) {
    const c = this.captured(turn, side),
      text = c?.serialized_input ?? c?.input ?? 'Input text not captured';
    return text.replace(/\s+/g, ' ').slice(0, 200);
  }
  compactOutput(turn: number, side: string) {
    return (this.captured(turn, side)?.output ?? 'Response text not captured')
      .replace(/\s+/g, ' ')
      .slice(0, 200);
  }
  compactCount(turn: number, side: string, stage: string) {
    const count = this.stageCount(turn, side, stage);
    return typeof count === 'number' ? formatCount(count) + ' tokens' : count;
  }
  togglePairedNode(turn: number, stage: string) {
    const reading = this.element.nativeElement.querySelector('.reading'),
      node = reading?.querySelector(
        `[data-node-turn="${turn}"][data-node-stage="${stage}"]`,
      );
    const viewport = reading?.getBoundingClientRect(),
      before = node?.getBoundingClientRect(),
      collapsed = this.nodeCollapsed(turn, 'target', stage);
    const anchor =
      before && viewport
        ? Math.max(
            viewport.top + 72,
            Math.min(before.top, viewport.bottom - 112),
          )
        : null;
    this.collapsed.update((set) => {
      const next = new Set(set);
      for (const side of ['ref', 'target']) {
        const key = `${turn}:${side}:${stage}`;
        collapsed ? next.delete(key) : next.add(key);
      }
      return next;
    });
    afterNextRender(
      () => {
        this.nodeNavigator?.mount();
        if (reading && node && anchor !== null)
          reading.scrollTop += node.getBoundingClientRect().top - anchor;
        this.virtualizer?.mount();
      },
      {injector: this.injector},
    );
  }
  nodeCollapsed(turn: number, run: string, phase: string) {
    return this.collapsed().has(`${turn}:${run}:${phase}`);
  }
  stageSelected(turn: number, stage: string) {
    const selected = this.selectedStage();
    return selected?.turn === turn && selected.stage === stage;
  }
  generatedCount(run: string) {
    const captures = this.turns().map((t) => this.captured(t.n, run));
    return captures.every((c) => c?.tokens !== undefined)
      ? captures.reduce((n, c) => n + c!.tokens!.length, 0)
      : 'Not captured';
  }
  stageCount(turn: number, run: string, stage: string) {
    const capture = this.captured(turn, run);
    return (
      (stage === 'prefill'
        ? capture?.input_token_count
        : capture?.tokens?.length) ?? 'Not captured'
    );
  }
  selectStagePointer(
    event: MouseEvent,
    turn: number,
    side: string,
    stage: 'prefill' | 'decode',
  ) {
    if (
      (event.target as Element).closest(
        'button,summary,a,input,select,[data-v-gap],[role="button"]',
      ) ||
      window.getSelection()?.toString()
    )
      return;
    this.selectStage(turn, side, stage, event.currentTarget as HTMLElement);
  }
  selectStageKey(
    event: KeyboardEvent,
    turn: number,
    side: string,
    stage: 'prefill' | 'decode',
  ) {
    if (
      event.target !== event.currentTarget ||
      !['Enter', ' '].includes(event.key)
    )
      return;
    event.preventDefault();
    this.selectStage(turn, side, stage, event.currentTarget as HTMLElement);
  }
  private selectStage(
    turn: number,
    side: string,
    stage: 'prefill' | 'decode',
    trigger: HTMLElement,
  ) {
    if (this.workspace.mode() !== 'Debug') return;
    const anchor = this.virtualizer?.capture();
    this.returnFocus = trigger;
    this.selection.set({kind: 'stage', turn, side, stage});
    this.details.set(true);
    this.state.selectTurn(turn);
    afterNextRender(
      () => {
        this.virtualizer?.mount();
        if (anchor) this.virtualizer?.restore(anchor);
      },
      {injector: this.injector},
    );
  }
  select(turn: number, index: number, event?: Event) {
    const current = this.selection();
    if (
      current?.kind === 'token' &&
      current.turn === turn &&
      current.index === index
    ) {
      this.closeDetails();
      return;
    }
    const trigger = event?.currentTarget ?? document.activeElement;
    if (trigger instanceof HTMLElement) this.returnFocus = trigger;
    const previewSide =
      trigger instanceof HTMLElement &&
      trigger.classList.contains('nodePreviewToken')
        ? trigger.closest('[data-node-side]')?.getAttribute('data-node-side')
        : null;
    const focusPreviewTarget = () => {
      if (previewSide) {
        const token = this.element.nativeElement.querySelector(
          `.turn[data-turn="${turn}"] [data-runtime="${previewSide}"] [data-row="${index}"]`,
        ) as HTMLElement | null;
        token?.focus({preventScroll: true});
        if (token) this.returnFocus = token;
      }
    };
    this.selection.set({kind: 'token', turn, index});
    this.details.set(true);
    this.state.selectTurn(turn);
    this.collapsed.update(
      (set) => new Set([...set].filter((k) => !k.startsWith(turn + ':'))),
    );
    afterNextRender(
      () => {
        if (this.virtualOutput(turn)) {
          const row = this.virtualModels()
              .get(turn)
              ?.rows.find((r) => r.index === index),
            side = row?.target != null ? 'target' : 'ref',
            step = row?.[side];
          if (step != null) {
            this.zone.runOutsideAngular(() => {
              this.virtualizer?.mount();
              this.virtualizer?.jump(turn, step, side, false);
            });
            focusPreviewTarget();
            return;
          }
        }
        focusPreviewTarget();
        this.element.nativeElement
          .querySelector(`[data-turn="${turn}"] [data-row="${index}"]`)
          ?.scrollIntoView({
            block: 'center',
            behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
              ? 'auto'
              : 'smooth',
          });
      },
      {injector: this.injector},
    );
  }
  closeDetails() {
    this.selection.set(null);
    this.returnFocus?.focus({preventScroll: true});
  }
  @HostListener('keydown.escape', ['$event']) escape(event: KeyboardEvent) {
    if (this.selection()) {
      event.stopPropagation();
      this.closeDetails();
    }
  }
  find(direction: number) {
    const results = this.mismatches();
    const i = direction === 0 ? 0 : this.resultIndex() + direction;
    if (i >= 0 && i < results.length)
      this.select(results[i].turn, results[i].index);
  }
  bookmark(addOnly = false) {
    const selected = this.selection();
    if (!selected) return;
    if (this.isBookmarked()) {
      if (!addOnly)
        this.workspace.setBookmarks(
          this.bookmarks().filter((b) => !this.bookmarkMatches(b)),
        );
      this.bookmarkNotice.set(
        addOnly ? 'Bookmark already saved' : 'Bookmark removed',
      );
      return;
    }
    const pair =
        selected.kind === 'input'
          ? this.selectedInputPair()
          : this.selectedPair(),
      token = pair?.target ?? pair?.ref;
    const title =
      selected.kind === 'stage'
        ? `Turn ${selected.turn} · ${selected.stage === 'prefill' ? 'Prefill' : 'Decode'}`
        : selected.kind === 'input'
          ? `Turn ${selected.turn} · Prefill · Token ${selected.index}${token ? ' “' + capturedTokenLabel(token).trim().slice(0, 80) + '”' : ''}`
          : `Turn ${selected.turn} · Decode · Token ${token?.step ?? selected.index}${token ? ' “' + capturedTokenLabel(token).trim().slice(0, 80) + '”' : ''}`;
    this.workspace.setBookmarks([
      ...this.bookmarks(),
      {
        id: crypto.randomUUID(),
        title,
        turn: selected.turn,
        index: selected.kind === 'token' ? selected.index : -1,
        alignment: this.alignment(),
        view: this.captureView(),
      },
    ]);
    this.bookmarkNotice.set('Bookmark added');
  }
  openBookmark(bookmark: ConversationBookmark) {
    const expected = {turn: bookmark.turn, alignment: bookmark.alignment};
    if (bookmark.view === undefined) {
      const view = {
        ...this.captureView(),
        mode: 'Debug' as const,
        alignment: bookmark.alignment,
        selection: {
          kind: 'token' as const,
          turn: bookmark.turn,
          index: bookmark.index,
        },
        identity: undefined,
      };
      if (this.applyView(view, 'legacy', expected)) {
        this.select(bookmark.turn, this.tokenSelection()!.index);
        this.bookmarkNotice.set(
          'Opened legacy bookmark; no saved display settings.',
        );
      }
      return;
    }
    if (this.applyView(bookmark.view, 'bookmark', expected))
      this.bookmarkNotice.set('Bookmark restored');
  }
  readonly tokenContent = tokenContent;
  readonly isTemplateToken = isTemplateToken;
  /** Plain token text for labels; a missing token reads as ∅. */
  tokenText(token: CapturedToken | undefined) {
    return token ? capturedTokenLabel(token) : '∅';
  }
  /** Debug token markup: real characters, with whitespace glyphs drawn in place when shown. */
  tokenHtml(token: CapturedToken | undefined) {
    return tokenMarkup(this.tokenText(token), false);
  }
  /** One entry per line break that follows this row, for both runtimes alike. */
  rowBreaks(pair: TokenPair) {
    const count = pairLineBreaks(pair);
    return count ? Array.from({length: count}, (_, i) => i) : NO_BREAKS;
  }
  zeroWidthText(text: string) {
    return /^[\r\n\u200b-\u200d\ufeff]*$/.test(text);
  }
  stageBatch(turn: number, stage: string) {
    return this.state
      .session()
      ?.batches.find((b) => b.turn === turn && b.phase === stage);
  }
  stageGraph(turn: number, stage: string) {
    const batch = this.stageBatch(turn, stage);
    if (batch) {
      this.workspace.graph(turn, stage);
      this.state.selectBatch(batch.batch);
    }
  }
  kvSnapshot(turn: number, stage: string, token?: CapturedToken) {
    const moment =
      stage === 'start'
        ? 'prefill_pre'
        : stage === 'prefill'
          ? 'prefill_post'
          : 'terminal';
    let candidates = (
      this.kvSnapshots().get(this.snapshotKey(turn)) ?? []
    ).filter((s) => s.run === 'target' && s.moment === moment);
    if (token) {
      const batch = this.linkedBatch(token);
      if (!batch) return undefined;
      candidates = (
        this.kvSnapshots().get(this.snapshotKey(turn)) ?? []
      ).filter(
        (s) =>
          s.run === 'target' &&
          s.forward_id === batchForwardId(batch, 'target') &&
          s.phase === batch.phase &&
          s.runtime === batch.runtime &&
          ['prefill_post', 'decode_post', 'terminal'].includes(s.moment),
      );
    }
    candidates = candidates.slice().sort((a, b) => a.forward_id - b.forward_id);
    return stage === 'start' ? candidates[0] : candidates.at(-1);
  }
  kvReason(turn: number, stage: string, token?: CapturedToken) {
    if (this.kvSnapshot(turn, stage, token))
      return 'Open the recorded cache observation';
    const error = this.kvLoadErrors().get(this.snapshotKey(turn));
    if (error) return error;
    const inferred = (
      this.kvSnapshots().get(this.snapshotKey(turn)) ?? []
    ).some((s) => s.run === 'target' && s.basis);
    return inferred && stage === 'start'
      ? 'No cache snapshot before Prefill: the runtime dump records the cache only after Prefill and at generation end'
      : 'No cache snapshot captured at this boundary';
  }
  openKv(turn: number, stage: string, token?: CapturedToken) {
    const snapshot = this.kvSnapshot(turn, stage, token);
    if (!snapshot) return;
    this.state.kvEntry.set({
      sessionId: this.state.captureId(),
      turn,
      moment: snapshot.moment,
      phase: snapshot.phase,
      step: snapshot.step,
      forward_id: snapshot.forward_id,
      runtime: snapshot.runtime,
    });
    this.state.selectTurn(turn);
    this.workspace.view.set('KV Diff');
    this.workspace.mode.set('Debug');
  }
  linkedBatch(token: CapturedToken | undefined) {
    return token?.batch == null
      ? undefined
      : this.state.preview.batches.find(
          (b) =>
            b.turn === this.selection()?.turn &&
            tokenMatchesBatch(token, b, 'target'),
        );
  }
  canInspect(token: CapturedToken | undefined) {
    return !!this.linkedBatch(token);
  }
  inspectToken(token: CapturedToken | undefined) {
    const batch = this.linkedBatch(token);
    if (!batch) return;
    this.workspace.graph(batch.turn, batch.phase);
    this.state.selectBatch(batch.batch);
  }
}
