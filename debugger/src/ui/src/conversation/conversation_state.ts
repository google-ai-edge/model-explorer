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
  ElementRef,
  Injectable,
  Injector,
  NgZone,
  afterNextRender,
  computed,
  inject,
  signal,
} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {WorkspaceStateService} from '../state/workspace_state_service';
import {CaptureStore} from '../server/capture_store';
import {isTerminal, type RuntimeJob} from '../server/contracts/jobs';
import type {CapturedConversation, CapturedTurn, TurnComparison} from '../server/contracts/capture';
import {capturedPhaseText, hasPhaseTokens, type TokenPhase} from './conversation_content';
import {ConversationMemory} from './conversation_memory';
import {ConversationReadingController} from './conversation_reading_controller';
import {createConversationScrollbar} from './conversation_scrollbar';
import {streamingConversation, virtualConversation} from './conversation_virtualization';
import type {FlowNodeNavigator} from './flow_node_navigation';
import type {FlowVirtualizer} from './flow_virtualization';
import {readTurns, rejectedTurns} from './rejected_turn/rejected_turn';
import type {AlignmentMode} from './token_alignment';

/**
 * The turn a job shows as it runs, or null once the capture shows it: a turn that ended and was
 * kept, read or rejected, shows as the capture has it, once. A turn that did not fit did not
 * happen: no turn is added, its text stays in the composer.
 */
export function liveTurn<T extends Pick<RuntimeJob, 'status' | 'turn' | 'does_not_fit'>>(
  job: T,
  turns: readonly Pick<CapturedTurn, 'n'>[],
): T | null {
  if (job.does_not_fit) {
    return null;
  }
  return isTerminal(job.status) && turns.some((t) => t.n === job.turn) ? null : job;
}

/**
 * What the Chat and the Debug readings of one conversation share: its sides and turns as the
 * server captured them, the generation in progress, how each reply ended, the long-text layout
 * both readings draw and the reading's scroll position and its tail-following.
 *
 * The Debug reading's alignment lives here because the long-text layout of a turn depends on it,
 * and the thinking a Chat reader opened because the saved Token Diff view keeps it. Provided by
 * ConversationPanel, so its element, injector and zone are the page's whichever part injects it
 * first.
 */
@Injectable()
export class ConversationState {
  readonly workspace = inject(WorkspaceStateService);
  readonly state = inject(CaptureStore);
  readonly element = inject(ElementRef<HTMLElement>);
  readonly injector = inject(Injector);
  readonly zone = inject(NgZone);
  readonly dialog = inject(MatDialog);
  /** The chat's capture (capture_id.ts): what the reading and the saved views are kept by. */
  readonly contextKey = this.workspace.captureId() ?? '';
  /** What this chat's conversation was left with when the page last showed it. */
  readonly memory = inject(ConversationMemory);
  /** The Debug reading's alignment of the two sides' tokens. */
  readonly alignment = signal<AlignmentMode>('content');
  readonly thinkingExpanded = signal(new Set<string>());
  /** The long-text layout of both readings, set up by ConversationPanel. */
  virtualizer: FlowVirtualizer | null = null;
  /** The Debug reading's pinned node titles, set up by ConversationPanel. */
  nodeNavigator: FlowNodeNavigator | null = null;
  scrollbar: ReturnType<typeof createConversationScrollbar> | null = null;
  readonly reading = new ConversationReadingController({
    context: this.contextKey,
    positions: this.memory.readingPositions,
    root: () => this.element.nativeElement,
    mode: () => this.workspace.mode(),
    virtualizer: () => this.virtualizer,
    afterRender: (work) =>
      afterNextRender(() => this.zone.runOutsideAngular(work), {injector: this.injector}),
    afterRestore: () => {
      this.nodeNavigator?.mount();
      this.scrollbar?.mount();
    },
  });
  readonly runs = computed(
    () =>
      this.state.session()?.runs ??
      this.workspace.runtime.selected()?.runs ??
      this.workspace.sessions.items().find((s) => s.id === this.workspace.selectedId())?.runs ??
      [],
  );
  /** The turns the capture index read; a rejected turn has no tokens and is not among them. */
  readonly turns = computed(() => readTurns(this.state.session()?.turns ?? []));
  /** The turns the capture index rejected; they follow the read ones, each with the server's text. */
  readonly rejectedTurns = computed(() => rejectedTurns(this.state.session()?.turns ?? []));
  readonly sideIds = computed(() => this.runs().map((run) => run.id));
  readonly liveGeneration = computed(() => {
    const job = this.workspace.jobs.job();
    if (this.workspace.mode() !== 'Chat' || !job || !this.ofThisChat(job)) {
      return null;
    }
    return liveTurn(job, this.state.session()?.turns ?? []);
  });
  readonly virtualModels = computed(() => {
    const models = new Map(
      this.turns().map((t) => [
        t.n,
        virtualConversation(
          this.captured(t.n, 'ref'),
          this.captured(t.n, 'target'),
          t.comparison,
          this.alignment(),
          this.workspace.mode() === 'Debug',
        ),
      ]),
    );
    const job = this.liveGeneration();
    if (
      job &&
      Math.max(job.prompt?.length ?? 0, ...Object.values(job.output).map((t) => t.length)) > 16000
    ) {
      models.set(job.turn, streamingConversation(job.prompt ?? '', job.output));
    }
    return models;
  });
  readonly usesVirtual = computed(() =>
    [...this.virtualModels().keys()].some((n) => this.virtualInput(n) || this.virtualOutput(n)),
  );
  /** Whether the reading follows the tail of the generation in progress. */
  private followLive = true;
  private resumeAtTail = true;
  private lastJob = '';
  /** The replies in progress, by `job:side`, whose thinking has been opened. */
  private readonly liveThinkingOpened = new Set<string>();
  /** Captured conversation items by `turn:run`; the template asks for them on every check. */
  private readonly capturedByKey = computed(() => {
    const items = new Map<string, CapturedConversation>();
    for (const item of this.state.session()?.conversation ?? []) {
      items.set(`${item.turn}:${item.run}`, item);
    }
    return items;
  });

  async readFullText(
    turn: number,
    side: string,
    kind: 'input' | 'output' | 'thinking',
  ): Promise<void> {
    const {FullTextDialog} = await import('../widgets/full_text_dialog/full_text_dialog');
    const capture = this.captured(turn, side);
    this.dialog.open(FullTextDialog, {
      width: '760px',
      maxWidth: 'calc(100vw - 24px)',
      data: {
        title:
          (side === 'ref' ? 'Reference' : 'Target') +
          ' · ' +
          (kind === 'input' ? 'User message' : kind === 'thinking' ? 'Thinking' : 'Model response'),
        text:
          kind === 'thinking'
            ? capturedPhaseText(capture, 'thinking')
            : (capture?.[kind] ??
              (kind === 'input'
                ? this.liveGeneration()?.prompt
                : this.liveGeneration()?.output[side]) ??
              ''),
      },
      ariaLabel: 'Read full captured text',
    });
  }

  /**
   * Follows what the reader asks of the reading's scroll position; returns whether the event
   * asks to scroll down, which the Debug reading treats as a scroll even when nothing moves.
   */
  trackReadingIntent(event: WheelEvent | KeyboardEvent | PointerEvent): boolean {
    let down = false;
    if (
      (event instanceof WheelEvent && event.deltaY < 0) ||
      (event instanceof KeyboardEvent && ['ArrowUp', 'PageUp', 'Home'].includes(event.key))
    ) {
      this.followLive = false;
      this.resumeAtTail = false;
    }
    if (
      (event instanceof WheelEvent && event.deltaY > 0) ||
      (event instanceof KeyboardEvent && ['ArrowDown', 'PageDown', 'End'].includes(event.key))
    ) {
      this.resumeAtTail = true;
      this.readingScrolled();
      down = true;
    }
    if (event instanceof PointerEvent) {
      const r = this.element.nativeElement.querySelector('.reading')?.getBoundingClientRect();
      if (r && event.clientX >= r.right - 16) {
        this.followLive = false;
        this.resumeAtTail = true;
      }
    }
    return down;
  }

  readingScrolled(): void {
    const r = this.element.nativeElement.querySelector('.reading');
    if (this.resumeAtTail && r && r.scrollHeight - r.clientHeight - r.scrollTop < 80) {
      this.followLive = true;
    }
  }

  /**
   * Keeps the tail of the generation in progress in view while the reader follows it. Run in
   * an effect.
   */
  followLiveGeneration(): void {
    const job = this.liveGeneration();
    if (!job) {
      return;
    }
    if (job.id !== this.lastJob) {
      this.followLive = true;
    }
    this.lastJob = job.id;
    // Lay out virtual text before following its tail. Native scroll intent is
    // separate from content growth, which can change scrollHeight by many pages.
    afterNextRender(
      () =>
        this.zone.runOutsideAngular(() => {
          const current = this.element.nativeElement.querySelector('.reading');
          if (current) {
            this.virtualizer?.mount();
            if (this.followLive) {
              current.scrollTop = current.scrollHeight;
              this.virtualizer?.paint();
            }
          }
        }),
      {injector: this.injector},
    );
  }

  virtualOutput(turn: number): boolean {
    const model = this.virtualModels().get(turn);
    return (
      (model?.rows.length ?? 0) > 2048 &&
      (this.workspace.mode() === 'Debug' || !!model?.responseTextComplete)
    );
  }

  virtualPhase(turn: number, phase: TokenPhase): boolean {
    return (
      this.virtualOutput(turn) &&
      ['ref', 'target'].some((side) => hasPhaseTokens(this.captured(turn, side), phase))
    );
  }

  virtualInput(turn: number): boolean {
    const t = this.virtualModels().get(turn);
    return Math.max(t?.inputTokens.ref.length ?? 0, t?.inputTokens.target.length ?? 0) > 2048;
  }

  /** The turn after a read turn, which may be a rejected one; undefined after the last. */
  turnAfter(index: number): CapturedTurn | undefined {
    return this.turns()[index + 1] ?? this.rejectedTurns()[0];
  }

  /** Whether a job is a turn of this chat. */
  private ofThisChat(
    job: {operation: string; session_id: string; chat?: number | null} | null,
  ): boolean {
    return (
      job?.operation === 'generate' &&
      job.session_id === this.workspace.captureRecordId() &&
      job.chat === this.workspace.chat()?.n
    );
  }

  /**
   * What a side of the turn in progress says of it: the job's status, on a side that failed
   * when the server names those, or its progress while it runs.
   */
  liveStatus(job: RuntimeJob, side: string): string {
    if (job.failed_sides?.length && !job.failed_sides.includes(side)) {
      return '';
    }
    return job.status_text || (this.workspace.runtime.busy() ? job.progress || 'Generating…' : '');
  }

  thinkingOpen(turn: number, run: string): boolean {
    return this.thinkingExpanded().has(`${turn}:${run}`);
  }

  /**
   * Opens a reply's thinking as it starts to stream, once per reply, under its turn: the turn
   * once captured shows it open, or as the reader left it, so nothing jumps. Run in an effect.
   */
  openLiveThinking(): void {
    const job = this.liveGeneration();
    if (!job) {
      return;
    }
    const opened = Object.keys(job.reasoning).filter(
      (run) => job.reasoning[run] && !this.liveThinkingOpened.has(`${job.id}:${run}`),
    );
    if (!opened.length) {
      return;
    }
    for (const run of opened) {
      this.liveThinkingOpened.add(`${job.id}:${run}`);
    }
    this.thinkingExpanded.update(
      (s) => new Set([...s, ...opened.map((run) => `${job.turn}:${run}`)]),
    );
  }

  toggleThinking(turn: number, run: string): void {
    const key = `${turn}:${run}`;
    this.thinkingExpanded.update((s) => {
      const n = new Set(s);
      if (n.has(key)) {
        n.delete(key);
      } else {
        n.add(key);
      }
      return n;
    });
  }

  turnEnding(turn: number): string {
    const job = this.workspace.jobs.job();
    if (job && this.ofThisChat(job) && job.turn === turn) {
      if (!['completed', 'failed', 'cancelled'].includes(job.status)) {
        return 'Generation in progress';
      }
      if (job.status === 'cancelled') {
        return 'Generation stopped';
      }
      if (job.status === 'failed') {
        return 'Generation failed';
      }
    }
    const reasons = this.runs().map((run) => this.captured(turn, run.id)?.stop_reason);
    if (reasons.some((r) => r === 'cancelled')) {
      return 'Generation stopped';
    }
    if (reasons.some((r) => r === 'error')) {
      return 'Generation failed';
    }
    // Complete when the server says why every side's reply ended.
    return reasons.length && this.runs().every((run) => this.stageEnding(turn, run.id))
      ? 'Conversation complete'
      : 'End of captured conversation';
  }

  /** The chat's notice from the server: why it ended, once it did. */
  chatNotice(): string {
    return this.workspace.chat()?.notice ?? '';
  }

  captured(turn: number, run: string): CapturedConversation | undefined {
    return this.capturedByKey().get(`${turn}:${run}`);
  }

  hasTokenCapture(turn: number, run: string): boolean {
    return this.captured(turn, run)?.tokens !== undefined;
  }

  /** The server's comparison of a read turn's two sides. */
  comparison(turn: number): TurnComparison | undefined {
    return this.turns().find((t) => t.n === turn)?.comparison;
  }

  /** Why one side's reply ended, as the server says it; empty when it does not. */
  stageEnding(turn: number, side: string): string {
    return this.captured(turn, side)?.stop_text ?? '';
  }
}
