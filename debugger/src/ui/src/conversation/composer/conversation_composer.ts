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

export const MAX_PROMPT_BYTES = 4 * 1024 * 1024;

/** The slice of WorkspaceStateService the composer reads and drives. */
export interface ComposerWorkspace {
  chatDraft(): string;
  setChatDraft(value: string): void;
  sessions: {refreshError(): string};
  /** The chat shown: whether the server allows sending to it now, and if not why. */
  chat(): {n: number; actions: {send: {allowed: boolean; reason: string}}} | null;
  runtime: {
    busy(): boolean;
    /** The attached Session. */
    selected(): object | undefined;
    start(operation: 'generate', chat: number, prompt: string): Promise<boolean>;
    stop(): Promise<void>;
  };
  jobs: {
    job(): {
      id?: string;
      operation: string;
      progress?: string;
      error?: string;
      does_not_fit?: unknown;
    } | null;
  };
}

/**
 * Message composer state: what blocks sending, what the notice says, and the key handling.
 * Whether the chat takes a turn now is the server's `send` action; this page only adds its own
 * request in flight, a stale list and the draft itself.
 */
export class ConversationComposerController {
  readonly promptError = computed(() =>
    new TextEncoder().encode(this.workspace.chatDraft()).length > MAX_PROMPT_BYTES
      ? 'Message exceeds the 4 MiB upload limit. Your full draft is retained.'
      : '',
  );
  private readonly sendAction = computed(() => this.workspace.chat()?.actions.send);
  /** Why the server refuses a turn to the chat now, while this page sends nothing itself. */
  readonly refusal = computed(() => {
    const action = this.sendAction();
    return !this.workspace.runtime.busy() && action && !action.allowed ? action.reason : '';
  });
  /** The message box takes input only while the chat can take a turn. */
  readonly inputDisabled = computed(
    () => this.workspace.runtime.busy() || !this.sendAction()?.allowed,
  );
  readonly canSend = computed(
    () =>
      !this.promptError() &&
      !!this.sendAction()?.allowed &&
      !this.workspace.sessions.refreshError() &&
      !this.workspace.runtime.busy() &&
      !!this.workspace.chatDraft().trim(),
  );
  readonly runtimeActivity = computed(() =>
    this.workspace.jobs.job()?.operation === 'initialize'
      ? 'Initializing runtime'
      : this.workspace.jobs.job()?.operation === 'chat'
        ? 'Starting chat'
        : // A generate's progress is its capture arriving, once the reply ended.
          this.workspace.jobs.job()?.progress || 'Generation in progress',
  );
  /** The turn whose refusal the text was changed after, by its job's id. */
  private readonly edited = signal<string | undefined>(undefined);
  /**
   * Why the last turn did not happen: neither side's context had room for it. The text stays in
   * the box; this shows under it until the text changes.
   */
  readonly doesNotFit = computed(() => {
    const job = this.workspace.jobs.job();
    return job?.does_not_fit && job.id !== this.edited() ? (job.error ?? '') : '';
  });
  /**
   * A Stop ends the chat, as the server ends it: it takes no more turns, and the Session's next
   * chat is a new one.
   */
  readonly stopLabel = 'Stop and end chat';
  readonly stopHint =
    'Stops both replies. The chat cannot continue after a Stop; start a new chat.';
  readonly sendNotice = computed(() =>
    !this.workspace.runtime.selected()
      ? 'This draft has no runtime context'
      : this.workspace.sessions.refreshError()
        ? 'Runner status is unknown'
        : this.workspace.runtime.busy()
          ? this.runtimeActivity()
          : !this.sendAction()?.allowed
            ? (this.sendAction()?.reason ?? '')
            : '',
  );

  constructor(private readonly workspace: ComposerWorkspace) {}

  draftInput(event: Event): void {
    const element = event.target as HTMLTextAreaElement;
    if (element.value !== this.workspace.chatDraft() && this.doesNotFit()) {
      this.edited.set(this.workspace.jobs.job()?.id);
    }
    this.workspace.setChatDraft(element.value);
    element.style.height = 'auto';
    element.style.height = Math.min(120, element.scrollHeight) + 'px';
  }
  draftKey(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void this.send();
    }
  }
  async send(): Promise<void> {
    const chat = this.workspace.chat();
    // The chat shown: a turn of one another page replaced is refused, never sent into the new.
    if (chat && this.canSend()) {
      await this.workspace.runtime.start('generate', chat.n, this.workspace.chatDraft());
    }
  }
  stop(): Promise<void> {
    return this.workspace.runtime.stop();
  }
}
