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

import {computed} from '@angular/core';

export const MAX_PROMPT_BYTES = 4 * 1024 * 1024;

/** The slice of WorkspaceStateService the composer reads and drives. */
export interface ComposerWorkspace {
  chatDraft(): string;
  setChatDraft(value: string): void;
  sessions: {refreshError(): string};
  runtime: {
    capabilities(): {available: boolean} | null;
    active(): boolean;
    busy(): boolean;
    selected(): unknown;
    execution(): {phase: string} | undefined;
    job(): {operation: string} | null;
    start(operation: 'generate', prompt: string): Promise<boolean>;
    stop(): Promise<void>;
  };
}

/** Message composer state: what blocks sending, what the notice says, and the key handling. */
export class ConversationComposerController {
  constructor(private readonly workspace: ComposerWorkspace) {}
  readonly promptError = computed(() =>
    new TextEncoder().encode(this.workspace.chatDraft()).length >
    MAX_PROMPT_BYTES
      ? 'Message exceeds the 4 MiB upload limit. Your full draft is retained.'
      : '',
  );
  readonly canSend = computed(
    () =>
      !this.promptError() &&
      !!this.workspace.runtime.capabilities()?.available &&
      this.workspace.runtime.active() &&
      !this.workspace.sessions.refreshError() &&
      !this.workspace.runtime.busy() &&
      !!this.workspace.chatDraft().trim(),
  );
  readonly runtimeActivity = computed(() =>
    this.workspace.runtime.job()?.operation === 'initialize'
      ? 'Initializing runtime'
      : this.workspace.runtime.job()?.operation === 'prepare'
        ? 'Preparing tensor capture'
        : 'Generation in progress',
  );
  readonly stopLabel = computed(() => 'Stop generation');
  readonly sendNotice = computed(() =>
    !this.workspace.runtime.selected()
      ? 'This draft has no runtime context'
      : this.workspace.sessions.refreshError()
        ? 'Runner status is unknown'
        : !this.workspace.runtime.capabilities()?.available
          ? 'Runtime unavailable'
          : this.workspace.runtime.execution()?.phase === 'ending'
            ? 'Ending Session…'
            : this.workspace.runtime.execution()?.phase === 'starting'
              ? 'Starting Session…'
              : this.workspace.runtime.busy()
                ? this.runtimeActivity()
                : !this.workspace.runtime.active()
                  ? 'Turn the Model Server on to send a message'
                  : '',
  );
  draftInput(event: Event) {
    const element = event.target as HTMLTextAreaElement;
    this.workspace.setChatDraft(element.value);
    element.style.height = 'auto';
    element.style.height = Math.min(120, element.scrollHeight) + 'px';
  }
  draftKey(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void this.send();
    }
  }
  async send() {
    if (this.canSend())
      await this.workspace.runtime.start(
        'generate',
        this.workspace.chatDraft(),
      );
  }
  stop() {
    return this.workspace.runtime.stop();
  }
}
