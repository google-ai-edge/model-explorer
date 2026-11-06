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

import {computed, inject, Injectable, signal, type WritableSignal} from '@angular/core';
import {JobsApi} from './jobs_api';
import {isTerminal} from './contracts/jobs';
import type {RuntimeJob, RuntimeJobEvent} from './contracts/jobs';

/** What a 'delta' event adds to a job: its text, to the side's reply or to its thinking, as its
 *  channel says. */
function appended(job: RuntimeJob, delta: RuntimeJobEvent): Partial<RuntimeJob> {
  const side = delta.runId ?? '';
  const field = delta.channel === 'reasoning' ? 'reasoning' : 'output';
  const said = job[field] ?? {};
  return {[field]: {...said, [side]: (said[side] ?? '') + (delta.text ?? '')}};
}

/** The attachment a followed job belongs to. */
export interface JobOwner {
  /** Whether the attachment that started following the job is still the current one. */
  current(): boolean;
  /** The owner's task line: connection trouble is shown here, and cleared once updates flow. */
  readonly error: WritableSignal<string>;
  /** The job reached a terminal status; the owner reads its final state. */
  finish(id: string): void;
}

/** Task progress: the current job, followed over its event stream, with a polling fallback. */
@Injectable({providedIn: 'root'})
export class JobsService {
  private readonly api = inject(JobsApi);
  readonly job = signal<RuntimeJob | null>(null);
  readonly generating = computed(() => {
    const current = this.job();
    return current?.operation === 'generate' && !isTerminal(current.status);
  });
  /** The turn's reply has ended and its capture is arriving: a generate's only progress. */
  readonly retrieving = computed(() => this.generating() && !!this.job()?.progress);
  private source: EventSource | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  /** Stops following and forgets the job, for a new attachment. */
  reset(): void {
    this.reconnectAttempts = 0;
    this.stopWatching();
    this.job.set(null);
  }
  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
  private stopWatching(): void {
    this.clearReconnectTimer();
    this.source?.close();
    this.source = null;
  }
  watch(job: RuntimeJob, owner: JobOwner): void {
    this.stopWatching();
    let sequence = job.sequence;
    const source = this.api.events(job.id, sequence);
    this.source = source;
    const current = () => owner.current() && source === this.source;
    const connected = () => {
      this.clearReconnectTimer();
      this.reconnectAttempts = 0;
      owner.error.set('');
    };
    source.onmessage = (event) => {
      if (!current()) {
        return;
      }
      let data: RuntimeJobEvent;
      try {
        data = JSON.parse(event.data) as RuntimeJobEvent;
      } catch {
        owner.error.set('Invalid task update. Reconnecting…');
        source.close();
        void this.recover(job, owner, source);
        return;
      }
      connected();
      if (data.sequence <= sequence) {
        return;
      }
      sequence = data.sequence;
      this.job.update((value) =>
        value
          ? {
              ...value,
              sequence,
              ...(data.type === 'status'
                ? {
                    status: data.status,
                    error: data.error ?? '',
                    note: data.note ?? '',
                    status_text: data.status_text ?? '',
                    failed_sides: data.failed_sides,
                  }
                : {}),
              ...(data.type === 'progress' ? {progress: data.message} : {}),
              ...(data.type === 'delta' ? appended(value, data) : {}),
            }
          : value,
      );
      if (data.type === 'status' && data.status && isTerminal(data.status)) {
        this.stopWatching();
        owner.finish(job.id);
      }
    };
    source.onopen = () => {
      if (current()) {
        connected();
      }
    };
    source.onerror = () => {
      if (!current()) {
        return;
      }
      // A stream the browser closed for good (HTTP 4xx) never reconnects by itself.
      if (source.readyState === EventSource.CLOSED) {
        void this.recover(job, owner, source);
        return;
      }
      const interrupted = () =>
        owner.error.set('Connection interrupted. Reconnecting to the local task…');
      // The server rotates healthy streams every 25 seconds; allow EventSource to reconnect.
      if (this.reconnectTimer === null) {
        this.reconnectTimer = setTimeout(() => {
          if (!current()) {
            return;
          }
          this.reconnectTimer = null;
          if (source.readyState !== EventSource.OPEN) {
            interrupted();
          }
        }, 5000);
      }
      void this.api
        .job(job.id)
        .then((jobState) => {
          if (!current()) {
            return;
          }
          if (isTerminal(jobState.status)) {
            this.stopWatching();
            owner.finish(job.id);
          }
        })
        .catch(() => {
          if (current() && source.readyState !== EventSource.OPEN) {
            interrupted();
          }
        });
    };
  }
  private watching(owner: JobOwner, source: EventSource): boolean {
    return owner.current() && source === this.source;
  }
  /** Re-create a closed stream with backoff; after five failures poll the job until it ends. */
  private async recover(job: RuntimeJob, owner: JobOwner, source: EventSource): Promise<void> {
    this.clearReconnectTimer();
    let state: RuntimeJob;
    try {
      state = await this.api.job(job.id);
    } catch (error) {
      if (!this.watching(owner, source)) {
        return;
      }
      const status = (error as {status?: unknown}).status;
      if (status === 400 || status === 404) {
        this.stopWatching();
        this.job.set(null);
        owner.error.set('The task no longer exists on the server. Reload the Session.');
        return;
      }
      owner.error.set('Connection interrupted. Retrying the local task…');
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        if (this.watching(owner, source)) {
          void this.recover(job, owner, source);
        }
      }, 2000);
      return;
    }
    if (!this.watching(owner, source)) {
      return;
    }
    this.job.update((value) =>
      value && value.id === state.id
        ? {
            ...value,
            sequence: Math.max(value.sequence, state.sequence),
            status: state.status,
            error: state.error,
            note: state.note,
            status_text: state.status_text,
            failed_sides: state.failed_sides,
            output: state.output,
            reasoning: state.reasoning,
            progress: state.progress ?? value.progress,
          }
        : value,
    );
    if (isTerminal(state.status)) {
      this.stopWatching();
      owner.finish(job.id);
      return;
    }
    const attempt = ++this.reconnectAttempts;
    if (attempt <= 5) {
      owner.error.set('Connection interrupted. Reconnecting to the local task…');
      this.reconnectTimer = setTimeout(
        () => {
          this.reconnectTimer = null;
          if (this.watching(owner, source)) {
            this.watch({...job, sequence: this.job()?.sequence ?? job.sequence}, owner);
          }
        },
        Math.min(16000, 1000 * 2 ** (attempt - 1)),
      );
    } else {
      owner.error.set('Live updates are unavailable; checking the task every few seconds.');
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        if (this.watching(owner, source)) {
          void this.recover(job, owner, source);
        }
      }, 2000);
    }
  }
}
