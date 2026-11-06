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

import {computed, DestroyRef, inject, Injectable, signal} from '@angular/core';
import {SessionsApi} from './sessions_api';
import type {Session} from './contracts/capture';

const EMPTY_SESSION: Session = {
  model: '',
  runs: [],
  turns: [],
  batches: [],
};

/** The Session evidence of the attached capture, loaded under one request lifetime. */
@Injectable({providedIn: 'root'})
export class CaptureStore {
  private readonly api = inject(SessionsApi);
  private readonly selectedCaptureId = signal<string | null>(null);
  readonly captureId = this.selectedCaptureId.asReadonly();
  private readonly sessionRequest = new RequestLifetime();
  private selectionCaptureId: string | null = null;
  readonly session = signal<Session | null>(null);
  readonly loading = signal(false);
  readonly error = signal('');
  /** The batch the reader observes: null until they choose one (selectTurn), never a default, so
   *  that an observation is only ever theirs. */
  readonly batchId = signal<number | null>(null);
  get preview(): Session {
    return this.session() ?? EMPTY_SESSION;
  }
  readonly batch = computed(() => this.preview.batches.find((b) => b.batch === this.batchId()));

  constructor() {
    inject(DestroyRef).onDestroy(() => this.sessionRequest.cancel());
  }

  attach(captureId: string | null): void {
    this.selectedCaptureId.set(captureId);
    this.clear();
  }

  clear(): void {
    this.sessionRequest.cancel();
    this.session.set(null);
    this.loading.set(false);
    this.error.set('');
    this.selectionCaptureId = null;
    this.batchId.set(null);
  }

  async load(): Promise<void> {
    const captureId = this.captureId();
    const request = this.sessionRequest.start();
    this.session.set(null);
    this.error.set('');
    this.loading.set(true);
    const current = () => request.current() && captureId === this.captureId();
    try {
      const session = await this.api.session(captureId, request.signal);
      if (!current()) {
        return;
      }
      const selectionValid =
        this.selectionCaptureId === captureId &&
        session.batches.some((batch) => batch.batch === this.batchId());
      if (!selectionValid) {
        this.batchId.set(null);
      }
      this.selectionCaptureId = captureId;
      this.session.set(session);
    } catch (error) {
      if (current()) {
        this.error.set(String(error));
      }
    } finally {
      if (current()) {
        this.loading.set(false);
      }
      request.finish();
    }
  }

  selectTurn(value: number): void {
    const candidates = this.preview.batches.filter((b) => b.turn === value);
    const next = candidates.find((b) => b.phase === this.batch()?.phase) || candidates[0];
    if (next) {
      this.batchId.set(next.batch);
    }
  }
}

interface RequestHandle {
  signal: AbortSignal;
  current: () => boolean;
  finish: () => void;
  cancel: () => void;
}

/** Aborting also invalidates the response token, even when a transport ignores AbortSignal. */
class RequestLifetime {
  private controller?: AbortController;
  cancel(): void {
    this.controller?.abort();
    this.controller = undefined;
  }
  start(): RequestHandle {
    this.cancel();
    const controller = new AbortController();
    this.controller = controller;
    const finish = () => {
      if (this.controller === controller) {
        this.controller = undefined;
      }
    };
    return {
      signal: controller.signal,
      current: () => this.controller === controller && !controller.signal.aborted,
      finish,
      cancel: () => {
        controller.abort();
        finish();
      },
    };
  }
}
