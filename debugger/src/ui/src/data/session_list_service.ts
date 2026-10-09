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
import {
  SessionConfig,
  SessionListing,
  SessionSummary,
} from './contracts/session_summary';
import {ReportApiService} from './report_api_service';
import {sessionExecutionPhase, sessionIsActive} from './session_status';

/** Home-page metadata is independent of tensor/graph loading. */
@Injectable({providedIn: 'root'})
export class SessionListService {
  private loadRequest = 0;
  private refreshing = false;
  private mutations = 0;
  private readonly api = inject(ReportApiService);
  readonly listing = signal<SessionListing | null>(null);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly refreshError = signal('');
  readonly allItems = computed(() => this.listing()?.sessions ?? []);
  readonly items = computed(() =>
    this.allItems().filter((s) => !s.parent_session_id),
  );
  constructor() {
    void this.load();
    // UI refresh only; Runner heartbeats belong to the Server.
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') void this.load(true);
    }, 3000);
    const visible = () => {
      if (document.visibilityState === 'visible') void this.load(true);
    };
    document.addEventListener('visibilitychange', visible);
    inject(DestroyRef).onDestroy(() => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', visible);
    });
  }
  async load(quiet = false) {
    if (this.mutations || (quiet && (this.loading() || this.refreshing)))
      return;
    this.refreshing = true;
    const request = ++this.loadRequest;
    this.loading.set(!quiet);
    if (!quiet) this.error.set('');
    try {
      const listing = await this.api.sessions();
      if (request === this.loadRequest) {
        this.listing.set(listing);
        this.error.set('');
        this.refreshError.set('');
      }
    } catch (error) {
      if (request === this.loadRequest) {
        const message = error instanceof Error ? error.message : String(error);
        if (this.listing()) this.refreshError.set(message);
        else this.error.set(message);
      }
    } finally {
      if (request === this.loadRequest) {
        this.loading.set(false);
        this.refreshing = false;
      }
    }
  }
  executionPhase = sessionExecutionPhase;
  active = sessionIsActive;
  async manage(
    operation:
      | 'create'
      | 'update'
      | 'duplicate'
      | 'chat'
      | 'chat-config'
      | 'delete'
      | 'restore',
    payload: Partial<SessionConfig> & {id?: string},
  ) {
    return this.change(
      () => this.api.manageSession(operation, payload),
      operation === 'delete',
    );
  }
  async rename(id: string, name: string) {
    return this.change(() => this.api.renameSession(id, name));
  }
  private async change(request: () => Promise<SessionSummary>, remove = false) {
    // A poll started before this mutation must never put the old row back.
    ++this.loadRequest;
    this.refreshing = false;
    this.loading.set(false);
    ++this.mutations;
    try {
      const updated = await request();
      this.listing.update((value) =>
        value
          ? {
              ...value,
              sessions: remove
                ? value.sessions.filter((s) => s.id !== updated.id)
                : value.sessions.some((s) => s.id === updated.id)
                  ? value.sessions.map((s) =>
                      s.id === updated.id ? updated : s,
                    )
                  : [...value.sessions, updated],
            }
          : value,
      );
      return updated;
    } finally {
      --this.mutations;
    }
  }
}
