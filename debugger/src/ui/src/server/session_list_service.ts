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
import {SessionListing, SessionConfig, SessionSummary} from './contracts/sessions';

/** A Session file being imported: how much of it the server has received. */
export interface SessionImport {
  name: string;
  loaded: number;
  total: number;
}

/** Home-page metadata is independent of tensor/graph loading. */
@Injectable({providedIn: 'root'})
export class SessionListService {
  private loadRequest = 0;
  private refreshing = false;
  private mutations = 0;
  private readonly api = inject(SessionsApi);
  readonly listing = signal<SessionListing | null>(null);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly refreshError = signal('');
  readonly items = computed(() => this.listing()?.sessions ?? []);
  /** The Session file being imported, while it is: the list's top row shows its progress. */
  readonly importing = signal<SessionImport | null>(null);
  constructor() {
    void this.load();
    // UI refresh only; Runner heartbeats belong to the Server.
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') {
        void this.load(true);
      }
    }, 3000);
    const visible = () => {
      if (document.visibilityState === 'visible') {
        void this.load(true);
      }
    };
    document.addEventListener('visibilitychange', visible);
    inject(DestroyRef).onDestroy(() => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', visible);
    });
  }
  async load(quiet = false): Promise<void> {
    if (this.mutations || (quiet && (this.loading() || this.refreshing))) {
      return;
    }
    this.refreshing = true;
    const request = ++this.loadRequest;
    this.loading.set(!quiet);
    if (!quiet) {
      this.error.set('');
    }
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
        if (this.listing()) {
          this.refreshError.set(message);
        } else {
          this.error.set(message);
        }
      }
    } finally {
      if (request === this.loadRequest) {
        this.loading.set(false);
        this.refreshing = false;
      }
    }
  }
  async manage(
    operation: 'create' | 'delete',
    payload: Partial<SessionConfig> & {id?: string},
  ): Promise<SessionSummary> {
    return this.change(() => this.api.manageSession(operation, payload), operation === 'delete');
  }
  async rename(id: string, name: string): Promise<SessionSummary> {
    return this.change(() => this.api.renameSession(id, name));
  }
  /** Where the browser saves a Session that has ended as one file. */
  exportUrl(id: string): string {
    return this.api.exportUrl(id);
  }
  /** Imports a Session another computer exported; it joins the list once the server has it. The
   *  list keeps polling meanwhile: an upload of several GB takes a while. */
  async importSession(file: File): Promise<SessionSummary> {
    const progress = (loaded: number, total: number) =>
      this.importing.set({name: file.name, loaded, total});
    progress(0, file.size);
    try {
      const imported = await this.api.importSession(file, {onProgress: progress});
      // A poll started before the import must not drop the new row.
      ++this.loadRequest;
      this.refreshing = false;
      this.put(imported);
      return imported;
    } finally {
      this.importing.set(null);
    }
  }
  private async change(
    request: () => Promise<SessionSummary>,
    remove = false,
  ): Promise<SessionSummary> {
    // A poll started before this mutation must never put the old row back.
    ++this.loadRequest;
    this.refreshing = false;
    this.loading.set(false);
    ++this.mutations;
    try {
      const updated = await request();
      this.put(updated, remove);
      return updated;
    } finally {
      --this.mutations;
    }
  }
  /** Puts a record the server answered into the list: replaced, added first (the list is the
   *  last active first), or removed. */
  private put(updated: SessionSummary, remove = false): void {
    this.listing.update((value) =>
      value
        ? {
            ...value,
            sessions: remove
              ? value.sessions.filter((s) => s.id !== updated.id)
              : value.sessions.some((s) => s.id === updated.id)
                ? value.sessions.map((s) => (s.id === updated.id ? updated : s))
                : [updated, ...value.sessions],
          }
        : value,
    );
  }
}
