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

import {signal} from '@angular/core';
import type {
  KvAnalysis,
  KvFindResponse,
  KvRangeBin,
  KvRangeDetails,
  KvRangeResponse,
} from '../../../data/contracts/kv';
import type {ReportApiService} from '../../../data/report_api_service';
import type {KvRangeSelection} from './kv_range_types';
import type {KvViewIdentity} from './kv_view_state';

/** One current request. Abort-insensitive adapters still cannot publish late data or errors. */
export class KvRequest<T> {
  readonly data = signal<T | null>(null);
  readonly loading = signal(false);
  readonly error = signal('');
  private cancelCurrent: (() => void) | null = null;

  load(
    read: ((signal: AbortSignal) => Promise<T>) | null,
    receive?: (value: T) => void,
    delay = 0,
  ) {
    this.cancelCurrent?.();
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const cancel = () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
      if (this.cancelCurrent === cancel) this.loading.set(false);
    };
    this.cancelCurrent = cancel;
    this.data.set(null);
    this.error.set('');
    this.loading.set(!!read);
    if (read) {
      const start = async () => {
        try {
          const value = await read(controller.signal);
          if (controller.signal.aborted) return;
          this.data.set(value);
          receive?.(value);
        } catch (error) {
          if (!controller.signal.aborted)
            this.error.set(
              error instanceof Error ? error.message : String(error),
            );
        } finally {
          if (!controller.signal.aborted) this.loading.set(false);
        }
      };
      if (delay) timer = setTimeout(() => void start(), delay);
      else void start();
    }
    return cancel;
  }
  destroy() {
    this.cancelCurrent?.();
  }
}

interface KvContextQuery extends KvViewIdentity {
  contextId: string;
}
export interface KvRangeQuery extends KvContextQuery {
  range: KvRangeBin;
  bins: number;
  kind: string;
  head: string;
  formula: string;
}
export interface KvSelectionQuery extends KvContextQuery {
  selection: KvRangeSelection;
}
export interface KvFindQuery extends KvContextQuery {
  formula: string;
  page: number;
}
type KvApi = Pick<
  ReportApiService,
  'kvMetadata' | 'kvRange' | 'kvSelection' | 'kvFind'
>;

/** Owns KV network lifecycles; the panel supplies current query values and renders resource state. */
export class KvDataController {
  readonly metadata = new KvRequest<KvAnalysis>();
  readonly scale = new KvRequest<KvRangeResponse>();
  readonly range = new KvRequest<KvRangeResponse>();
  readonly selection = new KvRequest<KvRangeDetails>();
  readonly find = new KvRequest<KvFindResponse>();
  constructor(private readonly api: KvApi) {}

  loadMetadata(
    identity: KvViewIdentity | null,
    receive: (analysis: KvAnalysis) => void,
  ) {
    return this.metadata.load(
      identity
        ? (signal) =>
            this.api.kvMetadata(identity.captureId, identity.turn, signal)
        : null,
      receive,
    );
  }
  private async readRange(query: KvRangeQuery, signal: AbortSignal) {
    const data = await this.api.kvRange(
      query.captureId,
      query.turn,
      query.contextId,
      query.range.start,
      query.range.end,
      query.bins,
      query.kind,
      query.head,
      query.formula,
      signal,
    );
    this.checkRange(data, query.contextId, query.range);
    return data;
  }
  private checkRange(
    data: KvRangeResponse | KvRangeDetails,
    contextId: string,
    range: KvRangeBin,
  ) {
    if (
      data.context_id !== contextId ||
      data.start !== range.start ||
      data.end !== range.end
    )
      throw new Error(
        'KV response did not match the requested observation and position range.',
      );
  }
  loadScale(query: KvRangeQuery | null) {
    // Full-observation colors remain stable across zoom, Find and chart changes.
    return this.scale.load(
      query
        ? (signal) => this.readRange({...query, bins: 1, formula: ''}, signal)
        : null,
    );
  }
  loadRange(query: KvRangeQuery | null) {
    // Inspector resizing changes the visible bin count repeatedly during a drag.
    return this.range.load(
      query ? (signal) => this.readRange(query, signal) : null,
      undefined,
      100,
    );
  }
  loadSelection(query: KvSelectionQuery | null) {
    return this.selection.load(
      query
        ? async (signal) => {
            const data = await this.api.kvSelection(
              query.captureId,
              query.turn,
              query.contextId,
              query.selection.start,
              query.selection.end,
              query.selection.layer,
              signal,
            );
            this.checkRange(data, query.contextId, query.selection);
            return data;
          }
        : null,
    );
  }
  loadFind(query: KvFindQuery | null, receive: (data: KvFindResponse) => void) {
    return this.find.load(
      query
        ? (signal) =>
            this.api.kvFind(
              query.captureId,
              query.turn,
              query.contextId,
              query.formula,
              query.page * 40,
              signal,
            )
        : null,
      receive,
    );
  }
  destroy() {
    for (const request of [
      this.metadata,
      this.scale,
      this.range,
      this.selection,
      this.find,
    ])
      request.destroy();
  }
}
