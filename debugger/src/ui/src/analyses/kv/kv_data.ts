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

import type {KvApi} from '../../server/kv_api';
import type {
  KvAnalysis,
  KvRangeResponse,
  KvRangeBin,
  KvPositionAlignment,
} from '../../server/contracts/kv';
import type {KvViewIdentity} from './kv_view_state';
import {LatestRequest} from '../../server/latest_request';

interface KvContextQuery extends KvViewIdentity {
  contextId: string;
  alignment?: KvPositionAlignment | null;
}
export interface KvRangeQuery extends KvContextQuery {
  range: KvRangeBin;
  bins: number;
  kind: string;
  head: string;
  heads?: number[] | null;
  /** How the bins follow the range under the other alignment, which is asked too (KvApi). */
  capacity?: number | 'fixed' | null;
}

/** Owns KV network lifecycles; the panel supplies current query values and renders resource state. */
export class KvDataController {
  readonly metadata = new LatestRequest<KvAnalysis>();
  readonly scale = new LatestRequest<KvRangeResponse>();
  readonly range = new LatestRequest<KvRangeResponse>();
  constructor(private readonly api: Pick<KvApi, 'kvMetadata' | 'kvRange'>) {}

  loadMetadata(identity: KvViewIdentity | null, receive: (analysis: KvAnalysis) => void): () => void {
    return this.metadata.load(
      identity ? (signal) => this.api.kvMetadata(identity.captureId, identity.turn, signal) : null,
      receive,
    );
  }
  private async readRange(query: KvRangeQuery, signal: AbortSignal): Promise<KvRangeResponse> {
    const data = await this.api.kvRange(
      query.captureId,
      query.turn,
      query.contextId,
      query.range.start,
      query.range.end,
      query.bins,
      query.kind,
      query.head,
      signal,
      query.alignment,
      query.heads,
      null,
      query.capacity ?? null,
    );
    this.checkRange(data, query.contextId, query.range);
    return data;
  }
  private checkRange(data: KvRangeResponse, contextId: string, range: KvRangeBin): void {
    if (data.context_id !== contextId || data.start !== range.start || data.end !== range.end) {
      throw new Error('KV response did not match the requested observation and position range.');
    }
  }
  loadScale(query: KvRangeQuery | null): () => void {
    // Full-observation colors remain stable across zoom and chart changes.
    return this.scale.load(query ? (signal) => this.readRange({...query, bins: 1}, signal) : null);
  }
  loadRange(query: KvRangeQuery | null): () => void {
    // Inspector resizing changes the visible bin count repeatedly during a drag.
    return this.range.load(
      query ? (signal) => this.readRange(query, signal) : null,
      undefined,
      100,
    );
  }
  destroy(): void {
    for (const request of [this.metadata, this.scale, this.range]) {
      request.destroy();
    }
  }
}
