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

import {inject, Injectable} from '@angular/core';
import {AnalysisCache} from './analysis_cache';
import {ServerClient} from './server_client';
import type {
  KvAnalysis,
  KvHeadsEvidence,
  KvPositionAlignment,
  KvRangeResponse,
} from './contracts/kv';

/** /api/kv-analysis, /api/kv-range and /api/kv-head: the KV cache comparison of a turn. With an
 *  alignment, a request names only its stage and alignment kind; the server reads each column at
 *  the positions its own comparison gives the row. kv-head answers every KV head at once. */
@Injectable({providedIn: 'root'})
export class KvApi {
  private readonly client = inject(ServerClient);
  private readonly cache = inject(AnalysisCache);

  kvMetadata(captureId: string | null, turn: number, signal: AbortSignal): Promise<KvAnalysis> {
    return this.client.getCapture<KvAnalysis>(
      captureId,
      `kv-analysis?mode=metadata&turn=${turn}`,
      signal,
    );
  }
  /** A kind's layers binned over columns; kept, and with the whole Decode asked, the whole Decode
   *  under the other alignment asked beside it (AnalysisCache): in its bins at the chart's
   *  `capacity`, or in as many bins when 'fixed' (one bin, a statistic of the stage, needs
   *  neither). */
  kvRange(
    captureId: string | null,
    turn: number,
    context: string,
    start: number,
    end: number,
    bins: number,
    kind: string,
    head: string,
    signal: AbortSignal,
    alignment: KvPositionAlignment | null = null,
    heads: number[] | null = null,
    layers: number[] | null = null,
    capacity: number | 'fixed' | null = null,
  ): Promise<KvRangeResponse> {
    const read = (
      rangeStart: number,
      rangeEnd: number,
      rangeBins: number,
      rangeAlignment: KvPositionAlignment | null,
      rangeSignal?: AbortSignal,
    ): Promise<KvRangeResponse> => {
      const params = new URLSearchParams({
        turn: String(turn),
        context,
        start: String(rangeStart),
        end: String(rangeEnd),
        bins: String(rangeBins),
        kind,
        head,
      });
      if (heads) {
        params.set('heads', JSON.stringify(heads));
      }
      if (layers) {
        params.set('layers', JSON.stringify(layers));
      }
      aligned(params, rangeAlignment);
      const path = `kv-range?${params}`;
      return this.cache.read(
        JSON.stringify([captureId, path]),
        (own) => this.client.getCapture<KvRangeResponse>(captureId, path, own),
        rangeSignal,
      );
    };
    const answer = read(start, end, bins, alignment, signal);
    return this.cache.prefetchRange(
      answer,
      {
        turn,
        stage: alignment?.stage ?? '',
        alignment: alignment?.alignment,
        analysis: 'kv',
        start,
        end,
        bins,
        capacity,
      },
      (twin, twinBins) =>
        read(twin.start, twin.end, twinBins, {
          stage: 'decode',
          alignment: twin.alignment,
          ...twin.positions,
        }),
      signal,
    );
  }
  kvHead(
    captureId: string | null,
    turn: number,
    contextId: string,
    layer: number,
    kind: string,
    position: number,
    batch: number | null,
    signal: AbortSignal,
    alignment: KvPositionAlignment | null = null,
  ): Promise<KvHeadsEvidence> {
    const params = new URLSearchParams({
      turn: String(turn),
      context_id: contextId,
      layer: String(layer),
      kind,
      position: String(position),
    });
    if (batch !== null) {
      params.set('batch', String(batch));
    }
    aligned(params, alignment);
    return this.client.getCapture<KvHeadsEvidence>(captureId, `kv-head?${params}`, signal);
  }
}

function aligned(params: URLSearchParams, alignment: KvPositionAlignment | null): void {
  if (!alignment) {
    return;
  }
  params.set('stage', alignment.stage);
  params.set('alignment', alignment.alignment);
}
