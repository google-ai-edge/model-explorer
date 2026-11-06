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
import type {KvPositionAlignment, KvRangeBin} from './contracts/kv';
import type {
  PositionAnalysis,
  PositionRangeResponse,
  PositionTensorRequest,
  PositionTensorResponse,
} from './contracts/position_analysis';

/** What a position-range request names besides its stage: the columns, how many bins, the sync
 *  points (null for every one) and how theirs combine, and the layers (null for every one). */
export interface PositionRangeQuery extends KvRangeBin {
  bins: number;
  anchors: readonly string[] | null;
  reduction: 'min' | 'max' | 'mean';
  layers: readonly number[] | null;
}

/** /api/position-analysis, /api/position-range and /api/position-tensor: a stage's intermediate
 *  tensors. A request names its turn and stage, and with an alignment its kind, as the KV requests
 *  do; the server reads each column at the positions its own comparison gives it. */
@Injectable({providedIn: 'root'})
export class PositionAnalysisApi {
  private readonly client = inject(ServerClient);
  private readonly cache = inject(AnalysisCache);
  /** A stage's places and cells; kept, and for a Decode, the Decode under the other alignment
   *  asked beside it (AnalysisCache). */
  positionAnalysis(
    captureId: string | null,
    turn: number,
    stage: string,
    signal: AbortSignal,
    alignment: KvPositionAlignment | null = null,
  ): Promise<PositionAnalysis> {
    const read = (
      targetAlignment: KvPositionAlignment | null,
      targetSignal?: AbortSignal,
    ): Promise<PositionAnalysis> => {
      const path = `position-analysis?${stageQuery(turn, stage, targetAlignment)}`;
      return this.cache.read(
        JSON.stringify([captureId, path]),
        (own) => this.client.getCapture<PositionAnalysis>(captureId, path, own),
        targetSignal,
        true,
      );
    };
    const answer = read(alignment, signal);
    const twin = this.cache.twin(turn, stage, alignment?.alignment, 'intermediate_tensors');
    return this.cache.prefetch(
      answer,
      twin ? () => read({stage: 'decode', alignment: twin.alignment, ...twin.positions}) : null,
      signal,
    );
  }
  /** Each layer's metrics over columns, binned; kept, and with the whole Decode asked, the whole
   *  Decode under the other alignment asked beside it (AnalysisCache), in its bins at the chart's
   *  `capacity` (one bin needs none). */
  positionRange(
    captureId: string | null,
    turn: number,
    stage: string,
    query: PositionRangeQuery,
    signal: AbortSignal,
    alignment: KvPositionAlignment | null = null,
    capacity: number | null = null,
  ): Promise<PositionRangeResponse> {
    const read = (
      rangeQuery: PositionRangeQuery,
      rangeAlignment: KvPositionAlignment | null,
      rangeSignal?: AbortSignal,
    ): Promise<PositionRangeResponse> => {
      const params = stageQuery(turn, stage, rangeAlignment);
      params.set('start', String(rangeQuery.start));
      params.set('end', String(rangeQuery.end));
      params.set('bins', String(rangeQuery.bins));
      params.set('reduction', rangeQuery.reduction);
      if (rangeQuery.anchors) {
        params.set('anchors', JSON.stringify(rangeQuery.anchors));
      }
      if (rangeQuery.layers) {
        params.set('layers', JSON.stringify(rangeQuery.layers));
      }
      const path = `position-range?${params}`;
      return this.cache.read(
        JSON.stringify([captureId, path]),
        (own) => this.client.getCapture<PositionRangeResponse>(captureId, path, own),
        rangeSignal,
      );
    };
    const answer = read(query, alignment, signal);
    return this.cache.prefetchRange(
      answer,
      {
        turn,
        stage,
        alignment: alignment?.alignment,
        analysis: 'intermediate_tensors',
        start: query.start,
        end: query.end,
        bins: query.bins,
        capacity,
      },
      (twin, bins) =>
        read(
          {...query, start: twin.start, end: twin.end, bins},
          {
            stage: 'decode',
            alignment: twin.alignment,
            ...twin.positions,
          },
        ),
      signal,
    );
  }
  positionTensor(
    captureId: string | null,
    turn: number,
    stage: string,
    query: PositionTensorRequest,
    signal: AbortSignal,
    alignment: KvPositionAlignment | null = null,
  ): Promise<PositionTensorResponse> {
    const params = stageQuery(turn, stage, alignment);
    params.set('position', String(query.position));
    params.set('layer', String(query.layer));
    params.set('anchor', query.anchor);
    return this.client.getCapture<PositionTensorResponse>(
      captureId,
      `position-tensor?${params}`,
      signal,
    );
  }
}

function stageQuery(
  turn: number,
  stage: string,
  alignment: KvPositionAlignment | null,
): URLSearchParams {
  const params = new URLSearchParams({turn: String(turn), stage});
  if (alignment) {
    params.set('alignment', alignment.alignment);
  }
  return params;
}
