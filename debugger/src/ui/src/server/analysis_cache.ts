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

import {inject, Injectable, untracked} from '@angular/core';
import {CaptureStore} from './capture_store';
import type {
  ComparisonAlignment,
  Session,
  StageAnalysisId,
  StagePositions,
} from './contracts/capture';

/** How many answers are kept: enough for every panel's views of both alignments. */
export const ANALYSIS_CACHE_SIZE = 256;
/** How many large answers are kept (a stage's intermediate tensors, cell by cell, are hundreds of
 *  megabytes): one stage's, under both alignments. */
export const LARGE_ANSWERS = 2;

/** The other alignment of a Decode stage: its kind, the server's positions of its rows and the
 *  columns it gives an analysis there. */
export interface AlignmentTwin {
  alignment: ComparisonAlignment;
  positions: StagePositions;
  start: number;
  end: number;
}

/** Parameters describing a binned range request eligible for opposite-alignment prefetching. */
export interface RangePrefetchRequest {
  turn: number;
  stage: string;
  alignment: string | null | undefined;
  analysis: StageAnalysisId;
  start: number;
  end: number;
  bins: number;
  capacity: number | 'fixed' | null;
}

/** The bins a request of `count` columns has at a chart's capacity (range_geometry's
 *  rangeBinCount, which the panels ask with). */
export function binsAt(count: number, capacity: number): number {
  const columns = Math.max(1, count);
  return Math.max(1, Math.floor(columns / Math.ceil(columns / Math.max(1, capacity))));
}

/**
 * The analysis answers of the attached capture, kept while its Session view is the same, so that
 * a view asked again, by any panel, is answered without asking the server.
 *
 * Every analysis of a Decode is read under both alignments up front: a request of the whole stage
 * under one alignment also asks the same view of the whole stage under the other (`twin`), and
 * is answered once both are, so that switching the alignment redraws from kept answers.
 */
@Injectable({providedIn: 'root'})
export class AnalysisCache {
  private readonly state = inject(CaptureStore);
  private session: Session | null = null;
  private readonly answers = new Map<string, Promise<unknown>>();
  private readonly large = new Set<string>();

  /** The answer to `key`: the kept one, else `fetch`'s, kept once asked. The caller's `signal`
   *  only stops its own wait; the request goes on for whoever asks next. */
  read<T>(
    key: string,
    fetch: (signal: AbortSignal) => Promise<T>,
    signal?: AbortSignal,
    large = false,
  ): Promise<T> {
    this.forgetStale();
    let answer = this.answers.get(key) as Promise<T> | undefined;
    if (answer) {
      // Most recently asked last: the oldest is dropped first.
      this.answers.delete(key);
      this.answers.set(key, answer);
    } else {
      answer = fetch(new AbortController().signal);
      this.answers.set(key, answer);
      answer.catch(() => {
        if (this.answers.get(key) === answer) {
          this.drop(key);
        }
      });
      if (large) {
        this.large.add(key);
        for (const kept of [...this.answers.keys()]
          .filter((k) => this.large.has(k))
          .reverse()
          .slice(LARGE_ANSWERS)) {
          this.drop(kept);
        }
      }
      while (this.answers.size > ANALYSIS_CACHE_SIZE) {
        const oldest = this.answers.keys().next().value;
        if (oldest === undefined) {
          break;
        }
        this.drop(oldest);
      }
    }
    return signal ? abortable(answer, signal) : answer;
  }

  /** Asks `twin` beside `first`, keeping its answer; `first`'s answer once `twin`'s is in too,
   *  so that a view counts as loaded only when both alignments are, and a switch never waits.
   *  A failure of `twin` is the panels' to see when they ask, never this prefetch's; the
   *  caller's `signal` stops its own wait. */
  prefetch<T>(
    first: Promise<T>,
    twin: (() => Promise<unknown>) | null,
    signal?: AbortSignal,
  ): Promise<T> {
    if (!twin) {
      return first;
    }
    // Both at once: the view waits for the slower, not for the two in turn.
    const other = twin().catch(() => undefined);
    const both = first.then((value) => other.then(() => value));
    return signal ? abortable(both, signal) : both;
  }

  /** Prefetches the matching full-stage range under the opposite Decode alignment when `request`
   *  covers the entire stage and has a computable bin count. */
  prefetchRange<T>(
    first: Promise<T>,
    request: RangePrefetchRequest,
    fetchTwin: (twin: AlignmentTwin, bins: number) => Promise<unknown>,
    signal?: AbortSignal,
  ): Promise<T> {
    const twin = this.whole(
      request.turn,
      request.stage,
      request.alignment,
      request.analysis,
      request.start,
      request.end,
    )
      ? this.twin(request.turn, request.stage, request.alignment, request.analysis)
      : null;
    const twinBins =
      twin &&
      (request.bins === 1 || request.capacity === 'fixed'
        ? request.bins
        : request.capacity
          ? binsAt(twin.end - twin.start, request.capacity)
          : null);
    return this.prefetch(first, twin && twinBins ? () => fetchTwin(twin, twinBins) : null, signal);
  }

  /** The other alignment of a Decode stage, and the columns it gives `analysis`; null for a
   *  Prefill (aligned by content alone) or a turn the Session view does not compare. */
  twin(
    turn: number,
    stage: string,
    alignment: string | null | undefined,
    analysis: StageAnalysisId,
  ): AlignmentTwin | null {
    if (stage !== 'decode' || (alignment !== 'content' && alignment !== 'steps')) {
      return null;
    }
    const other: ComparisonAlignment = alignment === 'content' ? 'steps' : 'content';
    const comparison = untracked(() => this.state.session())?.turns.find(
      (t) => t.n === turn,
    )?.comparison;
    const positions = comparison?.alignments[other]?.positions;
    const columns = positions?.columns?.[analysis];
    if (!positions || !columns || columns.end <= columns.start) {
      return null;
    }
    return {alignment: other, positions, start: columns.start, end: columns.end};
  }

  /** Whether [start, end) is the whole of the columns an alignment gives `analysis`. */
  whole(
    turn: number,
    stage: string,
    alignment: string | null | undefined,
    analysis: StageAnalysisId,
    start: number,
    end: number,
  ): boolean {
    if (stage !== 'decode' || (alignment !== 'content' && alignment !== 'steps')) {
      return false;
    }
    const columns = untracked(() => this.state.session())?.turns.find((t) => t.n === turn)
      ?.comparison?.alignments[alignment]?.positions.columns?.[analysis];
    return !!columns && columns.start === start && columns.end === end;
  }

  /** Forgets every answer: what they were answered from changed (a pair made by hand). */
  forget(): void {
    this.answers.clear();
    this.large.clear();
  }

  private drop(key: string): void {
    this.answers.delete(key);
    this.large.delete(key);
  }

  /** Forgets every answer once the Session view changes: a new view may answer differently. */
  private forgetStale(): void {
    const session = untracked(() => this.state.session());
    if (session !== this.session) {
      this.session = session;
      this.forget();
    }
  }
}

function abortable<T>(answer: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) {
    return Promise.reject(new DOMException('Aborted', 'AbortError'));
  }
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(new DOMException('Aborted', 'AbortError'));
    signal.addEventListener('abort', abort, {once: true});
    answer.then(
      (value) => {
        signal.removeEventListener('abort', abort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener('abort', abort);
        reject(error);
      },
    );
  });
}
