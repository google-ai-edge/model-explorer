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

import {Injectable} from '@angular/core';
import type {Session} from './contracts/capture';
import type {
  DeviceCandidates,
  RunnerBuildCatalog,
  RunnerDevice,
  RuntimeCapabilities,
  RuntimeJob,
} from './contracts/runtime';
import {
  SessionConfig,
  SessionListing,
  SessionSummary,
} from './contracts/session_summary';
import type {TapScan} from './contracts/tap_scan';
import {ResourcePreview, Telemetry} from './contracts/telemetry';
import {
  Comparison,
  NodeDetails,
  Overview,
  SavedMapping,
  Selection,
  SelectionResult,
  Semantic,
} from './contracts/types';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}
/** The server answers errors with `{"error": text}`; proxies may answer with HTML. */
function errorMessage(text: string, status: number): string {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    try {
      const body = JSON.parse(trimmed);
      if (typeof body?.error === 'string' && body.error) return body.error;
    } catch {
      /* not JSON after all */
    }
  }
  return `Backend request failed (${status})`;
}
export interface UploadOptions {
  signal?: AbortSignal;
  onProgress?: (loaded: number, total: number) => void;
}
@Injectable({providedIn: 'root'})
export class ReportApiService {
  private capturePath(captureId: string | null, path: string): string {
    if (!captureId) throw new Error('A capture record must be selected.');
    return (
      path +
      (path.includes('?') ? '&' : '?') +
      'session_id=' +
      encodeURIComponent(captureId)
    );
  }
  private async getCapture<T>(
    captureId: string | null,
    path: string,
    signal?: AbortSignal,
  ): Promise<T> {
    return this.get<T>(this.capturePath(captureId, path), signal);
  }
  private async postCapture<T>(
    captureId: string | null,
    path: string,
    payload: unknown,
    signal?: AbortSignal,
  ): Promise<T> {
    return this.post<T>(this.capturePath(captureId, path), payload, signal);
  }
  tapScan(id: string): Promise<TapScan> {
    return this.get<TapScan>(`artifacts/${id}/scan`);
  }
  capabilities(): Promise<RuntimeCapabilities> {
    return this.get<RuntimeCapabilities>('runtime/capabilities');
  }
  job(id: string): Promise<RuntimeJob> {
    return this.get<RuntimeJob>('jobs/' + id);
  }
  private get<T>(path: string, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {signal});
  }
  /** Single request path: server error text is surfaced; a 503 with Retry-After is retried once. */
  private async request<T>(
    path: string,
    init: RequestInit = {},
    retried = false,
  ): Promise<T> {
    const response = await fetch('/api/' + path, {cache: 'no-store', ...init});
    if (response.ok) return (await response.json()) as T;
    const message = errorMessage(
      await response.text().catch(() => ''),
      response.status,
    );
    if (response.status === 503 && !retried && !init.signal?.aborted) {
      const header = response.headers.get('Retry-After');
      const delay = header && header.trim() ? Number(header) : NaN;
      if (Number.isFinite(delay) && delay >= 0 && delay <= 5) {
        await new Promise((resolve) => setTimeout(resolve, delay * 1000));
        return this.request<T>(path, init, true);
      }
    }
    throw new ApiError(message, response.status);
  }
  devices() {
    return this.request<DeviceCandidates>('devices');
  }
  deviceBuilds(id: string) {
    return this.request<RunnerBuildCatalog>(
      'devices/builds?id=' + encodeURIComponent(id),
    );
  }
  inspectDevice(id: string) {
    return this.post<Partial<RunnerDevice>>('devices/inspect', {id});
  }
  /** Streams one artifact with upload progress; fetch cannot report progress. */
  upload(file: File, options: UploadOptions = {}): Promise<{artifact: string}> {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open('POST', '/api/artifacts/upload');
      request.setRequestHeader('Content-Type', 'application/octet-stream');
      request.setRequestHeader('X-File-Name', encodeURIComponent(file.name));
      const abort = () => request.abort();
      options.signal?.addEventListener('abort', abort, {once: true});
      const settle = () => options.signal?.removeEventListener('abort', abort);
      request.upload.onprogress = (event) => {
        if (event.lengthComputable)
          options.onProgress?.(event.loaded, event.total);
      };
      request.onload = () => {
        settle();
        const text = request.responseText ?? '';
        if (request.status >= 200 && request.status < 300) {
          try {
            resolve(JSON.parse(text));
          } catch {
            reject(
              new ApiError('Upload response was not JSON', request.status),
            );
          }
        } else
          reject(
            new ApiError(errorMessage(text, request.status), request.status),
          );
      };
      request.onerror = () => {
        settle();
        reject(
          new ApiError('Upload failed; the server could not be reached.', 0),
        );
      };
      request.onabort = () => {
        settle();
        reject(new DOMException('Upload cancelled', 'AbortError'));
      };
      request.send(file);
    });
  }
  node(
    captureId: string | null,
    layer: number,
    batch: number,
    semantic: string,
    signal: AbortSignal,
  ) {
    return this.getCapture<NodeDetails>(
      captureId,
      `node?layer=${layer}&batch=${batch}&semantic=${encodeURIComponent(semantic)}`,
      signal,
    );
  }
  post<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
      signal,
    });
  }
  compareSelection(
    captureId: string | null,
    payload: Selection,
    signal: AbortSignal,
  ) {
    return this.postCapture<SelectionResult>(
      captureId,
      'selection/compare',
      payload,
      signal,
    );
  }
  removeMapping(captureId: string | null, payload: Selection) {
    return this.postCapture<SavedMapping>(
      captureId,
      'mappings/remove',
      payload,
    );
  }
  saveMapping(captureId: string | null, payload: Selection) {
    return this.postCapture<SavedMapping>(captureId, 'mappings', payload);
  }
  manageSession(
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
    return this.post<SessionSummary>('sessions/' + operation, payload);
  }
  sessions() {
    return this.get<SessionListing>('sessions');
  }
  renameSession(id: string, name: string) {
    return this.post<SessionSummary>('sessions/rename', {id, name});
  }
  overview(captureId: string | null, signal?: AbortSignal) {
    return this.getCapture<Overview>(captureId, 'overview', signal);
  }
  session(captureId: string | null, signal?: AbortSignal) {
    return this.getCapture<Session>(captureId, 'session', signal);
  }
  semantic(captureId: string | null, signal?: AbortSignal) {
    return this.getCapture<Semantic>(captureId, 'semantic', signal);
  }
  explicitPairs(captureId: string | null, signal: AbortSignal) {
    return this.getCapture<import('./contracts/telemetry').ExplicitPair[]>(
      captureId,
      'explicit-pairs',
      signal,
    );
  }
  pairTensor(
    captureId: string | null,
    id: string,
    role: string,
    mode: string,
    offset: number,
    signal: AbortSignal,
  ) {
    return this.getCapture<ResourcePreview>(
      captureId,
      `pair-tensor?pair_id=${encodeURIComponent(id)}&role=${role}&mode=${mode}&offset=${offset}&limit=32`,
      signal,
    );
  }
  telemetry(captureId: string | null, turn: number, signal: AbortSignal) {
    return this.getCapture<Telemetry>(
      captureId,
      'telemetry?turn=' + turn,
      signal,
    );
  }
  tokenAnalysis(
    captureId: string | null,
    turn: number,
    pairs: {ref: number | null; target: number | null}[],
    signal: AbortSignal,
  ) {
    return this.postCapture<
      import('./contracts/token_analysis').TokenAnalysisResponse
    >(captureId, 'token-analysis', {turn, pairs}, signal);
  }
  kvMetadata(captureId: string | null, turn: number, signal: AbortSignal) {
    return this.getCapture<import('./contracts/kv').KvAnalysis>(
      captureId,
      'kv-analysis?mode=metadata&turn=' + turn,
      signal,
    );
  }
  kvRange(
    captureId: string | null,
    turn: number,
    context: string,
    start: number,
    end: number,
    bins: number,
    kind: string,
    head: string,
    formula: string,
    signal: AbortSignal,
  ) {
    const params = new URLSearchParams({
      turn: String(turn),
      context,
      start: String(start),
      end: String(end),
      bins: String(bins),
      kind,
      head,
    });
    if (formula) params.set('formula', formula);
    return this.getCapture<import('./contracts/kv').KvRangeResponse>(
      captureId,
      'kv-range?' + params,
      signal,
    );
  }
  kvSelection(
    captureId: string | null,
    turn: number,
    context: string,
    start: number,
    end: number,
    layer: number | null,
    signal: AbortSignal,
  ) {
    const params = new URLSearchParams({
      turn: String(turn),
      context,
      start: String(start),
      end: String(end),
    });
    if (layer !== null) params.set('layer', String(layer));
    return this.getCapture<import('./contracts/kv').KvRangeDetails>(
      captureId,
      'kv-selection?' + params,
      signal,
    );
  }
  kvFind(
    captureId: string | null,
    turn: number,
    context: string,
    formula: string,
    offset: number,
    signal: AbortSignal,
  ) {
    const params = new URLSearchParams({
      turn: String(turn),
      context,
      formula,
      offset: String(offset),
      limit: '40',
    });
    return this.getCapture<import('./contracts/kv').KvFindResponse>(
      captureId,
      'kv-find?' + params,
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
    head: number,
    batch: number | null,
    signal: AbortSignal,
  ) {
    const params = new URLSearchParams({
      turn: String(turn),
      context_id: contextId,
      layer: String(layer),
      kind,
      position: String(position),
      head: String(head),
    });
    if (batch !== null) params.set('batch', String(batch));
    return this.getCapture<import('./contracts/kv').KvHeadEvidence>(
      captureId,
      'kv-head?' + params,
      signal,
    );
  }
  comparison(captureId: string | null, batch: number, signal: AbortSignal) {
    return this.getCapture<Comparison>(
      captureId,
      'comparisons?batch=' + batch,
      signal,
    );
  }
}
