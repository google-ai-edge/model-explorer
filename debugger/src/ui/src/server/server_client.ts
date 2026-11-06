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
import {parseCaptureId} from './capture_id';

/** Where the Python server answers; every request the UI makes goes below it. */
const BASE_URL = '/api/';

/** Error thrown when an HTTP request to `/api/*` fails. */
export class ApiError extends Error {
  /** `body` is the error answer's JSON object, when it is one: `{"error": text, ...}`. */
  constructor(
    message: string,
    readonly status: number,
    readonly body: Record<string, unknown> | null = null,
  ) {
    super(message);
  }
}
/** The JSON object an answer holds; null for anything else (proxies may answer with HTML). */
function jsonObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{')) {
    return null;
  }
  try {
    const body = JSON.parse(trimmed);
    return body && typeof body === 'object' && !Array.isArray(body) ? body : null;
  } catch {
    return null;
  }
}
/** The server answers errors with `{"error": text}`; proxies may answer with HTML. */
function errorMessage(body: Record<string, unknown> | null, status: number): string {
  return typeof body?.['error'] === 'string' && body['error']
    ? body['error']
    : `Backend request failed (${status})`;
}
/** Options for streaming a file upload with abort and progress callbacks. */
export interface UploadOptions {
  signal?: AbortSignal;
  onProgress?: (loaded: number, total: number) => void;
}

/**
 * The UI's one connection to the Python server: the base URL, the fetch wrapper with its error
 * mapping and abort handling, the capture scoping of capture-scoped requests, the upload with
 * progress and the event streams. The request files of each /api topic go through it.
 */
@Injectable({providedIn: 'root'})
export class ServerClient {
  /** Sends a GET request to `/api/{path}`. */
  get<T>(path: string, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {signal});
  }
  /** Sends a JSON POST request to `/api/{path}`. */
  post<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
      signal,
    });
  }
  /** Sends a DELETE request to `/api/{path}`. */
  delete<T>(path: string, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {method: 'DELETE', signal});
  }
  /** The address of an API path, for what the browser fetches itself (a download). */
  url(path: string): string {
    return `${BASE_URL}${path}`;
  }
  /**
   * A request about one capture names its Session and chat (capture_id.ts); a missing capture
   * fails before any request.
   */
  async getCapture<T>(captureId: string | null, path: string, signal?: AbortSignal): Promise<T> {
    return this.get<T>(this.capturePath(captureId, path), signal);
  }
  /** Sends a capture-scoped JSON POST request naming the capture's Session and chat. */
  async postCapture<T>(
    captureId: string | null,
    path: string,
    payload: unknown,
    signal?: AbortSignal,
  ): Promise<T> {
    return this.post<T>(this.capturePath(captureId, path), payload, signal);
  }
  /** Streams one file with upload progress; fetch cannot report progress. */
  upload<T>(path: string, file: File, options: UploadOptions = {}): Promise<T> {
    if (options.signal?.aborted) {
      return Promise.reject(new DOMException('Upload cancelled', 'AbortError'));
    }
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open('POST', `${BASE_URL}${path}`);
      request.setRequestHeader('Content-Type', 'application/octet-stream');
      request.setRequestHeader('X-File-Name', encodeURIComponent(file.name));
      const abort = () => request.abort();
      options.signal?.addEventListener('abort', abort, {once: true});
      const settle = () => options.signal?.removeEventListener('abort', abort);
      request.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          options.onProgress?.(event.loaded, event.total);
        }
      };
      request.onload = () => {
        settle();
        const text = request.responseText ?? '';
        if (request.status >= 200 && request.status < 300) {
          try {
            resolve(JSON.parse(text));
          } catch {
            reject(new ApiError('Upload response was not JSON', request.status));
          }
        } else {
          const body = jsonObject(text);
          reject(new ApiError(errorMessage(body, request.status), request.status, body));
        }
      };
      request.onerror = () => {
        settle();
        reject(new ApiError('Upload failed; the server could not be reached.', 0));
      };
      request.onabort = () => {
        settle();
        reject(new DOMException('Upload cancelled', 'AbortError'));
      };
      request.send(file);
    });
  }
  /** A server-sent event stream; the caller owns it and closes it. */
  events(path: string): EventSource {
    return new EventSource(`${BASE_URL}${path}`);
  }
  private capturePath(captureId: string | null, path: string): string {
    const capture = captureId ? parseCaptureId(captureId) : null;
    if (!capture) {
      throw new Error('A chat must be selected.');
    }
    const separator = path.includes('?') ? '&' : '?';
    return `${path}${separator}session_id=${encodeURIComponent(capture.session)}&chat=${capture.chat}`;
  }
  /** Single request path: server error text is surfaced. */
  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${BASE_URL}${path}`, {cache: 'no-store', ...init});
    if (response.ok) {
      return (await response.json()) as T;
    }
    const body = jsonObject(await response.text().catch(() => ''));
    throw new ApiError(errorMessage(body, response.status), response.status, body);
  }
}
