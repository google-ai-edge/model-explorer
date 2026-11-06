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

/** One current request. Abort-insensitive adapters still cannot publish late data or errors.
 *  `data` is the current request's answer, null while it loads; `last` the last answer that
 *  arrived, kept while a newer request loads or fails, and cleared when nothing is asked. */
export class LatestRequest<T> {
  readonly data = signal<T | null>(null);
  readonly last = signal<T | null>(null);
  readonly loading = signal(false);
  readonly error = signal('');
  private cancelCurrent: (() => void) | null = null;

  load(
    read: ((signal: AbortSignal) => Promise<T>) | null,
    receive?: (value: T) => void,
    delay = 0,
  ): () => void {
    this.cancelCurrent?.();
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const cancel = () => {
      controller.abort();
      if (timer !== undefined) {
        clearTimeout(timer);
      }
      if (this.cancelCurrent === cancel) {
        this.loading.set(false);
      }
    };
    this.cancelCurrent = cancel;
    this.data.set(null);
    if (!read) {
      this.last.set(null);
    }
    this.error.set('');
    this.loading.set(!!read);
    if (read) {
      const start = async () => {
        try {
          const value = await read(controller.signal);
          if (controller.signal.aborted) {
            return;
          }
          this.data.set(value);
          this.last.set(value);
          receive?.(value);
        } catch (error) {
          if (!controller.signal.aborted) {
            this.error.set(error instanceof Error ? error.message : String(error));
          }
        } finally {
          if (!controller.signal.aborted) {
            this.loading.set(false);
          }
        }
      };
      if (delay) {
        timer = setTimeout(() => void start(), delay);
      } else {
        void start();
      }
    }
    return cancel;
  }
  /** Takes the last answer as the current one again, as when the view goes back to the request
   *  it answered: a newer request is cancelled, and its error forgotten. */
  reuse(): void {
    this.cancelCurrent?.();
    this.cancelCurrent = null;
    this.data.set(this.last());
    this.error.set('');
    this.loading.set(false);
  }
  destroy(): void {
    this.cancelCurrent?.();
  }
}
