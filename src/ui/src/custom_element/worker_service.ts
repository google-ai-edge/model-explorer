/**
 * @license
 * Copyright 2025 The Model Explorer Authors. All Rights Reserved.
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

import {Injectable, OnDestroy} from '@angular/core';

declare interface GlobalWithWorkerScriptPath {
  readonly modelExplorer?: {
    readonly workerScriptPath?: string;
  };
}

/**
 * Service managing the layout Web Worker for the custom element bundle.
 */
@Injectable()
export class WorkerService implements OnDestroy {
  private currentWorker?: Worker;

  /** Returns the active layout worker after `init()` has initialized it. */
  get worker(): Worker {
    if (!this.currentWorker) {
      throw new Error(
        'WorkerService.init() must be called before accessing worker.',
      );
    }
    return this.currentWorker;
  }

  /**
   * Initializes (or re-initializes) the layout worker, honoring
   * `window.modelExplorer.workerScriptPath` before falling back to
   * `'worker.js'`.
   *
   * Accepts `_onPortal` to match the main app's `WorkerService.init` contract
   * used via Angular CLI `fileReplacements`.
   */
  init(_onPortal: boolean): void {
    this.disposeWorker();

    const workerScriptPath = (globalThis as GlobalWithWorkerScriptPath)
      .modelExplorer?.workerScriptPath;
    const workerScriptUrl =
      workerScriptPath != null && workerScriptPath !== ''
        ? workerScriptPath
        : 'worker.js';
    this.currentWorker = new Worker(workerScriptUrl);
  }

  ngOnDestroy(): void {
    this.disposeWorker();
  }

  private disposeWorker(): void {
    this.currentWorker?.terminate();
    this.currentWorker = undefined;
  }
}
