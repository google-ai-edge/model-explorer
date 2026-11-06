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
import {ServerClient} from './server_client';
import type {RuntimeJob} from './contracts/jobs';

/** /api/jobs: one task's state, its cancellation, and its event stream. */
@Injectable({providedIn: 'root'})
export class JobsApi {
  private readonly client = inject(ServerClient);
  job(id: string): Promise<RuntimeJob> {
    return this.client.get<RuntimeJob>(`jobs/${id}`);
  }
  cancel(id: string): Promise<unknown> {
    return this.client.post(`jobs/${id}/cancel`, {});
  }
  /** Events after `sequence`; the caller closes the stream. */
  events(id: string, sequence: number): EventSource {
    return this.client.events(`jobs/${id}/events?after=${sequence}`);
  }
}
