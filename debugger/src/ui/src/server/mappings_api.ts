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
import type {ManualPairs, SavePair} from './contracts/mappings';

/** /api/mappings: the tensor pairs made by hand of the two models of a Session, kept by the server
 *  for every Session of those two models. Each call answers the list as it is then. */
@Injectable({providedIn: 'root'})
export class MappingsApi {
  private readonly client = inject(ServerClient);
  pairs(sessionId: string, signal?: AbortSignal): Promise<ManualPairs> {
    return this.client.get<ManualPairs>(path(sessionId), signal);
  }
  save(sessionId: string, pair: SavePair): Promise<ManualPairs> {
    return this.client.post<ManualPairs>(path(sessionId), pair);
  }
  delete(sessionId: string, pairId: string): Promise<ManualPairs> {
    return this.client.delete<ManualPairs>(path(sessionId, pairId));
  }
}

function path(sessionId: string, pairId = ''): string {
  const pair = pairId ? `/${encodeURIComponent(pairId)}` : '';
  return `mappings${pair}?session_id=${encodeURIComponent(sessionId)}`;
}
