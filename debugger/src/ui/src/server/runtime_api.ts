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
import type {RuntimeCapabilities} from './contracts/runtime';

/** /api/runtime: what the runtimes offer — models, backends, options and limits. */
@Injectable({providedIn: 'root'})
export class RuntimeApi {
  private readonly client = inject(ServerClient);
  capabilities(): Promise<RuntimeCapabilities> {
    return this.client.get<RuntimeCapabilities>('runtime/capabilities');
  }
}
