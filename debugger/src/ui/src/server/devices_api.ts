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
import type {DeviceCandidates, RunnerBuildCatalog, RunnerDevice} from './contracts/devices';

/** /api/devices: the devices a Runner can run on, their Runner builds, and a check of one. */
@Injectable({providedIn: 'root'})
export class DevicesApi {
  private readonly client = inject(ServerClient);
  devices(): Promise<DeviceCandidates> {
    return this.client.get<DeviceCandidates>('devices');
  }
  inspectDevice(id: string): Promise<Partial<RunnerDevice>> {
    return this.client.post<Partial<RunnerDevice>>('devices/inspect', {id});
  }
  /** The runner versions of a device, read without launching any. */
  deviceBuilds(id: string): Promise<RunnerBuildCatalog> {
    return this.client.get<RunnerBuildCatalog>(`devices/builds?id=${encodeURIComponent(id)}`);
  }
}
