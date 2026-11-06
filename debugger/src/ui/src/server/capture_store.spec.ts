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

import {TestBed} from '@angular/core/testing';
import {CaptureStore} from './capture_store';
import type {Session} from './contracts/capture';
import {SessionsApi} from './sessions_api';
import {deferred} from './testing/session_fixtures';

describe('CaptureStore', () => {
  it('report attachment rejects stale sessions; Chat loads only Session; retry keeps the capture identity', async () => {
    const pending = new Map<string, ReturnType<typeof deferred<Session>>>();
    const seen: Array<[string, string]> = [];
    const api = {
      session: (id: string) => {
        seen.push(['session', id]);
        const request = deferred<Session>();
        pending.set(id, request);
        return request.promise;
      },
    };
    const capture = (model: string) => ({model, batches: []}) as unknown as Session;
    TestBed.configureTestingModule({providers: [{provide: SessionsApi, useValue: api}]});
    const state = TestBed.inject(CaptureStore);
    state.attach('parent');
    const oldLoad = state.load();
    state.attach('child');
    const currentLoad = state.load();
    pending.get('parent')!.resolve(capture('Old'));
    await oldLoad;
    expect(state.session()).toBe(null);
    pending.get('child')!.resolve(capture('Current'));
    await currentLoad;
    expect(state.session()!.model).toBe('Current');
    expect(state.captureId()).toBe('child');
    state.attach('other');
    const nextLoad = state.load();
    pending.get('other')!.resolve(capture('Next'));
    await nextLoad;
    for (let i = 0; i < 8; i++) {
      await Promise.resolve();
      TestBed.tick();
    }
    expect(
      seen.every(([kind]) => kind === 'session'),
      'Chat attachment loads only Session',
    ).toBeTruthy();
    const retry = state.load();
    expect(seen.at(-1)![1]).toBe('other');
    pending.get('other')!.resolve(capture('Retried'));
    await retry;
    expect(state.session()!.model).toBe('Retried');
  });
});
