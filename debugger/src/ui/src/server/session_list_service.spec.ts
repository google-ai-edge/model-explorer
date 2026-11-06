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
import type {SessionListing, SessionSummary} from './contracts/sessions';
import {SessionListService} from './session_list_service';
import {SessionsApi} from './sessions_api';
import {deferred, listing, session} from './testing/session_fixtures';

/** Lets every promise chained so far settle, the fake's answers and the list's handling. */
function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe('SessionListService', () => {
  // The page is hidden: the list polls only when the spec asks it to.
  beforeEach(() =>
    Object.defineProperty(document, 'visibilityState', {configurable: true, get: () => 'hidden'}),
  );
  afterEach(() => Reflect.deleteProperty(document, 'visibilityState'));

  it('late rename/delete polls, mutation failure recovery, paused polls, last-known refresh error', async () => {
    // Resolve a real pre-mutation GET after a rename, then after a deletion.
    const reads: Array<ReturnType<typeof deferred<SessionListing>>> = [];
    let rename = deferred<SessionSummary>();
    const mutation = deferred<SessionSummary>();
    const api = {
      sessions() {
        const read = deferred<SessionListing>();
        reads.push(read);
        return read.promise;
      },
      renameSession() {
        return rename.promise;
      },
      manageSession() {
        return mutation.promise;
      },
    };
    TestBed.configureTestingModule({providers: [{provide: SessionsApi, useValue: api}]});
    const sessions = TestBed.inject(SessionListService);
    reads.shift()!.resolve(listing());
    await tick();
    const oldRead = sessions.load(true);
    const old = reads.shift()!;
    const renaming = sessions.rename('a', 'Renamed');
    rename.resolve(session('a', {name: 'Renamed'}));
    await renaming;
    old.resolve(listing());
    await oldRead;
    expect(sessions.items()[0].name, 'late poll must not revert completed rename').toBe('Renamed');
    const beforeDelete = sessions.load(true);
    const stale = reads.shift()!;
    const deleting = sessions.manage('delete', {id: 'a'});
    mutation.resolve(session());
    await deleting;
    stale.resolve(listing());
    await beforeDelete;
    expect(sessions.items().length, 'late poll must not resurrect deleted row').toBe(0);

    // Failed mutation still releases polling and keeps last-known data.
    sessions.listing.set(listing());
    rename = deferred<SessionSummary>();
    const failed = sessions.rename('a', 'Unsaved');
    const readCount = reads.length;
    await sessions.load(true);
    expect(reads.length, 'polls paused during mutation').toBe(readCount);
    rename.reject(new Error('test write failure'));
    await expect(failed).rejects.toThrow(/write failure/);
    const refresh = sessions.load(true);
    reads.shift()!.resolve(listing([session('a', {name: 'Server value'})]));
    await refresh;
    expect(sessions.items()[0].name).toBe('Server value');
    expect(sessions.loading()).toBe(false);
    const rejectedRead = sessions.load(true);
    reads.shift()!.reject(new Error('offline'));
    await rejectedRead;
    expect(sessions.items()[0].name).toBe('Server value');
    expect(sessions.refreshError()).toMatch(/offline/);
  });

  it('puts a Session just created first, as the server lists the last active first', async () => {
    const api = {
      sessions: () => new Promise<SessionListing>(() => {}),
      manageSession: async () => session('new'),
    };
    TestBed.configureTestingModule({providers: [{provide: SessionsApi, useValue: api}]});
    const sessions = TestBed.inject(SessionListService);
    sessions.listing.set(listing([session('a'), session('b')]));
    await sessions.manage('create', {});
    expect(sessions.items().map((s) => s.id)).toEqual(['new', 'a', 'b']);
  });
});
