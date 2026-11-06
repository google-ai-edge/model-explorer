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
import {TestBed} from '@angular/core/testing';
import {NavigationEnd, Router} from '@angular/router';
import {Subject} from 'rxjs';
import {captureId} from '../server/capture_id';
import {CaptureStore} from '../server/capture_store';
import {JobsService} from '../server/jobs_service';
import {RuntimeService} from '../server/runtime_service';
import {SessionListService} from '../server/session_list_service';
import type {RuntimeJob} from '../server/contracts/jobs';
import type {SessionSummary} from '../server/contracts/sessions';
import {allowed, chat, refused, session} from '../server/testing/session_fixtures';
import {WorkspaceStateService} from './workspace_state_service';

const startable = {...session().actions, new_chat: allowed};
/** Sessions whose live chat kept a turn: read in both modes. */
function captured(id: string, name: string) {
  return session(id, {
    name,
    has_capture: true,
    chats: [chat(1, {has_capture: true})],
    live_chat: 1,
    actions: startable,
  });
}
const records = [captured('parent', 'Parent'), captured('other', 'Other')];
const NO_TURN = 'This chat has no turn to inspect.';
const NO_CHAT = 'Start a chat, or choose one, to inspect it in Debug.';
/** A Session as the server lists it once it ended: its chats ended, no new one starts. */
function ended(id: string, chats = [chat(1, {state: 'ended', has_capture: true})]) {
  return session(id, {
    chats,
    live_chat: null,
    has_capture: chats.some((c) => c.has_capture),
    actions: {...session().actions, new_chat: refused('This Session has ended.')},
  });
}

/** A router that arrives at each address at once, as the browser's would after its navigation. */
function fakeRouter() {
  const events = new Subject<NavigationEnd>();
  const visited: Array<[string, boolean]> = [];
  const router = {
    url: '/',
    events,
    visited,
    navigateByUrl(url: string, extras: {replaceUrl?: boolean} = {}) {
      router.url = url;
      visited.push([url, !!extras.replaceUrl]);
      events.next(new NavigationEnd(visited.length, url, url));
      return Promise.resolve(true);
    },
  };
  return router;
}

describe('WorkspaceStateService', () => {
  // Nothing is stored: each page starts with no drafts.
  beforeEach(() => vi.stubGlobal('localStorage', {getItem: () => null, setItem() {}}));
  afterEach(() => vi.unstubAllGlobals());

  it("the attachments follow the resolved context automatically and only on change: the runtime the Session's, the capture its chat's", () => {
    const attached: Array<string | null> = [];
    const runtimeIds: Array<string | null> = [];
    const router = fakeRouter();
    const list = signal<SessionSummary[]>(records);
    TestBed.configureTestingModule({
      providers: [
        {provide: SessionListService, useValue: {items: list, listing: () => null}},
        {
          provide: CaptureStore,
          useValue: {
            attach: (id: string | null) => attached.push(id),
            load: async () => {},
            selectTurn: () => {},
            selectPhase: () => {},
          },
        },
        {
          provide: RuntimeService,
          useValue: {
            attach: async (id: string | null) => runtimeIds.push(id),
            selected: () => null,
            follow: async () => {},
          },
        },
        {provide: JobsService, useValue: {job: () => null}},
        {provide: Router, useValue: router},
      ],
    });
    const workspace = TestBed.inject(WorkspaceStateService);
    // Angular runs effects after each change; the spec runs them on request.
    const settle = () => TestBed.tick();
    const visit = (url: string) => {
      router.navigateByUrl(url);
      settle();
    };
    expect(workspace.home()).toBe(true);
    workspace.open('parent');
    settle();
    // The Session's address is replaced by that of the chat it shows.
    expect(router.url).toBe('/sessions/parent/chats/1');
    expect(router.visited.at(-1)).toStrictEqual(['/sessions/parent/chats/1', true]);
    expect(attached).toStrictEqual([captureId('parent', 1)]);
    expect(runtimeIds).toStrictEqual(['parent']);
    expect(workspace.session()!.id).toBe('parent');
    expect(workspace.chat()!.n).toBe(1);
    workspace.open('other');
    settle();
    expect(router.url).toBe('/sessions/other/chats/1');
    expect(attached.at(-1)).toBe(captureId('other', 1));
    expect(runtimeIds.at(-1)).toBe('other');
    // Any change of the chat re-attaches; polls, repeats and a change of mode do not.
    const before = attached.length;
    settle();
    settle();
    visit('/sessions/other/debug/turns/2');
    expect(workspace.mode()).toBe('Debug');
    visit('/sessions/other/chats/1/debug/turns/2');
    expect(attached.length, 'unchanged identity never re-attaches').toBe(before);
    // Another chat of the same Session: its capture, the Session's runtime still.
    list.set([
      {...records[0]},
      {...records[1], chats: [chat(1, {state: 'ended', has_capture: true}), chat(2)], live_chat: 2},
    ]);
    settle();
    expect(attached.at(-1), 'the chat the address names').toBe(captureId('other', 1));
    visit('/sessions/other');
    expect(attached.at(-1), 'the live chat').toBe(captureId('other', 2));
    expect(router.url).toBe('/sessions/other/chats/2');
    expect(runtimeIds.at(-1)).toBe('other');
    visit('/sessions/other/chats/9');
    expect(attached.at(-1), 'a chat the Session lacks detaches').toBe(null);
    expect(runtimeIds.at(-1)).toBe('other');
    visit('/sessions/missing');
    expect(attached.at(-1), 'an unknown Session detaches instead of keeping the last').toBe(null);
    expect(runtimeIds.at(-1)).toBe(null);
    expect(workspace.session()).toBe(null);
    visit('/sessions/parent');
    expect(attached.at(-1)).toBe(captureId('parent', 1));
    // An address of a chat by name names nothing of the Session: its page says so, and stays.
    visit('/sessions/parent/chats/child/debug');
    expect(workspace.home()).toBe(false);
    expect(workspace.address().unmatched).toBe('chats/child/debug');
    expect(router.url).toBe('/sessions/parent/chats/child/debug');
    expect(workspace.chat()).toBe(null);
    expect(workspace.newChat()).toBe(false);
    expect(attached.at(-1)).toBe(null);
    expect(runtimeIds.at(-1)).toBe('parent');
  });

  it('follows a job of the open Session that another tab started, as the list names it', () => {
    const followed: Array<string | undefined> = [];
    const list = signal(records);
    TestBed.configureTestingModule({
      providers: [
        {provide: SessionListService, useValue: {items: list, listing: () => null}},
        {provide: CaptureStore, useValue: {attach() {}, load: async () => {}}},
        {
          provide: RuntimeService,
          useValue: {
            attach: async () => {},
            selected: () => null,
            follow: async (id: string | undefined) => followed.push(id),
          },
        },
        {provide: JobsService, useValue: {job: () => null}},
        {provide: Router, useValue: fakeRouter()},
      ],
    });
    const workspace = TestBed.inject(WorkspaceStateService);
    workspace.open('parent');
    TestBed.tick();
    // Another tab sends a turn: the next poll names its job.
    list.set([{...records[0], job_id: 'turn-2'}, records[1]]);
    TestBed.tick();
    expect(followed.at(-1)).toBe('turn-2');
    // Another Session's job is not this page's.
    list.set([
      {...records[0], job_id: 'turn-2'},
      {...records[1], job_id: 'elsewhere'},
    ]);
    TestBed.tick();
    expect(followed).not.toContain('elsewhere');
  });

  it('modes return to where the Debug reading is, of the chat shown; navigation keeps the query', () => {
    const router = fakeRouter();
    TestBed.configureTestingModule({
      providers: [
        {provide: SessionListService, useValue: {items: () => records, listing: () => null}},
        {provide: CaptureStore, useValue: {attach() {}, load: async () => {}}},
        {
          provide: RuntimeService,
          useValue: {attach: async () => {}, selected: () => null, follow: async () => {}},
        },
        {provide: JobsService, useValue: {job: () => null}},
        {provide: Router, useValue: router},
      ],
    });
    const workspace = TestBed.inject(WorkspaceStateService);
    router.navigateByUrl('/sessions/parent?debugPalette=teal');
    workspace.debugPlace.set({turn: 2, stage: 'decode', analysis: 'kv'});
    // The address names the chat from then on: Debug stays with it whatever the default becomes.
    workspace.setMode('Debug');
    expect(router.url).toBe('/sessions/parent/chats/1/debug/turns/2/decode/kv?debugPalette=teal');
    workspace.setMode('Chat');
    expect(router.url).toBe('/sessions/parent/chats/1?debugPalette=teal');
    workspace.setMode('Debug');
    workspace.showDebugPlace({turn: 1, stage: null, analysis: null}, true);
    expect(router.visited.at(-1)).toStrictEqual([
      '/sessions/parent/chats/1/debug/turns/1?debugPalette=teal',
      true,
    ]);
    expect(workspace.debugAddress({turn: 3, stage: null, analysis: null})).toBe(
      '/sessions/parent/chats/1/debug/turns/3',
    );
    workspace.goHome();
    expect(router.url).toBe('/?debugPalette=teal');
    workspace.showDebugPlace({turn: 1, stage: null, analysis: null});
    expect(router.url, 'no Debug place off a Debug page').toBe('/?debugPalette=teal');
  });

  describe('the chats and modes the server allows', () => {
    function setUp(items: SessionSummary[]) {
      const router = fakeRouter();
      const list = signal(items);
      const job = signal<RuntimeJob | null>(null);
      TestBed.configureTestingModule({
        providers: [
          {provide: SessionListService, useValue: {items: list, listing: () => null}},
          {provide: CaptureStore, useValue: {attach() {}, load: async () => {}}},
          {
            provide: RuntimeService,
            useValue: {attach: async () => {}, selected: () => null, follow: async () => {}},
          },
          {provide: JobsService, useValue: {job}},
          {provide: Router, useValue: router},
        ],
      });
      const workspace = TestBed.inject(WorkspaceStateService);
      const visit = (url: string) => {
        router.navigateByUrl(url);
        TestBed.tick();
      };
      return {router, list, job, workspace, visit};
    }

    it('an ended Session opens at its newest chat; a chat with a turn is read in Chat and in Debug, one with no turn leaves Debug', () => {
      const {router, workspace, visit} = setUp([
        ended('done', [chat(1, {state: 'ended', has_capture: true}), chat(2, {state: 'ended'})]),
      ]);
      router.navigateByUrl('/?debugPalette=teal');
      workspace.open('done');
      TestBed.tick();
      expect(router.url, 'opened from home').toBe('/sessions/done/chats/2?debugPalette=teal');
      expect(workspace.chat()!.n).toBe(2);
      expect(workspace.newChat()).toBe(false);
      visit('/sessions/done/chats/1');
      expect(workspace.modes()).toStrictEqual({Chat: allowed, Debug: allowed});
      visit('/sessions/done/chats/1/debug/turns/2');
      expect(router.url).toBe('/sessions/done/chats/1/debug/turns/2');
      // A chat with no turn is read in Chat only: its Debug address is replaced.
      visit('/sessions/done/chats/2/debug/turns/1');
      expect(workspace.modes()).toStrictEqual({Chat: allowed, Debug: refused(NO_TURN)});
      expect(workspace.mode()).toBe('Chat');
      expect(router.visited.at(-1)).toStrictEqual(['/sessions/done/chats/2', true]);
      // Choosing a chat keeps Debug where that chat is read in it.
      visit('/sessions/done/chats/1/debug');
      workspace.showChat(2);
      expect(router.url).toBe('/sessions/done/chats/2');
      visit('/sessions/done/chats/1/debug');
      workspace.showChat(1);
      expect(router.url).toBe('/sessions/done/chats/1/debug');
    });

    it('a Session with no live chat shows the New chat form, which is not read in Debug', () => {
      const fresh = session('fresh', {chats: [], live_chat: null, actions: startable});
      const {router, workspace, visit} = setUp([fresh]);
      workspace.open('fresh');
      TestBed.tick();
      expect(workspace.newChat()).toBe(true);
      expect(workspace.chat()).toBe(null);
      expect(workspace.captureId()).toBe(null);
      expect(workspace.modes()).toStrictEqual({Chat: allowed, Debug: refused(NO_CHAT)});
      expect(router.visited.at(-1)).toStrictEqual(['/sessions/fresh/new', true]);
      visit('/sessions/fresh/debug');
      expect(router.url).toBe('/sessions/fresh/new');
    });

    it('the chat shown stays as it ends; the form stays while its chat starts; one started is shown once listed', () => {
      const {router, list, job, workspace, visit} = setUp(records);
      workspace.open('parent');
      TestBed.tick();
      expect(workspace.chat()!.n).toBe(1);
      // A Send failed, or was stopped: the chat ended, the Session goes on, and the page still
      // shows the chat, with what its turn replied.
      list.set([
        {...records[0], chats: [chat(1, {state: 'ended', has_capture: true})], live_chat: null},
        records[1],
      ]);
      TestBed.tick();
      expect(workspace.newChat()).toBe(false);
      expect(workspace.chat()!.n).toBe(1);
      expect(router.url).toBe('/sessions/parent/chats/1');
      visit('/sessions/parent/new');
      // While the chat starts, no chat starts (new_chat refused): the form stays.
      job.set({
        id: 'chat-job',
        session_id: 'parent',
        operation: 'chat',
        status: 'starting',
        error: '',
        sequence: 1,
        output: {},
        reasoning: {},
        turn: 0,
        chat: 2,
      });
      list.set([
        {
          ...records[0],
          chats: [chat(1, {state: 'ended', has_capture: true})],
          live_chat: null,
          actions: {...startable, new_chat: refused('Wait for the current task to finish.')},
        },
        records[1],
      ]);
      TestBed.tick();
      expect(workspace.newChat()).toBe(true);
      expect(router.url).toBe('/sessions/parent/new');
      const started: RuntimeJob = {
        id: 'chat-job',
        session_id: 'parent',
        operation: 'chat',
        status: 'completed',
        error: '',
        sequence: 3,
        output: {},
        reasoning: {},
        turn: 0,
        chat: 2,
      };
      job.set(started);
      TestBed.tick();
      expect(router.url, 'not before the list has it').toBe('/sessions/parent/new');
      list.set([
        {
          ...records[0],
          chats: [chat(1, {state: 'ended', has_capture: true}), chat(2)],
          live_chat: 2,
        },
        records[1],
      ]);
      TestBed.tick();
      expect(router.visited.at(-1)).toStrictEqual(['/sessions/parent/chats/2', true]);
      expect(workspace.chat()!.n).toBe(2);
      // Shown once: the reader may go back to the form.
      visit('/sessions/parent/new');
      expect(router.url).toBe('/sessions/parent/new');
    });
  });
});
