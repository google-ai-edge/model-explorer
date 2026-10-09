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

import {strict as assert} from 'node:assert';

import {angularContext, bundleSources, deferred, withGlobals} from './lib/angular_harness.mjs';

const {module: m} = await bundleSources(
    {
      resolveWorkspaceContext: 'src/app/workspace_context',
      ReportApiService: 'src/data/report_api_service',
      ReportStateService: 'src/data/report_state_service',
      RuntimeService: 'src/data/runtime_service',
      SessionListService: 'src/data/session_list_service',
      ThemeService: 'src/theme/theme_service',
      WorkspaceStateService: 'src/app/workspace_state_service',
    },
    'capture-context-',
);
const records = [
  {id: 'parent', name: 'Parent', has_capture: true},
  {id: 'child', parent_session_id: 'parent', name: 'Child', has_capture: true},
  {id: 'other', name: 'Other', has_capture: true},
  {
    id: 'other-child',
    parent_session_id: 'other',
    name: 'Other child',
    has_capture: true
  },
];
{
  const parent = m.resolveWorkspaceContext('parent', 'capture', records),
        child = m.resolveWorkspaceContext('parent', 'child', records);
  assert.equal(parent.captureRecordId, 'parent');
  assert.equal(parent.activeChatId, 'capture');
  assert.equal(parent.parentSession.id, 'parent');
  assert.equal(child.captureRecordId, 'child');
  assert.equal(child.parentSession.id, 'parent');
  assert.equal(child.activeRecord.parent_session_id, 'parent');
  for (const [parentId, chatId] of [
           ['parent', 'other-child'],
           ['parent', 'legacy-local-draft'],
           ['missing', 'child'],
           ['child', 'capture'],
  ]) {
    assert.equal(
        m.resolveWorkspaceContext(parentId, chatId, records).captureRecordId,
        null);
  }
  console.log(
      'PASS: parent, active Chat, and capture identities remain distinct; stale/foreign/local identities do not inherit a capture',
  );
}
const calls = [];
await withGlobals(
    {
      fetch: (url, options) => {
        const response = deferred();
        calls.push({url, options, response});
        return response.promise;
      },
    },
    async () => {
      const api = new m.ReportApiService(),
            signal = new AbortController().signal;
      const first = api.telemetry('parent&capture', 1, signal),
            second = api.telemetry('child', 2, signal);
      assert.equal(
          calls[0].url, '/api/telemetry?turn=1&session_id=parent%26capture');
      assert.equal(calls[1].url, '/api/telemetry?turn=2&session_id=child');
      calls[1].response.resolve(
          {ok: true, json: async () => ({identity: 'child'})});
      assert.equal((await second).identity, 'child');
      calls[0].response.resolve(
          {ok: true, json: async () => ({identity: 'parent'})});
      assert.equal((await first).identity, 'parent');
      const missing = [
        () => api.session(null),
        () => api.overview(null),
        () => api.semantic(''),
        () => api.kvMetadata(null, 1, signal),
        () => api.tokenAnalysis(null, 1, [], signal),
        () => api.saveMapping(null, {}),
      ];
      for (const request of missing) {
        await assert.rejects(request, /capture record must be selected/);
      }
      assert.equal(calls.length, 2, 'missing capture must reject before fetch');
      const post = api.post('sessions/child/turns', {prompt: 'request'});
      assert.equal(calls[2].url, '/api/sessions/child/turns');
      calls[2].response.resolve({ok: true, json: async () => ({})});
      await post;
      assert.equal('selectedId' in api, false);
      console.log(
          'PASS: requests keep explicit capture IDs across out-of-order responses; missing IDs never query server default; lifecycle POST stays unscoped',
      );
    },
);
{
  const pending = new Map();
  const seen = [];
  const api = {
    session: (id) => {
      seen.push(['session', id]);
      const request = deferred();
      pending.set(id, request);
      return request.promise;
    },
    semantic: async (id) => {
      seen.push(['semantic', id]);
      return {semantic_graph: [], layers: []};
    },
    overview: async (id) => {
      seen.push(['overview', id]);
      return {batches: []};
    },
  };
  const context = angularContext([
    {provide: m.ReportApiService, useValue: api},
    {provide: m.ThemeService, useValue: {dark: () => false}},
  ]);
  const state = context.run(() => new m.ReportStateService());
  state.attach('parent');
  const oldLoad = state.load();
  state.attach('child');
  const currentLoad = state.load();
  pending.get('parent').resolve({model: 'Old', batches: []});
  await oldLoad;
  assert.equal(state.session(), null);
  pending.get('child').resolve({model: 'Current', batches: []});
  await currentLoad;
  assert.equal(state.session().model, 'Current');
  assert.equal(state.captureId(), 'child');
  state.attach('other');
  const nextLoad = state.load();
  pending.get('other').resolve({model: 'Next', batches: []});
  await nextLoad;
  assert.equal(state.overview(), null);
  await context.flush();
  assert.ok(
      seen.every(([kind]) => kind === 'session'),
      'Chat attachment loads only Session',
  );
  const retry = state.load();
  assert.equal(seen.at(-1)[1], 'other');
  pending.get('other').resolve({model: 'Retried', batches: []});
  await retry;
  assert.equal(state.session().model, 'Retried');
  context.destroy();
  console.log(
      'PASS: report attachment rejects stale sessions; Chat loads only Session; retry keeps the capture identity',
  );
}
await withGlobals({localStorage: {getItem: () => null, setItem() {}}}, async () => {
  const managed = [];
  const attached = [];
  const runtimeIds = [];
  const sessions = {
    allItems: () => records,
    items: () => records.filter((r) => !r.parent_session_id),
    listing: () => null,
    manage: (operation, payload) => {
      const response = deferred();
      managed.push({operation, payload, response});
      return response.promise;
    },
  };
  const report = {
    attach: (id) => attached.push(id),
    load: async () => {},
    selectTurn: () => {},
    selectPhase: () => {},
  };
  const runtime = {
    attach: async (id) => runtimeIds.push(id),
    selected: () => null,
    job: () => null,
  };
  const context = angularContext([
    {provide: m.SessionListService, useValue: sessions},
    {provide: m.ReportStateService, useValue: report},
    {provide: m.RuntimeService, useValue: runtime},
  ]);
  const workspace = context.run(() => new m.WorkspaceStateService());
  // Angular runs effects after each change; the harness runs them on request.
  const settle = () => context.settle();
  workspace.open('parent');
  settle();
  assert.deepEqual(attached, ['parent']);
  workspace.localChats.set({parent: [{id: 'legacy', name: 'Legacy draft'}]});
  const migrating = workspace.chooseChat('legacy');
  settle();
  assert.equal(attached.at(-1), null);
  assert.equal(runtimeIds.at(-1), null);
  const chosen = workspace.chooseChat('child');
  settle();
  await chosen;
  assert.equal(attached.at(-1), 'child');
  managed[0].response.resolve(
      {id: 'converted', parent_session_id: 'parent', name: 'Converted'});
  await migrating;
  settle();
  assert.equal(workspace.activeChat(), 'child');
  assert.equal(attached.at(-1), 'child');
  assert.equal(runtimeIds.at(-1), 'child');
  const newChat = workspace.newChat();
  workspace.open('other');
  settle();
  managed[1].response.resolve(
      {id: 'new-parent-chat', parent_session_id: 'parent'});
  await newChat;
  settle();
  assert.equal(workspace.selectedId(), 'other');
  assert.equal(attached.at(-1), 'other');
  assert.equal(runtimeIds.at(-1), 'other');
  console.log(
      'PASS: legacy migration detaches old capture immediately; a late Chat creation cannot override a newer Chat or parent navigation',
  );
  // Any change of the resolved context re-attaches; polls and repeats do not.
  const before = attached.length;
  workspace.activeChat.set('other-child');
  settle();
  assert.deepEqual(attached.slice(before), ['other-child']);
  assert.equal(runtimeIds.at(-1), 'other-child');
  settle();
  settle();
  workspace.activeChat.set('other-child');
  settle();
  assert.equal(
      attached.length, before + 1, 'unchanged identity never re-attaches');
  workspace.activeChat.set('missing-chat');
  settle();
  assert.equal(
      attached.at(-1), null,
      'an unknown Chat detaches instead of inheriting the parent');
  context.destroy();
  console.log(
      'PASS: the capture attachment follows the resolved context automatically and only on change',
  );
});
