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
import {strict as assert} from 'node:assert';

import {angularContext, bundleSources} from './lib/angular_harness.mjs';

// The real service with a fake Server: what the Model Server switch asks for,
// and in which order.
const {
  module: {
    RuntimeService,
    ReportApiService,
    ReportStateService,
    SessionListService
  },
} =
    await bundleSources(
        {
          RuntimeService: 'src/data/runtime_service',
          ReportApiService: 'src/data/report_api_service',
          ReportStateService: 'src/data/report_state_service',
          SessionListService: 'src/data/session_list_service',
        },
        'model-server-',
    );
const session = (extra = {}) => ({
  id: 'a',
  name: 'A',
  model: 'test',
  created_at: null,
  status: 'draft',
  has_capture: false,
  runs: [],
  ...extra,
});
function fixture(initial, refuse = '') {
  const items = signal([initial]), posts = [];
  const api = {
    capabilities: async () => null,
    post: async (path) => {
      posts.push(path);
      if (refuse) throw new Error(refuse);
      return {id: 'job', status: 'queued'};
    },
  };
  const sessions = {
    allItems: items,
    items,
    listing: () => ({generation_available: true}),
    load: async () => {},
  };
  const context = angularContext([
    {provide: ReportApiService, useValue: api},
    {provide: SessionListService, useValue: sessions},
    {
      provide: ReportStateService,
      useValue: {captureId: () => null, load: async () => {}}
    },
  ]);
  const service = context.run(() => new RuntimeService());
  return {
    service,
    posts,
    // What the next list poll shows.
    async poll(next) {
      items.set([next]);
      await context.flush();
    },
    flush: () => context.flush(),
  };
}

{
  const {service, posts} = fixture(session());
  await service.toggleModelServer(session());
  assert.deepEqual(
      posts, ['sessions/a/initialize'], 'Off turns on with one initialize');
}
{
  const taps = {tap_profile: 'custom-outputs-v1', tap_prepared: false};
  const {service, posts, poll} = fixture(session(taps));
  await service.toggleModelServer(session(taps));
  assert.deepEqual(
      posts, ['sessions/a/prepare'], 'selected outputs are prepared first');
  await poll(session({...taps, status: 'preparing'}));
  assert.deepEqual(
      posts,
      ['sessions/a/prepare'],
      'nothing starts while the model is being prepared',
  );
  await poll(session({...taps, tap_prepared: true}));
  assert.deepEqual(posts, ['sessions/a/prepare', 'sessions/a/initialize']);
  await poll(session({...taps, tap_prepared: true}));
  assert.equal(posts.length, 2, 'the queued start fires once');
}
{
  const taps = {tap_profile: 'custom-outputs-v1', tap_prepared: false};
  const {service, posts, poll} = fixture(session(taps));
  service.startWhenPrepared('a');
  await poll(session({...taps, status: 'preparing'}));
  await poll(session({...taps, status: 'failed'}));
  await poll(session({...taps, tap_prepared: true}));
  assert.deepEqual(posts, [], 'a failed preparation drops the queued start');
}
{
  const taps = {
    tap_profile: 'custom-outputs-v1',
    tap_prepared: false,
    status: 'preparing'
  };
  const {service, posts, poll} = fixture(session(taps));
  service.startWhenPrepared('a');
  await service.toggleModelServer(session(taps));
  await poll(session({...taps, status: 'draft', tap_prepared: true}));
  assert.deepEqual(
      posts, [], 'cancelling while preparing leaves the Model Server off');
}
for (const phase of ['starting', 'active', 'unavailable']) {
  const running = session({execution: {phase, runners: []}});
  const {service, posts} = fixture(running);
  await service.toggleModelServer(running);
  assert.deepEqual(
      posts, ['sessions/a/close'], phase + ' turns off with one close');
}
{
  const ending = session({execution: {phase: 'ending', runners: []}});
  const {service, posts} = fixture(ending);
  await service.toggleModelServer(ending);
  assert.deepEqual(
      posts, [], 'a Session that is releasing its devices ignores the switch');
}
{
  const {service} = fixture(session(), 'Device is occupied by another Session');
  assert.equal(service.modelServerState(session()), 'off');
  await service.toggleModelServer(session());
  assert.equal(service.modelServerState(session()), 'failure');
  assert.equal(
      service.lifecycleErrors()['a'], 'Device is occupied by another Session');
}
console.log(
    'PASS: Model Server switch turns on, prepares first, fires a queued start once, cancels, turns off and reports a refusal',
);
