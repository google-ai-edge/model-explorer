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

import {angularContext, bundleSources, deferred, flush, recordHistory,} from './lib/angular_harness.mjs';

// Run the real service with only browser time and transport replaced.
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
        'runtime-reconnect-',
    );
const realGlobals = {
  setTimeout: globalThis.setTimeout,
  clearTimeout: globalThis.clearTimeout,
  EventSource: globalThis.EventSource,
};
process.once('exit', () => Object.assign(globalThis, realGlobals));
const warning = 'Connection interrupted. Reconnecting to the local task…';
async function fixture() {
  let now = 0, nextId = 0;
  const timers = new Map(), sources = [];
  const clock = {
    setTimeout(callback, delay) {
      const id = ++nextId;
      timers.set(id, {at: now + delay, callback});
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    tick(ms) {
      const end = now + ms;
      while (true) {
        const due = [...timers]
                        .filter(([, timer]) => timer.at <= end)
                        .sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        now = due[1].at;
        timers.delete(due[0]);
        due[1].callback();
      }
      now = end;
    },
  };
  class FakeEventSource {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 2;
    readyState = FakeEventSource.CONNECTING;
    constructor(url) {
      this.url = url;
      sources.push(this);
    }
    open() {
      this.readyState = FakeEventSource.OPEN;
      this.onopen?.({});
    }
    fail() {
      this.readyState = FakeEventSource.CONNECTING;
      this.onerror?.({});
    }
    closed() {
      this.readyState = FakeEventSource.CLOSED;
      this.onerror?.({});
    }
    message(data) {
      this.readyState = FakeEventSource.OPEN;
      this.onmessage?.({data: JSON.stringify(data)});
    }
    close() {
      this.readyState = FakeEventSource.CLOSED;
    }
  }
  const job = {
    id: 'job-a',
    session_id: 'one',
    operation: 'initialize',
    status: 'running',
    error: '',
    sequence: 1,
    output: {},
    turn: 0,
  };
  const api = {
    capabilities: async () => ({available: true, models: []}),
    job: async () => ({...job}),
  };
  let loads = 0;
  const sessions = {
    allItems: () => [{id: 'one', job_id: job.id}, {id: 'two'}],
    load: async () => {
      loads++;
    },
  };
  const report = {captureId: () => job.session_id, load: async () => {}};
  globalThis.EventSource = FakeEventSource;
  globalThis.setTimeout = clock.setTimeout;
  globalThis.clearTimeout = clock.clearTimeout;
  const context = angularContext([
    {provide: ReportApiService, useValue: api},
    {provide: SessionListService, useValue: sessions},
    {provide: ReportStateService, useValue: report},
  ]);
  const service = context.run(() => new RuntimeService());
  recordHistory(service.error);
  await service.attach('one');
  await flush();
  return {
    service,
    api,
    job,
    sessions,
    report,
    clock,
    timers,
    sources,
    source: sources[0],
    loads: () => loads,
  };
}

{
  const {service, source, clock, timers} = await fixture();
  source.open();
  for (let i = 0; i < 3; i++) {
    clock.tick(25000);
    source.fail();
    await flush();
    assert.equal(service.error(), '');
    clock.tick(3000);
    source.open();
    clock.tick(5000);
    assert.equal(service.error(), '');
    assert.equal(timers.size, 0);
  }
  assert.equal(service.error.history.includes(warning), false);
  console.log(
      'PASS: three normal 25-second rotations reconnect without flashing an error');
}
{
  const {service, source, clock, timers} = await fixture();
  source.open();
  source.fail();
  await flush();
  clock.tick(3000);
  source.fail();
  await flush();
  clock.tick(1999);
  assert.equal(service.error(), '');
  clock.tick(1);
  assert.equal(
      service.error(), warning,
      'repeated errors must not postpone the initial deadline');
  source.message({sequence: 2, type: 'progress', message: 'Still running'});
  assert.equal(service.error(), '');
  assert.equal(timers.size, 0);
  assert.equal(service.job().progress, 'Still running');
  console.log(
      'PASS: persistent SSE loss stays visible despite healthy REST; a valid update clears it',
  );
}
{
  const {service, source, api} = await fixture();
  source.open();
  api.job = async () => {
    throw new Error('offline');
  };
  source.fail();
  await flush();
  assert.equal(service.error(), warning);
  source.open();
  assert.equal(service.error(), '');
  const pending = deferred();
  api.job = () => pending.promise;
  source.fail();
  source.open();
  pending.reject(new Error('late failure'));
  await flush();
  assert.equal(service.error(), '');
  console.log(
      'PASS: failed REST fallback is visible; a stale failure after reconnect cannot restore the warning',
  );
}
{
  const {service, source, api, clock, timers, loads} = await fixture();
  source.open();
  const pending = deferred();
  api.job = () => pending.promise;
  source.fail();
  await service.attach('two');
  assert.equal(timers.size, 0);
  assert.equal(source.readyState, 2);
  service.error.set('New Session error');
  source.onerror({});
  source.onopen({});
  source.message({sequence: 99, type: 'progress', message: 'Stale'});
  pending.resolve({status: 'completed'});
  await flush();
  clock.tick(10000);
  assert.equal(service.error(), 'New Session error');
  assert.equal(service.job(), null);
  assert.equal(loads(), 0);
  assert.equal(timers.size, 0);
  console.log(
      'PASS: attach closes the old source and isolates old timers, callbacks and HTTP results by epoch',
  );
}
{
  const {service, source, job, sources, clock, timers} = await fixture();
  source.open();
  source.fail();
  await flush();
  service.watch({...job, id: 'job-b'}, service.epoch);
  const replacement = sources.at(-1);
  replacement.open();
  service.error.set('Current task error');
  source.onopen({});
  source.onerror({});
  clock.tick(10000);
  assert.equal(service.error(), 'Current task error');
  assert.equal(timers.size, 0);
  console.log(
      'PASS: source identity also isolates a replacement watcher in the same epoch');
}
for (const via of ['message', 'fallback']) {
  const {service, source, api, job, clock, timers, loads} = await fixture();
  source.open();
  const completed = {...job, status: 'completed'};
  if (via === 'message') {
    source.fail();
    await flush();
    api.job = async () => completed;
    source.message({sequence: 2, type: 'status', status: 'completed'});
  } else {
    api.job = async () => completed;
    source.fail();
  }
  await flush();
  assert.equal(service.job().status, 'completed');
  assert.equal(source.readyState, 2);
  assert.equal(timers.size, 0);
  assert.equal(loads(), 1);
  clock.tick(10000);
  assert.equal(service.error(), '');
}
console.log(
    'PASS: terminal SSE and fallback status both finish the job and cancel pending warnings',
);

{
  const {service, api, job, sessions} = await fixture();
  const calls = [];
  sessions.allItems = () =>
      [{id: 'parent', execution: {phase: 'active'}},
       {id: 'child', parent_session_id: 'parent'},
  ];
  await service.attach('child');
  assert.equal(service.captureRecordId(), 'child');
  assert.equal(service.selected().id, 'child');
  assert.equal(service.ownerSession().id, 'parent');
  api.post = async (path, payload) => {
    calls.push({path, payload});
    return {
      ...job,
      id: 'child-job',
      session_id: 'child',
      operation: 'generate',
      status: 'running'
    };
  };
  await service.start('generate', 'child prompt');
  assert.equal(calls[0].path, 'sessions/child/turns');
  await service.stop();
  assert.equal(calls[1].path, 'jobs/child-job/cancel');
  await service.toggleLifecycle(service.ownerSession());
  assert.equal(calls[2].path, 'sessions/parent/close');
  await service.attach(null);
  assert.equal(service.captureRecordId(), null);
  assert.equal(await service.start('generate', 'missing'), false);
  assert.equal(calls.length, 3);
  console.log(
      'PASS: child Chat generation/cancel keeps its record; ending the owner Session closes it; detached runtime cannot start',
  );
}
{
  const {service, source} = await fixture();
  source.open();
  source.message({sequence: 2, type: 'delta', runId: 'ref', text: 'A'});
  source.message({sequence: 2, type: 'delta', runId: 'ref', text: 'A'});
  source.message({sequence: 1, type: 'delta', runId: 'ref', text: 'old'});
  assert.equal(service.job().output.ref, 'A');
  console.log(
      'PASS: duplicate and older SSE deltas do not append output twice');
}

{
  const {service, source, api, job, report} = await fixture();
  source.open();
  const reading = deferred();
  let reloaded = false;
  api.job = async () => ({...job, operation: 'generate', status: 'completed'});
  report.load = () => {
    reloaded = true;
    return reading.promise;
  };
  source.message({sequence: 2, type: 'status', status: 'completed'});
  await flush();
  assert.equal(reloaded, true);
  await service.attach('two');
  service.error.set('Current capture error');
  reading.resolve();
  await flush();
  assert.equal(service.error(), 'Current capture error');
  console.log(
      'PASS: a completed old generation cannot overwrite current runtime state after its report reload resolves',
  );
}

{
  const {service, source, api, job, sources, clock} = await fixture();
  source.open();
  source.closed();
  await flush();
  assert.equal(
      sources.length, 1, 'the replacement waits for the backoff timer');
  assert.match(service.error(), /Reconnecting/);
  clock.tick(1000);
  assert.equal(sources.length, 2, 'a permanently closed stream is re-created');
  const second = sources.at(-1);
  assert.match(second.url, /after=1$/);
  second.open();
  assert.equal(service.error(), '');
  second.closed();
  await flush();
  clock.tick(1000);
  assert.equal(sources.length, 3, 'a successful open resets the backoff');
  api.job = async () => ({...job, status: 'completed'});
  sources.at(-1).closed();
  await flush();
  assert.equal(service.job().status, 'completed');
  assert.equal(service.error(), '');
  console.log(
      'PASS: a closed EventSource is re-created with backoff and a terminal REST state finishes the job',
  );
}
{
  const {service, source, api, sources, clock} = await fixture();
  source.open();
  api.job = async () => {
    const error = new Error('Unknown task');
    error.status = 400;
    throw error;
  };
  source.closed();
  await flush();
  assert.equal(service.job(), null);
  assert.match(service.error(), /no longer exists/);
  clock.tick(60000);
  await flush();
  assert.equal(sources.length, 1);
  console.log(
      'PASS: a stream closed because the task is unknown clears the job instead of retrying forever',
  );
}
{
  const {service, source, api, job, sources, clock} = await fixture();
  source.open();
  let polls = 0;
  api.job = async () => {
    polls++;
    return {...job};
  };
  let current = source;
  for (let attempt = 1; attempt <= 5; attempt++) {
    current.closed();
    await flush();
    clock.tick(16000);
    current = sources.at(-1);
  }
  assert.equal(sources.length, 6);
  current.closed();
  await flush();
  assert.match(service.error(), /every few seconds/);
  const before = polls;
  clock.tick(2000);
  await flush();
  assert.equal(polls, before + 1);
  assert.equal(sources.length, 6, 'polling mode opens no further streams');
  api.job = async () => ({...job, status: 'completed'});
  clock.tick(2000);
  await flush();
  assert.equal(service.job().status, 'completed');
  console.log(
      'PASS: after five failed re-creations the service polls the job until it ends');
}
