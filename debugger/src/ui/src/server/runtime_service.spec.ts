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

import {signal, WritableSignal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {captureId} from './capture_id';
import {CaptureStore} from './capture_store';
import type {RuntimeJob} from './contracts/jobs';
import type {ModelServerState, SessionSummary} from './contracts/sessions';
import {JobsService} from './jobs_service';
import {RuntimeService} from './runtime_service';
import {ServerClient} from './server_client';
import {SessionListService} from './session_list_service';
import {deferred, flushMicrotasks as flush} from './testing/session_fixtures';

/** A client whose GETs and POSTs go to the spec's fakes; everything else is the real client. */
function fakeClient(
  get: (path: string) => Promise<unknown>,
  post: (path: string, payload: unknown) => Promise<unknown> = async (path) => {
    throw new Error('unexpected POST ' + path);
  },
) {
  const client = new ServerClient();
  client.get = get as ServerClient['get'];
  client.post = post as ServerClient['post'];
  return client;
}

describe('RuntimeService: the Model Server switch', () => {
  // The real service with a fake Server: what the Model Server switch asks for, and in which order.
  const button = (allowed = true, reason = '') => ({label: 'Turn off', allowed, reason});
  const server = (state: ModelServerState, switchButton: ReturnType<typeof button> | null) => ({
    state,
    text: state,
    error: '',
    button: switchButton,
  });
  const row = (extra: Partial<SessionSummary> = {}) => {
    const listed: Partial<SessionSummary> = {
      id: 'a',
      name: 'A',
      model: 'test',
      created_at: null,
      status_text: 'Active',
      model_server: server('on', button()),
      has_capture: false,
      runs: [],
      ...extra,
    };
    return listed as SessionSummary;
  };

  function fixture(initial: SessionSummary, refuse = '') {
    const items = signal([initial]);
    const posts: string[] = [];
    const client = fakeClient(
      async (path) => {
        if (path !== 'runtime/capabilities') {
          throw new Error('unexpected GET ' + path);
        }
        return null;
      },
      async (path) => {
        posts.push(path);
        if (refuse) {
          throw new Error(refuse);
        }
        return {id: 'job', status: 'queued'};
      },
    );
    TestBed.configureTestingModule({
      providers: [
        {provide: ServerClient, useValue: client},
        {provide: SessionListService, useValue: {items, load: async () => {}}},
        {provide: CaptureStore, useValue: {captureId: () => null, load: async () => {}}},
      ],
    });
    const service = TestBed.inject(RuntimeService);
    return {
      service,
      posts,
    };
  }

  for (const state of ['starting', 'on'] as const) {
    it(`${state} turns off with one close`, async () => {
      const running = row({model_server: server(state, button())});
      const {service, posts} = fixture(running);
      await service.closeModelServer(running);
      expect(posts, state + ' turns off with one close').toStrictEqual(['sessions/a/close']);
    });
  }

  it('a switch the server refuses, or an ended Session, sends nothing', async () => {
    const releasing = row({model_server: server('on', button(false, 'Releasing both devices.'))});
    const ended = row({model_server: server('ended', null)});
    const {service, posts} = fixture(releasing);
    await service.closeModelServer(releasing);
    await service.closeModelServer(ended);
    expect(posts).toStrictEqual([]);
  });

  it('reports a refusal of the server', async () => {
    const {service} = fixture(row(), 'Releasing both devices.');
    await service.closeModelServer(row());
    expect(service.lifecycleErrors()['a']).toBe('Releasing both devices.');
  });

  it('capabilities follow the device listing', async () => {
    // No device when the page loads; the local runner appears later. Listing the devices again (the
    // editor does on opening, and on Refresh devices) reads the capabilities again after it.
    const listed: {devices: object[]} = {devices: []};
    const gets: string[] = [];
    const client = fakeClient(async (path) => {
      gets.push(path);
      if (path === 'devices') {
        return {devices: listed.devices};
      }
      if (path === 'runtime/capabilities') {
        return {available: listed.devices.length > 0, reason: '', models: []};
      }
      throw new Error('unexpected GET ' + path);
    });
    TestBed.configureTestingModule({
      providers: [
        {provide: ServerClient, useValue: client},
        {provide: SessionListService, useValue: {items: signal([]), load: async () => {}}},
        {provide: CaptureStore, useValue: {captureId: () => null, load: async () => {}}},
      ],
    });
    const service = TestBed.inject(RuntimeService);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(service.capabilities()!.available, 'no device at load').toBe(false);
    listed.devices = [{id: 'local', name: 'This computer', statusText: 'Connected'}];
    gets.length = 0;
    await service.refreshDevices();
    expect(gets, 'capabilities follow the listing').toStrictEqual([
      'devices',
      'runtime/capabilities',
    ]);
    expect(service.capabilities()!.available, 'the device that appeared is offered').toBe(true);
    expect(service.devices().length).toBe(1);
  });
});

describe('RuntimeService: the job stream', () => {
  // Run the real service with only browser time and transport replaced.
  const warning = 'Connection interrupted. Reconnecting to the local task…';
  afterEach(() => vi.unstubAllGlobals());

  /** Records every value a writable signal is set to, for assertions on transient states. */
  function recordHistory<T>(signalRef: WritableSignal<T>) {
    const history: T[] = [];
    const set = signalRef.set;
    signalRef.set = (value: T) => {
      history.push(value);
      set(value);
    };
    signalRef.update = (fn: (value: T) => T) => signalRef.set(fn(signalRef()));
    return history;
  }

  type Handler = ((event: object) => void) | null;
  /** The browser's EventSource as far as the service uses it; the spec drives each stream. */
  class FakeEventSource {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSED = 2;
    static sources: FakeEventSource[] = [];
    readyState = FakeEventSource.CONNECTING;
    onopen: Handler = null;
    onerror: Handler = null;
    onmessage: ((event: {data: string}) => void) | null = null;
    constructor(readonly url: string) {
      FakeEventSource.sources.push(this);
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
    message(data: object) {
      this.readyState = FakeEventSource.OPEN;
      this.onmessage?.({data: JSON.stringify(data)});
    }
    close() {
      this.readyState = FakeEventSource.CLOSED;
    }
  }

  async function fixture() {
    let now = 0;
    let nextId = 0;
    const timers = new Map<number, {at: number; callback: () => void}>();
    FakeEventSource.sources = [];
    const sources = FakeEventSource.sources;
    const clock = {
      setTimeout(callback: () => void, delay: number) {
        const id = ++nextId;
        timers.set(id, {at: now + delay, callback});
        return id;
      },
      clearTimeout(id: number) {
        timers.delete(id);
      },
      tick(ms: number) {
        const end = now + ms;
        while (true) {
          const due = [...timers]
            .filter(([, timer]) => timer.at <= end)
            .sort((a, b) => a[1].at - b[1].at)[0];
          if (!due) {
            break;
          }
          now = due[1].at;
          timers.delete(due[0]);
          due[1].callback();
        }
        now = end;
      },
    };
    const job: RuntimeJob = {
      id: 'job-a',
      session_id: 'one',
      operation: 'initialize',
      status: 'running',
      error: '',
      sequence: 1,
      output: {},
      reasoning: {},
      turn: 0,
    };
    const api = {
      capabilities: async (): Promise<unknown> => ({available: true, models: []}),
      job: async (_id: string): Promise<unknown> => ({...job}),
      post: async (path: string, _payload: unknown): Promise<unknown> => {
        throw new Error('unexpected POST ' + path);
      },
    };
    let loads = 0;
    const sessions = {
      items: (): object[] => [
        {id: 'one', job_id: job.id, chats: [{n: 1, has_capture: true}]},
        {id: 'two', chats: []},
      ],
      load: async () => {
        loads++;
      },
    };
    const report = {
      captureId: () => captureId(job.session_id, 1),
      load: async (): Promise<unknown> => undefined,
    };
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('setTimeout', clock.setTimeout);
    vi.stubGlobal('clearTimeout', clock.clearTimeout);
    // The real request files over a client whose requests go to `api`; streams stay real.
    const client = fakeClient(
      (path) =>
        path === 'runtime/capabilities' ? api.capabilities() : api.job(path.replace(/^jobs\//, '')),
      (path, payload) => api.post(path, payload),
    );
    TestBed.configureTestingModule({
      providers: [
        {provide: ServerClient, useValue: client},
        {provide: SessionListService, useValue: sessions},
        {provide: CaptureStore, useValue: report},
      ],
    });
    const service = TestBed.inject(RuntimeService);
    const jobs = TestBed.inject(JobsService);
    const errors = recordHistory(service.error);
    await service.attach('one');
    await flush();
    return {
      service,
      jobs,
      errors,
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

  it('three normal 25-second rotations reconnect without flashing an error', async () => {
    const {service, errors, source, clock, timers} = await fixture();
    source.open();
    for (let i = 0; i < 3; i++) {
      clock.tick(25000);
      source.fail();
      await flush();
      expect(service.error()).toBe('');
      clock.tick(3000);
      source.open();
      clock.tick(5000);
      expect(service.error()).toBe('');
      expect(timers.size).toBe(0);
    }
    expect(errors.includes(warning)).toBe(false);
  });

  it('persistent SSE loss stays visible despite healthy REST; a valid update clears it', async () => {
    const {service, jobs, source, clock, timers} = await fixture();
    source.open();
    source.fail();
    await flush();
    clock.tick(3000);
    source.fail();
    await flush();
    clock.tick(1999);
    expect(service.error()).toBe('');
    clock.tick(1);
    expect(service.error(), 'repeated errors must not postpone the initial deadline').toBe(warning);
    source.message({sequence: 2, type: 'progress', message: 'Still running'});
    expect(service.error()).toBe('');
    expect(timers.size).toBe(0);
    expect(jobs.job()!.progress).toBe('Still running');
  });

  it('failed REST fallback is visible; a stale failure after reconnect cannot restore the warning', async () => {
    const {service, source, api} = await fixture();
    source.open();
    api.job = async () => {
      throw new Error('offline');
    };
    source.fail();
    await flush();
    expect(service.error()).toBe(warning);
    source.open();
    expect(service.error()).toBe('');
    const pending = deferred();
    api.job = () => pending.promise;
    source.fail();
    source.open();
    pending.reject(new Error('late failure'));
    await flush();
    expect(service.error()).toBe('');
  });

  it('follows a job another tab started, as a reload would; the one it follows, once', async () => {
    const {service, jobs, api, job, sources} = await fixture();
    const asked: string[] = [];
    api.job = async (id: string) => {
      asked.push(id);
      return {...job, id, operation: 'generate', prompt: 'Hi', turn: 2};
    };
    await service.follow('job-a');
    expect(asked, 'the job it follows already is not fetched again').toStrictEqual([]);
    await service.follow('job-b');
    await flush();
    expect(asked).toStrictEqual(['job-b']);
    expect(jobs.job()!.id).toBe('job-b');
    expect(jobs.job()!.prompt).toBe('Hi');
    expect(sources.length, 'its live reply is followed').toBe(2);
    expect(sources[0].readyState, 'the old stream is closed').toBe(2);
    await service.follow('job-b');
    expect(asked).toStrictEqual(['job-b']);
  });

  it('attach closes the old source and isolates old timers, callbacks and HTTP results by epoch', async () => {
    const {service, jobs, source, api, clock, timers, loads} = await fixture();
    source.open();
    const pending = deferred();
    api.job = () => pending.promise;
    source.fail();
    await service.attach('two');
    expect(timers.size).toBe(0);
    expect(source.readyState).toBe(2);
    service.error.set('New Session error');
    source.onerror!({});
    source.onopen!({});
    source.message({sequence: 99, type: 'progress', message: 'Stale'});
    pending.resolve({status: 'completed'});
    await flush();
    clock.tick(10000);
    expect(service.error()).toBe('New Session error');
    expect(jobs.job()).toBe(null);
    expect(loads()).toBe(0);
    expect(timers.size).toBe(0);
  });

  it('source identity also isolates a replacement watcher in the same epoch', async () => {
    const {service, source, job, sources, clock, timers} = await fixture();
    source.open();
    source.fail();
    await flush();
    service['watch']({...job, id: 'job-b'}, service['epoch']);
    const replacement = sources.at(-1)!;
    replacement.open();
    service.error.set('Current task error');
    source.onopen!({});
    source.onerror!({});
    clock.tick(10000);
    expect(service.error()).toBe('Current task error');
    expect(timers.size).toBe(0);
  });

  for (const via of ['message', 'fallback']) {
    it(`terminal SSE and fallback status both finish the job and cancel pending warnings: ${via}`, async () => {
      const {service, jobs, source, api, job, clock, timers, loads} = await fixture();
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
      expect(jobs.job()!.status).toBe('completed');
      expect(source.readyState).toBe(2);
      expect(timers.size).toBe(0);
      expect(loads()).toBe(1);
      clock.tick(10000);
      expect(service.error()).toBe('');
    });
  }

  it("a new chat, a turn and its cancel go to the Session's record; turning it off closes it; detached runtime cannot start", async () => {
    const {service, api, job, sessions} = await fixture();
    const calls: Array<{path: string; payload: unknown}> = [];
    sessions.items = () => [
      {
        id: 'session',
        execution: {phase: 'active'},
        model_server: {
          state: 'on',
          text: 'On',
          error: '',
          button: {label: 'Turn off', allowed: true, reason: ''},
        },
      },
    ];
    await service.attach('session');
    expect(service.captureRecordId()).toBe('session');
    expect(service.selected()!.id).toBe('session');
    expect(service.execution()!.phase).toBe('active');
    api.post = async (path, payload) => {
      calls.push({path, payload});
      // A new chat whose start completed at once; a turn still running.
      return path.endsWith('/chats')
        ? {...job, id: 'chat-job', session_id: 'session', operation: 'chat', status: 'completed'}
        : {
            ...job,
            id: 'turn-job',
            session_id: 'session',
            operation: 'generate',
            status: 'running',
          };
    };
    api.job = async (id: string) =>
      id === 'chat-job'
        ? {...job, id, session_id: 'session', operation: 'chat', status: 'completed'}
        : {...job};
    const generation = {ref: {}, target: {samplerType: 'greedy'}};
    expect(await service.startChat(generation)).toBe(true);
    expect(calls[0]).toStrictEqual({path: 'sessions/session/chats', payload: {generation}});
    await service.start('generate', 1, 'a prompt');
    expect(calls[1]).toStrictEqual({
      path: 'sessions/session/turns',
      payload: {prompt: 'a prompt', chat: 1},
    });
    await service.stop();
    expect(calls[2].path).toBe('jobs/turn-job/cancel');
    await service.closeModelServer(service.selected()!);
    expect(calls[3].path).toBe('sessions/session/close');
    await service.attach(null);
    expect(service.captureRecordId()).toBe(null);
    expect(await service.start('generate', 1, 'missing')).toBe(false);
    expect(await service.startChat(generation)).toBe(false);
    expect(calls.length).toBe(4);
  });

  it('duplicate and older SSE deltas do not append output twice', async () => {
    const {jobs, source} = await fixture();
    source.open();
    source.message({sequence: 2, type: 'delta', channel: 'content', runId: 'ref', text: 'A'});
    source.message({sequence: 2, type: 'delta', channel: 'content', runId: 'ref', text: 'A'});
    source.message({sequence: 1, type: 'delta', channel: 'content', runId: 'ref', text: 'old'});
    expect(jobs.job()!.output['ref']).toBe('A');
  });

  it("a 'reasoning' delta adds to the side's thinking, not to its reply", async () => {
    const {jobs, source} = await fixture();
    source.open();
    source.message({sequence: 2, type: 'delta', channel: 'reasoning', runId: 'ref', text: 'Hm'});
    source.message({sequence: 3, type: 'delta', channel: 'reasoning', runId: 'ref', text: 'm'});
    source.message({sequence: 4, type: 'delta', channel: 'content', runId: 'ref', text: '4'});
    expect(jobs.job()!.reasoning).toStrictEqual({ref: 'Hmm'});
    expect(jobs.job()!.output['ref']).toBe('4');
  });

  it("a stopped generation keeps each side's streamed output, its prompt and the server's note", async () => {
    const {jobs, source, api, job} = await fixture();
    source.open();
    const note = 'This turn has no debug data: a stopped turn is not captured.';
    source.message({
      sequence: 2,
      type: 'delta',
      channel: 'content',
      runId: 'ref',
      text: 'Once upon',
    });
    source.message({sequence: 3, type: 'delta', channel: 'content', runId: 'target', text: 'Once'});
    // The server's stopped job: the output it streamed, its prompt and its note.
    api.job = async () => ({
      ...job,
      operation: 'generate',
      prompt: 'Tell a story',
      status: 'cancelled',
      error: 'Generation stopped.',
      output: {ref: 'Once upon', target: 'Once'},
      note,
    });
    source.message({sequence: 4, type: 'status', status: 'cancelled', error: 'Stopped.', note});
    expect(jobs.job()!.note).toBe(note);
    await flush();
    expect(jobs.job()!.status).toBe('cancelled');
    expect(jobs.job()!.note).toBe(note);
    expect(jobs.job()!.prompt).toBe('Tell a story');
    expect(jobs.job()!.output).toStrictEqual({ref: 'Once upon', target: 'Once'});
  });

  it('a turn that did not fit shows no runtime error: the composer says why', async () => {
    const {jobs, service, source, api, job} = await fixture();
    source.open();
    const reason =
      "This input doesn't fit: this turn needs 2100 tokens, the chat has 1990 left. Shorten it and send again.";
    const doesNotFit = {ref: {tokens: 2100, left: 1990}, target: {tokens: 2100, left: 1990}};
    api.job = async () => ({
      ...job,
      operation: 'generate',
      status: 'failed',
      error: reason,
      does_not_fit: doesNotFit,
    });
    source.message({sequence: 2, type: 'status', status: 'failed', error: reason});
    await flush();
    expect(jobs.job()!.does_not_fit).toStrictEqual(doesNotFit);
    expect(service.error()).toBe('');
  });

  it('a completed old generation cannot overwrite current runtime state after its report reload resolves', async () => {
    const {service, source, api, job, report} = await fixture();
    source.open();
    const reading = deferred();
    let reloaded = false;
    api.job = async () => ({...job, operation: 'generate', status: 'completed', chat: 1});
    report.load = () => {
      reloaded = true;
      return reading.promise;
    };
    source.message({sequence: 2, type: 'status', status: 'completed'});
    await flush();
    expect(reloaded).toBe(true);
    await service.attach('two');
    service.error.set('Current capture error');
    reading.resolve(undefined);
    await flush();
    expect(service.error()).toBe('Current capture error');
  });

  it('a failed turn of a chat that kept a capture reads the chat anew; one without none', async () => {
    // A rejected turn fails its job but is kept: the page waits for the chat's capture.
    for (const kept of [true, false]) {
      const {source, api, job, sessions, report} = await fixture();
      sessions.items = () => [{id: 'one', job_id: job.id, chats: [{n: 1, has_capture: kept}]}];
      let reloaded = false;
      report.load = async () => {
        reloaded = true;
      };
      source.open();
      api.job = async () => ({...job, operation: 'generate', status: 'failed', chat: 1});
      source.message({sequence: 2, type: 'status', status: 'failed', error: 'Rejected.'});
      await flush();
      expect(reloaded).toBe(kept);
      TestBed.resetTestingModule();
    }
  });

  it('a closed EventSource is re-created with backoff and a terminal REST state finishes the job', async () => {
    const {service, jobs, source, api, job, sources, clock} = await fixture();
    source.open();
    source.closed();
    await flush();
    expect(sources.length, 'the replacement waits for the backoff timer').toBe(1);
    expect(service.error()).toMatch(/Reconnecting/);
    clock.tick(1000);
    expect(sources.length, 'a permanently closed stream is re-created').toBe(2);
    const second = sources.at(-1)!;
    expect(second.url).toMatch(/after=1$/);
    second.open();
    expect(service.error()).toBe('');
    second.closed();
    await flush();
    clock.tick(1000);
    expect(sources.length, 'a successful open resets the backoff').toBe(3);
    api.job = async () => ({...job, status: 'completed'});
    sources.at(-1)!.closed();
    await flush();
    expect(jobs.job()!.status).toBe('completed');
    expect(service.error()).toBe('');
  });

  it('a stream closed because the task is unknown clears the job instead of retrying forever', async () => {
    const {service, jobs, source, api, sources, clock} = await fixture();
    source.open();
    api.job = async () => {
      throw Object.assign(new Error('Unknown task'), {status: 400});
    };
    source.closed();
    await flush();
    expect(jobs.job()).toBe(null);
    expect(service.error()).toMatch(/no longer exists/);
    clock.tick(60000);
    await flush();
    expect(sources.length).toBe(1);
  });

  it('after five failed re-creations the service polls the job until it ends', async () => {
    const {service, jobs, source, api, job, sources, clock} = await fixture();
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
      current = sources.at(-1)!;
    }
    expect(sources.length).toBe(6);
    current.closed();
    await flush();
    expect(service.error()).toMatch(/every few seconds/);
    const before = polls;
    clock.tick(2000);
    await flush();
    expect(polls).toBe(before + 1);
    expect(sources.length, 'polling mode opens no further streams').toBe(6);
    api.job = async () => ({...job, status: 'completed'});
    clock.tick(2000);
    await flush();
    expect(jobs.job()!.status).toBe('completed');
  });
});
