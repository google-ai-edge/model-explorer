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
import {ArtifactsApi} from './artifacts_api';
import {captureId} from './capture_id';
import {JobsApi} from './jobs_api';
import {KvApi} from './kv_api';
import {ApiError, ServerClient} from './server_client';
import {SessionsApi} from './sessions_api';
import {deferred} from './testing/session_fixtures';
import {TokenAnalysisApi} from './token_analysis_api';

describe('ServerClient and the request files', () => {
  // The real client and request files; only the browser's transport is replaced.
  afterEach(() => vi.unstubAllGlobals());

  /** Answers each fetch with the next of `queue`: a response, or a function that answers. */
  function fakeFetch() {
    const calls: Array<{url: string; init: RequestInit}> = [];
    const queue: Array<Response | (() => Response)> = [];
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      calls.push({url, init});
      const next = queue.shift();
      if (!next) {
        throw new Error('unexpected fetch ' + url);
      }
      return typeof next === 'function' ? next() : next;
    });
    return {calls, queue};
  }
  const respond = (status: number, body: string, headers: Record<string, string> = {}) =>
    new Response(body, {status, headers});

  it('server error bodies become the message for GET; non-JSON errors keep the status', async () => {
    const {queue} = fakeFetch();
    const jobs = TestBed.inject(JobsApi);
    const sessions = TestBed.inject(SessionsApi);
    queue.push(
      respond(400, JSON.stringify({error: 'Analysis queue is full; retry shortly'}), {
        'content-type': 'application/json',
      }),
    );
    await expect(jobs.job('x')).rejects.toSatisfy(
      (error) =>
        error instanceof ApiError &&
        error.status === 400 &&
        error.message === 'Analysis queue is full; retry shortly',
    );
    queue.push(respond(400, JSON.stringify({error: 'unknown_batch'})));
    await expect(sessions.sessions()).rejects.toThrow(/unknown_batch/);
    queue.push(respond(504, '<html>Gateway Timeout</html>', {'content-type': 'text/html'}));
    await expect(sessions.sessions()).rejects.toSatisfy(
      (error) =>
        (error as ApiError).status === 504 &&
        (error as ApiError).message === 'Backend request failed (504)',
    );
  });

  it('POST bodies, POST errors and aborted requests keep their meaning', async () => {
    const {calls, queue} = fakeFetch();
    const client = TestBed.inject(ServerClient);
    const sessions = TestBed.inject(SessionsApi);
    queue.push(respond(200, JSON.stringify({id: 'job'})));
    expect(await client.post('sessions/x/turns', {prompt: 'p'})).toStrictEqual({id: 'job'});
    expect(calls.at(-1)!.init.method).toBe('POST');
    expect(JSON.parse(calls.at(-1)!.init.body as string).prompt).toBe('p');
    expect(calls.at(-1)!.url).toBe('/api/sessions/x/turns');
    queue.push(respond(400, JSON.stringify({error: 'Type a prompt.'})));
    await expect(client.post('sessions/x/turns', {})).rejects.toThrow(/Type a prompt\./);
    queue.push(() => {
      throw new DOMException('aborted', 'AbortError');
    });
    await expect(sessions.sessions()).rejects.toMatchObject({name: 'AbortError'});
  });

  it('XHR upload reports progress, surfaces server errors and honours AbortSignal', async () => {
    class FakeXHR {
      static last: FakeXHR;
      upload: {onprogress?: (event: object) => void} = {};
      status = 0;
      responseText = '';
      headers: Record<string, string> = {};
      aborted = false;
      method = '';
      url = '';
      body: unknown;
      onload?: () => void;
      onabort?: () => void;
      constructor() {
        FakeXHR.last = this;
      }
      open(method: string, url: string) {
        this.method = method;
        this.url = url;
      }
      setRequestHeader(name: string, value: string) {
        this.headers[name] = value;
      }
      send(body: unknown) {
        this.body = body;
      }
      abort() {
        this.aborted = true;
        this.onabort?.();
      }
      respond(status: number, text: string) {
        this.status = status;
        this.responseText = text;
        this.onload?.();
      }
    }
    vi.stubGlobal('XMLHttpRequest', FakeXHR);
    const artifacts = TestBed.inject(ArtifactsApi);
    const file = (name: string, size: number) => ({name, size}) as File;
    const progress: [number, number][] = [];
    const controller = new AbortController();
    const uploading = artifacts.upload(file('m.litertlm', 10), {
      signal: controller.signal,
      onProgress: (loaded, total) => progress.push([loaded, total]),
    });
    const request = FakeXHR.last;
    request.upload.onprogress!({lengthComputable: true, loaded: 5, total: 10});
    request.respond(200, JSON.stringify({artifact: 'artifacts/m.litertlm'}));
    expect(await uploading).toStrictEqual({artifact: 'artifacts/m.litertlm'});
    expect(progress).toStrictEqual([[5, 10]]);
    expect(request.method).toBe('POST');
    expect(request.url).toBe('/api/artifacts/upload');
    expect(request.headers['X-File-Name']).toBe('m.litertlm');
    expect((request.body as File).size).toBe(10);
    const failing = artifacts.upload(file('m', 1));
    FakeXHR.last.respond(
      400,
      JSON.stringify({error: 'Expected an artifact file of at most 20 GiB'}),
    );
    await expect(failing).rejects.toSatisfy(
      (error) => error instanceof ApiError && /at most 20 GiB/.test(error.message),
    );
    const cancelled = new AbortController();
    const aborting = artifacts.upload(file('m', 1), {signal: cancelled.signal});
    cancelled.abort();
    await expect(aborting).rejects.toMatchObject({name: 'AbortError'});
    expect(FakeXHR.last.aborted).toBe(true);
    const beforeCall = FakeXHR.last;
    await expect(
      artifacts.upload(file('m', 1), {signal: cancelled.signal}),
    ).rejects.toMatchObject({name: 'AbortError'});
    expect(FakeXHR.last).toBe(beforeCall);
  });

  it('a job event stream opens below the same base URL', () => {
    class FakeEventSource {
      constructor(readonly url: string) {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);
    expect(TestBed.inject(JobsApi).events('job', 3).url).toBe('/api/jobs/job/events?after=3');
  });

  it('requests name the Session and chat of their capture across out-of-order responses; missing IDs never query server default; lifecycle POST stays unscoped', async () => {
    const calls: Array<{
      url: string;
      options: RequestInit;
      response: ReturnType<typeof deferred<object>>;
    }> = [];
    vi.stubGlobal('fetch', (url: string, options: RequestInit) => {
      const response = deferred<object>();
      calls.push({url, options, response});
      return response.promise;
    });
    const client = TestBed.inject(ServerClient);
    const sessions = TestBed.inject(SessionsApi);
    const kv = TestBed.inject(KvApi);
    const tokens = TestBed.inject(TokenAnalysisApi);
    const signal = new AbortController().signal;
    const first = kv.kvMetadata(captureId('parent&capture', 1), 1, signal);
    const second = kv.kvMetadata(captureId('child', 3), 2, signal);
    expect(calls[0].url).toBe(
      '/api/kv-analysis?mode=metadata&turn=1&session_id=parent%26capture&chat=1',
    );
    expect(calls[1].url).toBe('/api/kv-analysis?mode=metadata&turn=2&session_id=child&chat=3');
    calls[1].response.resolve({ok: true, json: async () => ({identity: 'child'})});
    expect(((await second) as unknown as {identity: string}).identity).toBe('child');
    calls[0].response.resolve({ok: true, json: async () => ({identity: 'parent'})});
    expect(((await first) as unknown as {identity: string}).identity).toBe('parent');
    const missing = [
      () => sessions.session(null),
      // A Session alone names no capture: each chat has its own.
      () => sessions.session('child'),
      () => kv.kvMetadata(null, 1, signal),
      () =>
        tokens.tokenAnalysis(
          null,
          {turn: 1, stage: 'decode', alignment: 'content', rows: []},
          signal,
        ),
    ];
    for (const request of missing) {
      await expect(request()).rejects.toThrow(/chat must be selected/);
    }
    expect(calls.length, 'missing capture must reject before fetch').toBe(2);
    const post = sessions.turn('child', 2, 'request');
    expect(calls[2].url).toBe('/api/sessions/child/turns');
    // A turn sends only its prompt and its chat: the reply's length is the chat's setting.
    expect(JSON.parse(calls[2].options.body as string)).toStrictEqual({
      prompt: 'request',
      chat: 2,
    });
    calls[2].response.resolve({ok: true, json: async () => ({})});
    await post;
    // A new chat sends each side's settings.
    const generation = {ref: {samplerType: 'greedy'}, target: {}};
    const started = sessions.startChat('child', generation);
    expect(calls[3].url).toBe('/api/sessions/child/chats');
    expect(JSON.parse(calls[3].options.body as string)).toStrictEqual({generation});
    calls[3].response.resolve({ok: true, json: async () => ({})});
    await started;
    for (const api of [client, sessions, kv, tokens]) {
      expect('selectedId' in api).toBe(false);
    }
  });
});
