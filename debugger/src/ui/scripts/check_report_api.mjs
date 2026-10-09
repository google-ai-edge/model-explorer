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

import {build} from 'esbuild';
import {strict as assert} from 'node:assert';
import {fileURLToPath} from 'node:url';

// The real service with only the Angular decorator replaced.
const {outputFiles} = await build({
  entryPoints: [fileURLToPath(
      new URL('../src/data/report_api_service.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [
    {
      name: 'angular-shim',
      setup(build) {
        build.onResolve(
            {filter: /^@angular\/core$/},
            (args) => ({path: args.path, namespace: 'shim'}));
        build.onLoad(
            {filter: /.*/, namespace: 'shim'},
            () => ({
              contents: 'export const Injectable=()=>target=>target;',
            }));
      },
    },
  ],
});
const {ReportApiService, ApiError} = await import(
    'data:text/javascript;base64,' +
    Buffer.from(outputFiles[0].text).toString('base64'));
const api = new ReportApiService();
const calls = [];
let queue = [];
const respond = (status, body, headers = {}) =>
    new Response(body, {status, headers});
globalThis.fetch = async (url, init) => {
  calls.push({url, init});
  const next = queue.shift();
  if (!next) throw new Error('unexpected fetch ' + url);
  return typeof next === 'function' ? next() : next;
};

queue = [respond(
    400, JSON.stringify({error: 'Analysis queue is full; retry shortly'}),
    {'content-type': 'application/json'})];
await assert.rejects(
    api.job('x'),
    (error) => error instanceof ApiError && error.status === 400 &&
        error.message === 'Analysis queue is full; retry shortly');
queue = [respond(400, JSON.stringify({error: 'unknown_batch'}))];
await assert.rejects(api.sessions(), /unknown_batch/);
queue = [respond(
    504, '<html>Gateway Timeout</html>', {'content-type': 'text/html'})];
await assert.rejects(
    api.sessions(),
    (error) => error.status === 504 &&
        error.message === 'Backend request failed (504)');
console.log(
    'PASS: server error bodies become the message for GET; non-JSON errors keep the status');

queue = [
  respond(503, JSON.stringify({error: 'busy'}), {'retry-after': '0'}),
  respond(200, JSON.stringify({ok: true}))
];
const before = calls.length;
assert.deepEqual(await api.sessions(), {ok: true});
assert.equal(calls.length - before, 2);
queue = [respond(503, JSON.stringify({error: 'Analysis deadline exceeded'}))];
await assert.rejects(
    api.sessions(),
    (error) =>
        error.status === 503 && error.message === 'Analysis deadline exceeded');
queue = [
  respond(503, JSON.stringify({error: 'busy'}), {'retry-after': '1'}),
  respond(503, JSON.stringify({error: 'still busy'}), {'retry-after': '1'})
];
await assert.rejects(api.sessions(), /still busy/);
console.log(
    'PASS: 503 with Retry-After is retried exactly once; without it the error surfaces at once');

queue = [respond(200, JSON.stringify({id: 'job'}))];
assert.deepEqual(
    await api.post('sessions/x/turns', {prompt: 'p'}), {id: 'job'});
assert.equal(calls.at(-1).init.method, 'POST');
assert.equal(JSON.parse(calls.at(-1).init.body).prompt, 'p');
assert.equal(calls.at(-1).url, '/api/sessions/x/turns');
queue = [respond(400, JSON.stringify({error: 'request_id is required'}))];
await assert.rejects(
    api.post('sessions/x/turns', {}), /request_id is required/);
queue = [() => {
  throw new DOMException('aborted', 'AbortError');
}];
await assert.rejects(api.sessions(), {name: 'AbortError'});
console.log(
    'PASS: POST bodies, POST errors and aborted requests keep their meaning');

class FakeXHR {
  static last = null;
  upload = {};
  status = 0;
  responseText = '';
  headers = {};
  aborted = false;
  constructor() {
    FakeXHR.last = this;
  }
  open(method, url) {
    this.method = method;
    this.url = url;
  }
  setRequestHeader(name, value) {
    this.headers[name] = value;
  }
  send(body) {
    this.body = body;
  }
  abort() {
    this.aborted = true;
    this.onabort?.();
  }
  respond(status, text) {
    this.status = status;
    this.responseText = text;
    this.onload?.();
  }
}
globalThis.XMLHttpRequest = FakeXHR;
const progress = [];
const controller = new AbortController();
const uploading = api.upload({name: 'm.litertlm', size: 10}, {
  signal: controller.signal,
  onProgress: (loaded, total) => progress.push([loaded, total])
});
const request = FakeXHR.last;
request.upload.onprogress({lengthComputable: true, loaded: 5, total: 10});
request.respond(200, JSON.stringify({artifact: 'artifacts/m.litertlm'}));
assert.deepEqual(await uploading, {artifact: 'artifacts/m.litertlm'});
assert.deepEqual(progress, [[5, 10]]);
assert.equal(request.method, 'POST');
assert.equal(request.url, '/api/artifacts/upload');
assert.equal(request.headers['X-File-Name'], 'm.litertlm');
assert.equal(request.body.size, 10);
const failing = api.upload({name: 'm', size: 1});
FakeXHR.last.respond(
    400,
    JSON.stringify({error: 'Expected an artifact file of at most 20 GiB'}));
await assert.rejects(
    failing,
    (error) =>
        error instanceof ApiError && /at most 20 GiB/.test(error.message));
const cancelled = new AbortController();
const aborting = api.upload({name: 'm', size: 1}, {signal: cancelled.signal});
cancelled.abort();
await assert.rejects(aborting, {name: 'AbortError'});
assert.equal(FakeXHR.last.aborted, true);
console.log(
    'PASS: XHR upload reports progress, surfaces server errors and honours AbortSignal');
