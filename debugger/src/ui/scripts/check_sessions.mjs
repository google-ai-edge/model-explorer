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

import '@angular/compiler';

import {ChangeDetectorRef, createEnvironmentInjector, ElementRef, runInInjectionContext,} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {build} from 'esbuild';
import {strict as assert} from 'node:assert';
import {mkdir, mkdtemp, rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const cache = path.join(root, 'node_modules/.cache');
await mkdir(cache, {recursive: true});
const temp = await mkdtemp(path.join(cache, 'session-regressions-'));
const originalDocument = globalThis.document;
const originalFetch = globalThis.fetch;
const originalFrame = globalThis.requestAnimationFrame;
globalThis.document = {
  visibilityState: 'hidden',
  addEventListener() {},
  removeEventListener() {}
};
globalThis.requestAnimationFrame = (fn) => {
  fn();
  return 1;
};
const disposals = [];
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
};
const session = (id = 'a', extra = {}) => ({
  id,
  name: 'Original',
  model: 'test',
  created_at: null,
  status: 'draft',
  has_capture: false,
  runs: [
    {
      id: 'ref',
      runtime: 'LiteRT-LM',
      source: 'path',
      artifact: 'model.litertlm'
    },
    {
      id: 'target',
      runtime: 'LiteRT-LM',
      source: 'path',
      artifact: 'model.litertlm'
    },
  ],
  notice: '',
  ...extra,
});
const listing = (sessions = [session()]) => ({
  sessions,
  configuration_template: sessions[0],
  capabilities: {create: true, rename: true, duplicate: true, delete: true},
  generation_available: false,
  unavailable_reason: '',
});
try {
  const bundle = path.join(temp, 'sessions.mjs');
  await build({
    stdin: {
      contents:
          `export {SessionListService} from './src/data/session_list_service'; export {ReportApiService} from './src/data/report_api_service'; export {SessionEditor} from './src/features/sessions/session_editor/session_editor'; export {RuntimeService} from './src/data/runtime_service'; export {HfFilePicker} from './src/features/sessions/hf_file_picker/hf_file_picker'; export * from './src/data/session_status';`,
      resolveDir: root,
      loader: 'ts',
    },
    bundle: true,
    packages: 'external',
    platform: 'node',
    format: 'esm',
    outfile: bundle,
    logLevel: 'silent',
  });
  const {
    SessionListService,
    ReportApiService,
    SessionEditor,
    RuntimeService,
    HfFilePicker,
    sessionStatusLabel,
    modelServerState,
    sessionEnterable,
    sessionConfigurationLocked,
    copiedSessionName,
  } = await import(pathToFileURL(bundle));
  function service(api) {
    const injector =
        createEnvironmentInjector([{provide: ReportApiService, useValue: api}]);
    disposals.push(() => injector.destroy());
    return runInInjectionContext(injector, () => new SessionListService());
  }

  // Resolve a real pre-mutation GET after a rename, then after a deletion.
  const reads = [];
  let rename = deferred();
  let mutation = deferred();
  const api = {
    sessions() {
      const read = deferred();
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
  const sessions = service(api);
  reads.shift().resolve(listing());
  await tick();
  const oldRead = sessions.load(true);
  const old = reads.shift();
  const renaming = sessions.rename('a', 'Renamed');
  rename.resolve(session('a', {name: 'Renamed'}));
  await renaming;
  old.resolve(listing());
  await oldRead;
  assert.equal(
      sessions.items()[0].name, 'Renamed',
      'late poll must not revert completed rename');
  const beforeDelete = sessions.load(true);
  const stale = reads.shift();
  const deleting = sessions.manage('delete', {id: 'a'});
  mutation.resolve(session());
  await deleting;
  stale.resolve(listing());
  await beforeDelete;
  assert.equal(
      sessions.items().length, 0, 'late poll must not resurrect deleted row');

  // Failed mutation still releases polling and keeps last-known data.
  sessions.listing.set(listing());
  rename = deferred();
  const failed = sessions.rename('a', 'Unsaved');
  const readCount = reads.length;
  await sessions.load(true);
  assert.equal(reads.length, readCount, 'polls paused during mutation');
  rename.reject(new Error('test write failure'));
  await assert.rejects(failed, /write failure/);
  const refresh = sessions.load(true);
  reads.shift().resolve(listing([session('a', {name: 'Server value'})]));
  await refresh;
  assert.equal(sessions.items()[0].name, 'Server value');
  assert.equal(sessions.loading(), false);
  const rejectedRead = sessions.load(true);
  reads.shift().reject(new Error('offline'));
  await rejectedRead;
  assert.equal(sessions.items()[0].name, 'Server value');
  assert.match(sessions.refreshError(), /offline/);
  console.log(
      'PASS: late rename/delete polls, mutation failure recovery, paused polls, last-known refresh error',
  );

  assert.equal(sessionStatusLabel(session('a', {status: 'failed'})), 'Failed');
  for (const [phase, label] of Object.entries({
         starting: 'Starting',
         active: 'Active',
         ending: 'Ending',
         ended: 'Ended',
         interrupted: 'Interrupted',
         unavailable: 'Unavailable',
       }))
    assert.equal(
        sessionStatusLabel(session('a', {execution: {phase, runners: []}})),
        label);
  // The list shows one Model Server state per Session: failure, starting, on or
  // off.
  for (const [extra, state] of [
           [{}, 'off'],
           [{status: 'saved', has_capture: true}, 'off'],
           [{status: 'preparing'}, 'starting'],
           [{status: 'initializing'}, 'starting'],
           [{execution: {phase: 'starting', runners: []}}, 'starting'],
           [{execution: {phase: 'active', runners: []}}, 'on'],
           [{execution: {phase: 'ending', runners: []}}, 'on'],
           [{execution: {phase: 'ended', runners: []}}, 'off'],
           [{status: 'failed'}, 'failure'],
           [{execution: {phase: 'interrupted', runners: []}}, 'failure'],
           [{execution: {phase: 'unavailable', runners: []}}, 'failure'],
  ])
    assert.equal(
        modelServerState(session('a', extra)), state, JSON.stringify(extra));
  assert.equal(
      modelServerState(session('a'), 'Server refused the request'), 'failure');
  // Entering needs something to do: a Model Server that is on, or captured
  // data.
  assert.equal(sessionEnterable(session('a')), false);
  assert.equal(sessionEnterable(session('a', {status: 'preparing'})), false);
  assert.equal(sessionEnterable(session('a', {status: 'failed'})), false);
  assert.equal(
      sessionEnterable(
          session('a', {execution: {phase: 'active', runners: []}})),
      true);
  assert.equal(
      sessionEnterable(session('a', {status: 'saved', has_capture: true})),
      true);
  assert.equal(
      sessionConfigurationLocked(
          session('a', {execution: {phase: 'unavailable', runners: []}})),
      true,
      'an unavailable execution still owns its slot',
  );
  assert.equal(
      sessionConfigurationLocked(
          session('a', {execution: {phase: 'active', runners: []}})),
      true,
  );
  assert.equal(
      sessionConfigurationLocked(session('a', {status: 'preparing'})), true);
  assert.equal(copiedSessionName('a'.repeat(80)).length, 80);
  assert.equal(copiedSessionName('Example'), 'Example (copy)');
  console.log(
      'PASS: shared execution/status semantics, immutable active configuration, bounded duplicate name',
  );

  const listingState = {value: listing()};
  let requests = [];
  let prepareCalls = 0;
  let failure = false;
  const sessionMock = {
    listing: () => listingState.value,
    items: () => listingState.value.sessions,
    manage: async (operation, payload) => {
      requests.push({operation, payload});
      if (failure) throw new Error('save failed');
      return session(
          payload.id ?? 'saved', {...payload, id: payload.id ?? 'saved'});
    },
    load: async () => {},
  };
  const runtimeMock = {
    devices: () => [],
    capabilities: () => null,
    refreshCapabilities: async () => {},
    api: {
      post: async () => {
        prepareCalls++;
        throw new Error('prepare failed');
      },
    },
  };
  const element = {nativeElement: {querySelector: () => null}};
  const injector = createEnvironmentInjector([
    {provide: RuntimeService, useValue: runtimeMock},
    {provide: SessionListService, useValue: sessionMock},
    {provide: ElementRef, useValue: element},
    {provide: ChangeDetectorRef, useValue: {markForCheck() {}}},
    {provide: MatDialog, useValue: {}},
  ]);
  disposals.push(() => injector.destroy());
  const editor = runInInjectionContext(injector, () => new SessionEditor());
  editor.request = () => ({mode: 'create'});
  editor.config = {
    name: 'Offline',
    model: 'test',
    runs: session().runs,
    tap_profile: 'custom-outputs-v1',
    tap_points: {'model.litertlm': ['out']},
  };
  assert.equal(editor.deviceConfigurationError(), '');
  assert.equal(editor.captureReady(), true);
  await editor.save();
  assert.equal(requests.length, 1);
  assert.equal(
      prepareCalls, 0,
      'offline saved configuration must not attempt Runner preparation');
  assert.equal(editor.saving(), false);
  failure = true;
  editor.savedId = undefined;
  await editor.save();
  assert.equal(editor.config.name, 'Offline');
  assert.match(editor.error(), /save failed/);
  assert.equal(editor.saving(), false);
  failure = false;
  listingState.value = {...listing(), generation_available: true};
  editor.deviceConfigurationError = () => '';
  editor.setScanReady('model.litertlm', true);
  editor.savedId = undefined;
  requests = [];
  await editor.save();
  assert.equal(editor.savedId, 'saved');
  assert.match(editor.error(), /Configuration saved.*preparation failed/);
  await editor.save();
  assert.deepEqual(
      requests.map((r) => r.operation),
      ['create', 'update'],
  );
  assert.equal(requests[1].payload.id, 'saved');
  console.log(
      'PASS: offline configuration save without Runner, save error retains draft, prepare retry reuses saved identity',
  );

  // Creating a Session starts its Model Server: right away, or queued behind
  // the preparation of selected outputs. Saving an existing configuration never
  // starts anything.
  {
    const calls = [];
    const starting = {
      ...runtimeMock,
      lifecycle: async (id, operation) => {
        calls.push(operation + ':' + id);
        return {id: 'job', status: 'queued'};
      },
      startWhenPrepared: (id) => calls.push('startWhenPrepared:' + id),
      toggleModelServer: async (item) => calls.push('toggle:' + item.id),
    };
    const list = {...sessionMock, allItems: () => [session('saved')]};
    const startInjector = createEnvironmentInjector([
      {provide: RuntimeService, useValue: starting},
      {provide: SessionListService, useValue: list},
      {provide: ElementRef, useValue: element},
      {provide: ChangeDetectorRef, useValue: {markForCheck() {}}},
      {provide: MatDialog, useValue: {}},
    ]);
    disposals.push(() => startInjector.destroy());
    const create = (mode, tap_profile) => {
      const item =
          runInInjectionContext(startInjector, () => new SessionEditor());
      item.request = () =>
          ({mode, session: mode === 'update' ? session('saved') : undefined});
      item.config = {
        name: 'Live',
        model: 'test',
        runs: session().runs,
        tap_profile,
        tap_points: tap_profile ? {'model.litertlm': ['out']} : {},
      };
      item.deviceConfigurationError = () => '';
      if (tap_profile) item.setScanReady('model.litertlm', true);
      const created = [];
      item.created.subscribe((id) => created.push(id));
      return {item, created};
    };
    const plain = create('create', '');
    await plain.item.save();
    assert.deepEqual(calls, ['toggle:saved']);
    assert.deepEqual(plain.created, ['saved']);
    calls.length = 0;
    const tapped = create('create', 'custom-outputs-v1');
    await tapped.item.save();
    assert.deepEqual(calls, ['prepare:saved', 'startWhenPrepared:saved']);
    assert.match(tapped.item.saveNotice(), /starts its Model Server/);
    calls.length = 0;
    const update = create('update', '');
    await update.item.save();
    assert.deepEqual(calls, []);
    assert.deepEqual(update.created, []);
    assert.doesNotMatch(update.item.saveNotice(), /Model Server/);
    console.log(
        'PASS: creating a Session starts its Model Server, queued behind preparation when outputs are selected',
    );
  }

  let observedSignal;
  const pending = deferred();
  runtimeMock.api.upload = (file, options) => {
    observedSignal = options.signal;
    return pending.promise;
  };
  const run = {id: 'ref', runtime: 'LiteRT-LM', artifact: ''};
  const file = {name: 'same.litertlm'};
  const input = {files: [file], value: 'C:\\fakepath\\same.litertlm'};
  const upload = editor.upload({target: input}, run);
  assert.equal(input.value, '');
  assert.equal(editor.uploading(), 'ref');
  editor.cancelUpload();
  assert.equal(observedSignal.aborted, true);
  // A transport that resolves even after abort cannot apply its stale result.
  pending.resolve({artifact: 'late.litertlm'});
  await upload;
  assert.equal(run.artifact, '');
  assert.equal(editor.uploading(), null);
  const progress = [];
  runtimeMock.api.upload = async (file, options) => {
    options.onProgress(5, 10);
    progress.push(editor.uploadProgress());
    return {artifact: 'same.litertlm'};
  };
  input.value = 'C:\\fakepath\\same.litertlm';
  await editor.upload({target: input}, run);
  assert.equal(run.artifact, 'same.litertlm');
  assert.equal(input.value, '');
  assert.deepEqual(progress, [0.5]);
  assert.equal(editor.uploadProgress(), null);
  console.log(
      'PASS: same-file input reset, upload cancellation, ignored late result, successful same-file retry, progress',
  );

  let uploads = 0;
  runtimeMock.api.upload = async () => {
    uploads++;
    return {artifact: 'never.litertlm'};
  };
  runtimeMock.capabilities = () => ({upload_limit_bytes: 10});
  input.files = [{name: 'big.litertlm', size: 11}];
  input.value = 'C:\\fakepath\\big.litertlm';
  await editor.upload(
      {target: input}, {id: 'ref', runtime: 'LiteRT-LM', artifact: ''});
  assert.equal(
      uploads, 0, 'oversize files are rejected before any transfer starts');
  assert.match(editor.error(), /accepts files up to/);
  assert.equal(editor.uploading(), null);
  runtimeMock.capabilities = () => null;
  console.log(
      'PASS: upload size limit from runtime capabilities is enforced client-side');

  const hfInjector = createEnvironmentInjector([]);
  disposals.push(() => hfInjector.destroy());
  const hf = runInInjectionContext(hfInjector, () => new HfFilePicker());
  hf.repository.set('org/model');
  hf.path.set('../secret.litertlm');
  assert.equal(hf.selection(), null);
  hf.path.set('sub/model.litertlm');
  assert.equal(hf.selection()?.artifact, 'sub/model.litertlm');
  console.log(
      'PASS: Hugging Face parent traversal rejected after directory migration');
} finally {
  for (const dispose of disposals.reverse()) dispose();
  globalThis.document = originalDocument;
  globalThis.fetch = originalFetch;
  globalThis.requestAnimationFrame = originalFrame;
  await rm(temp, {recursive: true, force: true});
}
