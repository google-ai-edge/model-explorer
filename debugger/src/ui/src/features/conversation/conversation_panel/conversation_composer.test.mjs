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

import assert from 'node:assert/strict';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL('./conversation_composer.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [
    {
      name: 'angular-stub',
      setup(builder) {
        builder.onResolve({filter: /^@angular\/core$/}, () => ({path: 'core', namespace: 'stub'}));
        builder.onLoad({filter: /.*/, namespace: 'stub'}, () => ({
          contents: `export const computed=read=>read;`,
        }));
      },
    },
  ],
});
const {ConversationComposerController} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
function workspace(overrides = {}) {
  const state = {
    draft: 'hello',
    refreshError: '',
    capabilities: {available: true},
    active: true,
    busy: false,
    selected: {id: 's'},
    execution: {phase: 'active'},
    job: null,
    started: [],
    stopped: 0,
    ...overrides,
  };
  return {
    state,
    chatDraft: () => state.draft,
    setChatDraft: (value) => (state.draft = value),
    sessions: {refreshError: () => state.refreshError},
    runtime: {
      capabilities: () => state.capabilities,
      active: () => state.active,
      busy: () => state.busy,
      selected: () => state.selected,
      execution: () => state.execution,
      job: () => state.job,
      start: async (operation, prompt) => {
        state.started.push([operation, prompt]);
        return true;
      },
      stop: async () => {
        state.stopped++;
      },
    },
  };
}

test('sending needs an active, idle runtime, a fresh listing and a draft within 4 MiB', async () => {
  const w = workspace();
  const composer = new ConversationComposerController(w);
  assert.equal(composer.canSend(), true);
  await composer.send();
  assert.deepEqual(w.state.started, [['generate', 'hello']]);
  for (const blocker of [
    {draft: '  '},
    {active: false},
    {busy: true},
    {refreshError: 'stale'},
    {capabilities: {available: false}},
    {draft: 'x'.repeat(4 * 1024 * 1024 + 1)},
  ]) {
    const blocked = new ConversationComposerController(workspace(blocker));
    assert.equal(blocked.canSend(), false, JSON.stringify(Object.keys(blocker)));
  }
  assert.match(
    new ConversationComposerController(
      workspace({draft: '😀'.repeat(1024 * 1024 + 1)}),
    ).promptError(),
    /4 MiB/,
  );
});

test('the notice names the most specific blocker first', () => {
  const cases = [
    [{selected: null}, 'This draft has no runtime context'],
    [{refreshError: 'x'}, 'Runner status is unknown'],
    [{capabilities: {available: false}}, 'Runtime unavailable'],
    [{execution: {phase: 'ending'}}, 'Ending Session…'],
    [{execution: {phase: 'starting'}}, 'Starting Session…'],
    [{busy: true, job: {operation: 'initialize'}}, 'Initializing runtime'],
    [{busy: true, job: {operation: 'prepare'}}, 'Preparing tensor capture'],
    [{busy: true, job: {operation: 'generate'}}, 'Generation in progress'],
    [{active: false}, 'Turn the Model Server on to send a message'],
    [{}, ''],
  ];
  for (const [overrides, expected] of cases)
    assert.equal(new ConversationComposerController(workspace(overrides)).sendNotice(), expected);
});

test('Enter sends, Shift+Enter and IME composition do not', async () => {
  const w = workspace();
  const composer = new ConversationComposerController(w);
  const key = (init) => {
    const event = {
      key: 'Enter',
      shiftKey: false,
      isComposing: false,
      prevented: 0,
      preventDefault() {
        this.prevented++;
      },
      ...init,
    };
    composer.draftKey(event);
    return event;
  };
  assert.equal(key({shiftKey: true}).prevented, 0);
  assert.equal(key({isComposing: true}).prevented, 0);
  assert.equal(key({key: 'a'}).prevented, 0);
  assert.equal(key({}).prevented, 1);
  await Promise.resolve();
  assert.equal(w.state.started.length, 1);
  await composer.stop();
  assert.equal(w.state.stopped, 1);
});
