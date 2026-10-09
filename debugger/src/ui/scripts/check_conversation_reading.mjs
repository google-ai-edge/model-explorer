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

const {outputFiles} = await build({
  entryPoints: [
    fileURLToPath(
        new URL(
            '../src/features/conversation/conversation_reading_controller.ts',
            import.meta.url),
        ),
  ],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {ConversationReadingController} = await import(
    'data:text/javascript;base64,' +
    Buffer.from(outputFiles[0].text).toString('base64'));
function fixture() {
  let mode = 'Chat';
  let anchor = null;
  const tasks = [];
  const events = [];
  const positions = new Map();
  const reading = {
    scrollTop: 0,
    getBoundingClientRect: () => ({top: 0, bottom: 800, left: 0, right: 1000}),
    querySelector: (selector) => selector.includes('data-node-turn="2"') ||
            selector === '.turn[data-turn="2"]' ?
        {getBoundingClientRect: () => ({top: 1400 - reading.scrollTop})} :
        null,
  };
  const root = {
    querySelector: (selector) => (selector === '.reading' ? reading : null)
  };
  const virtualizer = {
    capture: () => {
      events.push('capture');
      return structuredClone(anchor);
    },
    mount: () => events.push('mount'),
    paint: () => events.push('paint'),
    restore: (next) => {
      events.push(['restore', structuredClone(next)]);
      anchor = structuredClone(next);
    },
    destroy: () => {
      events.push('destroy');
      anchor = null;
    },
  };
  const controller = new ConversationReadingController({
    context: 'parent:child',
    positions,
    root: () => root,
    mode: () => mode,
    virtualizer: () => virtualizer,
    afterRender: (work) => tasks.push(work),
    afterRestore: () => events.push('restored'),
  });
  return {
    controller,
    reading,
    positions,
    events,
    tasks,
    setMode: (next) => (mode = next),
    setAnchor: (next) => (anchor = structuredClone(next)),
    flush: () => {
      for (const task of tasks.splice(0)) task();
    },
  };
}
const chatAnchor = {
  kind: 'input',
  ti: 1,
  position: 40,
  stage: 'Prefill',
  offset: 25
};
const debugAnchor = {
  kind: 'token',
  ti: 1,
  side: 'target',
  step: 2049,
  stage: 'Output',
  offset: 70
};
{
  const f = fixture();
  f.controller.syncMode();
  f.flush();
  f.reading.scrollTop = 300;
  f.setAnchor(chatAnchor);
  f.setMode('Debug');
  f.controller.syncMode();
  f.flush();
  assert.equal(f.reading.scrollTop, 0);
  assert.equal(
      f.events.some((event) => Array.isArray(event)),
      false,
      'first visit cannot restore outgoing anchor',
  );
  f.reading.scrollTop = 9000;
  f.setAnchor(debugAnchor);
  f.setMode('Chat');
  f.controller.syncMode();
  f.flush();
  assert.equal(f.reading.scrollTop, 300);
  assert.deepEqual(f.events.filter(Array.isArray).at(-1)[1], chatAnchor);
  f.setMode('Debug');
  f.controller.syncMode();
  f.flush();
  assert.equal(f.reading.scrollTop, 9000);
  assert.deepEqual(f.events.filter(Array.isArray).at(-1)[1], debugAnchor);
  console.log(
      'PASS: controller restores each rendered mode offset and semantic anchor independently',
  );
}
{
  const f = fixture();
  f.controller.syncMode();
  f.flush();
  f.reading.scrollTop = 350;
  f.setAnchor(chatAnchor);
  f.setMode('Debug');
  f.controller.syncMode();
  f.setMode('Chat');
  f.controller.syncMode();
  f.flush();
  assert.equal(f.reading.scrollTop, 350);
  assert.deepEqual(f.events.filter(Array.isArray).at(-1)[1], chatAnchor);
  assert.equal(
      f.events.filter((event) => event === 'restored').length,
      2,
      'initial layout and newest callback only',
  );
  console.log(
      'PASS: a rapid mode round-trip cancels obsolete callbacks without assigning old DOM geometry to a new mode',
  );
}
{
  const f = fixture();
  f.setMode('Debug');
  f.controller.apply(
      {mode: 'Debug', scrollTop: 9000, anchor: debugAnchor}, false, {
        turn: 2,
        phase: 'prefill',
      });
  f.controller.syncMode();
  f.flush();
  assert.equal(f.reading.scrollTop, 1392);
  assert.equal(
      f.events.some(Array.isArray),
      false,
      'cross-Turn return must not restore the stale token anchor',
  );
  console.log(
      'PASS: shared Turn/phase return locates the current stage without replaying the old token anchor',
  );
}
{
  const f = fixture();
  f.controller.syncMode();
  f.flush();
  f.reading.scrollTop = 500;
  f.setAnchor(chatAnchor);
  f.controller.apply(
      {mode: 'Chat', scrollTop: 9000, anchor: debugAnchor}, true, null);
  f.controller.destroy();
  const count = f.events.length;
  f.flush();
  assert.equal(
      f.events.length, count,
      'destroyed controller must reject pending restore');
  assert.equal(f.events.at(-2), 'capture');
  assert.equal(f.events.at(-1), 'destroy');
  assert.deepEqual(
      f.controller.capture(),
      {mode: 'Chat', scrollTop: 500, anchor: chatAnchor});
  const stored = [...f.positions.values()];
  assert(stored.some(
      (value) => value.scrollTop === 500 && value.anchor.position === 40));
  console.log(
      'PASS: disposal saves final semantic location before virtualizer destruction and suppresses delayed restoration',
  );
}
