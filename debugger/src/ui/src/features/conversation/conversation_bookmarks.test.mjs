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
  entryPoints: [fileURLToPath(new URL('./conversation_bookmarks.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [
    {
      name: 'angular-signal-stub',
      setup(builder) {
        builder.onResolve({filter: /^@angular\/core$/}, () => ({path: 'core', namespace: 'stub'}));
        builder.onLoad({filter: /.*/, namespace: 'stub'}, () => ({
          contents: `export function signal(value){const read=()=>value;read.set=next=>value=next;read.update=fn=>value=fn(value);return read;}`,
        }));
      },
    },
  ],
});
const {CONVERSATION_BOOKMARK_CODEC: codec} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const storeBuild = await build({
  entryPoints: [
    fileURLToPath(new URL('../../shared/bookmark_store/bookmark_store.ts', import.meta.url)),
  ],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [
    {
      name: 'angular-signal-stub',
      setup(builder) {
        builder.onResolve({filter: /^@angular\/core$/}, () => ({path: 'core', namespace: 'stub'}));
        builder.onLoad({filter: /.*/, namespace: 'stub'}, () => ({
          contents: `export function signal(value){const read=()=>value;read.set=next=>value=next;read.update=fn=>value=fn(value);return read;}`,
        }));
      },
    },
  ],
});
const {decodeBookmarks} = await import(
  'data:text/javascript;base64,' + Buffer.from(storeBuild.outputFiles[0].text).toString('base64')
);
const parse = (raw) => decodeBookmarks(raw, codec);
const legacy = {turn: 1, index: 3, alignment: 'content'};
const stage = {
  id: 'stage',
  turn: 2,
  index: -1,
  alignment: 'steps',
  view: {version: 2, selection: {kind: 'stage', stage: 'prefill'}},
};
const expired = {id: 'expired', turn: 99, index: 0, alignment: 'content', view: {version: 99}};

test('corrupt entries are isolated; legacy, stage and expired bookmarks stay; duplicates go', () => {
  assert.deepEqual(
    parse(
      JSON.stringify([
        null,
        legacy,
        false,
        {turn: -1, index: -8, alignment: 'content'},
        stage,
        expired,
        legacy,
      ]),
    ),
    [legacy, stage, expired],
  );
});

test('unparseable or non-list storage reads as no bookmarks', () => {
  for (const raw of [null, '{', 'null', '{}', '[false,null,0]']) assert.deepEqual(parse(raw), []);
});

test('bad ids, titles and stage entries without a view are dropped', () => {
  assert.deepEqual(
    parse(
      JSON.stringify([
        {...legacy, id: 3},
        {...legacy, title: {}},
        {...legacy, index: -1},
      ]),
    ),
    [],
  );
});

test('identity prefers the id and falls back to the location', () => {
  assert.equal(codec.identity(stage), 'stage');
  assert.equal(codec.identity(legacy), '1:content:3');
});
