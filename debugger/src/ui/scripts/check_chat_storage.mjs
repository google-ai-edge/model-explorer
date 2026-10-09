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

import {parseStoredChats, pruneStoredChats} from '../src/features/sessions/chat_storage.ts';

const value = {
  version: 1,
  chats: {s: [{id: 'legacy', name: 'Draft'}]},
  drafts: {'s:server-chat': 'saved\nmessage'},
  active: {s: 'server-chat'}
};
assert.deepEqual(
    parseStoredChats(JSON.stringify(value)),
    {chats: value.chats, drafts: value.drafts, active: value.active});
assert.deepEqual(
    parseStoredChats('{invalid'), {chats: {}, drafts: {}, active: {}});
assert.equal(
    parseStoredChats(JSON.stringify({...value, active: {s: 42}})).active.s,
    undefined);
assert.equal(
    parseStoredChats(JSON.stringify({...value, drafts: {x: 'a'.repeat(16001)}}))
        .drafts.x.length,
    16000);
console.log(
    'PASS: draft roundtrip, server chat ID restoration, corrupt storage, invalid types, input limit');
const stored = {
  chats: {keep: [{id: 'c', name: 'C'}], gone: [{id: 'd', name: 'D'}]},
  drafts: {'keep:c': 'draft', 'keep:capture': '', 'gone:d': 'old'},
  active: {keep: 'c', gone: 'd'}
};
assert.deepEqual(pruneStoredChats(stored, ['keep']), {
  chats: {keep: [{id: 'c', name: 'C'}]},
  drafts: {'keep:c': 'draft'},
  active: {keep: 'c'}
});
assert.deepEqual(
    pruneStoredChats(stored, []), {chats: {}, drafts: {}, active: {}});
console.log('PASS: pruning keeps only listed sessions and drops empty drafts');
