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
            '../src/features/conversation/conversation_view_state.ts',
            import.meta.url),
        ),
  ],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {
  validViewState,
  readViewState,
  captureViewState,
  prepareViewRestore,
  pairIdentity,
  samePairIdentity,
  readingPositionKey,
  savedReadingPosition,
  switchReadingPosition,
} =
    await import(
        'data:text/javascript;base64,' +
        Buffer.from(outputFiles[0].text).toString('base64'));
const state = {
  version: 3,
  mode: 'Debug',
  observation: {turn: 1, phase: 'prefill'},
  formula: 'NOT token_match',
  alignment: 'content',
  selection: {kind: 'token', turn: 1, index: 3},
  details: true,
  boundaries: true,
  whitespace: false,
  color: 'token_match',
  collapsed: ['1:target:prefill'],
  thinking: ['1:ref'],
  infoExpanded: true,
  groups: {token: false},
  panelWidth: 390,
  scrollTop: 1000,
  anchor: {
    kind: 'token',
    ti: 1,
    side: 'target',
    step: 65536,
    stage: 'Output',
    offset: 108
  },
  trends: {'1:content': {open: true, metric: 'relative_l2', start: 2, end: 7}},
};
assert.equal(validViewState(state), true);
// An admitted Prefill token is a selection of its own, addressed by its aligned
// pair index.
assert.equal(
    validViewState({...state, selection: {kind: 'input', turn: 1, index: 4}}),
    true);
assert.equal(
    validViewState({...state, selection: {kind: 'input', turn: 1, index: -2}}),
    false);
const invalid = [
  {version: 99},
  {version: '3'},
  {formula: 'NOT ('},
  {mode: 'Graph'},
  {mode: {}},
  {observation: {turn: 0, phase: 'decode'}},
  {observation: {turn: 1, phase: null}},
  {observation: {turn: 1}},
  {panelWidth: 99999},
  {scrollTop: -5},
  {collapsed: ['bad']},
  {
    anchor: {
      kind: 'token',
      ti: 1,
      side: '<script>',
      step: 3,
      stage: 'Output',
      offset: 0
    }
  },
  {anchor: {kind: 'input', ti: 1, position: -1, stage: 'Prefill', offset: 0}},
  {
    anchor: {
      kind: 'token',
      ti: -1,
      side: 'ref',
      step: 1,
      stage: 'Output',
      offset: 0
    }
  },
  {selection: {kind: 'token', turn: 1, index: -1}},
  {selection: undefined},
  {color: true},
  {color: 'invalid'},
  {identity: {ref: {step: '0', text: 'X'}}},
  {identity: {}},
  {trends: []},
  {trends: {'1:content': {open: true, metric: 'unknown', start: 0, end: 10}}},
  {trends: {'1:content': {open: true, metric: 'js', start: 5, end: 2}}},
];
for (const delta of invalid) {
  const value = {...state, ...delta};
  assert.equal(validViewState(value), false, JSON.stringify(delta));
  assert.equal(readViewState(value), null);
  let inspected = false;
  assert.equal(
      prepareViewRestore(value, {
        hasTurn: () => {
          inspected = true;
          return true;
        },
        rows: () => {
          inspected = true;
          return [];
        },
      }).status,
      'invalid',
  );
  assert.equal(
      inspected,
      false,
      'invalid snapshot must be rejected before looking at selection or capture data',
  );
}
console.log(
    'PASS: unknown versions and partial/invalid snapshots are rejected atomically before query, selection, or anchor application',
);
const legacy = {
  ...state,
  version: 2,
  color: true
};
delete legacy.mode;
const migrated = readViewState(legacy);
assert.equal(migrated.version, 3);
assert.equal(migrated.mode, 'Debug');
assert.equal(migrated.color, 'token_match');
assert.equal(readViewState({...legacy, color: false}).color, 'none');
assert.equal(validViewState(legacy), false);
assert.equal(readViewState({...legacy, version: 1}), null);
console.log(
    'PASS: valid v2 bookmarks migrate to explicit Debug mode and normalized display colors; earlier and future versions remain unsupported',
);
const original = {
  index: 3,
  ref: {id: 4, step: 2, text: 'A'},
  target: {id: 9, step: 3, text: null},
  match: null,
};
const identity = pairIdentity(original);
assert.equal(samePairIdentity({...original, index: 100}, identity), true);
assert.equal(
    samePairIdentity(
        {...original, target: {id: 9, step: 3, text: 'new B'}}, identity),
    false,
);
assert.equal(
    samePairIdentity({...original, target: undefined}, identity), false);
let requestedAlignment;
const context = {
  hasTurn: (turn) => turn === 1,
  rows: (_turn, alignment) => {
    requestedAlignment = alignment;
    return [{...original, index: 100}];
  },
};
const ready = prepareViewRestore({...state, identity}, context);
assert.equal(ready.status, 'ready');
assert.equal(ready.state.selection.index, 100);
assert.equal(requestedAlignment, 'content');
assert.equal(ready.state.identity.target.text, null);
assert.equal(
    prepareViewRestore({...state, identity}, context, {
      source: 'bookmark',
      expected: {turn: 2, alignment: 'content'},
    }).status,
    'inconsistent-turn',
);
assert.equal(
    prepareViewRestore({...state, identity}, {...context, hasTurn: () => false})
        .status,
    'missing-turn',
);
assert.equal(
    prepareViewRestore(
        {...state, identity},
        {
          ...context,
          rows: () =>
              [{...original, target: {...original.target, text: 'changed'}}]
        },
        )
        .status,
    'missing-token',
);
assert.equal(
    prepareViewRestore(
        {...state, identity},
        {
          ...context,
          rows: () => {
            throw Error('alignment unavailable');
          },
        },
        )
        .status,
    'missing-token',
);
assert.equal(
    prepareViewRestore(
        {
          ...state,
          selection: {kind: 'stage', turn: 1, stage: 'prefill', side: 'ref'}
        },
        {
          ...context,
          rows: () => {
            throw Error('stage restore must not align tokens');
          },
        },
        )
        .status,
    'ready',
);
assert.equal(
    prepareViewRestore({...state, selection: null}, context).status, 'ready');
console.log(
    'PASS: the shared restore path resolves stable captured-token identity across alignment changes and preserves unknown text, stage selection, and unavailable captures',
);
const snapshot = captureViewState({...state, identity});
snapshot.groups.token = true;
snapshot.trends['1:content'].start = 4;
snapshot.anchor.offset = 0;
snapshot.identity.target.text = 'mutated';
assert.equal(state.groups.token, false);
assert.equal(state.trends['1:content'].start, 2);
assert.equal(state.anchor.offset, 108);
assert.equal(identity.target.text, null);
assert.deepEqual(readViewState(state), state);
console.log(
    'PASS: snapshots preserve Find, Display, Alignment, groups, Trends, selection, and reading fields without sharing mutable references',
);
const positions = new Map();
const chat = {
  scrollTop: 300,
  anchor: {kind: 'input', ti: 1, position: 40, stage: 'Prefill', offset: 25},
},
      debug = {
        scrollTop: 9000,
        anchor: state.anchor
      };
assert.deepEqual(
    switchReadingPosition(
        positions, 'session:chat', null, 'Chat', {scrollTop: 0, anchor: null}),
    {scrollTop: 0, anchor: null},
);
assert.deepEqual(
    switchReadingPosition(positions, 'session:chat', 'Chat', 'Debug', chat),
    {scrollTop: 0, anchor: null},
    'first Debug visit must not inherit the outgoing Chat anchor',
);
assert.deepEqual(
    switchReadingPosition(positions, 'session:chat', 'Debug', 'Chat', debug),
    chat);
const back = switchReadingPosition(positions, 'session:chat', 'Chat', 'Debug', {
  ...chat,
  scrollTop: 350,
});
assert.deepEqual(back, debug);
back.anchor.offset = -100;
assert.equal(
    savedReadingPosition(positions, 'session:chat', 'Debug').anchor.offset,
    108);
assert.deepEqual(
    savedReadingPosition(positions, 'other-session:chat', 'Debug'), {
      scrollTop: 0,
      anchor: null,
    });
positions.set(
    readingPositionKey('session:chat', 'Debug'), {scrollTop: -1, anchor: null});
assert.deepEqual(savedReadingPosition(positions, 'session:chat', 'Debug'), {
  scrollTop: 0,
  anchor: null,
});
console.log(
    'PASS: Chat and Debug restore their own offsets and semantic anchors, isolate Sessions/Chats, and ignore corrupt reading positions',
);

const crossTurn = prepareViewRestore(
    {...state, identity},
    {
      hasTurn: (turn) => turn === 2,
      rows: () => {
        throw Error('outgoing Turn must not be resolved');
      },
    },
    {source: 'return', observation: {turn: 2, phase: 'prefill'}},
);
assert.equal(crossTurn.status, 'ready');
assert.equal(crossTurn.state.selection, null);
assert.equal(crossTurn.state.identity, undefined);
assert.deepEqual(crossTurn.observation, {turn: 2, phase: 'prefill'});
for (const field
         of ['formula',
             'alignment',
             'details',
             'boundaries',
             'whitespace',
             'color',
             'collapsed',
             'thinking',
             'infoExpanded',
             'groups',
             'panelWidth',
             'trends',
])
  assert.deepEqual(
      crossTurn.state[field],
      state[field],
      field + ' should survive cross-Turn return',
  );
const sameTurn = prepareViewRestore({...state, identity}, context, {
  source: 'return',
  observation: {turn: 1, phase: 'prefill'},
});
assert.equal(sameTurn.status, 'ready');
assert.equal(sameTurn.state.selection.index, 100);
assert.equal(sameTurn.observation, null);
const noSelection = {
  ...state,
  selection: null
};
const movedWithoutSelection = prepareViewRestore(noSelection, context, {
  source: 'return',
  observation: {turn: 2, phase: 'decode'},
});
assert.equal(movedWithoutSelection.status, 'ready');
assert.equal(movedWithoutSelection.state.selection, null);
assert.deepEqual(movedWithoutSelection.observation, {turn: 2, phase: 'decode'});
assert.equal(movedWithoutSelection.state.scrollTop, state.scrollTop);
assert.equal(movedWithoutSelection.state.formula, state.formula);
const unmovedWithoutSelection = prepareViewRestore(noSelection, context, {
  source: 'return',
  observation: {turn: 1, phase: 'prefill'},
});
assert.equal(
    unmovedWithoutSelection.observation,
    null,
    'unchanged shared observation preserves independent reading even with no token selection',
);
const changedPhase = prepareViewRestore({...state, identity}, context, {
  source: 'return',
  observation: {turn: 1, phase: 'decode'},
});
assert.deepEqual(changedPhase.observation, {turn: 1, phase: 'decode'});
assert.equal(
    changedPhase.state.selection.index,
    100,
    'a phase change preserves a valid selected token within the same Turn',
);
const oldV3 = {...noSelection};
delete oldV3.observation;
assert.equal(readViewState(oldV3).observation, null);
assert.equal(
    prepareViewRestore(oldV3, context, {
      source: 'return',
      observation: {turn: 2, phase: 'decode'},
    }).observation,
    null,
    'older snapshots cannot fabricate an observation change',
);
const clonedObservation = captureViewState(noSelection);
clonedObservation.observation.turn = 9;
assert.equal(noSelection.observation.turn, 1);
console.log(
    'PASS: shared observation changes navigate without token selection; unchanged reading, same-Turn token identity, phase navigation and old-v3 migration remain independent',
);
const bookmark = prepareViewRestore({...state, identity}, context, {
  source: 'bookmark',
  expected: {turn: 1, alignment: 'content'},
});
assert.equal(bookmark.status, 'ready');
assert.equal(bookmark.state.selection.turn, 1);
console.log(
    'PASS: ordinary return preserves the shared observation, clears an old Turn selection, restores local settings, and retains valid selection within the same Turn',
);
