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

import type {CapturedToken} from '../server/contracts/capture';
import type {TokenSelection} from './conversation_content';
import {
  captureViewState,
  pairIdentity,
  prepareViewRestore,
  readingPositionKey,
  readViewState,
  samePairIdentity,
  savedReadingPosition,
  switchReadingPosition,
  validViewState,
  type ConversationReadingPosition,
  type TokenDiffViewState,
  type ViewRestoreResult,
} from './conversation_view_state';
import type {AlignmentMode, TokenPair} from './token_alignment';

/** A restore the spec expects ready; its status is checked beside it. */
type Ready = Extract<ViewRestoreResult, {status: 'ready'}>;
const asReady = (result: ViewRestoreResult) => result as Ready;
/** The token selection a restored state holds. */
const selected = (state: TokenDiffViewState) => state.selection as TokenSelection;
/** A saved view as a page reads it back: any shape, which the code checks. */
type Saved = Record<string, unknown>;

describe('conversation view state', () => {
  const state: TokenDiffViewState = {
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
    anchor: {kind: 'token', ti: 1, side: 'target', step: 65536, stage: 'Output', offset: 108},
  };
  const original: TokenPair = {
    index: 3,
    ref: {id: 4, step: 2, text: 'A'} as CapturedToken,
    target: {id: 9, step: 3, text: null} as CapturedToken,
    match: null,
  };
  const identity = pairIdentity(original)!;
  let requestedAlignment: AlignmentMode | undefined;
  const context = {
    hasTurn: (turn: number) => turn === 1,
    rows: (_turn: number, alignment: AlignmentMode) => {
      requestedAlignment = alignment;
      return [{...original, index: 100}];
    },
  };

  it('unknown versions and partial/invalid snapshots are rejected atomically before query, selection, or anchor application', () => {
    expect(validViewState(state)).toBe(true);
    // A view saved while views still kept logit states restores: unknown keys are ignored.
    expect(
      validViewState({...state, logitStates: {'1:content': {open: true, metric: 'relative_l2'}}}),
    ).toBe(true);
    // An admitted Prefill token is a selection of its own, addressed by its aligned pair index.
    expect(validViewState({...state, selection: {kind: 'input', turn: 1, index: 4}})).toBe(true);
    expect(validViewState({...state, selection: {kind: 'input', turn: 1, index: -2}})).toBe(false);
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
      {anchor: {kind: 'token', ti: 1, side: '<script>', step: 3, stage: 'Output', offset: 0}},
      {anchor: {kind: 'input', ti: 1, position: -1, stage: 'Prefill', offset: 0}},
      {anchor: {kind: 'token', ti: -1, side: 'ref', step: 1, stage: 'Output', offset: 0}},
      {selection: {kind: 'token', turn: 1, index: -1}},
      {selection: undefined},
      {color: true},
      {color: 'invalid'},
      {identity: {ref: {step: '0', text: 'X'}}},
      {identity: {}},
    ];
    for (const delta of invalid) {
      const value = {...state, ...delta};
      expect(validViewState(value), JSON.stringify(delta)).toBe(false);
      expect(readViewState(value)).toBe(null);
      let inspected = false;
      expect(
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
      ).toBe('invalid');
      expect(
        inspected,
        'invalid snapshot must be rejected before looking at selection or capture data',
      ).toBe(false);
    }
  });

  it('only version 3 is read; earlier and future versions are ignored', () => {
    const old: Saved = {...state, version: 2, color: true};
    delete old.mode;
    expect(validViewState(old)).toBe(false);
    expect(readViewState(old)).toBe(null);
    expect(readViewState({...old, version: 1})).toBe(null);
  });

  it('a saved panel that is not valid is dropped alone, not the whole view', () => {
    // One panel no stage can hold (a token panel of a Prefill) is dropped; the view is kept.
    const panel = (id: string, stage: string, kind: string) => ({
      id,
      turn: 1,
      stage,
      kind,
      open: true,
    });
    const saved = {
      ...state,
      analysisPanels: [
        panel('analysis-1', 'decode', 'kv'),
        panel('analysis-2', 'prefill', 'tokens'),
      ],
      inspectedPanel: 'analysis-2',
    };
    expect(validViewState(saved)).toBe(false);
    const read = readViewState(saved)!;
    expect(read.analysisPanels).toStrictEqual([panel('analysis-1', 'decode', 'kv')]);
    expect(read.inspectedPanel).toBe(null);
    expect(read.formula).toBe(state.formula);
    // A panel the view names that it never held still makes the view invalid.
    expect(readViewState({...saved, inspectedPanel: 'analysis-9'})).toBe(null);
  });

  it('the shared restore path resolves stable captured-token identity across alignment changes and preserves unknown text, stage selection, and unavailable captures', () => {
    expect(samePairIdentity({...original, index: 100}, identity)).toBe(true);
    expect(
      samePairIdentity(
        {...original, target: {id: 9, step: 3, text: 'new B'} as CapturedToken},
        identity,
      ),
    ).toBe(false);
    expect(samePairIdentity({...original, target: undefined}, identity)).toBe(false);
    const ready = asReady(prepareViewRestore({...state, identity}, context));
    expect(ready.status).toBe('ready');
    expect(selected(ready.state).index).toBe(100);
    expect(requestedAlignment).toBe('content');
    expect(ready.state.identity!.target!.text).toBe(null);
    expect(
      prepareViewRestore({...state, identity}, {...context, hasTurn: () => false}).status,
    ).toBe('missing-turn');
    expect(
      prepareViewRestore(
        {...state, identity},
        {...context, rows: () => [{...original, target: {...original.target!, text: 'changed'}}]},
      ).status,
    ).toBe('missing-token');
    expect(
      prepareViewRestore(
        {...state, identity},
        {
          ...context,
          rows: () => {
            throw Error('alignment unavailable');
          },
        },
      ).status,
    ).toBe('missing-token');
    expect(
      prepareViewRestore(
        {...state, selection: {kind: 'stage', turn: 1, stage: 'prefill', side: 'ref'}},
        {
          ...context,
          rows: () => {
            throw Error('stage restore must not align tokens');
          },
        },
      ).status,
    ).toBe('ready');
    expect(prepareViewRestore({...state, selection: null}, context).status).toBe('ready');
  });

  it('snapshots preserve Find, Display, Alignment, groups, selection, and reading fields without sharing mutable references', () => {
    const snapshot = captureViewState({...state, identity});
    snapshot.groups.token = true;
    (snapshot.anchor as {offset: number}).offset = 0;
    snapshot.identity!.target!.text = 'mutated';
    expect(state.groups.token).toBe(false);
    expect((state.anchor as {offset: number}).offset).toBe(108);
    expect(identity.target!.text).toBe(null);
    expect(readViewState(state)).toStrictEqual(state);
  });

  it('Chat and Debug restore their own offsets and semantic anchors, isolate Sessions, and ignore corrupt reading positions', () => {
    const positions = new Map<string, ConversationReadingPosition>();
    const chat: ConversationReadingPosition = {
        scrollTop: 300,
        anchor: {kind: 'input', ti: 1, position: 40, stage: 'Prefill', offset: 25},
      },
      debug: ConversationReadingPosition = {scrollTop: 9000, anchor: state.anchor};
    expect(
      switchReadingPosition(positions, 'session', null, 'Chat', {scrollTop: 0, anchor: null}),
    ).toStrictEqual({scrollTop: 0, anchor: null});
    expect(
      switchReadingPosition(positions, 'session', 'Chat', 'Debug', chat),
      'first Debug visit must not inherit the outgoing Chat anchor',
    ).toStrictEqual({scrollTop: 0, anchor: null});
    expect(switchReadingPosition(positions, 'session', 'Debug', 'Chat', debug)).toStrictEqual(chat);
    const back = switchReadingPosition(positions, 'session', 'Chat', 'Debug', {
      ...chat,
      scrollTop: 350,
    });
    expect(back).toStrictEqual(debug);
    (back.anchor as {offset: number}).offset = -100;
    expect(
      (savedReadingPosition(positions, 'session', 'Debug').anchor as {offset: number}).offset,
    ).toBe(108);
    expect(savedReadingPosition(positions, 'other-session', 'Debug')).toStrictEqual({
      scrollTop: 0,
      anchor: null,
    });
    positions.set(readingPositionKey('session', 'Debug'), {scrollTop: -1, anchor: null});
    expect(savedReadingPosition(positions, 'session', 'Debug')).toStrictEqual({
      scrollTop: 0,
      anchor: null,
    });
  });

  it('shared observation changes navigate without token selection; unchanged reading, same-Turn token identity, phase navigation and old-v3 migration remain independent', () => {
    const crossTurn = asReady(
      prepareViewRestore(
        {...state, identity},
        {
          hasTurn: (turn: number) => turn === 2,
          rows: () => {
            throw Error('outgoing Turn must not be resolved');
          },
        },
        {turn: 2, phase: 'prefill'},
      ),
    );
    expect(crossTurn.status).toBe('ready');
    expect(crossTurn.state.selection).toBe(null);
    expect(crossTurn.state.identity).toBe(undefined);
    expect(crossTurn.observation).toStrictEqual({turn: 2, phase: 'prefill'});
    for (const field of [
      'formula',
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
    ] as const)
      expect(crossTurn.state[field], field + ' should survive cross-Turn return').toStrictEqual(
        state[field],
      );
    const sameTurn = asReady(
      prepareViewRestore({...state, identity}, context, {turn: 1, phase: 'prefill'}),
    );
    expect(sameTurn.status).toBe('ready');
    expect(selected(sameTurn.state).index).toBe(100);
    expect(sameTurn.observation).toBe(null);
    const noSelection: TokenDiffViewState = {...state, selection: null};
    const movedWithoutSelection = asReady(
      prepareViewRestore(noSelection, context, {turn: 2, phase: 'decode'}),
    );
    expect(movedWithoutSelection.status).toBe('ready');
    expect(movedWithoutSelection.state.selection).toBe(null);
    expect(movedWithoutSelection.observation).toStrictEqual({turn: 2, phase: 'decode'});
    expect(movedWithoutSelection.state.scrollTop).toBe(state.scrollTop);
    expect(movedWithoutSelection.state.formula).toBe(state.formula);
    const unmovedWithoutSelection = asReady(
      prepareViewRestore(noSelection, context, {turn: 1, phase: 'prefill'}),
    );
    expect(
      unmovedWithoutSelection.observation,
      'unchanged shared observation preserves independent reading even with no token selection',
    ).toBe(null);
    const changedPhase = asReady(
      prepareViewRestore({...state, identity}, context, {turn: 1, phase: 'decode'}),
    );
    expect(changedPhase.observation).toStrictEqual({turn: 1, phase: 'decode'});
    expect(
      selected(changedPhase.state).index,
      'a phase change preserves a valid selected token within the same Turn',
    ).toBe(100);
    const oldV3: Saved = {...noSelection};
    delete oldV3.observation;
    expect(readViewState(oldV3)!.observation).toBe(null);
    expect(
      asReady(prepareViewRestore(oldV3, context, {turn: 2, phase: 'decode'})).observation,
      'older snapshots cannot fabricate an observation change',
    ).toBe(null);
    const clonedObservation = captureViewState(noSelection);
    clonedObservation.observation!.turn = 9;
    expect(noSelection.observation!.turn).toBe(1);
  });

  it('ordinary return preserves the shared observation, clears an old Turn selection, restores local settings, and retains valid selection within the same Turn', () => {
    const kept = asReady(prepareViewRestore({...state, identity}, context));
    expect(kept.status).toBe('ready');
    expect(selected(kept.state).turn).toBe(1);
  });

  it('a saved view keeps the identity of tokens captured without their text', () => {
    const pair: TokenPair = {
      index: 5,
      ref: {step: 5, text: null, id: 106} as CapturedToken,
      target: {step: 6, text: null, id: 106} as CapturedToken,
      match: null,
    };
    expect(samePairIdentity(pair, pairIdentity(pair)!)).toBe(true);
    const saved = {
      version: 3,
      mode: 'Debug',
      formula: 'NOT token_match',
      alignment: 'steps',
      selection: {kind: 'token', turn: 1, index: 5},
      identity: pairIdentity(pair),
      details: true,
      boundaries: true,
      whitespace: false,
      color: 'token_match',
      collapsed: [],
      thinking: [],
      infoExpanded: true,
      groups: {},
      panelWidth: 390,
      scrollTop: 0,
      anchor: null,
    };
    expect(validViewState(readViewState(saved))).toBe(true);
    expect(readViewState(saved)!.identity!.target!.text).toBe(null);
  });
});
