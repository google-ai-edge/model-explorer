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

import {computed, signal} from '@angular/core';
import {HOME, NO_PLACE, type PageAddress} from '../../state/page_address/page_address';
import {DebugAddress, readingPlace} from './debug_address';

const SESSION: PageAddress = {...HOME, session: 's', mode: 'Debug'};

/** The controller over stubbed parts: a conversation of turns 1 and 2, whose address the test
 *  sets as the router would and whose writes it records. */
function setup() {
  const address = signal<PageAddress>(SESSION);
  const selection = signal<{turn: number} | null>(null);
  const overview = signal<{turn: number; stage: 'decode'; kind: 'kv'; id: string} | null>(null);
  const written: [unknown, boolean][] = [];
  const workspace = {
    address,
    debugPlace: signal(NO_PLACE),
    showDebugPlace: (place: unknown, replace = false) => written.push([place, replace]),
  };
  const panels = {
    overviewPanel: overview,
    pendingInspector: signal<string | null>(null),
    tokenMetricRange: signal(null),
    panelsFor: () => [],
    analysisUnavailable: () => 'Not captured',
  };
  const tokens = {
    selection,
    analysisOverview: signal(null),
    metricInspector: signal(null),
    tokenMetricInspector: signal(null),
  };
  const views = {restoring: signal(false)};
  const dom = {openStage: vi.fn((turn: number) => selection.set({turn}))};
  const analyses = {showAnalysis: vi.fn(), addAnalysis: vi.fn()};
  const controller = Object.assign(Object.create(DebugAddress.prototype), {
    conversation: {state: {session: () => ({})}, turns: () => [{n: 1}, {n: 2}]},
    workspace,
    tokens,
    panels,
    analyses,
    analysisInfo: {analysisLabel: () => 'KV cache'},
    views,
    dom,
    notice: signal(''),
    place: computed(() => readingPlace(overview(), selection())),
    arrived: null,
    requested: [],
    replace: false,
  }) as DebugAddress;
  return {controller, address, selection, overview, written, views, dom};
}

describe('the Debug reading and its address', () => {
  it('opens the turn an address names, at its Prefill', () => {
    const {controller, address, dom, written} = setup();
    address.set({...SESSION, turn: 2});
    controller.follow();
    expect(dom.openStage).toHaveBeenCalledWith(2, 'prefill');
    controller.follow();
    expect(written).toEqual([]);
  });

  it('says why it cannot go to a turn the Chat lacks, and puts its own place instead', () => {
    const {controller, address, written} = setup();
    address.set({...SESSION, turn: 9});
    controller.follow();
    expect(controller.notice()).toBe('This Session has no turn 9.');
    expect(written).toEqual([[NO_PLACE, true]]);
  });

  it('says why an analysis cannot be added, and adds none', () => {
    const {controller, address, written} = setup();
    address.set({...SESSION, turn: 1, stage: 'prefill', analysis: 'kv'});
    controller.follow();
    expect(controller.notice()).toBe('KV cache cannot be added to Turn 1 · Prefill: Not captured');
    expect(written).toEqual([[NO_PLACE, true]]);
  });

  it('writes where the reader goes, and does not follow its own address back', () => {
    const {controller, address, selection, overview, dom, written} = setup();
    controller.follow();
    selection.set({turn: 1});
    controller.follow();
    expect(written).toEqual([[{turn: 1, stage: null, analysis: null}, false]]);
    overview.set({turn: 2, stage: 'decode', kind: 'kv', id: 'analysis-1'});
    controller.follow();
    expect(written.at(-1)).toEqual([{turn: 2, stage: 'decode', analysis: 'kv'}, false]);
    // The first write arrives after the second was asked for: the reading stays.
    address.set({...SESSION, turn: 1});
    controller.follow();
    expect(dom.openStage).not.toHaveBeenCalled();
    expect(overview()).not.toBeNull();
  });

  it('puts where the reading is when its own address arrives after the reading moved back', () => {
    // The reader opens turn 2 and returns to turn 1 before turn 2's address arrives: turn 1 is
    // the address already, so nothing is asked for, and the address of turn 2 then arrives.
    const {controller, address, selection, dom, written} = setup();
    address.set({...SESSION, turn: 1});
    controller.follow();
    expect(dom.openStage).toHaveBeenCalledWith(1, 'prefill');
    selection.set({turn: 2});
    controller.follow();
    expect(written).toEqual([[{turn: 2, stage: null, analysis: null}, false]]);
    selection.set({turn: 1});
    controller.follow();
    expect(written.length).toBe(1);
    address.set({...SESSION, turn: 2});
    controller.follow();
    expect(selection()).toEqual({turn: 1});
    expect(written.at(-1)).toEqual([{turn: 1, stage: null, analysis: null}, true]);
    // Turn 1's address arriving is its own too: nothing more is written or followed.
    address.set({...SESSION, turn: 1});
    controller.follow();
    expect(written.length).toBe(2);
    expect(dom.openStage).toHaveBeenCalledTimes(1);
  });

  it('follows Back, and a later link to a place it once asked for, after its request was cancelled', () => {
    // The reader opens turn 2; before its address arrives the reader goes Back, which cancels
    // that navigation, to the overview. A link to turn 2 later is followed, not taken for the
    // reading's own request.
    const {controller, address, selection, dom, written} = setup();
    address.set({...SESSION, turn: 1});
    controller.follow();
    selection.set({turn: 2});
    controller.follow();
    expect(written).toEqual([[{turn: 2, stage: null, analysis: null}, false]]);
    address.set(SESSION);
    controller.follow();
    expect(selection()).toBeNull();
    address.set({...SESSION, turn: 2});
    controller.follow();
    expect(dom.openStage).toHaveBeenLastCalledWith(2, 'prefill');
    expect(selection()).toEqual({turn: 2});
    expect(written.length).toBe(1);
  });

  it('leaves the address alone in Chat mode and while a restored view settles', () => {
    const {controller, address, selection, views, written} = setup();
    address.set({...SESSION, mode: 'Chat'});
    selection.set({turn: 1});
    controller.follow();
    expect(written).toEqual([]);
    address.set({...SESSION, turn: 1});
    controller.follow();
    views.restoring.set(true);
    selection.set({turn: 2});
    controller.follow();
    expect(written).toEqual([]);
    views.restoring.set(false);
    controller.follow();
    expect(written).toEqual([[{turn: 2, stage: null, analysis: null}, false]]);
  });
});
