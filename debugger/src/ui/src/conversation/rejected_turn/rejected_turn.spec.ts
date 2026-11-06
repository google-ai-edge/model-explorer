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
import type {CapturedTurn, TurnStage} from '../../server/contracts/capture';
import {RejectedTurn, type RejectedTurnInfo, readTurns, rejectedTurns} from './rejected_turn';

const TEXT =
  'Target: Turn 2 could not be read: a.safetensors holds 9 bytes; the capture recorded 10.';
const FOLLOWING = 'Turn 3 could not be read: It follows turn 2, which was rejected.';
const read: CapturedTurn = {
  n: 1,
  step_start: 12,
};
/** A rejected turn's stage, as the server sends it: every analysis unavailable, for `text`. */
function rejectedStage(moment: string, text: string): TurnStage {
  const why = {available: false, reasons: [{code: 'turn_unreadable', side: null, text}]};
  return {
    positions: {ref: null, target: null},
    kv: {moment, context_id: null},
    analyses: {final_logits: why, sampling: why, kv: why, intermediate_tensors: why},
  };
}
const rejected: RejectedTurnInfo = {
  n: 2,
  prompt: 'And more?',
  step_start: null,
  rejected: {code: 'turn_unreadable', side: 'target', text: TEXT},
  stages: {
    prefill: rejectedStage('prefill_post', TEXT),
    decode: rejectedStage('terminal', TEXT),
  },
};
const following: RejectedTurnInfo = {
  ...rejected,
  n: 3,
  prompt: 'Thanks.',
  rejected: {code: 'turn_unreadable', side: null, text: FOLLOWING},
};

describe('rejected turns', () => {
  it('splits the turns the index read from the ones it rejected, by turn number', () => {
    const turns = [read, following, rejected];
    expect(readTurns(turns)).toEqual([read]);
    expect(rejectedTurns(turns).map((turn) => turn.n)).toEqual([2, 3]);
  });

  function render(turn: RejectedTurnInfo, debug: boolean) {
    const fixture = TestBed.createComponent(RejectedTurn);
    fixture.componentRef.setInput('turn', turn);
    fixture.componentRef.setInput('sides', ['ref', 'target']);
    fixture.componentRef.setInput('debug', debug);
    fixture.detectChanges();
    return {fixture, host: fixture.nativeElement as HTMLElement};
  }

  it('shows the prompt on both sides and the server text, marked as rejected', () => {
    const {fixture, host} = render(rejected, false);
    expect(host.getAttribute('data-turn')).toBe('2');
    expect(host.getAttribute('aria-label')).toBe('Turn 2 · Rejected');
    const sides = Array.from(host.querySelectorAll('article.side'));
    expect(sides.map((side) => side.getAttribute('data-runtime'))).toEqual(['ref', 'target']);
    for (const side of sides) expect(side.querySelector('p')!.textContent).toBe('And more?');
    const notice = host.querySelector('.rejection')!;
    expect(notice.querySelector('strong')!.textContent).toBe('Turn 2 · Rejected');
    expect(notice.querySelector('p')!.textContent).toBe(TEXT);
    // Chat has no analysis buttons, for any turn.
    expect(host.querySelector('button')).toBeNull();
    fixture.destroy();
  });

  it('shows no prompt when the server sends none', () => {
    const {fixture, host} = render({...rejected, prompt: undefined}, false);
    expect(host.querySelector('article.side')).toBeNull();
    expect(host.querySelector('.rejection p')!.textContent).toBe(TEXT);
    fixture.destroy();
  });

  it('offers no analysis in Debug: the turn has no tokens to analyse', () => {
    const {fixture, host} = render(rejected, true);
    expect(host.querySelector('button')).toBeNull();
    expect(host.textContent).not.toContain('analysis view');
    expect(host.querySelector('.rejection p')!.textContent).toBe(TEXT);
    fixture.destroy();
  });
});
