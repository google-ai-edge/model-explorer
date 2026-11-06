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

import {Component, computed, input} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import type {CapturedTurn, CapturedTurnRejection} from '../../server/contracts/capture';

/** A turn the capture index rejected: it has no tokens, and the server says why. */
export type RejectedTurnInfo = CapturedTurn & {rejected: CapturedTurnRejection};

/** The turns the capture index read, in the server's order; only these have tokens. */
export function readTurns(turns: readonly CapturedTurn[]): CapturedTurn[] {
  return turns.filter((turn) => !turn.rejected);
}

/** The turns the capture index rejected, by turn number. */
export function rejectedTurns(turns: readonly CapturedTurn[]): RejectedTurnInfo[] {
  return turns
    .filter((turn): turn is RejectedTurnInfo => !!turn.rejected)
    .sort((a, b) => a.n - b.n);
}

/**
 * One rejected turn in the conversation: its prompt on both sides, when the server sends it,
 * and the server's text, marked. It offers no analysis, in Debug neither: it has no tokens to
 * analyse. The UI never works out why a turn was rejected.
 */
@Component({
  selector: 'rejected-turn',
  imports: [MatIconModule],
  templateUrl: './rejected_turn.ng.html',
  styleUrl: './rejected_turn.scss',
  host: {
    role: 'region',
    '[attr.data-turn]': 'turn().n',
    '[attr.aria-label]': "'Turn ' + turn().n + ' · Rejected'",
    '[class.debug]': 'debug()',
  },
})
export class RejectedTurn {
  readonly turn = input.required<RejectedTurnInfo>();
  /** The side columns, in the conversation's order. */
  readonly sides = input.required<readonly string[]>();
  /** Debug lays it out as Debug's turns are. */
  readonly debug = input(false);
  readonly textId = computed(() => `rejected-turn-${this.turn().n}-text`);
}
