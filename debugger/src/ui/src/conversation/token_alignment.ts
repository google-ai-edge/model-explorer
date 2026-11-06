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

import type {
  CapturedToken,
  ComparisonRow,
  FirstDivergence,
  InputMessage,
} from '../server/contracts/capture';
export type AlignmentMode = 'content' | 'steps';
/** One row of the server's alignment, with the captured tokens it names. */
export interface TokenPair {
  index: number;
  ref?: CapturedToken;
  target?: CapturedToken;
  /** The server's `same`: null when a token's text was not captured. */
  match: boolean | null;
}
const pairsOfRows = new WeakMap<readonly ComparisonRow[], TokenPair[]>();
/**
 * The server's rows of one turn with the tokens they name; the rows come with their capture, so
 * one rows array always names the same tokens.
 */
export function comparisonPairs(
  rows: readonly ComparisonRow[] | undefined,
  ref: readonly CapturedToken[] | undefined,
  target: readonly CapturedToken[] | undefined,
): TokenPair[] {
  if (!rows) {
    return [];
  }
  let pairs = pairsOfRows.get(rows);
  if (!pairs) {
    pairs = rows.map((row, index) => ({
      index,
      ...(row.ref !== null && ref?.[row.ref] ? {ref: ref[row.ref]} : {}),
      ...(row.target !== null && target?.[row.target] ? {target: target[row.target]} : {}),
      match: row.same,
    }));
    pairsOfRows.set(rows, pairs);
  }
  return pairs;
}
/** One message of the chat template over items that carry a token. */
export interface InputRowsOf<T> {
  role: InputMessage['role'];
  leading: T[];
  text: T[];
  trailing: T[];
}
const rowsOfMessages = new WeakMap<
  readonly InputMessage[],
  WeakMap<readonly unknown[], InputRowsOf<unknown>[]>
>();
/**
 * The server's messages over a list of items (tokens or aligned pairs) they index; one messages
 * array over one items array is grouped once, as both come with their capture.
 */
export function messageRows<T>(
  messages: readonly InputMessage[] | undefined,
  items: readonly T[],
): InputRowsOf<T>[] | null {
  if (!messages?.length) {
    return null;
  }
  let byItems = rowsOfMessages.get(messages);
  if (!byItems) {
    byItems = new WeakMap();
    rowsOfMessages.set(messages, byItems);
  }
  let rows = byItems.get(items) as InputRowsOf<T>[] | undefined;
  if (!rows) {
    const of = (indices: number[]) => indices.map((k) => items[k]);
    rows = messages.map((message) => ({
      role: message.role,
      leading: of(message.leading),
      text: of(message.text),
      trailing: of(message.trailing),
    }));
    byItems.set(items, rows);
  }
  return rows;
}
export interface ForkPair {
  /** The alignment row that holds the first differing token. */
  index: number;
  /** The two tokens of that generation step. */
  pair: TokenPair;
  /** The side the alignment left empty in that row and `pair` fills by generation step. */
  filled: 'ref' | 'target' | null;
  /** The row where the filled token is drawn, or null when nothing was filled. */
  partner: number | null;
}
/** The server's first differing generation step over the turn's pairs. */
export function forkPair(
  fork: FirstDivergence | null | undefined,
  pairs: readonly TokenPair[],
  ref: readonly CapturedToken[] | undefined,
  target: readonly CapturedToken[] | undefined,
): ForkPair | null {
  const row = fork ? pairs[fork.row] : undefined;
  if (!fork || !row) {
    return null;
  }
  return {
    index: fork.row,
    pair: fork.filled
      ? {...row, ref: ref?.[fork.ref], target: target?.[fork.target], match: false}
      : row,
    filled: fork.filled,
    partner: fork.partner,
  };
}
