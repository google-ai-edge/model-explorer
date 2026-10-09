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

import type {CapturedToken} from '../../data/contracts/capture';
export type AlignmentMode = 'content' | 'steps';
export interface CapturedAlignmentRow {
  ref: number | null;
  target: number | null;
}
export interface TokenPair {
  index: number;
  ref?: CapturedToken;
  target?: CapturedToken;
  match: boolean | null;
}
const sameText = (a: CapturedToken, b: CapturedToken) =>
  a.text !== null && b.text !== null && a.text === b.text;
/** Adapted from debugger-shell's conversation-step-zero.html alignedPairs.
 * Align phases independently. Missing capture is unknown, not a mismatch.
 * Token IDs are only meaningful within their own tokenizer; compare text.
 */
export function alignTokens(
  ref: CapturedToken[] | undefined,
  target: CapturedToken[] | undefined,
  mode: AlignmentMode,
  precomputed?: CapturedAlignmentRow[],
): TokenPair[] {
  const rows: Omit<TokenPair, 'index' | 'match'>[] = [];
  if (ref === undefined || target === undefined) {
    for (const token of ref ?? target ?? []) {
      rows.push(ref ? {ref: token} : {target: token});
    }
    return rows.map((row, index) => ({...row, index, match: null}));
  }
  if (precomputed && mode === 'content') {
    let previousRef = -1;
    let previousTarget = -1;
    for (const row of precomputed) {
      if (row.ref == null && row.target == null) {
        throw new Error('Invalid captured alignment: empty row');
      }
      for (const side of ['ref', 'target'] as const) {
        const index = row[side];
        const tokens = side === 'ref' ? ref : target;
        const previous = side === 'ref' ? previousRef : previousTarget;
        if (index != null) {
          if (
            !Number.isInteger(index) ||
            index !== previous + 1 ||
            index >= tokens.length
          ) {
            throw new Error(
              'Invalid captured alignment: token indices must cover each side in order',
            );
          }
          if (side === 'ref') {
            previousRef = index;
          } else {
            previousTarget = index;
          }
        }
      }
      rows.push({
        ref: row.ref == null ? undefined : ref[row.ref],
        target: row.target == null ? undefined : target[row.target],
      });
    }
    if (previousRef + 1 !== ref.length || previousTarget + 1 !== target.length) {
      throw new Error('Invalid captured alignment: incomplete token coverage');
    }
  } else if (mode === 'steps') {
    const a = new Map(ref.map((t) => [t.step, t]));
    const b = new Map(target.map((t) => [t.step, t]));
    for (const step of [...new Set([...a.keys(), ...b.keys()])].sort(
      (a, b) => a - b,
    )) {
      rows.push({ref: a.get(step), target: b.get(step)});
    }
  } else {
    for (const phase of ['thinking', 'response']) {
      let a = ref.filter((t) => (t.phase ?? 'response') === phase);
      let b = target.filter((t) => (t.phase ?? 'response') === phase);
      // Trim equal context before allocating the LCS matrix. Long captures often differ in a small window.
      let prefix = 0;
      let suffix = 0;
      while (
        prefix < a.length &&
        prefix < b.length &&
        sameText(a[prefix], b[prefix])
      ) {
        prefix++;
      }
      while (
        suffix < a.length - prefix &&
        suffix < b.length - prefix &&
        sameText(a[a.length - 1 - suffix], b[b.length - 1 - suffix])
      ) {
        suffix++;
      }
      for (let k = 0; k < prefix; k++) {
        rows.push({ref: a[k], target: b[k]});
      }
      const tail = a
        .slice(a.length - suffix)
        .map((token, k) => ({ref: token, target: b[b.length - suffix + k]}));
      a = a.slice(prefix, a.length - suffix);
      b = b.slice(prefix, b.length - suffix);
      if ((a.length + 1) * (b.length + 1) > 4_000_000) {
        throw new Error(
          'Context alignment exceeds the supported size. Select Steps alignment.',
        );
      }
      const dp = Array.from(
        {length: a.length + 1},
        () => new Uint32Array(b.length + 1),
      );
      for (let i = a.length - 1; i >= 0; i--) {
        for (let j = b.length - 1; j >= 0; j--) {
          dp[i][j] = sameText(a[i], b[j])
            ? 1 + dp[i + 1][j + 1]
            : Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
      }
      let i = 0;
      let j = 0;
      let removed: CapturedToken[] = [];
      let added: CapturedToken[] = [];
      const flush = () => {
        for (let k = 0; k < Math.max(removed.length, added.length); k++) {
          rows.push({ref: removed[k], target: added[k]});
        }
        removed = [];
        added = [];
      };
      while (i < a.length || j < b.length) {
        if (i < a.length && j < b.length && sameText(a[i], b[j])) {
          flush();
          rows.push({ref: a[i++], target: b[j++]});
        } else if (
          i < a.length &&
          (j === b.length || dp[i + 1][j] >= dp[i][j + 1])
        ) {
          removed.push(a[i++]);
        } else {
          added.push(b[j++]);
        }
      }
      flush();
      for (const row of tail) {
        rows.push(row);
      }
    }
  }
  return rows.map((row, index) => ({
    ...row,
    index,
    match:
      row.ref?.text === null || row.target?.text === null
        ? null
        : !!row.ref &&
          !!row.target &&
          sameText(row.ref, row.target) &&
          (row.ref.phase ?? 'response') === (row.target.phase ?? 'response'),
  }));
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
/** The first row where the generations differ. Both runtimes sampled that step after the same
 * generated tokens, so its two tokens belong together even when the text alignment, which matches
 * content, leaves one side of the row empty. Null when no row differs, when text is missing, or
 * when the rows before it do not pair equal generation steps.
 */
export function forkPair(
  rows: readonly TokenPair[],
  ref: readonly CapturedToken[] | undefined,
  target: readonly CapturedToken[] | undefined,
): ForkPair | null {
  const index = rows.findIndex((row) => row.match !== true);
  if (index < 0 || rows[index].match !== false) {
    return null;
  }
  for (let k = 0; k < index; k++) {
    if (rows[k].ref?.step !== rows[k].target?.step) {
      return null;
    }
  }
  const row = rows[index];
  const step = (row.ref ?? row.target)?.step;
  if (typeof step !== 'number') {
    return null;
  }
  if (row.ref && row.target) {
    return row.ref.step === row.target.step
      ? {index, pair: row, filled: null, partner: null}
      : null;
  }
  const filled = row.ref ? 'target' : 'ref';
  const token = (filled === 'ref' ? ref : target)?.find((t) => t.step === step);
  if (!token) {
    return null;
  }
  const partner = rows.findIndex((r) => r[filled] === token);
  return {
    index,
    pair: {...row, [filled]: token, match: false},
    filled,
    partner: partner < 0 ? null : partner,
  };
}
export interface DecodeOutcome {
  /** `unknown`: token text is missing or the rows do not prove which step differs first. */
  kind: 'identical' | 'diverged' | 'unknown' | 'uncaptured';
  /** The first differing generation step, for `diverged`. */
  step: number | null;
  /** Leading rows where both runtimes produced the same token. */
  prefix: number;
}
/** What one turn's generations did relative to each other, from its aligned rows. */
export function decodeOutcome(
  rows: readonly TokenPair[],
  fork: ForkPair | null,
  ref: readonly CapturedToken[] | undefined,
  target: readonly CapturedToken[] | undefined,
): DecodeOutcome {
  if (!ref || !target) {
    return {kind: 'uncaptured', step: null, prefix: 0};
  }
  const first = rows.findIndex((row) => row.match !== true);
  const prefix = first < 0 ? rows.length : first;
  if (first < 0) {
    return {kind: 'identical', step: null, prefix};
  }
  const divergedStep = fork?.pair.target?.step ?? fork?.pair.ref?.step ?? null;
  return fork && divergedStep !== null
    ? {kind: 'diverged', step: divergedStep, prefix}
    : {kind: 'unknown', step: null, prefix};
}
