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
    fileURLToPath(new URL(
        '../src/features/conversation/token_alignment.ts', import.meta.url)),
  ],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {alignTokens, forkPair, decodeOutcome} = await import(
    'data:text/javascript;base64,' +
    Buffer.from(outputFiles[0].text).toString('base64'));
const tokens = (words) => words.map((text, step) => ({text, step}));
const rows = alignTokens(
    tokens(['a', 'b', 'c']), tokens(['a', 'x', 'b', 'c']), 'content');
assert.deepEqual(
    rows.map((r) => [r.ref?.text, r.target?.text, r.match]),
    [
      ['a', 'a', true],
      [undefined, 'x', false],
      ['b', 'b', true],
      ['c', 'c', true],
    ],
);
assert.equal(alignTokens(undefined, tokens(['a']), 'content')[0].match, null);
assert.equal(alignTokens([], tokens(['a']), 'content')[0].match, false);
assert.equal(
    alignTokens([{text: 'a', step: 3}], [{text: 'a', step: 5}], 'steps').length,
    2);
assert.equal(
    alignTokens(
        [{text: 'a', step: 0, phase: 'thinking'}],
        [{text: 'a', step: 0, phase: 'response'}],
        'content',
        )
        .length,
    2,
);
assert.equal(
    alignTokens(
        [{text: 'a', step: 0, id: 1}], [{text: 'a', step: 0, id: 2}],
        'content')[0]
        .match,
    true,
);
assert.throws(
    () => alignTokens(
        tokens(Array(2000).fill('a')), tokens(Array(2000).fill('b')),
        'content'),
    /Select Steps/,
);
console.log(
    'PASS: insertion, unknown capture, empty capture, sparse steps, phase boundaries, tokenizer IDs, size guard',
);

const longRef = tokens(Array.from({length: 131072}, (_, i) => 't' + i)),
      longTarget = tokens(longRef.map((t) => t.text));
longTarget[65536] = {
  ...longTarget[65536],
  text: 'changed'
};
const longRows = alignTokens(longRef, longTarget, 'content');
assert.equal(longRows.length, 131072);
assert.equal(longRows.filter((r) => r.match === false).length, 1);
assert.equal(longRows[65536].target.text, 'changed');
const mapped =
    alignTokens(tokens(['a', 'b']), tokens(['a', 'inserted', 'b']), 'content', [
      {ref: 0, target: 0},
      {ref: null, target: 1},
      {ref: 1, target: 2},
    ]);
assert.equal(mapped.length, 3);
assert.equal(mapped[1].match, false);
assert.throws(
    () => alignTokens(
        tokens(['a', 'b']), tokens(['a']), 'content', [{ref: 0, target: 0}]),
    /incomplete token coverage/,
);
assert.throws(
    () => alignTokens(
        tokens(['a']), tokens(['a']), 'content',
        [
          {ref: 0, target: 0},
          {ref: 0, target: null},
        ]),
    /indices/,
);
console.log(
    'PASS: 131072-token sparse divergence, validated precomputed insertion, incomplete and duplicate mapping rejection',
);

const unknown = {
  text: null,
  id: 106,
  step: 1
};
for (const mode of ['content', 'steps']) {
  const missing = alignTokens(
      [{text: 'The', step: 0}, unknown],
      [{text: 'The', step: 0}, {...unknown}],
      mode,
  );
  assert.equal(missing[0].match, true);
  assert.equal(missing[1].match, null);
  assert.equal(missing[1].ref.text, null);
  assert.equal(missing[1].target.id, 106);
}
assert.equal(
    alignTokens([unknown], [{text: '<turn|>', id: 106, step: 1}], 'steps')[0]
        .match,
    null);
console.log(
    'PASS: null spellings remain unknown despite equal IDs; no invented EOS spelling');
// The first differing row is one generation step on both runtimes. Text
// alignment may leave one side of it empty; the fork pair fills that side with
// the token of the same step.
{
  const ref = tokens(['a', 'b', 'c']),
        target = tokens(['a', 'x', 'y', 'b', 'c']),
        content = alignTokens(ref, target, 'content'),
        fork = forkPair(content, ref, target);
  assert.deepEqual(
      content.slice(0, 2).map((r) => [r.ref?.text, r.target?.text]),
      [
        ['a', 'a'],
        [undefined, 'x'],
      ],
  );
  assert.deepEqual(
      [
        fork.index, fork.filled, fork.partner, fork.pair.ref.step,
        fork.pair.target.step
      ],
      [1, 'ref', 3, 1, 1],
  );
  assert.equal(fork.pair.ref.text, 'b');
  assert.equal(fork.pair.match, false);
  // Under Steps alignment the row already pairs the step, so nothing is filled.
  const steps = forkPair(alignTokens(ref, target, 'steps'), ref, target);
  assert.deepEqual([steps.index, steps.filled, steps.partner], [1, null, null]);
  // A substitution row is the fork pair as it stands.
  const swap = forkPair(
      alignTokens(tokens(['a', 'b']), tokens(['a', 'x']), 'content'),
      tokens(['a', 'b']),
      tokens(['a', 'x']),
  );
  assert.deepEqual([swap.index, swap.filled], [1, null]);
  // No fork: identical generations, a side without a token at that step, or
  // missing text.
  assert.equal(forkPair(alignTokens(ref, ref, 'content'), ref, ref), null);
  assert.equal(
      forkPair(alignTokens([], tokens(['a']), 'content'), [], tokens(['a'])),
      null);
  const unknown = [{text: null, step: 0}];
  assert.equal(
      forkPair(
          alignTokens(unknown, tokens(['a']), 'steps'), unknown, tokens(['a'])),
      null,
  );
  // Rows before the fork must pair equal steps.
  const shifted = [
    {text: 'a', step: 1},
    {text: 'b', step: 2},
  ];
  assert.equal(
      forkPair(
          alignTokens(tokens(['a', 'c']), shifted, 'content'),
          tokens(['a', 'c']), shifted),
      null,
  );
}
// A turn's outcome comes from the same rows: identical, diverged at a step,
// unknown or uncaptured.
{
  const ref = tokens(['a', 'b', 'c']),
        target = tokens(['a', 'x', 'y', 'b', 'c']),
        rows = alignTokens(ref, target, 'content');
  assert.deepEqual(
      decodeOutcome(rows, forkPair(rows, ref, target), ref, target), {
        kind: 'diverged',
        step: 1,
        prefix: 1,
      });
  const same = alignTokens(ref, ref, 'content');
  assert.deepEqual(
      decodeOutcome(same, null, ref, ref),
      {kind: 'identical', step: null, prefix: 3});
  const unknown = [{text: null, step: 0}],
        unknownRows = alignTokens(unknown, tokens(['a']), 'steps');
  assert.equal(
      decodeOutcome(unknownRows, null, unknown, tokens(['a'])).kind, 'unknown');
  assert.equal(decodeOutcome([], null, undefined, ref).kind, 'uncaptured');
}
console.log(
    'PASS: first differing step pairs by generation step; turn outcome follows the rows');
