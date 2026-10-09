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
        '../src/features/conversation/conversation_content.ts',
        import.meta.url)),
  ],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {
  splitPhasePairs,
  capturedPhaseText,
  hasPhaseTokens,
  capturedInputSegments,
  capturedInputRows,
  capturedInputPairRows,
  historyStatus,
  tokenContent,
  isTemplateToken,
  tokenMarkup,
  pairLineBreaks,
  isInvisibleOnly,
  spokenTokenText,
  capturedTokenContent,
} =
    await import(
        'data:text/javascript;base64,' +
        Buffer.from(outputFiles[0].text).toString('base64'));
const thinking = {
  text: 'think',
  step: 7,
  phase: 'thinking'
},
      response = {
        text: 'answer',
        step: 7,
        phase: 'response'
      };
const cross = {
  index: 4,
  ref: thinking,
  target: response,
  match: false
};
const insertion = {
  index: 5,
  target: {...thinking, step: 9},
  match: false
};
const ordinary = {
  index: 6,
  ref: {text: 'legacy', step: 10},
  target: {text: 'legacy', step: 10},
  match: true,
};
const split = splitPhasePairs([cross, insertion, ordinary]);
assert.deepEqual(split.ref.thinking, [cross, insertion]);
assert.deepEqual(split.target.thinking, [insertion]);
assert.deepEqual(split.ref.response, [ordinary]);
assert.deepEqual(split.target.response, [cross, ordinary]);
assert.equal(split.target.response[0].index, 4);
assert.equal(split.target.response[0].target.step, 7);
const capture = {
  turn: 1,
  run: 'ref',
  tokens: [thinking, response],
  output: 'answer'
};
assert.equal(capturedPhaseText(capture, 'thinking'), 'think');
assert.equal(capturedPhaseText(capture, 'response'), 'answer');
assert.equal(
    capturedPhaseText({...capture, thinking: 'complete thought'}, 'thinking'),
    'complete thought',
);
assert.equal(hasPhaseTokens(capture, 'thinking'), true);
assert.equal(
    hasPhaseTokens({...capture, tokens: [response]}, 'thinking'), false);
assert.equal(capturedPhaseText(undefined, 'thinking'), '');
console.log(
    'PASS: per-side phase separation, Steps boundary mismatch, one-sided phase gap, stable pair/step identity, captured text fallback',
);
const serialized =
    '<bos><system>Be precise.\n</system><user>你好😀</user><model>';
const input = {
  turn: 1,
  run: 'ref',
  input: '你好😀',
  serialized_input: serialized,
  messages: [
    {role: 'system', content: 'Be precise.\n'},
    {role: 'user', content: '你好😀'},
  ],
};
const segments = capturedInputSegments(input);
assert.equal(segments.map((s) => s.text).join(''), serialized);
assert.deepEqual(
    segments.filter((s) => s.kind === 'message').map((s) => s.role),
    ['system', 'user'],
);
assert.equal(capturedInputSegments(input), segments);
assert.deepEqual(
    capturedInputSegments({
      ...input,
      messages: [{role: 'user', content: 'not in serialization'}]
    }),
    [{kind: 'serialized', text: serialized}],
);
assert.deepEqual(
    capturedInputSegments({turn: 1, run: 'ref', input: 'literal <bos>'}), [
      {kind: 'message', role: 'user', text: 'literal <bos>'},
    ]);
console.log(
    'PASS: exact serialized input, captured System/User roles, conservative unmatched fallback, no invented template markers',
);

assert.equal(
    tokenContent({text: '<eos>', kind: 'special', step: 900}), 'Special token');
assert.equal(
    tokenContent({text: '<turn>', kind: 'template', step: 4}), 'Chat template');
assert.equal(isTemplateToken({text: '<turn>', step: 4}), false);
assert.equal(tokenContent(undefined), 'Not captured');
console.log(
    'PASS: captured template/special semantics; literal text is never inferred');

// Admitted input rows: Gemma 4 template as the Runner tokenized it (BOS, user
// turn, generation prompt).
const T = (id, text, kind) => ({id, text, kind, step: 0});
// The engine tokenizes `<|turn>` and the role separately; a coarse single
// marker is accepted too.
const admitted = [
  T(2, '<bos>', 'special'),
  T(105, '<|turn>', 'template'),
  T(2364, 'user', 'template'),
  T(107, '\n', 'template'),
  T(40654, 'Reply', 'text'),
  T(607, ' with', 'text'),
  T(236761, '.', 'text'),
  T(106, '<turn|>', 'template'),
  T(107, '\n', 'template'),
  T(105, '<|turn>', 'template'),
  T(4368, 'model', 'template'),
  T(107, '\n', 'template'),
];
const rows = capturedInputRows({turn: 1, run: 'ref', input_tokens: admitted});
assert.equal(rows.length, 1);
assert.equal(rows[0].role, 'user');
assert.deepEqual(
    rows[0].leading.map((t) => t.text),
    ['<bos>', '<|turn>', 'user', '\n'],
);
assert.deepEqual(
    rows[0].text.map((t) => t.text),
    ['Reply', ' with', '.'],
);
assert.deepEqual(
    rows[0].trailing.map((t) => t.text),
    ['<turn|>', '\n', '<|turn>', 'model', '\n'],
);
const withSystem = capturedInputRows({
  turn: 1,
  run: 'ref',
  input_tokens: [
    T(2, '<bos>', 'special'),
    T(105, '<|turn>system', 'template'),
    T(1, 'Be brief.', 'text'),
    T(106, '<turn|>', 'template'),
    ...admitted.slice(1),
  ],
});
assert.deepEqual(
    withSystem.map((r) => r.role),
    ['system', 'user'],
);
assert.deepEqual(
    withSystem[0].leading.map((t) => t.text),
    ['<bos>', '<|turn>system'],
);
assert.deepEqual(
    withSystem[1].trailing.map((t) => t.text),
    ['<turn|>', '\n', '<|turn>', 'model', '\n'],
);
assert.equal(
    capturedInputRows(
        {turn: 1, run: 'ref', input_tokens: [T(5, 'x', undefined)]}),
    null,
    'tokens without kinds never get invented roles',
);
assert.equal(capturedInputRows({turn: 1, run: 'ref'}), null);
const same = {
  turn: 1,
  run: 'ref',
  input_tokens: admitted
};
assert.equal(capturedInputRows(same), capturedInputRows(same));
console.log(
    'PASS: admitted input rows keep BOS with the first role, text between its markers, and fold the generation prompt into the closing markers',
);

// History status: earlier inputs and consumed outputs on both sides; the
// filtered stop token never counts.
const hello = (run, extra = []) => ({
  turn: 1,
  run,
  input: 'hi',
  tokens: [{id: 5, text: 'hello', step: 0}, ...extra],
});
const stop = {
  id: 106,
  text: '<turn|>',
  step: 1,
  kind: 'special',
  released: false,
  stop: true
};
assert.equal(
    historyStatus([hello('ref'), hello('target')], 1),
    null,
    'the first turn has no history',
);
assert.equal(
    historyStatus([hello('ref', [stop]), hello('target')], 2),
    'same',
    'an unreleased stop token is not history',
);
assert.equal(
    historyStatus(
        [
          hello('ref'),
          {...hello('target'), tokens: [{id: 6, text: 'hey', step: 0}]}
        ],
        2),
    'differs',
);
assert.equal(
    historyStatus([hello('ref'), {...hello('target'), input: 'other'}], 2),
    'differs');
assert.equal(
    historyStatus([hello('ref'), {turn: 1, run: 'target', input: 'hi'}], 2),
    'unknown',
    'missing tokens never claim sameness',
);
assert.equal(historyStatus([hello('ref')], 2), 'unknown');
assert.equal(
    historyStatus(
        [
          hello('ref'),
          hello('target'),
          {turn: 2, run: 'ref', input: 'x', tokens: []},
          {
            turn: 2,
            run: 'target',
            input: 'x',
            tokens: [{id: 1, text: 'a', step: 0}]
          },
        ],
        3,
        ),
    'differs',
);
console.log(
    'PASS: history status compares earlier inputs and consumed outputs only, and stays unknown without captured tokens',
);

const SP = '<span class="invisible-char space-char marked-space"> </span>';
const LF =
    '<span class="invisible-char line-break-char"><span class="whitespace-glyph" aria-hidden="true">↵</span></span>\n';
const code = (hex, ch) =>
    `<span class="invisible-char code-char"><span class="whitespace-code" aria-hidden="true">U+${
        hex}</span>${ch}</span>`;
assert.equal(tokenMarkup('a<b&c'), 'a&lt;b&amp;c');
assert.equal(
    tokenMarkup(' x'), SP + 'x', 'a space next to text is always bracketed');
assert.equal(
    tokenMarkup(' '), SP, 'a token that is only a space is never an empty box');
assert.equal(tokenMarkup('\n\n'), LF + LF);
assert.equal(tokenMarkup('a\nb'), 'a' + LF + 'b');
// A token drawn as its own box keeps the mark and drops the character: the row
// breaks the line.
assert.equal(tokenMarkup('\n\n', false), (LF + LF).replaceAll('\n', ''));
assert.ok(
    !tokenMarkup('a\r\nb', false).includes('\n') &&
        !tokenMarkup('a\r\nb', false).includes('\r'),
);
assert.equal(tokenMarkup('a b', false), tokenMarkup('a b'));
// Both runtimes break after the same row, by the side with more line breaks.
assert.equal(
    pairLineBreaks({ref: {text: 'a', step: 0}, target: {text: 'b', step: 0}}),
    0);
assert.equal(
    pairLineBreaks(
        {ref: {text: '我', step: 18}, target: {text: '\n\n', step: 18}}),
    2);
assert.equal(pairLineBreaks({ref: {text: '.\r\n', step: 1}}), 1);
assert.equal(pairLineBreaks({target: {text: null, step: 1}}), 0);
assert.equal(
    tokenMarkup('a\tb'),
    'a<span class="invisible-char"><span class="whitespace-glyph" aria-hidden="true">⇥</span>\t</span>b',
);
assert.equal(tokenMarkup('\u200b'), code('200B', '\u200b'));
assert.equal(tokenMarkup('a\u00a0b'), 'a' + code('00A0', '\u00a0') + 'b');
assert.equal(isInvisibleOnly(''), false);
assert.equal(isInvisibleOnly(' \n\t\u200b'), true);
assert.equal(isInvisibleOnly(' a'), false);
assert.equal(spokenTokenText('\n\n'), 'line feed × 2');
assert.equal(spokenTokenText(' \u200b'), 'space, U+200B');
assert.equal(spokenTokenText(' ocean'), ' ocean');
// A generated stop token recorded as special is a chat-template token when its
// run admitted the same ID as one.
const stopToken = {
  id: 106,
  text: '<turn|>',
  kind: 'special',
  step: 3,
  stop: true,
  released: false
};
assert.equal(
    capturedTokenContent(
        {
          turn: 1,
          run: 'ref',
          input_tokens: [{id: 106, text: '<turn|>', kind: 'template', step: 5}]
        },
        stop,
        ),
    'Chat template',
);
assert.equal(
    capturedTokenContent(
        {
          turn: 1,
          run: 'ref',
          input_tokens: [{id: 2, text: null, kind: 'special', step: 0}]
        },
        {...stopToken, id: 1, text: '<eos>'},
        ),
    'Special token',
);
assert.equal(capturedTokenContent(undefined, stopToken), 'Special token');
console.log(
    'PASS: Debug token markup always draws invisible characters, escapes text, names invisible tokens; stop tokens read as chat template from capture evidence',
);

{
  // Aligned Prefill rows: the same template on both sides pairs marker by
  // marker; a differing user token is a mismatch, an extra token on one side is
  // a gap on the other.
  const t = (text, kind, id) => ({text, kind, id, step: 0});
  const opener = [
    t('<bos>', 'special', 2),
    t('<|turn>', 'template', 105),
    t('user', 'template', 1645),
    t('\n', 'template', 107),
  ];
  const closer = [
    t('<turn|>', 'template', 106),
    t('\n', 'template', 107),
    t('<|turn>', 'template', 105),
    t('model', 'template', 2516),
    t('\n', 'template', 107),
  ];
  const ref = {
    turn: 2,
    run: 'ref',
    input_tokens: [
      ...opener, t('Reply', 'text', 40654), t(' now', 'text', 1490), ...closer
    ],
  };
  const target = {
    turn: 2,
    run: 'target',
    input_tokens: [
      ...opener,
      t('Reply', 'text', 40654),
      t(' later', 'text', 3050),
      t(' please', 'text', 4000),
      ...closer,
    ],
  };
  const rows = capturedInputPairRows(ref, target);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].role, 'user');
  assert.deepEqual(
      rows[0].leading.map((p) => p.ref?.text),
      ['<bos>', '<|turn>', 'user', '\n'],
  );
  assert.deepEqual(
      rows[0].text.map(
          (p) => [p.ref?.text ?? null, p.target?.text ?? null, p.match]),
      [
        ['Reply', 'Reply', true],
        [' now', ' later', false],
        [null, ' please', false],
      ],
  );
  assert.deepEqual(
      rows[0].trailing.map((p) => p.target?.text),
      ['<turn|>', '\n', '<|turn>', 'model', '\n'],
  );
  assert.equal(
      capturedInputPairRows(ref, target), rows,
      'memoized per pair of captures');
  assert.equal(
      capturedInputPairRows(ref, {
        turn: 2,
        run: 'target',
        input_tokens: [{text: 'x', id: 1, step: 0}],
      }),
      null,
      'no kinds on one side: no pairing',
  );
  assert.equal(
      capturedInputRows(ref)[0].text.length, 2, 'per-side rows are unchanged');
  console.log(
      'PASS: Prefill pairs align both admitted inputs by content and fall back without token kinds',
  );
}
