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
            '../src/features/conversation/conversation_virtualization.ts',
            import.meta.url),
        ),
  ],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {
  virtualConversation,
  streamingConversation,
  textChunks,
  escapeVirtualText,
  renderInputToken,
  renderOutputToken,
} =
    await import(
        'data:text/javascript;base64,' +
        Buffer.from(outputFiles[0].text).toString('base64'));
const text = '中文😀\r\n\t<unsafe>&  ' +
    'x'.repeat(200000);
assert.equal(textChunks(text).join(''), text);
assert.equal(
    escapeVirtualText('<img onerror="x">'),
    '&lt;img onerror=&quot;x&quot;&gt;');
const tokens =
    Array.from({length: 131072}, (_, step) => ({
                                   text: ' token',
                                   step: step * 2,
                                   phase: step === 0 ? 'thinking' : 'response',
                                 }));
const ref = {
  turn: 1,
  run: 'ref',
  tokens,
  input: text
};
const target = {
  turn: 1,
  run: 'target',
  tokens: tokens.map((t) => ({...t})),
  input: text
};
target.tokens[65536].text = ' changed';
const steps = virtualConversation(ref, target, 'steps', true);
assert.equal(steps.rows.length, 131072);
assert.equal(steps.bySide.target.get(65536).index, 65536);
assert.equal(steps.pairs[65536].target.step, 131072);
assert.equal(steps.pairs.filter((p) => p.match === false).length, 1);
assert.equal(steps.inputTokens.ref.join(''), text);
assert.equal(virtualConversation(ref, target, 'steps', true), steps);
const content = virtualConversation(ref, target, 'content', false);
assert.equal(virtualConversation(ref, target, 'content', true), content);
assert.equal(virtualConversation(ref, target, 'steps', false), content);
assert.notEqual(steps, content);
assert.equal(content.alignmentUnavailable, false);
assert.equal(content.alignments.steps, content.rows);
assert.equal(content.pairs.filter((p) => p.match === false).length, 1);
assert.equal(content.steps.ref.length, 131072);
assert.equal(content.rows[1].index, 1);
const unknown = virtualConversation(ref, undefined, 'steps', true);
assert.equal(unknown.hasCapturedTokens, false);
assert(unknown.pairs.every((p) => p.match === null));
const completeRef = {
  ...ref,
  output: tokens.slice(1).map((t) => t.text).join(''),
};
const completeTarget = {
  ...target,
  output: target.tokens.slice(1).map((t) => t.text).join(''),
};
assert.equal(
    virtualConversation(completeRef, completeTarget, 'steps', false)
        .responseTextComplete,
    true,
);
assert.equal(
    virtualConversation(
        {...completeRef, output: completeRef.output + ' uncaptured tail'},
        completeTarget,
        'steps',
        false,
        )
        .responseTextComplete,
    false,
);
console.log(
    'PASS: 128k rows, sparse steps, missing capture, partial response guard, long Context alignment, immutable cache, exact Unicode text preservation and HTML escaping',
);

const streamText = ' stream'.repeat(131072) + '中文😀';
const streaming = streamingConversation(
    text, {ref: streamText, target: streamText + ' tail'});
assert.equal(streaming.steps.ref.join(''), streamText);
assert.equal(streaming.steps.target.join(''), streamText + ' tail');
assert.equal(streaming.inputTokens.ref.join(''), text);
assert.equal(streaming.hasCapturedTokens, false);
assert.equal(streaming.pairs.length, 0);
assert.equal(streaming.rows.length, streaming.steps.target.length);
console.log(
    'PASS: 128k streaming text preserves complete Unicode and never invents captured tokens or matches',
);
const serializedRef = {
  ...completeRef,
  serialized_input: '<system>' + text + '</system><user>' + text + '</user>',
};
const serializedTarget = {
  ...completeTarget,
  serialized_input: serializedRef.serialized_input
};
const serializedDebug =
    virtualConversation(serializedRef, serializedTarget, 'content', true);
const serializedChat =
    virtualConversation(serializedRef, serializedTarget, 'content', false);
assert.notEqual(serializedDebug, serializedChat);
assert.equal(
    serializedDebug.inputTokens.ref.join(''), serializedRef.serialized_input);
assert.equal(serializedChat.inputTokens.ref.join(''), text);
console.log(
    'PASS: long captured serialization stays exact in Debug while Chat retains only the user input',
);

const base = {
  interactive: true,
  debug: true,
  side: 'target',
  numeric: null,
  first: false,
  partner: false,
  template: false,
  mismatch: false,
  active: false,
  boundaries: false,
  palette: 'logits',
  heat: null,
  empty: false,
};
assert.equal(
    renderInputToken('<b>', 3, 'ref'),
    '<span class="readingToken" data-input-position="3" data-side="ref">&lt;b&gt;</span>',
);
const button = renderOutputToken(
    ' a\tb',
    {turn: 2, side: 'target', step: 7, row: 5},
    {
      ...base,
      numeric: 'token',
      first: true,
      mismatch: true,
      active: true,
      boundaries: true,
      heat: '40%',
    },
);
assert.match(
    button,
    /^<button type="button" class="token numeric-token first-divergence  mismatch chat-mismatch selected selected-token boundaries "/,
);
assert.match(
    button,
    /data-turn="2" data-row="5" data-step="7" data-side="target" aria-pressed="true"/,
);
assert.match(
    button,
    /aria-label="target aligned position 5:  a\tb · First divergence"/);
assert.match(
    button,
    /><span class="invisible-char space-char marked-space"> <\/span>a<span class="invisible-char"><span class="whitespace-glyph" aria-hidden="true">⇥<\/span>\t<\/span>b<\/button>$/,
);
assert.match(button, /style="--heat-level:40%"/);
// The token the details pair, by generation step, with the selected first
// differing row.
assert.match(
    renderOutputToken(
        'b', {turn: 2, side: 'ref', step: 7, row: 10},
        {...base, partner: true}),
    /class="token [^"]*step-partner/,
);
const span = renderOutputToken(
    '<i>',
    {turn: 1, side: 'ref', step: 0, row: undefined},
    {...base, interactive: false, debug: false},
);
assert.equal(
    span,
    '<span class="readingToken      " data-step="0" data-turn="1" data-side="ref" title="&lt;i&gt;">&lt;i&gt;</span>',
);
assert.match(
    renderOutputToken(
        '',
        {turn: 1, side: 'ref', step: 0, row: 0},
        {...base, empty: true, numeric: 'missing'},
        ),
    /class="token numeric-missing      empty-token"/,
);
console.log(
    'PASS: token renderers escape text, expose data attributes and aria labels, and mark whitespace only in Debug',
);
