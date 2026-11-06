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
  CapturedConversation,
  CapturedToken,
  TurnComparison,
} from '../server/contracts/capture';
import {
  escapeVirtualText,
  renderInputToken,
  renderOutputToken,
  streamingConversation,
  textChunks,
  virtualConversation,
} from './conversation_virtualization';

describe('conversation virtualization', () => {
  const text = '中文😀\r\n\t<unsafe>&  ' + 'x'.repeat(200000);
  const tokens: CapturedToken[] = Array.from({length: 131072}, (_, step) => ({
    text: ' token',
    step: step * 2,
    phase: step === 0 ? 'thinking' : 'response',
  }));
  const ref: CapturedConversation = {turn: 1, run: 'ref', tokens, input: text};
  const target = {turn: 1, run: 'target', tokens: tokens.map((t) => ({...t})), input: text};
  target.tokens[65536].text = ' changed';
  // The server's rows (analysis/alignment.py): both alignments pair the tokens in order here.
  const serverRows = () => tokens.map((_, k) => ({ref: k, target: k, same: k !== 65536}));
  const comparison = {
    alignments: {content: {rows: serverRows()}, steps: {rows: serverRows()}},
  } as TurnComparison;
  const completeRef = {
    ...ref,
    output: tokens
      .slice(1)
      .map((t) => t.text)
      .join(''),
  };
  const completeTarget = {
    ...target,
    output: target.tokens
      .slice(1)
      .map((t) => t.text)
      .join(''),
  };

  it('128k server rows, sparse steps, missing capture, partial response guard, immutable cache, exact Unicode text preservation and HTML escaping', () => {
    expect(textChunks(text).join('')).toBe(text);
    expect(escapeVirtualText('<img onerror="x">')).toBe('&lt;img onerror=&quot;x&quot;&gt;');
    const steps = virtualConversation(ref, target, comparison, 'steps', true);
    expect(steps.rows.length).toBe(131072);
    expect(steps.bySide.target.get(65536)!.index).toBe(65536);
    expect(steps.pairs[65536].target!.step).toBe(131072);
    expect(steps.pairs.filter((p) => p.match === false).length).toBe(1);
    expect(steps.inputTokens.ref.join('')).toBe(text);
    expect(virtualConversation(ref, target, comparison, 'steps', true)).toBe(steps);
    const content = virtualConversation(ref, target, comparison, 'content', false);
    expect(virtualConversation(ref, target, comparison, 'content', true)).toBe(content);
    expect(virtualConversation(ref, target, comparison, 'steps', false)).toBe(content);
    expect(steps).not.toBe(content);
    expect(content.alignments.steps).toBe(content.rows);
    expect(content.pairs.filter((p) => p.match === false).length).toBe(1);
    expect(content.steps.ref.length).toBe(131072);
    expect(content.rows[1].index).toBe(1);
    const unknown = virtualConversation(ref, undefined, undefined, 'steps', true);
    expect(unknown.hasCapturedTokens).toBe(false);
    expect(unknown.pairs.every((p) => p.match === null)).toBeTruthy();
    expect(
      virtualConversation(completeRef, completeTarget, comparison, 'steps', false)
        .responseTextComplete,
    ).toBe(true);
    expect(
      virtualConversation(
        {...completeRef, output: completeRef.output + ' uncaptured tail'},
        completeTarget,
        comparison,
        'steps',
        false,
      ).responseTextComplete,
    ).toBe(false);
  });

  it('128k streaming text preserves complete Unicode and never invents captured tokens or matches', () => {
    const streamText = ' stream'.repeat(131072) + '中文😀';
    const streaming = streamingConversation(text, {ref: streamText, target: streamText + ' tail'});
    expect(streaming.steps.ref.join('')).toBe(streamText);
    expect(streaming.steps.target.join('')).toBe(streamText + ' tail');
    expect(streaming.inputTokens.ref.join('')).toBe(text);
    expect(streaming.hasCapturedTokens).toBe(false);
    expect(streaming.pairs.length).toBe(0);
    expect(streaming.rows.length).toBe(streaming.steps.target.length);
  });

  it('token renderers escape text, expose data attributes and aria labels, and mark whitespace only in Debug', () => {
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
    expect(renderInputToken('<b>', 3, 'ref')).toBe(
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
    expect(button).toMatch(
      /^<button type="button" class="token numeric-token first-divergence  mismatch chat-mismatch selected selected-token boundaries "/,
    );
    expect(button).toMatch(
      /data-turn="2" data-row="5" data-step="7" data-side="target" aria-pressed="true"/,
    );
    expect(button).toMatch(/aria-label="target aligned position 5:  a\tb · First divergence"/);
    expect(button).toMatch(
      /><span class="invisible-char space-char marked-space"> <\/span>a<span class="invisible-char"><span class="whitespace-glyph" aria-hidden="true">⇥<\/span>\t<\/span>b<\/button>$/,
    );
    expect(button).toMatch(/style="--heat-level:40%"/);
    // The token the details pair, by generation step, with the selected first differing row.
    expect(
      renderOutputToken('b', {turn: 2, side: 'ref', step: 7, row: 10}, {...base, partner: true}),
    ).toMatch(/class="token [^"]*step-partner/);
    const span = renderOutputToken(
      '<i>',
      {turn: 1, side: 'ref', step: 0, row: undefined},
      {...base, interactive: false, debug: false},
    );
    expect(span).toBe(
      '<span class="readingToken      " data-step="0" data-turn="1" data-side="ref" title="&lt;i&gt;">&lt;i&gt;</span>',
    );
    expect(
      renderOutputToken(
        '',
        {turn: 1, side: 'ref', step: 0, row: 0},
        {...base, empty: true, numeric: 'missing'},
      ),
    ).toMatch(/class="token numeric-missing      empty-token"/);
  });

  it('a reading of more than 2048 tokens labels a token captured without its text by its id', () => {
    const long = Array.from(
      {length: 2050},
      (_, step) =>
        ({
          text: step === 2049 ? null : 'a',
          id: step === 2049 ? 106 : 818,
          step,
        }) as CapturedToken,
    );
    const rows = long.map((t, k) => ({ref: k, target: k, same: t.text === null ? null : true}));
    const virtual = virtualConversation(
      {turn: 1, run: 'ref', tokens: long, output: 'a'.repeat(2049)},
      {turn: 1, run: 'target', tokens: long.map((t) => ({...t})), output: 'a'.repeat(2049)},
      {alignments: {content: {rows}, steps: {rows}}} as TurnComparison,
      'steps',
      true,
    );
    expect(virtual.steps.ref.at(-1)).toBe('Token #106');
    expect(virtual.pairs.at(-1)!.match).toBe(null);
    expect(virtual.responseTextComplete).toBe(false);
  });
});
