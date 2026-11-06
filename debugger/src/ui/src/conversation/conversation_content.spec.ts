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

import type {CapturedConversation, CapturedToken} from '../server/contracts/capture';
import {
  capturedInputSegments,
  capturedPhaseText,
  capturedTokenLabel,
  isUnknownStop,
  UNKNOWN_STOP_HINT,
  hasPhaseTokens,
  isInvisibleOnly,
  isTemplateToken,
  pairLineBreaks,
  splitPhasePairs,
  spokenTokenText,
  tokenMarkup,
  tokenTitleText,
} from './conversation_content';
import type {TokenPair} from './token_alignment';

describe('conversation content', () => {
  /** A captured token or conversation with only the fields a case needs. */
  const token = (fields: object) => fields as CapturedToken;
  const conversation = (fields: object) => fields as CapturedConversation;

  it('per-side phase separation, Steps boundary mismatch, one-sided phase gap, stable pair/step identity, captured text fallback', () => {
    const thinking = token({text: 'think', step: 7, phase: 'thinking'}),
      response = token({text: 'answer', step: 7, phase: 'response'});
    const cross: TokenPair = {index: 4, ref: thinking, target: response, match: false};
    const insertion: TokenPair = {index: 5, target: {...thinking, step: 9}, match: false};
    const ordinary: TokenPair = {
      index: 6,
      ref: token({text: 'legacy', step: 10}),
      target: token({text: 'legacy', step: 10}),
      match: true,
    };
    const split = splitPhasePairs([cross, insertion, ordinary]);
    expect(split.ref.thinking).toStrictEqual([cross, insertion]);
    expect(split.target.thinking).toStrictEqual([insertion]);
    expect(split.ref.response).toStrictEqual([ordinary]);
    expect(split.target.response).toStrictEqual([cross, ordinary]);
    expect(split.target.response[0].index).toBe(4);
    expect(split.target.response[0].target!.step).toBe(7);
    const capture = conversation({
      turn: 1,
      run: 'ref',
      tokens: [thinking, response],
      output: 'answer',
    });
    expect(capturedPhaseText(capture, 'thinking')).toBe('think');
    expect(capturedPhaseText(capture, 'response')).toBe('answer');
    expect(capturedPhaseText({...capture, thinking: 'complete thought'}, 'thinking')).toBe(
      'complete thought',
    );
    expect(hasPhaseTokens(capture, 'thinking')).toBe(true);
    expect(hasPhaseTokens({...capture, tokens: [response]}, 'thinking')).toBe(false);
    expect(capturedPhaseText(undefined, 'thinking')).toBe('');
  });

  it('captured System/User roles, turn 1 opening messages only, no invented template markers', () => {
    // Turn 1 carries the messages the chat opened with; a later turn, its prompt only.
    const input = conversation({
      turn: 1,
      run: 'ref',
      input: '你好😀',
      messages: [{role: 'system', content: 'Be precise.\n'}],
    });
    const segments = capturedInputSegments(input);
    expect(segments).toStrictEqual([
      {kind: 'message', role: 'system', text: 'Be precise.\n'},
      {kind: 'message', role: 'user', text: '你好😀'},
    ]);
    expect(capturedInputSegments(input)).toBe(segments);
    expect(capturedInputSegments(conversation({turn: 2, run: 'ref', input: 'More'}))).toStrictEqual(
      [{kind: 'message', role: 'user', text: 'More'}],
    );
    expect(
      capturedInputSegments(conversation({turn: 1, run: 'ref', input: 'literal <bos>'})),
    ).toStrictEqual([{kind: 'message', role: 'user', text: 'literal <bos>'}]);
  });

  it('captured template/special semantics; literal text is never inferred', () => {
    expect(isTemplateToken({text: '<turn>', step: 4})).toBe(false);
  });

  it('Debug token markup always draws invisible characters, escapes text, names invisible tokens, writes token text in titles', () => {
    const SP = '<span class="invisible-char space-char marked-space"> </span>';
    const LF =
      '<span class="invisible-char line-break-char"><span class="whitespace-glyph" aria-hidden="true">↵</span></span>\n';
    const code = (hex: string, ch: string) =>
      `<span class="invisible-char code-char"><span class="whitespace-code" aria-hidden="true">U+${hex}</span>${ch}</span>`;
    expect(tokenMarkup('a<b&c')).toBe('a&lt;b&amp;c');
    expect(tokenMarkup(' x'), 'a space next to text is always bracketed').toBe(SP + 'x');
    expect(tokenMarkup(' '), 'a token that is only a space is never an empty box').toBe(SP);
    expect(tokenMarkup('\n\n')).toBe(LF + LF);
    expect(tokenMarkup('a\nb')).toBe('a' + LF + 'b');
    // A token drawn as its own box keeps the mark and drops the character: the row breaks the line.
    expect(tokenMarkup('\n\n', false)).toBe((LF + LF).replaceAll('\n', ''));
    expect(
      !tokenMarkup('a\r\nb', false).includes('\n') && !tokenMarkup('a\r\nb', false).includes('\r'),
    ).toBeTruthy();
    expect(tokenMarkup('a b', false)).toBe(tokenMarkup('a b'));
    // Both runtimes break after the same row, by the side with more line breaks.
    expect(pairLineBreaks({ref: {text: 'a', step: 0}, target: {text: 'b', step: 0}})).toBe(0);
    expect(pairLineBreaks({ref: {text: '我', step: 18}, target: {text: '\n\n', step: 18}})).toBe(2);
    expect(pairLineBreaks({ref: {text: '.\r\n', step: 1}})).toBe(1);
    expect(pairLineBreaks({target: {text: null, step: 1}})).toBe(0);
    expect(tokenMarkup('a\tb')).toBe(
      'a<span class="invisible-char"><span class="whitespace-glyph" aria-hidden="true">⇥</span>\t</span>b',
    );
    expect(tokenMarkup('\u200b')).toBe(code('200B', '\u200b'));
    expect(tokenMarkup('a\u00a0b')).toBe('a' + code('00A0', '\u00a0') + 'b');
    expect(isInvisibleOnly('')).toBe(false);
    expect(isInvisibleOnly(' \n\t\u200b')).toBe(true);
    expect(isInvisibleOnly(' a')).toBe(false);
    expect(spokenTokenText('\n\n')).toBe('line feed × 2');
    expect(spokenTokenText(' \u200b')).toBe('space, U+200B');
    expect(spokenTokenText(' ocean')).toBe(' ocean');
    // A title shows token text in one readable line of at most `limit` code points.
    expect(tokenTitleText(' Paris ', 80)).toBe('Paris');
    expect(tokenTitleText('a\nb\tc\r\n', 80)).toBe('a↵b⇥c␍↵');
    expect(tokenTitleText('a\x1b[31mb\u0085c\u2028', 80)).toBe('aU+001B[31mbU+0085cU+2028');
    expect(tokenTitleText('\n\n', 80)).toBe('line feed × 2');
    expect(tokenTitleText('\u200d', 80)).toBe('U+200D');
    expect(tokenTitleText('🙂'.repeat(80), 80)).toBe('🙂'.repeat(80));
    for (const text of ['x' + '🙂'.repeat(90), 'xx' + '🙂'.repeat(90)]) {
      const title = tokenTitleText(text, 80);
      expect(Array.from(title).length, text).toBe(80);
      expect(title.endsWith('🙂…'), title).toBeTruthy();
      expect(
        !/[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/.test(title),
      ).toBeTruthy();
    }
    // A code point drawn as U+001B is cut whole, not inside its mark.
    expect(tokenTitleText('x'.repeat(76) + '\x1b' + 'y', 80)).toBe('x'.repeat(76) + '…');
  });

  it('a token captured without its text is labelled by its id, else as not captured', () => {
    expect(capturedTokenLabel(token({text: null, id: 106, step: 5}))).toBe('Token #106');
    expect(capturedTokenLabel(token({text: null, step: 5}))).toBe('Not captured');
    expect(capturedTokenLabel(token({text: '', id: 106, step: 5}))).toBe('');
    expect(capturedTokenLabel(token({text: '<turn|>', id: 106, step: 5}))).toBe('<turn|>');
  });

  it('a stop token whose id the runtime did not report reads stop, and only it', () => {
    const unknown = token({text: null, stop: true, step: 7});
    expect(isUnknownStop(unknown)).toBe(true);
    expect(capturedTokenLabel(unknown)).toBe('stop');
    expect(UNKNOWN_STOP_HINT).toBe('Stop token: the runtime did not report its id.');
    // Named by the next turn, or by greedy sampling: the token itself.
    const named = token({text: '<turn|>', id: 106, stop: true, step: 7});
    expect(isUnknownStop(named)).toBe(false);
    expect(capturedTokenLabel(named)).toBe('<turn|>');
    expect(isUnknownStop(token({text: null, step: 7}))).toBe(false);
  });
});
