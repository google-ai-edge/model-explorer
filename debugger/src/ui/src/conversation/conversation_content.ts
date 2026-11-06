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
import type {TokenPair} from './token_alignment';

export type TokenPhase = 'thinking' | 'response';
export type TokenSelection = {kind: 'token'; turn: number; index: number};
export type StageSelection = {
  range?: {start: number; end: number};
  kind: 'stage';
  turn: number;
  stage: 'prefill' | 'decode';
  side: string;
};
/** An admitted Prefill token, addressed by its aligned pair index. */
export type InputSelection = {kind: 'input'; turn: number; index: number};
export type ConversationSelection = TokenSelection | StageSelection | InputSelection;
export interface CapturedInputSegment {
  kind: 'template' | 'message' | 'serialized';
  text: string;
  role?: string;
}
const inputSegments = new WeakMap<CapturedConversation, CapturedInputSegment[]>();
/** A side's turn as the user wrote it: the system messages the chat opened with, then its prompt. */
export function capturedInputSegments(
  capture: CapturedConversation | undefined,
): CapturedInputSegment[] {
  if (!capture) {
    return [];
  }
  const known = inputSegments.get(capture);
  if (known) {
    return known;
  }
  const segments: CapturedInputSegment[] = [];
  for (const message of capture.messages ?? []) {
    if (message.role === 'system') {
      segments.push({kind: 'message', role: message.role, text: message.content});
    }
  }
  if (capture.input !== undefined) {
    segments.push({kind: 'message', role: 'user', text: capture.input});
  }
  inputSegments.set(capture, segments);
  return segments;
}
export const tokenPhase = (token: CapturedToken | undefined): TokenPhase =>
  token?.phase ?? 'response';
/** A stop token the runtime named no id of: the last turn's, which no next turn carries in. */
export function isUnknownStop(token: CapturedToken | undefined): boolean {
  return !!token?.stop && token.id == null && token.text == null;
}
/** What a stop token of unknown id says on hover. */
export const UNKNOWN_STOP_HINT = 'Stop token: the runtime did not report its id.';
/** A missing token spelling is not a captured EOS/template string; a stop token of unknown id
 *  reads 'stop'. */
export function capturedTokenLabel(token: CapturedToken | undefined): string {
  if (isUnknownStop(token)) {
    return 'stop';
  }
  return token?.text ?? (token?.id != null ? `Token #${token.id}` : 'Not captured');
}

const phases = new WeakMap<CapturedConversation, Set<TokenPhase>>();
export function hasPhaseTokens(
  capture: CapturedConversation | undefined,
  phase: TokenPhase,
): boolean {
  if (!capture) {
    return false;
  }
  let available = phases.get(capture);
  if (!available) {
    available = new Set(capture.tokens?.map(tokenPhase));
    phases.set(capture, available);
  }
  return available.has(phase);
}

/** Keep alignment identity while placing each side in its captured phase. */
export function splitPhasePairs(pairs: TokenPair[]): {
  ref: {thinking: TokenPair[]; response: TokenPair[]};
  target: {thinking: TokenPair[]; response: TokenPair[]};
} {
  const result = {
    ref: {thinking: [] as TokenPair[], response: [] as TokenPair[]},
    target: {thinking: [] as TokenPair[], response: [] as TokenPair[]},
  };
  for (const pair of pairs) {
    result.ref[tokenPhase(pair.ref ?? pair.target)].push(pair);
    result.target[tokenPhase(pair.target ?? pair.ref)].push(pair);
  }
  return result;
}

const contents = new WeakMap<CapturedConversation, {thinking: string; response: string}>();
export function capturedPhaseText(
  capture: CapturedConversation | undefined,
  phase: TokenPhase,
): string {
  if (!capture) {
    return '';
  }
  let text = contents.get(capture);
  if (!text) {
    text = {
      thinking:
        capture.thinking ??
        capture.tokens
          ?.filter((t) => tokenPhase(t) === 'thinking')
          .map((t) => t.text)
          .join('') ??
        '',
      response:
        capture.output ??
        capture.tokens
          ?.filter((t) => tokenPhase(t) === 'response')
          .map((t) => t.text)
          .join('') ??
        '',
    };
    contents.set(capture, text);
  }
  return text[phase];
}

export const isTemplateToken = (token: CapturedToken | undefined): boolean =>
  token?.kind === 'template' || token?.kind === 'special';

const HTML_ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};
export const escapeHtml = (text: string): string =>
  text.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c] ?? c);

const WHITESPACE_GLYPHS: Record<string, string> = {'\n': '↵', '\r': '␍', '\t': '⇥'};
/** Characters that draw nothing: whitespace, zero-width and format characters, controls. */
const INVISIBLE =
  /^[\s\u0000-\u001f\u007f-\u009f\u00ad\u200b-\u200f\u2028-\u202f\u2060-\u2064\ufeff]$/u;
export const isInvisibleChar = (ch: string): boolean => INVISIBLE.test(ch);
/** A token made only of invisible characters has nothing to show unless its glyphs are drawn. */
export function isInvisibleOnly(text: string): boolean {
  return text.length > 0 && Array.from(text).every(isInvisibleChar);
}
const codePoint = (ch: string): string =>
  'U+' + (ch.codePointAt(0) ?? 0).toString(16).toUpperCase().padStart(4, '0');
const INVISIBLE_NAMES: Record<string, string> = {
  ' ': 'space',
  '\n': 'line feed',
  '\r': 'carriage return',
  '\t': 'tab',
};
/** How an invisible-only token is named in labels, e.g. "line feed × 2"; other text is returned as is. */
export function spokenTokenText(text: string): string {
  if (!isInvisibleOnly(text)) {
    return text;
  }
  const runs: {name: string; count: number}[] = [];
  for (const ch of text) {
    const name = INVISIBLE_NAMES[ch] ?? codePoint(ch);
    const last = runs[runs.length - 1];
    if (last?.name === name) {
      last.count++;
    } else {
      runs.push({name, count: 1});
    }
  }
  return runs.map((run) => (run.count > 1 ? `${run.name} × ${run.count}` : run.name)).join(', ');
}
/** Characters that break a line of a title or print nothing there: controls and the Unicode line
 *  and paragraph separators. */
const TITLE_UNPRINTABLE = /^[\u0000-\u001f\u007f-\u009f\u2028\u2029]$/u;
/**
 * Token text as it reads in one line of a title, at most `limit` code points: a token of
 * invisible characters only is named ("line feed × 2"); in other text the spaces at its ends go,
 * a line break or tab draws its glyph and another control spells its code point (U+001B), and
 * text longer than `limit` is cut to end in "…" at `limit`, never inside a character or a mark.
 */
export function tokenTitleText(text: string, limit: number): string {
  const pieces = Array.from(
    isInvisibleOnly(text) ? spokenTokenText(text) : text.replace(/^ +| +$/g, ''),
    (ch) => (TITLE_UNPRINTABLE.test(ch) ? (WHITESPACE_GLYPHS[ch] ?? codePoint(ch)) : ch),
  );
  const size = (piece: string) => Array.from(piece).length;
  if (pieces.reduce((total, piece) => total + size(piece), 0) <= limit) {
    return pieces.join('');
  }
  let out = '';
  let length = 0;
  for (const piece of pieces) {
    if (length + size(piece) > limit - 1) {
      break;
    }
    out += piece;
    length += size(piece);
  }
  return out + '…';
}
/**
 * Debug token markup. Tokens keep their real characters and every invisible one is drawn in
 * place: a space owns a fixed slot with a bracket, line breaks and tabs draw a glyph, anything
 * else spells its code point. There is no setting: in a token view whitespace is data, and a
 * token made only of it must never be an empty box. Text is escaped; spans carry only classes.
 */
/** `keepBreaks: false` draws a line break's mark without the character itself. A token drawn as
 *  its own box (a Decode token, a details chip) would otherwise grow a second, empty line; the
 *  row breaks the line for both runtimes instead (see `pairLineBreaks`). */
export function tokenMarkup(text: string, keepBreaks = true): string {
  let out = '';
  for (const ch of text) {
    if (ch === ' ') {
      out += '<span class="invisible-char space-char marked-space"> </span>';
    } else if (!isInvisibleChar(ch)) {
      out += escapeHtml(ch);
    } else if (ch === '\n' || ch === '\r') {
      out += `<span class="invisible-char line-break-char"><span class="whitespace-glyph" aria-hidden="true">${WHITESPACE_GLYPHS[ch]}</span></span>${keepBreaks ? ch : ''}`;
    } else if (ch === '\t') {
      out += `<span class="invisible-char"><span class="whitespace-glyph" aria-hidden="true">⇥</span>\t</span>`;
    } else {
      out += `<span class="invisible-char code-char"><span class="whitespace-code" aria-hidden="true">${codePoint(ch)}</span>${escapeHtml(ch)}</span>`;
    }
  }
  return out;
}

/** Line breaks that follow an aligned row: as many as the side with more of them, so that both
 *  runtimes break at the same row and stay on shared lines. `\r\n` counts once. */
export function pairLineBreaks(pair: {ref?: CapturedToken; target?: CapturedToken}): number {
  const count = (token: CapturedToken | undefined) =>
    token?.text ? (token.text.replace(/\r\n/g, '\n').match(/[\n\r]/g) ?? []).length : 0;
  return Math.max(count(pair.ref), count(pair.target));
}
