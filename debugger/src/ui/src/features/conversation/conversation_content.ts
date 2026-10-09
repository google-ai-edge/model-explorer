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
} from '../../data/contracts/capture';
import {alignTokens, TokenPair} from './token_alignment';

export type TokenPhase = 'thinking' | 'response';
export type TokenSelection = {kind: 'token'; turn: number; index: number};
export type StageSelection = {
  kind: 'stage';
  turn: number;
  stage: 'prefill' | 'decode';
  side: string;
};
/** An admitted Prefill token, addressed by its aligned pair index. */
export type InputSelection = {kind: 'input'; turn: number; index: number};
export type ConversationSelection =
  | TokenSelection
  | StageSelection
  | InputSelection;
export interface CapturedInputSegment {
  kind: 'template' | 'message' | 'serialized';
  text: string;
  role?: string;
}
const inputSegments = new WeakMap<
  CapturedConversation,
  CapturedInputSegment[]
>();
/** Only split a recorded serialization when every recorded message can be located. */
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
  const serialized = capture.serialized_input;
  const messages = capture.messages ?? [];
  let segments: CapturedInputSegment[] = [];
  if (serialized === undefined) {
    if (capture.turn === 1) {
      for (const message of messages.filter((m) => m.role === 'system')) {
        segments.push({
          kind: 'message',
          role: message.role,
          text: message.content,
        });
      }
    }
    if (capture.input !== undefined) {
      segments.push({kind: 'message', role: 'user', text: capture.input});
    }
  } else {
    let cursor = 0;
    let matched = 0;
    for (const message of messages) {
      if (!message.content) {
        continue;
      }
      const start = serialized.indexOf(message.content, cursor);
      if (start < 0) {
        matched = -1;
        break;
      }
      if (start > cursor) {
        segments.push({
          kind: 'template',
          text: serialized.slice(cursor, start),
        });
      }
      segments.push({
        kind: 'message',
        role: message.role,
        text: message.content,
      });
      cursor = start + message.content.length;
      matched++;
    }
    if (matched <= 0) {
      segments = [{kind: 'serialized', text: serialized}];
    } else if (cursor < serialized.length) {
      segments.push({kind: 'template', text: serialized.slice(cursor)});
    }
  }
  inputSegments.set(capture, segments);
  return segments;
}
/** One rendered message of the admitted input: its role markers around the message text. */
export interface CapturedInputRow {
  role: 'system' | 'user' | 'model' | 'unknown';
  /** Template/special tokens before the first text token (BOS, `<|turn>user`, newlines). */
  leading: CapturedToken[];
  text: CapturedToken[];
  /** Template tokens after the text, including a trailing generation prompt such as `<|turn>model`. */
  trailing: CapturedToken[];
}
const inputRows = new WeakMap<
  CapturedConversation,
  CapturedInputRow[] | null
>();
/** Message rows over any items that carry a token: admitted tokens, or aligned pairs. */
export interface InputRowsOf<T> {
  role: 'system' | 'user' | 'model' | 'unknown';
  leading: T[];
  text: T[];
  trailing: T[];
}
/** Rows of aligned Prefill pairs; a side without a token at a position is a gap. */
export type CapturedInputPairRow = InputRowsOf<TokenPair>;
const pairAnchor = (pair: TokenPair): CapturedToken => {
  const token = pair.ref ?? pair.target;
  if (!token) {
    throw new Error('Aligned token pair is empty');
  }
  return token;
};
const ROLES = new Set(['system', 'user', 'model']);
/** A turn opener is `<|turn>user` as one token or `<|turn>` followed by the role token. */
function turnOpener(
  tokens: CapturedToken[],
  index: number,
): {role: CapturedInputRow['role']; length: number} | null {
  const token = tokens[index];
  if (token.kind === 'text') {
    return null;
  }
  const text = token.text ?? '';
  const single = /^<\|turn>(system|user|model)$/.exec(text);
  if (single) {
    return {role: single[1] as CapturedInputRow['role'], length: 1};
  }
  const next = tokens[index + 1];
  const role = next?.kind !== 'text' ? (next?.text ?? '').trim() : '';
  if (text === '<|turn>' && ROLES.has(role)) {
    return {role: role as CapturedInputRow['role'], length: 2};
  }
  return null;
}
function groupInputRows<T>(
  items: T[],
  tokenOf: (item: T) => CapturedToken,
): InputRowsOf<T>[] {
  const tokens = items.map(tokenOf);
  const rows: InputRowsOf<T>[] = [];
  let current: InputRowsOf<T> | null = null;
  const preamble: T[] = [];
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    const opener = turnOpener(tokens, index);
    if (opener) {
      const markers = items.slice(index, index + opener.length);
      current = {
        role: opener.role,
        leading: [...preamble.splice(0), ...markers],
        text: [],
        trailing: [],
      };
      rows.push(current);
      index += opener.length - 1;
      continue;
    }
    if (!current) {
      preamble.push(items[index]);
      continue;
    }
    if (token.kind === 'text') {
      if (current.trailing.length) {
        // Text after a closing marker starts a new unlabeled message rather than merging.
        current = {
          role: 'unknown',
          leading: [],
          text: [items[index]],
          trailing: [],
        };
        rows.push(current);
      } else {
        current.text.push(items[index]);
      }
    } else {
      (current.text.length ? current.trailing : current.leading).push(
        items[index],
      );
    }
  }
  if (preamble.length) {
    rows.push({role: 'unknown', leading: preamble, text: [], trailing: []});
  }
  // The generation prompt (`<|turn>model` and its newline) closes the previous message.
  const last = rows[rows.length - 1];
  if (
    rows.length > 1 &&
    last.role === 'model' &&
    !last.text.length &&
    !last.trailing.length
  ) {
    rows.pop();
    rows[rows.length - 1].trailing.push(...last.leading);
  }
  return rows;
}
const kindedTokens = (capture: CapturedConversation | undefined) => {
  const tokens = capture?.input_tokens;
  return tokens?.length && tokens.every((token) => token.kind !== undefined)
    ? tokens
    : null;
};
/** Group admitted input tokens into message rows; null unless every token carries its kind. */
export function capturedInputRows(
  capture: CapturedConversation | undefined,
): CapturedInputRow[] | null {
  if (!capture) {
    return null;
  }
  const known = inputRows.get(capture);
  if (known !== undefined) {
    return known;
  }
  const tokens = kindedTokens(capture);
  const rows = tokens ? groupInputRows(tokens, (token) => token) : null;
  inputRows.set(capture, rows);
  return rows;
}
const inputPairRows = new WeakMap<
  CapturedConversation,
  WeakMap<CapturedConversation, CapturedInputPairRow[]>
>();
/** Both sides' admitted inputs aligned by content and grouped into message rows, so a differing
 *  history or template shows as gaps and mismatches; null unless both sides carry token kinds. */
export function capturedInputPairRows(
  ref: CapturedConversation | undefined,
  target: CapturedConversation | undefined,
): CapturedInputPairRow[] | null {
  if (!ref || !target) {
    return null;
  }
  const known = inputPairRows.get(ref)?.get(target);
  if (known) {
    return known;
  }
  const refTokens = kindedTokens(ref);
  const targetTokens = kindedTokens(target);
  if (!refTokens || !targetTokens) {
    return null;
  }
  const rows = groupInputRows(
    alignTokens(refTokens, targetTokens, 'content'),
    pairAnchor,
  );
  let byTarget = inputPairRows.get(ref);
  if (!byTarget) {
    byTarget = new WeakMap();
    inputPairRows.set(ref, byTarget);
  }
  byTarget.set(target, rows);
  return rows;
}
export type HistoryStatus = 'same' | 'differs' | 'unknown';
const consumed = (capture: CapturedConversation) =>
  capture.tokens
    ?.filter((token) => token.released !== false)
    .map((token) => token.id ?? token.text);
/** Whether both runtimes entered `turn` with the same conversation: identical earlier inputs and
 *  identical earlier outputs (sampled stop tokens are never consumed). Null for the first turn. */
export function historyStatus(
  conversation: CapturedConversation[] | undefined,
  turn: number,
): HistoryStatus | null {
  const earlier = (conversation ?? []).filter((c) => c.turn < turn);
  if (!earlier.length) {
    return null;
  }
  const turns = [...new Set(earlier.map((c) => c.turn))].sort((a, b) => a - b);
  let status: HistoryStatus = 'same';
  for (const n of turns) {
    const ref = earlier.find((c) => c.turn === n && c.run === 'ref');
    const target = earlier.find((c) => c.turn === n && c.run === 'target');
    if (!ref || !target) {
      return 'unknown';
    }
    const refTokens = consumed(ref);
    const targetTokens = consumed(target);
    if (!refTokens || !targetTokens) {
      return 'unknown';
    }
    if (
      ref.input !== target.input ||
      JSON.stringify(refTokens) !== JSON.stringify(targetTokens)
    ) {
      status = 'differs';
    }
  }
  return status;
}
export const tokenPhase = (token: CapturedToken | undefined): TokenPhase =>
  token?.phase ?? 'response';
/** A missing token spelling is not a captured EOS/template string. */
export function capturedTokenLabel(token: CapturedToken | undefined): string {
  return (
    token?.text ?? (token?.id != null ? `Token #${token.id}` : 'Not captured')
  );
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
export function splitPhasePairs(pairs: TokenPair[]) {
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

const contents = new WeakMap<
  CapturedConversation,
  {thinking: string; response: string}
>();
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

/** Content semantics come from capture metadata, never marker-looking text. */
export function tokenContent(token: CapturedToken | undefined): string {
  if (!token) {
    return 'Not captured';
  }
  return token.kind === 'template'
    ? 'Chat template'
    : token.kind === 'special'
      ? 'Special token'
      : tokenPhase(token) === 'thinking'
        ? 'Thinking'
        : 'Response';
}
/** A generated stop token such as `<turn|>` is recorded as special by older captures; when the
 *  same ID is a chat-template token of that run's admitted input, it is a chat-template token. */
export function capturedTokenContent(
  capture: CapturedConversation | undefined,
  token: CapturedToken | undefined,
): string {
  if (
    token?.kind === 'special' &&
    token.id != null &&
    capture?.input_tokens?.some(
      (input) => input.id === token.id && input.kind === 'template',
    )
  ) {
    return 'Chat template';
  }
  return tokenContent(token);
}
export const isTemplateToken = (token: CapturedToken | undefined) =>
  token?.kind === 'template' || token?.kind === 'special';

const HTML_ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};
export const escapeHtml = (text: string) =>
  text.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c] ?? c);

const WHITESPACE_GLYPHS: Record<string, string> = {
  '\n': '↵',
  '\r': '␍',
  '\t': '⇥',
};
/** Characters that draw nothing: whitespace, zero-width and format characters, controls. */
const INVISIBLE =
  /^[\s\u0000-\u001f\u007f-\u009f\u00ad\u200b-\u200f\u2028-\u202f\u2060-\u2064\ufeff]$/u;
export const isInvisibleChar = (ch: string) => INVISIBLE.test(ch);
/** A token made only of invisible characters has nothing to show unless its glyphs are drawn. */
export function isInvisibleOnly(text: string): boolean {
  return text.length > 0 && Array.from(text).every(isInvisibleChar);
}
const codePoint = (ch: string) =>
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
  return runs
    .map((run) => (run.count > 1 ? `${run.name} × ${run.count}` : run.name))
    .join(', ');
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
export function pairLineBreaks(pair: {
  ref?: CapturedToken;
  target?: CapturedToken;
}): number {
  const count = (token: CapturedToken | undefined) =>
    token?.text
      ? (token.text.replace(/\r\n/g, '\n').match(/[\n\r]/g) ?? []).length
      : 0;
  return Math.max(count(pair.ref), count(pair.target));
}
