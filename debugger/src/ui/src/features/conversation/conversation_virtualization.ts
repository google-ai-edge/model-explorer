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

import type {CapturedConversation} from '../../data/contracts/capture';
import {
  capturedTokenLabel,
  escapeHtml,
  tokenMarkup,
} from './conversation_content';
import {FlowRow, FlowTurn} from './flow_virtualization';
import {AlignmentMode, alignTokens, TokenPair} from './token_alignment';

/** Display chunks preserve the text; they are not claimed to be model tokens. */
export function textChunks(text: string): string[] {
  return text.match(/\r\n|\r|\n|[^\S\r\n]+|[^\s]{1,8}/gu) ?? [];
}
const inputChunks = new WeakMap<CapturedConversation, Map<string, string[]>>();
function capturedInputChunks(
  capture: CapturedConversation | undefined,
  text: string,
): string[] {
  if (!capture) return [];
  let versions = inputChunks.get(capture);
  if (!versions) {
    versions = new Map();
    inputChunks.set(capture, versions);
  }
  if (!versions.has(text)) versions.set(text, textChunks(text));
  return versions.get(text)!;
}
export interface VirtualConversation extends FlowTurn {
  rows: FlowRow[];
  pairs: TokenPair[];
  bySide: Record<string, Map<number, FlowRow>>;
  alignmentUnavailable: boolean;
  hasCapturedTokens: boolean;
  responseTextComplete: boolean;
}
/** Streaming chunks are layout units, never fabricated captured model tokens. */
export function streamingConversation(
  input: string,
  output: Record<string, string>,
): VirtualConversation {
  const steps = {
    ref: textChunks(output['ref'] ?? ''),
    target: textChunks(output['target'] ?? ''),
  };
  const inputTokens = textChunks(input);
  const rows = Array.from(
    {length: Math.max(steps.ref.length, steps.target.length)},
    (_, index) => ({
      index,
      ref: index < steps.ref.length ? index : null,
      target: index < steps.target.length ? index : null,
    }),
  );
  return {
    steps,
    inputTokens: {ref: inputTokens, target: inputTokens},
    alignments: {content: rows, steps: rows},
    rows,
    pairs: [],
    bySide: {ref: new Map(), target: new Map()},
    alignmentUnavailable: false,
    hasCapturedTokens: false,
    responseTextComplete: true,
  };
}
function buildVirtualConversation(
  ref: CapturedConversation | undefined,
  target: CapturedConversation | undefined,
  mode: AlignmentMode,
  debug: boolean,
): VirtualConversation {
  const captured = {ref: ref?.tokens, target: target?.tokens};
  const hasCapturedTokens = !!captured.ref && !!captured.target;
  // Partial token captures must never replace the complete Chat response text.
  const responseTextComplete = [ref, target].every(
    (c) =>
      c?.output !== undefined &&
      c.tokens
        ?.filter((t) => (t.phase ?? 'response') === 'response')
        .every((t) => t.text !== null) &&
      c.tokens
        ?.filter((t) => (t.phase ?? 'response') === 'response')
        .map((t) => t.text)
        .join('') === c.output,
  );
  const inputs = {
    ref: (debug ? ref?.serialized_input : undefined) ?? ref?.input ?? '',
    target:
      (debug ? target?.serialized_input : undefined) ?? target?.input ?? '',
  };
  const longInput = Math.max(inputs.ref.length, inputs.target.length) > 16000;
  const inputTokens = {
    ref: longInput ? capturedInputChunks(ref, inputs.ref) : [],
    target: longInput ? capturedInputChunks(target, inputs.target) : [],
  };
  // Small turns retain their existing Angular templates and alignment path.
  if (
    Math.max(captured.ref?.length ?? 0, captured.target?.length ?? 0) <= 2048
  ) {
    return {
      steps: {ref: [], target: []},
      inputTokens,
      alignments: {},
      rows: [],
      pairs: [],
      bySide: {ref: new Map(), target: new Map()},
      alignmentUnavailable: false,
      hasCapturedTokens,
      responseTextComplete,
    };
  }
  const tokens = {ref: captured.ref ?? [], target: captured.target ?? []};
  let pairs: TokenPair[] = [],
    alignmentUnavailable = false;
  try {
    pairs = alignTokens(
      captured.ref,
      captured.target,
      debug ? mode : 'content',
      ref?.alignments?.content ?? target?.alignments?.content,
    );
  } catch {
    alignmentUnavailable = true;
    pairs = alignTokens(captured.ref, captured.target, 'steps');
  }
  const indices = {
    ref: new Map(tokens.ref.map((token, index) => [token, index])),
    target: new Map(tokens.target.map((token, index) => [token, index])),
  };
  const rows: FlowRow[] = pairs.map((pair) => ({
    index: pair.index,
    ref: pair.ref ? (indices.ref.get(pair.ref) ?? null) : null,
    target: pair.target ? (indices.target.get(pair.target) ?? null) : null,
  }));
  const bySide = {
    ref: new Map<number, FlowRow>(),
    target: new Map<number, FlowRow>(),
  };
  for (const row of rows)
    for (const side of ['ref', 'target'] as const)
      if (row[side] != null) bySide[side].set(row[side]!, row);
  return {
    steps: {
      ref: tokens.ref.map(capturedTokenLabel),
      target: tokens.target.map(capturedTokenLabel),
    },
    inputTokens,
    alignments: {steps: rows, content: rows},
    rows,
    pairs,
    bySide,
    alignmentUnavailable,
    hasCapturedTokens,
    responseTextComplete,
  };
}
export const escapeVirtualText = escapeHtml;

// Captures are immutable; keep per-mode adapters while the capture is retained.
const models = new WeakMap<
  CapturedConversation,
  WeakMap<CapturedConversation, Map<string, VirtualConversation>>
>();
export function virtualConversation(
  ref: CapturedConversation | undefined,
  target: CapturedConversation | undefined,
  mode: AlignmentMode,
  debug: boolean,
): VirtualConversation {
  if (!ref || !target)
    return buildVirtualConversation(ref, target, mode, debug);
  let targets = models.get(ref);
  if (!targets) {
    targets = new WeakMap();
    models.set(ref, targets);
  }
  let variants = targets.get(target);
  if (!variants) {
    variants = new Map();
    targets.set(target, variants);
  }
  // Chat and Context Debug share the same immutable alignment. Mode changes
  // only replace their visible markup, not all 128k rows and lookup maps.
  const alignment = debug ? mode : 'content';
  const key =
    alignment +
    (ref.serialized_input !== undefined || target.serialized_input !== undefined
      ? debug
        ? ':serialized'
        : ':chat'
      : '');
  if (!variants.has(key))
    variants.set(key, buildVirtualConversation(ref, target, alignment, debug));
  return variants.get(key)!;
}

export function renderInputToken(
  word: string,
  index: number,
  side: string,
): string {
  return `<span class="readingToken" data-input-position="${index}" data-side="${side}">${escapeVirtualText(word)}</span>`;
}

export interface OutputTokenStyle {
  /** Debug mode with a usable alignment renders a button; otherwise a plain span. */
  interactive: boolean;
  debug: boolean;
  side: string;
  /** null when the numeric color is off, otherwise whether this token has a metric. */
  numeric: 'token' | 'missing' | null;
  first: boolean;
  /** The details pair this token, by generation step, with the selected first differing row. */
  partner: boolean;
  template: boolean;
  mismatch: boolean;
  active: boolean;
  boundaries: boolean;
  palette: string;
  heat: string | null;
  empty: boolean;
}
export function outputTokenClasses(style: OutputTokenStyle): string {
  return [
    style.numeric
      ? style.numeric === 'missing'
        ? 'numeric-missing'
        : 'numeric-token'
      : '',
    style.first ? 'first-divergence' : '',
    style.template ? 'template-token' : '',
    style.mismatch ? 'mismatch chat-mismatch' : '',
    style.active ? 'selected selected-token' : '',
    style.boundaries ? 'boundaries' : '',
  ]
    .join(' ')
    .concat(style.partner ? ' step-partner' : '');
}
export function renderOutputToken(
  word: string,
  position: {turn: number; side: string; step: number; row: number | undefined},
  style: OutputTokenStyle,
): string {
  const classes = outputTokenClasses(style);
  const content = style.debug ? tokenMarkup(word) : escapeVirtualText(word);
  if (!style.interactive)
    return `<span class="readingToken ${classes}" data-step="${position.step}" data-turn="${position.turn}" data-side="${position.side}" title="${escapeVirtualText(word)}">${content}</span>`;
  const label = `${position.side} aligned position ${position.row}: ${word}${style.first ? ' · First divergence' : ''}`;
  return `<button type="button" class="token ${classes} ${style.empty ? 'empty-token' : ''}" data-palette="${style.palette}" style="--heat-level:${style.heat ?? '0%'}" data-turn="${position.turn}" data-row="${position.row}" data-step="${position.step}" data-side="${position.side}" aria-pressed="${style.active}" aria-label="${escapeVirtualText(label)}">${content}</button>`;
}
