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

import type {ConversationSelection} from './conversation_content';
import type {FlowAnchor} from './flow_virtualization';
import type {AlignmentMode, TokenPair} from './token_alignment';
import {normalizeColor, type TokenColor} from './token_analysis';
import {TOKEN_METRICS} from './token_metric_metadata';
import {compileQuery} from './token_query';
import {SIDE_TREND_METRICS, type TokenTrendState} from './token_trends';

/** Details panel width bounds; the snapshot validator and the resize handle share them. */
export const PANEL_WIDTH = {min: 280, max: 600} as const;

export type ConversationMode = 'Chat' | 'Debug';
export interface ConversationObservation {
  turn: number;
  phase: string;
}
export interface ConversationReadingPosition {
  scrollTop: number;
  anchor: FlowAnchor | null;
}
export interface TokenIdentity {
  step: number;
  id?: number;
  text: string | null;
}
export interface TokenDiffViewState extends ConversationReadingPosition {
  version: 3;
  mode: ConversationMode;
  /** Absent in older v3 snapshots; null means no captured observation was available. */
  observation?: ConversationObservation | null;
  formula: string;
  alignment: AlignmentMode;
  selection: ConversationSelection | null;
  identity?: {ref?: TokenIdentity; target?: TokenIdentity};
  details: boolean;
  boundaries: boolean;
  whitespace: boolean;
  color: TokenColor;
  collapsed: string[];
  thinking: string[];
  infoExpanded: boolean;
  groups: Record<string, boolean>;
  panelWidth: number;
  trends: Record<string, TokenTrendState>;
}

export function pairIdentity(
  pair: TokenPair | null | undefined,
): TokenDiffViewState['identity'] {
  return pair
    ? {
        ...(pair.ref
          ? {ref: {step: pair.ref.step, id: pair.ref.id, text: pair.ref.text}}
          : {}),
        ...(pair.target
          ? {
              target: {
                step: pair.target.step,
                id: pair.target.id,
                text: pair.target.text,
              },
            }
          : {}),
      }
    : undefined;
}
export function samePairIdentity(
  pair: TokenPair,
  identity: NonNullable<TokenDiffViewState['identity']>,
) {
  return (['ref', 'target'] as const).every((side) => {
    const saved = identity[side],
      token = pair[side];
    return saved
      ? !!token &&
          saved.step === token.step &&
          saved.id === token.id &&
          saved.text === token.text
      : !token;
  });
}
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const integer = (value: unknown, min = 0): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= min;
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const metricKeys: readonly string[] = TOKEN_METRICS.map((metric) => metric.key);

/** Anchors are data used in DOM queries, so validate coordinates and vocabulary. */
export function validFlowAnchor(value: unknown): value is FlowAnchor | null {
  if (value === null) return true;
  if (!object(value)) return false;
  if (value.kind === 'absolute')
    return finite(value.scrollTop) && value.scrollTop >= 0;
  if (!integer(value.ti, 1) || !finite(value.offset)) return false;
  if (
    value.windowLine !== undefined &&
    value.windowLine !== null &&
    !integer(value.windowLine)
  )
    return false;
  if (value.kind === 'input')
    return value.stage === 'Prefill' && integer(value.position);
  return (
    value.kind === 'token' &&
    integer(value.step) &&
    typeof value.side === 'string' &&
    ['ref', 'target'].includes(value.side) &&
    typeof value.stage === 'string' &&
    ['Thinking', 'Output'].includes(value.stage)
  );
}
export function validReadingPosition(
  value: unknown,
): value is ConversationReadingPosition {
  return (
    object(value) &&
    finite(value.scrollTop) &&
    value.scrollTop >= 0 &&
    validFlowAnchor(value.anchor)
  );
}
function validSelection(value: unknown): value is ConversationSelection | null {
  if (value === null) return true;
  if (!object(value) || !integer(value.turn, 1)) return false;
  return value.kind === 'token' || value.kind === 'input'
    ? integer(value.index)
    : value.kind === 'stage' &&
        typeof value.stage === 'string' &&
        ['prefill', 'decode'].includes(value.stage) &&
        typeof value.side === 'string' &&
        ['ref', 'target'].includes(value.side);
}
function validSnapshot(value: unknown): value is Record<string, unknown> {
  if (
    !object(value) ||
    typeof value.version !== 'number' ||
    ![2, 3].includes(value.version)
  )
    return false;
  if (
    value.version === 3 &&
    (typeof value.mode !== 'string' || !['Chat', 'Debug'].includes(value.mode))
  )
    return false;
  if (
    value.observation !== undefined &&
    value.observation !== null &&
    (!object(value.observation) ||
      !integer(value.observation.turn, 1) ||
      typeof value.observation.phase !== 'string' ||
      !value.observation.phase)
  )
    return false;
  if (
    typeof value.alignment !== 'string' ||
    !['content', 'steps'].includes(value.alignment) ||
    typeof value.formula !== 'string'
  )
    return false;
  try {
    compileQuery(value.formula);
  } catch {
    return false;
  }
  if (
    !['details', 'boundaries', 'whitespace', 'infoExpanded'].every(
      (key) => typeof value[key] === 'boolean',
    )
  )
    return false;
  if (
    !(
      typeof value.color === 'string' &&
      ['none', 'token_match', ...metricKeys].includes(value.color)
    ) &&
    !(value.version === 2 && typeof value.color === 'boolean')
  )
    return false;
  if (
    !finite(value.panelWidth) ||
    value.panelWidth < PANEL_WIDTH.min ||
    value.panelWidth > PANEL_WIDTH.max ||
    !validReadingPosition(value) ||
    !validSelection(value.selection)
  )
    return false;
  if (
    !Array.isArray(value.collapsed) ||
    !value.collapsed.every(
      (key) =>
        typeof key === 'string' &&
        /^\d+:(ref|target):(prefill|decode)$/.test(key),
    )
  )
    return false;
  if (
    !Array.isArray(value.thinking) ||
    !value.thinking.every(
      (key) => typeof key === 'string' && /^\d+:(ref|target)$/.test(key),
    )
  )
    return false;
  if (
    !object(value.groups) ||
    !Object.values(value.groups).every((item) => typeof item === 'boolean') ||
    !object(value.trends)
  )
    return false;
  if (
    !Object.entries(value.trends).every(
      ([key, trend]) =>
        /^\d+:(content|steps)$/.test(key) &&
        object(trend) &&
        typeof trend.open === 'boolean' &&
        typeof trend.metric === 'string' &&
        (metricKeys.includes(trend.metric) ||
          SIDE_TREND_METRICS.some((metric) => metric.key === trend.metric)) &&
        integer(trend.start) &&
        integer(trend.end) &&
        trend.end > trend.start,
    )
  )
    return false;
  if (value.identity !== undefined) {
    if (!object(value.identity) || !Object.keys(value.identity).length)
      return false;
    if (
      !Object.entries(value.identity).every(
        ([side, token]) =>
          ['ref', 'target'].includes(side) &&
          object(token) &&
          integer(token.step) &&
          (typeof token.text === 'string' || token.text === null) &&
          (token.id === undefined || integer(token.id)),
      )
    )
      return false;
  }
  return true;
}
export function validViewState(value: unknown): value is TokenDiffViewState {
  return validSnapshot(value) && value.version === 3;
}

/** Valid v2 bookmarks were captured in Debug. Unknown versions are never partly applied. */
export function readViewState(value: unknown): TokenDiffViewState | null {
  if (!validSnapshot(value)) return null;
  const raw = value;
  const state: Omit<TokenDiffViewState, 'version'> = {
    scrollTop: Number(raw['scrollTop']),
    anchor: (raw['anchor'] as FlowAnchor | null | undefined) ?? null,
    mode: raw['version'] === 2 ? 'Debug' : (raw['mode'] as ConversationMode),
    color: normalizeColor(raw['color'] as boolean | TokenColor),
    observation:
      (raw['observation'] as ConversationObservation | null | undefined) ??
      null,
    formula: String(raw['formula']),
    alignment: raw['alignment'] as AlignmentMode,
    selection:
      (raw['selection'] as ConversationSelection | null | undefined) ?? null,
    identity: raw['identity'] as
      | {ref?: TokenIdentity; target?: TokenIdentity}
      | undefined,
    details: Boolean(raw['details']),
    boundaries: Boolean(raw['boundaries']),
    whitespace: Boolean(raw['whitespace']),
    collapsed: raw['collapsed'] as string[],
    thinking: raw['thinking'] as string[],
    infoExpanded: Boolean(raw['infoExpanded']),
    groups: raw['groups'] as Record<string, boolean>,
    trends: raw['trends'] as Record<string, TokenTrendState>,
    panelWidth: Number(raw['panelWidth']),
  };
  return captureViewState(state);
}
export function captureViewState(
  state: Omit<TokenDiffViewState, 'version'>,
): TokenDiffViewState {
  return structuredClone({
    ...state,
    observation: state.observation ?? null,
    version: 3,
  });
}

export type ViewRestoreResult =
  | {
      status: 'ready';
      state: TokenDiffViewState;
      observation: ConversationObservation | null;
    }
  | {
      status:
        | 'invalid'
        | 'missing-turn'
        | 'missing-token'
        | 'inconsistent-turn';
    };
/** Resolve against captured tokens before changing any live display state. */
export function prepareViewRestore(
  value: unknown,
  context: {
    hasTurn: (turn: number) => boolean;
    rows: (turn: number, alignment: AlignmentMode) => readonly TokenPair[];
  },
  request:
    | {source: 'return'; observation: ConversationObservation | null}
    | {
        source: 'bookmark' | 'legacy';
        expected: {turn: number; alignment: AlignmentMode};
      } = {
    source: 'return',
    observation: null,
  },
): ViewRestoreResult {
  const state = readViewState(value);
  if (!state) return {status: 'invalid'};
  let observation: ConversationObservation | null = null;
  // Observation changes are independent of token selection. Ordinary return keeps
  // local reading only while the shared Turn/phase has stayed the same.
  if (request.source === 'return' && request.observation) {
    const current = request.observation;
    const changed =
      state.observation &&
      (state.observation.turn !== current.turn ||
        state.observation.phase !== current.phase);
    const staleSelection =
      state.selection && state.selection.turn !== current.turn;
    if (staleSelection) {
      state.selection = null;
      state.identity = undefined;
    }
    if (changed || staleSelection) observation = {...current};
  }
  const expected = request.source === 'return' ? undefined : request.expected;
  if (
    expected &&
    ((state.selection && state.selection.turn !== expected.turn) ||
      state.alignment !== expected.alignment)
  )
    return {status: 'inconsistent-turn'};
  const turn = expected?.turn ?? state.selection?.turn;
  if (turn !== undefined && !context.hasTurn(turn))
    return {status: 'missing-turn'};
  if (state.selection?.kind === 'token') {
    const selection = state.selection;
    let rows: readonly TokenPair[];
    try {
      rows = context.rows(selection.turn, state.alignment);
    } catch {
      return {status: 'missing-token'};
    }
    const pair = state.identity
      ? rows.find((row) => samePairIdentity(row, state.identity!))
      : rows.find((row) => row.index === selection.index);
    if (!pair) return {status: 'missing-token'};
    state.selection = {...selection, index: pair.index};
  }
  return {status: 'ready', state, observation};
}

export function readingPositionKey(
  context: string,
  mode: ConversationMode,
): string {
  return JSON.stringify([context, mode]);
}
export function captureReadingPosition(
  scrollTop: number,
  anchor: FlowAnchor | null,
): ConversationReadingPosition {
  return structuredClone({scrollTop: Math.max(0, scrollTop), anchor});
}
export function savedReadingPosition(
  positions: ReadonlyMap<string, ConversationReadingPosition>,
  context: string,
  mode: ConversationMode,
): ConversationReadingPosition {
  const saved = positions.get(readingPositionKey(context, mode));
  return validReadingPosition(saved)
    ? structuredClone(saved)
    : {scrollTop: 0, anchor: null};
}
/** Save the outgoing layout, then restore only the incoming mode's own location. */
export function switchReadingPosition(
  positions: Map<string, ConversationReadingPosition>,
  context: string,
  outgoing: ConversationMode | null,
  incoming: ConversationMode,
  current: ConversationReadingPosition,
): ConversationReadingPosition {
  if (outgoing)
    positions.set(
      readingPositionKey(context, outgoing),
      structuredClone(current),
    );
  return savedReadingPosition(positions, context, incoming);
}
