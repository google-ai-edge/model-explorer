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
  KvAnalysis,
  KvAnalysisContext,
  KvMetric,
  KvRangeBin,
} from '../../../data/contracts/kv';
import {compileKvQuery} from './kv_query';

export interface KvHeadViewState {
  view: 'index' | 'scatter';
  numericalExpanded: boolean;
  channelsExpanded: boolean;
}

/** Only KV-owned view state. Shared capture/Turn/phase are never restored from this value. */
export interface KvViewSnapshot {
  version: 1;
  contextId: string;
  kind: string;
  metric: KvMetric;
  colorScale: 'auto' | 'thresholds';
  head: string;
  view: 'heatmap' | 'trends';
  formula: string;
  layer: number;
  position: number;
  selectionEnd: number | null;
  column: boolean;
  zoomRange: KvRangeBin | null;
  zoomHistory: KvRangeBin[];
  panelWidth: number;
  showDetails: boolean;
  showResults: boolean;
  infoExpanded: boolean;
  actionsExpanded: boolean;
  savedExpanded: boolean;
  expandedHeads: Record<string, boolean>;
  headViews: Record<string, KvHeadViewState>;
  detailHeadPages: Record<string, number>;
  tensorExpanded: Record<string, boolean>;
}

export interface KvViewIdentity {
  captureId: string;
  turn: number;
}
export interface KvExplicitEntry {
  sessionId: string | null;
  turn: number;
  moment: string;
  phase: string;
  step: number | null;
  forward_id: number;
  runtime: string;
}

export function defaultKvView(showDetails = true): KvViewSnapshot {
  return {
    version: 1,
    contextId: '',
    kind: 'key',
    metric: 'relative_l2',
    colorScale: 'auto',
    head: 'max',
    view: 'heatmap',
    formula: '',
    layer: 0,
    position: 0,
    selectionEnd: null,
    column: false,
    zoomRange: null,
    zoomHistory: [],
    panelWidth: 350,
    showDetails,
    showResults: false,
    infoExpanded: true,
    actionsExpanded: true,
    savedExpanded: false,
    expandedHeads: {},
    headViews: {},
    detailHeadPages: {key: 0, value: 0},
    tensorExpanded: {key: true, value: true},
  };
}

const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const integer = (value: unknown): value is number =>
  Number.isSafeInteger(value) && Number(value) >= 0;
const interval = (value: unknown): value is KvRangeBin =>
  record(value) &&
  integer(value['start']) &&
  integer(value['end']) &&
  value['end'] > value['start'];
const booleans = (value: unknown): value is Record<string, boolean> =>
  record(value) &&
  Object.values(value).every((item) => typeof item === 'boolean');

/** Reject unsupported or malformed snapshots before applying any fields. */
export function parseKvView(value: unknown): KvViewSnapshot | null {
  if (!record(value) || value['version'] !== 1) return null;
  if (
    !['contextId', 'kind', 'head', 'formula'].every(
      (key) => typeof value[key] === 'string',
    )
  )
    return null;
  if (
    !['key', 'value'].includes(String(value['kind'])) ||
    !['relative_l2', 'max_abs', 'cosine_distance'].includes(
      String(value['metric']),
    ) ||
    !['auto', 'thresholds'].includes(String(value['colorScale'])) ||
    !['heatmap', 'trends'].includes(String(value['view'])) ||
    !/^(max|mean|\d+)$/.test(String(value['head']))
  )
    return null;
  if (
    !integer(value['layer']) ||
    !integer(value['position']) ||
    !(value['selectionEnd'] === null || integer(value['selectionEnd'])) ||
    !Number.isFinite(value['panelWidth']) ||
    Number(value['panelWidth']) < 100 ||
    Number(value['panelWidth']) > 10000
  )
    return null;
  if (
    ![
      'column',
      'showDetails',
      'showResults',
      'infoExpanded',
      'actionsExpanded',
      'savedExpanded',
    ].every((key) => typeof value[key] === 'boolean')
  )
    return null;
  if (
    !(value['zoomRange'] === null || interval(value['zoomRange'])) ||
    !Array.isArray(value['zoomHistory']) ||
    !value['zoomHistory'].every(interval) ||
    !booleans(value['expandedHeads']) ||
    !booleans(value['tensorExpanded']) ||
    !record(value['detailHeadPages']) ||
    !Object.values(value['detailHeadPages']).every(integer) ||
    !record(value['headViews']) ||
    !Object.values(value['headViews']).every(
      (item) =>
        record(item) &&
        ['index', 'scatter'].includes(String(item['view'])) &&
        typeof item['numericalExpanded'] === 'boolean' &&
        typeof item['channelsExpanded'] === 'boolean',
    )
  )
    return null;
  try {
    compileKvQuery(String(value['formula']));
  } catch {
    return null;
  }
  return {
    version: 1,
    contextId: String(value['contextId']),
    kind: String(value['kind']),
    metric: value['metric'] as KvMetric,
    colorScale: value['colorScale'] as 'auto' | 'thresholds',
    head: String(value['head']),
    view: value['view'] as 'heatmap' | 'trends',
    formula: String(value['formula']),
    layer: Number(value['layer']),
    position: Number(value['position']),
    selectionEnd:
      value['selectionEnd'] === null ? null : Number(value['selectionEnd']),
    column: Boolean(value['column']),
    zoomRange:
      value['zoomRange'] === null
        ? null
        : structuredClone(value['zoomRange'] as KvRangeBin),
    zoomHistory: structuredClone(value['zoomHistory'] as KvRangeBin[]),
    panelWidth: Number(value['panelWidth']),
    showDetails: Boolean(value['showDetails']),
    showResults: Boolean(value['showResults']),
    infoExpanded: Boolean(value['infoExpanded']),
    actionsExpanded: Boolean(value['actionsExpanded']),
    savedExpanded: Boolean(value['savedExpanded']),
    expandedHeads: structuredClone(
      value['expandedHeads'] as Record<string, boolean>,
    ),
    headViews: structuredClone(
      value['headViews'] as Record<string, KvHeadViewState>,
    ),
    detailHeadPages: structuredClone(
      value['detailHeadPages'] as Record<string, number>,
    ),
    tensorExpanded: structuredClone(
      value['tensorExpanded'] as Record<string, boolean>,
    ),
  };
}

/** Positions are derived from captured metadata; no absent tensor becomes a zero-valued tensor. */
export function kvContextRange(
  context: KvAnalysisContext | undefined,
): KvRangeBin | null {
  const bounds = (context?.layers ?? []).flatMap((row) => {
    const values: KvRangeBin[] = [];
    if (
      row.position_start !== null &&
      row.position_count !== null &&
      row.position_count > 0
    )
      values.push({
        start: row.position_start,
        end: row.position_start + row.position_count,
      });
    for (const source of [
      row.token_structure?.ref,
      row.token_structure?.target,
    ]) {
      if (
        source?.status === 'ok' &&
        source.position_start !== null &&
        source.position_count !== null &&
        source.position_count > 0
      )
        values.push({
          start: source.position_start,
          end: source.position_start + source.position_count,
        });
    }
    return values;
  });
  return bounds.length
    ? {
        start: Math.min(...bounds.map((range) => range.start)),
        end: Math.max(...bounds.map((range) => range.end)),
      }
    : null;
}

export function restoreKvView(
  value: unknown,
  analysis: KvAnalysis,
  identity: KvViewIdentity,
  entry: KvExplicitEntry | null = null,
  showDetails = true,
): {state: KvViewSnapshot; notice: string; entryMissing: boolean} {
  const parsed = parseKvView(value),
    state = parsed ?? defaultKvView(showDetails);
  const contexts = analysis.contexts.filter(
    (context) => context.turn === identity.turn,
  );
  const explicit =
    entry?.sessionId === identity.captureId && entry.turn === identity.turn;
  // A native context keyed by logical identity carries no step/forward coordinates of its
  // own; its target snapshot still does, so the entry resolves through that snapshot.
  const context = explicit
    ? contexts.find(
        (context) =>
          context.source === 'snapshot' &&
          context.moment === entry.moment &&
          context.runtime === entry.runtime &&
          ((context.phase === entry.phase &&
            context.step === entry.step &&
            context.forward_id === entry.forward_id) ||
            (context.snapshots?.target ?? []).some(
              (snapshot) =>
                snapshot.forward_id === entry.forward_id &&
                snapshot.phase === entry.phase &&
                snapshot.moment === entry.moment,
            )),
      )
    : (contexts.find((context) => context.id === state.contextId) ??
      contexts[0]);
  const changedContext = !context || context.id !== state.contextId;
  let notice =
    value != null && !parsed
      ? 'Saved KV settings were invalid. Defaults were restored.'
      : '';
  if (parsed?.contextId && changedContext && !explicit)
    notice =
      'The saved KV observation is no longer available. Showing an available observation.';
  if (explicit && !context)
    notice =
      'The requested KV observation is no longer available. Choose another captured observation.';
  state.contextId = context?.id ?? '';
  if (changedContext || explicit) {
    state.position = 0;
    state.selectionEnd = null;
    state.column = false;
    state.zoomRange = null;
    state.zoomHistory = [];
  }
  const rows = context?.layers ?? [];
  if (!rows.some((row) => row.kind === state.kind))
    state.kind =
      rows.find((row) => row.kind === 'key' || row.kind === 'value')?.kind ??
      'key';
  const selected =
    rows.find((row) => row.kind === state.kind && row.layer === state.layer) ??
    rows.find((row) => row.kind === state.kind);
  state.layer = selected?.layer ?? 0;
  const heads = Math.max(0, ...rows.map((row) => row.head_count ?? 0));
  if (!['max', 'mean'].includes(state.head) && Number(state.head) >= heads)
    state.head = 'max';
  const range = kvContextRange(context);
  const within = (candidate: KvRangeBin) =>
    !!range && candidate.start >= range.start && candidate.end <= range.end;
  if (!range || state.position < range.start || state.position >= range.end) {
    state.position = range?.start ?? 0;
    state.selectionEnd = null;
  }
  if (
    state.selectionEnd !== null &&
    (!range ||
      state.selectionEnd <= state.position ||
      state.selectionEnd > range.end)
  )
    state.selectionEnd = null;
  if (state.zoomRange && !within(state.zoomRange)) state.zoomRange = null;
  state.zoomHistory = state.zoomHistory.filter(within);
  const validHead = (key: string) => {
    const match = /^(key|value):(\d+)$/.exec(key);
    if (!match) return false;
    const row = rows.find(
      (row) => row.kind === match[1] && row.layer === state.layer,
    );
    const count = Math.max(
      row?.head_count ?? 0,
      row?.token_structure?.ref?.head_count ?? 0,
      row?.token_structure?.target?.head_count ?? 0,
    );
    return Number(match[2]) < count;
  };
  state.expandedHeads = Object.fromEntries(
    Object.entries(state.expandedHeads).filter(([key]) => validHead(key)),
  );
  state.headViews = Object.fromEntries(
    Object.entries(state.headViews).filter(([key]) => validHead(key)),
  );
  state.detailHeadPages = Object.fromEntries(
    ['key', 'value'].map((kind) => {
      const count =
        rows.find((row) => row.kind === kind && row.layer === state.layer)
          ?.head_count ?? 0;
      return [
        kind,
        Math.min(
          state.detailHeadPages[kind] ?? 0,
          Math.max(0, Math.ceil(count / 64) - 1),
        ),
      ];
    }),
  );
  state.tensorExpanded = {
    key: state.tensorExpanded['key'] ?? true,
    value: state.tensorExpanded['value'] ?? true,
  };
  if (!state.formula) state.showResults = false;
  return {state, notice, entryMissing: !!explicit && !context};
}

/** Bounded in-memory return history, distinct from user-created persistent bookmarks. */
export class KvViewCache {
  private readonly views = new Map<string, KvViewSnapshot>();
  private key(identity: KvViewIdentity) {
    return JSON.stringify([identity.captureId, identity.turn]);
  }
  save(identity: KvViewIdentity, value: KvViewSnapshot) {
    const parsed = parseKvView(value);
    if (!parsed) return;
    const key = this.key(identity);
    this.views.delete(key);
    this.views.set(key, parsed);
    if (this.views.size > 80)
      this.views.delete(this.views.keys().next().value!);
  }
  read(identity: KvViewIdentity): KvViewSnapshot | undefined {
    const value = this.views.get(this.key(identity));
    return value ? structuredClone(value) : undefined;
  }
}
