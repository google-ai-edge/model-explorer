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

import {validValueRange, type ValueRange} from '../../widgets/charts/scatter_model';
import type {KvAnalysis, KvAnalysisContext, KvMetric, KvRangeBin} from '../../server/contracts/kv';

/**
 * Only KV-owned view state. Shared capture/Turn/phase are never restored from this value.
 * Snapshots saved by older versions may carry fields this one no longer has; they are ignored.
 */
export interface KvViewSnapshot {
  version: 1;
  contextId: string;
  kind: string;
  metric: KvMetric;
  colorScale: 'auto' | 'thresholds';
  head: string;
  headSelection?: number[] | null;
  positionReduction?: 'min' | 'max' | 'mean' | null;
  view: 'heatmap' | 'lineChart';
  tensor?: {
    active: boolean;
    plot: 'index' | 'scatter';
    position: number | null;
    layer: number | null;
    head: number | null;
    channelRange?: KvRangeBin | null;
    valueRange?: ValueRange | null;
    channelReduction?: 'min' | 'max' | 'mean';
    visibleRuntimes?: string[];
  };
  layer: number;
  position: number;
  selectionEnd: number | null;
  lineChartLayers?: number[] | null;
  heatmapLayers?: number[] | null;
  bothTensors?: boolean;
  column: boolean;
  zoomRange: KvRangeBin | null;
}

export interface KvViewIdentity {
  captureId: string;
  turn: number;
}

export function defaultKvView(): KvViewSnapshot {
  return {
    version: 1,
    contextId: '',
    kind: 'key',
    metric: 'cosine_similarity',
    colorScale: 'auto',
    head: 'min',
    view: 'heatmap',
    layer: 0,
    position: 0,
    selectionEnd: null,
    column: false,
    zoomRange: null,
  };
}

function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}
function integer(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}
function interval(value: unknown): value is KvRangeBin {
  return (
    record(value) &&
    integer(value['start']) &&
    integer(value['end']) &&
    value['end'] > value['start']
  );
}

function isKvViewSnapshot(value: unknown): value is KvViewSnapshot {
  if (!record(value) || value['version'] !== 1) {
    return false;
  }
  const tensor = value['tensor'];
  if (
    tensor !== undefined &&
    (!record(tensor) ||
      typeof tensor['active'] !== 'boolean' ||
      (tensor['plot'] !== 'index' && tensor['plot'] !== 'scatter') ||
      (tensor['valueRange'] != null && !validValueRange(tensor['valueRange'])) ||
      (tensor['channelRange'] != null && !interval(tensor['channelRange'])) ||
      (tensor['channelReduction'] !== undefined &&
        tensor['channelReduction'] !== 'min' &&
        tensor['channelReduction'] !== 'max' &&
        tensor['channelReduction'] !== 'mean') ||
      (tensor['visibleRuntimes'] !== undefined &&
        (!Array.isArray(tensor['visibleRuntimes']) ||
          !tensor['visibleRuntimes'].every((side) => side === 'ref' || side === 'target'))) ||
      ![tensor['position'], tensor['layer'], tensor['head']].every(
        (coordinate) => coordinate === null || integer(coordinate),
      ))
  ) {
    return false;
  }
  const headSelection = value['headSelection'];
  if (
    headSelection != null &&
    (!Array.isArray(headSelection) ||
      !headSelection.length ||
      !headSelection.every(integer) ||
      new Set(headSelection).size !== headSelection.length)
  ) {
    return false;
  }
  for (const selected of [value['lineChartLayers'], value['heatmapLayers']]) {
    if (
      selected != null &&
      (!Array.isArray(selected) ||
        !selected.every((n) => Number.isInteger(n) && n >= 0) ||
        new Set(selected).size !== selected.length)
    ) {
      return false;
    }
  }
  const positionReduction = value['positionReduction'];
  if (
    positionReduction != null &&
    positionReduction !== 'min' &&
    positionReduction !== 'max' &&
    positionReduction !== 'mean'
  ) {
    return false;
  }
  if (value['bothTensors'] !== undefined && typeof value['bothTensors'] !== 'boolean') {
    return false;
  }
  if (!['contextId', 'kind', 'head'].every((key) => typeof value[key] === 'string')) {
    return false;
  }
  if (
    !['key', 'value'].includes(String(value['kind'])) ||
    !['relative_l2', 'max_abs', 'cosine_distance', 'cosine_similarity'].includes(
      String(value['metric']),
    ) ||
    !['auto', 'thresholds'].includes(String(value['colorScale'])) ||
    !['heatmap', 'lineChart'].includes(String(value['view'])) ||
    !/^(max|min|mean|\d+)$/.test(String(value['head']))
  ) {
    return false;
  }
  if (
    !integer(value['layer']) ||
    !integer(value['position']) ||
    !(value['selectionEnd'] === null || integer(value['selectionEnd']))
  ) {
    return false;
  }
  if (typeof value['column'] !== 'boolean') {
    return false;
  }
  if (!(value['zoomRange'] === null || interval(value['zoomRange']))) {
    return false;
  }
  return true;
}

/** Reject unsupported or malformed snapshots before applying any fields. */
export function parseKvView(value: unknown): KvViewSnapshot | null {
  return isKvViewSnapshot(value) ? structuredClone(value) : null;
}

/** Positions are derived from captured metadata; no absent tensor becomes a zero-valued tensor. */
export function kvContextRange(context: KvAnalysisContext | undefined): KvRangeBin | null {
  const bounds = (context?.layers ?? []).flatMap((row) => {
    const values: KvRangeBin[] = [];
    if (row.position_start !== null && row.position_count !== null && row.position_count > 0) {
      values.push({start: row.position_start, end: row.position_start + row.position_count});
    }
    for (const source of [row.token_structure?.ref, row.token_structure?.target]) {
      if (
        source?.status === 'ok' &&
        source.position_start !== null &&
        source.position_count !== null &&
        source.position_count > 0
      ) {
        values.push({
          start: source.position_start,
          end: source.position_start + source.position_count,
        });
      }
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
): {state: KvViewSnapshot; notice: string} {
  const parsed = parseKvView(value);
  const state = parsed ?? defaultKvView();
  const contexts = analysis.contexts.filter((context) => context.turn === identity.turn);
  const context = contexts.find((context) => context.id === state.contextId) ?? contexts[0];
  const changedContext = !context || context.id !== state.contextId;
  let notice =
    value != null && !parsed ? 'Saved KV settings were invalid. Defaults were restored.' : '';
  if (parsed?.contextId && changedContext) {
    notice = 'The saved KV observation is no longer available. Showing an available observation.';
  }
  state.contextId = context?.id ?? '';
  if (changedContext) {
    state.position = 0;
    state.selectionEnd = null;
    state.column = false;
    state.zoomRange = null;
  }
  const rows = context?.layers ?? [];
  if (!rows.some((row) => row.kind === state.kind)) {
    state.kind = rows.find((row) => row.kind === 'key' || row.kind === 'value')?.kind ?? 'key';
  }
  const selected =
    rows.find((row) => row.kind === state.kind && row.layer === state.layer) ??
    rows.find((row) => row.kind === state.kind);
  state.layer = selected?.layer ?? 0;
  const heads = Math.max(0, ...rows.map((row) => row.head_count ?? 0));
  if (state.headSelection) {
    state.headSelection = state.headSelection.filter((head) => head < heads);
    if (!state.headSelection.length) {
      state.headSelection = null;
    }
  }
  if (!['max', 'min', 'mean'].includes(state.head) && Number(state.head) >= heads) {
    state.head = 'max';
  }
  const range = kvContextRange(context);
  const within = (candidate: KvRangeBin): boolean =>
    !!range && candidate.start >= range.start && candidate.end <= range.end;
  if (!range || state.position < range.start || state.position >= range.end) {
    state.position = range?.start ?? 0;
    state.selectionEnd = null;
  }
  if (
    state.selectionEnd !== null &&
    (!range || state.selectionEnd <= state.position || state.selectionEnd > range.end)
  ) {
    state.selectionEnd = null;
  }
  if (state.zoomRange && !within(state.zoomRange)) {
    state.zoomRange = null;
  }
  const tensor = state.tensor;
  if (tensor) {
    if (
      changedContext ||
      !range ||
      tensor.position === null ||
      tensor.position < range.start ||
      tensor.position >= range.end
    ) {
      tensor.position = null;
    }
    if (!rows.some((row) => row.layer === tensor.layer)) {
      tensor.layer = null;
    }
    const count = Math.max(
      0,
      ...rows
        .filter((row) => row.layer === tensor.layer)
        .flatMap((row) => [
          row.head_count ?? 0,
          row.token_structure?.ref?.head_count ?? 0,
          row.token_structure?.target?.head_count ?? 0,
        ]),
    );
    if (tensor.head !== null && tensor.head >= count) {
      tensor.head = null;
    }
  }
  return {state, notice};
}
