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

import type {Semantic} from '../../../data/contracts/types';
import type {ArchitectureViewport} from '../architecture_view/architecture_renderer';
import {emptyGraphQuery, type GraphQuery} from './graph_query';

export type GraphViewport = ArchitectureViewport;

/** Reading state only. Captured data, comparison results and mapping drafts are never cached here. */
export interface GraphViewState {
  version: 1;
  query: GraphQuery;
  metric: string;
  details: boolean;
  results: boolean;
  trends: boolean;
  detailsWidth: number;
  trendHeight: number;
  context: {layer: number; semantic: string} | null;
  execution: boolean;
  viewport: GraphViewport | null;
}

export function graphViewKey(capture: string, batch: number): string {
  return JSON.stringify([capture, batch]);
}

export function parseGraphView(value: unknown): GraphViewState | null {
  if (!value || typeof value !== 'object') return null;
  const data = value as GraphViewState;
  const query = data.query;
  if (
    data.version !== 1 ||
    !query ||
    !['text', 'anchor', 'metric', 'threshold'].every(
      (key) => typeof query[key as keyof GraphQuery] === 'string',
    ) ||
    !['lt', 'gt'].includes(query.operator) ||
    typeof query.withMetrics !== 'boolean' ||
    typeof data.metric !== 'string' ||
    ![data.details, data.results, data.trends, data.execution].every(
      (v) => typeof v === 'boolean',
    ) ||
    !Number.isFinite(data.detailsWidth) ||
    !Number.isFinite(data.trendHeight)
  )
    return null;
  if (
    data.context &&
    (!Number.isInteger(data.context.layer) ||
      data.context.layer < 0 ||
      typeof data.context.semantic !== 'string')
  )
    return null;
  const viewport = data.viewport;
  if (
    viewport &&
    (!Number.isInteger(viewport.layer) ||
      viewport.layer < 0 ||
      ![viewport.zoom, viewport.left, viewport.top].every(Number.isFinite) ||
      viewport.zoom < 0.15 ||
      viewport.zoom > 3 ||
      viewport.left < 0 ||
      viewport.top < 0)
  )
    return null;
  return {
    version: 1,
    query: {...emptyGraphQuery(), ...query},
    metric: data.metric,
    details: data.details,
    results: data.results,
    trends: data.trends,
    detailsWidth: Math.max(300, Math.min(560, data.detailsWidth)),
    trendHeight: Math.max(80, Math.min(320, data.trendHeight)),
    context: data.context ? {...data.context} : null,
    execution: data.execution,
    viewport: viewport ? {...viewport} : null,
  };
}

/** A normal return never changes the observation or layer selected by another view. */
export function resolveGraphView(
  value: unknown,
  model: Semantic,
  layer: number,
  metrics: readonly string[],
): GraphViewState | null {
  const view = parseGraphView(value);
  if (!view) return null;
  if (!metrics.includes(view.metric)) view.metric = metrics[0] ?? 'CosSim';
  if (!metrics.includes(view.query.metric)) view.query.metric = view.metric;
  if (view.context) {
    const definition =
      model.semantic_graph[model.layers[view.context.layer]?.def];
    const id = view.context.semantic;
    if (
      view.context.layer !== layer ||
      !definition ||
      !(
        definition.nodes.some((node) => node.id === id) ||
        definition.anchors.some((anchor) => 'anchor:' + anchor.id === id)
      )
    )
      view.context = null;
  }
  if (!view.context) view.execution = false;
  if (view.viewport?.layer !== layer) view.viewport = null;
  return view;
}
