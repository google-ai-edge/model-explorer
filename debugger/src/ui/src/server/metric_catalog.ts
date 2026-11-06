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

import {computed, DestroyRef, inject, Injectable, InjectionToken, signal} from '@angular/core';
import {MetricsApi} from './metrics_api';
import type {MetricAnalysis, MetricCatalog, MetricDescription} from './contracts/metrics';

/** A catalog to start with instead of asking the server: tests and regressions provide one. */
export const METRIC_CATALOG = new InjectionToken<MetricCatalog>('METRIC_CATALOG');

/** One metric as an analysis offers it: its description and its label there. */
export interface MetricChoice {
  id: string;
  /** The label with the analysis's group, as `Logits · Kullback–Leibler divergence`. */
  label: string;
  description: MetricDescription;
}

/** Looks a catalog up: the descriptions and analyses the server gave. */
export class MetricLookup {
  private readonly metrics: ReadonlyMap<string, MetricDescription>;
  private readonly analyses: ReadonlyMap<string, MetricAnalysis>;
  private readonly reasons: ReadonlyMap<string, string>;
  /** `error` says why there is no catalog, once asking for it has failed; '' otherwise. */
  constructor(
    readonly catalog: MetricCatalog | null,
    readonly error = '',
  ) {
    this.metrics = new Map((catalog?.metrics ?? []).map((metric) => [metric.id, metric]));
    this.analyses = new Map((catalog?.analyses ?? []).map((analysis) => [analysis.id, analysis]));
    this.reasons = new Map((catalog?.reasons ?? []).map((reason) => [reason.code, reason.text]));
  }
  metric(id: string): MetricDescription | undefined {
    return this.metrics.get(id);
  }
  analysis(id: string): MetricAnalysis | undefined {
    return this.analyses.get(id);
  }
  /** The metric's label in an analysis, with the group it belongs to there; with no description,
   *  its id, and why the catalog is missing if it could not be read. */
  label(id: string, analysis?: string): string {
    const label = this.metric(id)?.label ?? (this.error ? `${id} (${this.error})` : id);
    const group = analysis ? this.analysis(analysis)?.groups[id] : undefined;
    return group ? `${group} · ${label}` : label;
  }
  /** The metrics an analysis offers, in its order. */
  choices(analysis: string): MetricChoice[] {
    return (this.analysis(analysis)?.metrics ?? []).flatMap((id) => {
      const description = this.metric(id);
      return description ? [{id, label: this.label(id, analysis), description}] : [];
    });
  }
  /** The catalog's words for a reason code a response gives alone; '' for a code it lacks. */
  reason(code: string): string {
    return this.reasons.get(code) ?? '';
  }
  /** What a metric compares in an analysis: its own inputs, else the analysis's. */
  inputs(id: string, analysis?: string): string | undefined {
    return this.metric(id)?.inputs ?? (analysis ? this.analysis(analysis)?.inputs : undefined);
  }
}

/** How long to wait before each new request of the catalog after a failed one, in ms. */
export const METRIC_CATALOG_RETRIES = [500, 1000, 2000, 4000];
/** How long to wait between requests once those have failed too, in ms. */
export const METRIC_CATALOG_SLOW_RETRY = 30_000;

/** The metric catalog of GET /api/metrics, asked for until it arrives: a few times soon, then
 *  every METRIC_CATALOG_SLOW_RETRY. Until it arrives lookups find nothing and the panels show bare
 *  metric ids; once the first few requests have failed, `error` says why, and labels say it with
 *  the id, until a later request brings the catalog. */
@Injectable({providedIn: 'root'})
export class MetricCatalogStore {
  private readonly api = inject(MetricsApi);
  private readonly controller = new AbortController();
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly catalog = signal<MetricCatalog | null>(null);
  readonly error = signal('');
  /** The current catalog, looked up; a new lookup whenever the catalog or its error changes. */
  readonly lookup = computed(() => new MetricLookup(this.catalog(), this.error()));
  constructor() {
    inject(DestroyRef).onDestroy(() => {
      this.controller.abort();
      if (this.retryTimer !== null) {
        clearTimeout(this.retryTimer);
        this.retryTimer = null;
      }
    });
    const preset = inject(METRIC_CATALOG, {optional: true});
    if (preset) {
      this.catalog.set(preset);
    } else {
      void this.load();
    }
  }
  private async load(): Promise<void> {
    const {signal} = this.controller;
    for (let attempt = 0; !signal.aborted; attempt++) {
      try {
        const catalog = await this.api.catalog(signal);
        if (signal.aborted) {
          return;
        }
        this.catalog.set(catalog);
        this.error.set('');
        return;
      } catch (error) {
        if (signal.aborted) {
          return;
        }
        if (attempt >= METRIC_CATALOG_RETRIES.length) {
          const detail = error instanceof Error ? error.message : String(error);
          this.error.set(`metric descriptions could not be read: ${detail}`);
        }
      }
      const delay = METRIC_CATALOG_RETRIES[attempt] ?? METRIC_CATALOG_SLOW_RETRY;
      const cancelled = await new Promise<boolean>((resolve) => {
        const onAbort = () => {
          if (this.retryTimer !== null) {
            clearTimeout(this.retryTimer);
            this.retryTimer = null;
          }
          resolve(true);
        };
        signal.addEventListener('abort', onAbort, {once: true});
        this.retryTimer = setTimeout(() => {
          this.retryTimer = null;
          signal.removeEventListener('abort', onAbort);
          resolve(false);
        }, delay);
      });
      if (cancelled) {
        return;
      }
    }
  }
}
