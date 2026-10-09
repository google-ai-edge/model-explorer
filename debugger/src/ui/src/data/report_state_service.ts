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

import {
  computed,
  DestroyRef,
  effect,
  inject,
  Injectable,
  signal,
  untracked,
} from '@angular/core';
import {formatSignificant} from '../shared/format/format';
import {ThemeService} from '../theme/theme_service';
import type {CapturedLayer, Session} from './contracts/capture';
import type {
  Comparison,
  Definition,
  Overview,
  Semantic,
} from './contracts/types';
import {ReportApiService} from './report_api_service';

const EMPTY_SESSION: Session = {
  model: '',
  runs: [],
  turns: [],
  phases: [],
  batches: [],
  layers: [],
  anchors: [],
};
const EMPTY_DEFINITION: Definition = {
  kind: '',
  inputs: [],
  nodes: [],
  anchors: [],
};

/** Session evidence loads independently; Graph owns its own request lifetime and projection. */
@Injectable({providedIn: 'root'})
export class ReportStateService {
  readonly theme = inject(ThemeService);
  private readonly api = inject(ReportApiService);
  private readonly selectedCaptureId = signal<string | null>(null);
  readonly captureId = this.selectedCaptureId.asReadonly();
  private readonly graphActive = signal(false);
  private readonly sessionRequest = new RequestLifetime();
  private readonly semanticRequest = new RequestLifetime();
  private readonly overviewRequest = new RequestLifetime();
  private readonly comparisonRequest = new RequestLifetime();
  private selectionCaptureId: string | null = null;
  private comparisonRevision = -1;
  private readonly graphLayers = signal<CapturedLayer[] | null>(null);
  readonly kvEntry = signal<{
    sessionId: string | null;
    turn: number;
    moment: string;
    phase: string;
    step: number | null;
    forward_id: number;
    runtime: string;
  } | null>(null);
  readonly session = signal<Session | null>(null);
  readonly loading = signal(false);
  readonly semantic = signal<Semantic | null>(null);
  readonly graphLoading = signal(false);
  readonly graphError = signal('');
  readonly error = signal('');
  readonly revision = signal(0);
  readonly comparisonError = signal('');
  readonly overview = signal<Overview | null>(null);
  readonly overviewError = signal('');
  readonly comparison = signal<Comparison | null>(null);
  readonly comparing = signal(false);
  readonly layer = signal(0);
  readonly batchId = signal(0);
  readonly metric = signal('CosSim');
  readonly dark = this.theme.dark;
  readonly metrics = [
    'CosSim',
    'Max abs error',
    'Mean abs error',
    'RMSE',
    'Relative L2',
  ];
  private readonly graphSession = computed(() => {
    const session = this.session(),
      layers = this.graphLayers();
    return session && layers ? {...session, layers} : session;
  });
  /** Legacy Graph consumers get enriched layers without modifying the raw Session response. */
  get preview(): Session {
    return this.graphSession() ?? EMPTY_SESSION;
  }
  readonly batch = computed(
    () => this.preview.batches.find((b) => b.batch === this.batchId())!,
  );
  readonly definition = computed(() => {
    const semantic = this.semantic(),
      layer = semantic?.layers[this.layer()];
    return layer ? semantic!.semantic_graph[layer.def] : EMPTY_DEFINITION;
  });
  readonly rows = computed(
    () => this.comparison()?.rows.filter((r) => r.layer === this.layer()) ?? [],
  );

  constructor() {
    effect((onCleanup) => {
      if (
        !this.graphActive() ||
        !this.session()?.batches.length ||
        !this.semantic()?.layers.length
      )
        return;
      const captureId = this.captureId(),
        batch = this.batchId(),
        revision = this.revision();
      if (
        untracked(() => this.comparison()?.batch === batch) &&
        this.comparisonRevision === revision
      )
        return;
      const request = this.comparisonRequest.start();
      this.comparing.set(true);
      this.comparison.set(null);
      this.comparisonError.set('');
      const current = () =>
        request.current() &&
        this.graphActive() &&
        captureId === this.captureId() &&
        batch === this.batchId();
      this.api
        .comparison(captureId, batch, request.signal)
        .then((result) => {
          if (current()) {
            this.comparisonRevision = revision;
            this.comparison.set(result);
          }
        })
        .catch((error) => {
          if (current()) this.comparisonError.set(String(error));
        })
        .finally(() => {
          if (current()) this.comparing.set(false);
          request.finish();
        });
      onCleanup(request.cancel);
    });
    inject(DestroyRef).onDestroy(() => {
      this.sessionRequest.cancel();
      this.cancelGraphRequests();
    });
  }

  attach(captureId: string | null) {
    this.selectedCaptureId.set(captureId);
    this.clear();
  }

  clear() {
    this.sessionRequest.cancel();
    this.resetGraph();
    this.session.set(null);
    this.loading.set(false);
    this.error.set('');
    this.selectionCaptureId = null;
    this.batchId.set(0);
    this.layer.set(0);
  }

  /** ReportPage sets this from navigation; leaving Graph cancels pending work immediately. */
  setGraphActive(active: boolean) {
    if (active === this.graphActive()) return;
    this.graphActive.set(active);
    if (active) void this.loadGraph();
    else this.cancelGraphRequests();
  }

  private cancelGraphRequests() {
    this.semanticRequest.cancel();
    this.overviewRequest.cancel();
    this.comparisonRequest.cancel();
    this.graphLoading.set(false);
    this.comparing.set(false);
  }

  private resetGraph() {
    this.cancelGraphRequests();
    this.semantic.set(null);
    this.graphLayers.set(null);
    this.overview.set(null);
    this.comparison.set(null);
    this.comparisonRevision = -1;
    this.graphError.set('');
    this.overviewError.set('');
    this.comparisonError.set('');
  }

  async load() {
    const captureId = this.captureId(),
      request = this.sessionRequest.start();
    this.resetGraph();
    this.session.set(null);
    this.error.set('');
    this.loading.set(true);
    const current = () => request.current() && captureId === this.captureId();
    try {
      const session = await this.api.session(captureId, request.signal);
      if (!current()) return;
      const selectionValid =
        this.selectionCaptureId === captureId &&
        session.batches.some((batch) => batch.batch === this.batchId());
      if (!selectionValid) this.batchId.set(session.batches[0]?.batch ?? 0);
      this.selectionCaptureId = captureId;
      // Publish the raw response before any Graph dependency is requested.
      this.session.set(session);
      this.loading.set(false);
      if (this.graphActive()) void this.loadGraph();
    } catch (error) {
      if (current()) this.error.set(String(error));
    } finally {
      if (current()) this.loading.set(false);
      request.finish();
    }
  }

  private async loadGraph() {
    const captureId = this.captureId(),
      session = this.session();
    if (
      !this.graphActive() ||
      !captureId ||
      !session?.batches.length ||
      this.loading()
    )
      return;
    if (this.semantic()) {
      if (
        this.semantic()!.layers.length &&
        !this.overview() &&
        !this.overviewRequest.pending
      )
        this.refreshOverview();
      return;
    }
    if (this.semanticRequest.pending) return;
    const request = this.semanticRequest.start();
    this.graphLoading.set(true);
    this.graphError.set('');
    const current = () =>
      request.current() &&
      this.graphActive() &&
      captureId === this.captureId() &&
      session === this.session();
    try {
      const semantic = await this.api.semantic(captureId, request.signal);
      if (!current()) return;
      const layers = semantic.layers.map((layer, index): CapturedLayer => {
        const definition = semantic.semantic_graph[layer.def];
        if (!definition)
          throw new Error('Semantic definition unavailable for layer ' + index);
        const attrs = layer.attrs['attn']?.ops ?? {};
        const span =
          (attrs['attend'] as {span?: {kind: string}})?.span?.kind ?? 'decoder';
        const shared = (attrs['k_proj'] as {skip?: boolean})?.skip;
        const dimension = definition.inputs[0]?.shape.at(-1);
        const hidden =
          typeof dimension === 'number'
            ? dimension
            : typeof dimension === 'string' && /^\d+$/.test(dimension)
              ? Number(dimension)
              : null;
        return {
          index,
          label: span + (shared ? ' · shared KV' : ''),
          hidden: hidden != null && Number.isFinite(hidden) ? hidden : null,
        };
      });
      this.graphLayers.set(layers);
      this.layer.set(Math.max(0, Math.min(this.layer(), layers.length - 1)));
      this.semantic.set(semantic);
      if (layers.length) this.refreshOverview();
    } catch (error) {
      if (current()) this.graphError.set(String(error));
    } finally {
      if (current()) this.graphLoading.set(false);
      request.finish();
    }
  }

  retryGraph() {
    this.resetGraph();
    if (this.graphActive()) void this.loadGraph();
  }

  private refreshOverview() {
    const captureId = this.captureId();
    if (!this.graphActive() || !captureId || !this.semantic()?.layers.length)
      return;
    const request = this.overviewRequest.start();
    this.overviewError.set('');
    const current = () =>
      request.current() && this.graphActive() && captureId === this.captureId();
    this.api
      .overview(captureId, request.signal)
      .then((result) => {
        if (current()) this.overview.set(result);
      })
      .catch((error) => {
        if (current()) this.overviewError.set(String(error));
      })
      .finally(request.finish);
  }

  refreshMetrics() {
    this.comparisonRequest.cancel();
    this.comparing.set(false);
    this.comparison.set(null);
    this.comparisonRevision = -1;
    this.comparisonError.set('');
    this.overview.set(null);
    this.overviewError.set('');
    this.revision.update((value) => value + 1);
    this.refreshOverview();
  }
  summary(layer: number) {
    return this.comparison()?.layers.find((l) => l.layer === layer)?.metrics[
      this.metric()
    ];
  }
  format(value: number | null | undefined) {
    return formatSignificant(value, 5);
  }
  selectLayer(value: number) {
    this.layer.set(
      Math.max(0, Math.min(this.preview.layers.length - 1, value)),
    );
  }
  selectBatch(value: number) {
    if (this.preview.batches.some((b) => b.batch === value))
      this.batchId.set(value);
  }
  selectTurn(value: number) {
    const candidates = this.preview.batches.filter((b) => b.turn === value);
    const next =
      candidates.find((b) => b.phase === this.batch()?.phase) || candidates[0];
    if (next) this.batchId.set(next.batch);
  }
  selectPhase(value: string) {
    const next = this.preview.batches.find(
      (b) => b.turn === this.batch()?.turn && b.phase === value,
    );
    if (next) this.batchId.set(next.batch);
  }
}

/** Aborting also invalidates the response token, even when a transport ignores AbortSignal. */
class RequestLifetime {
  private controller?: AbortController;
  get pending() {
    return this.controller !== undefined;
  }
  cancel() {
    this.controller?.abort();
    this.controller = undefined;
  }
  start() {
    this.cancel();
    const controller = new AbortController();
    this.controller = controller;
    const finish = () => {
      if (this.controller === controller) this.controller = undefined;
    };
    return {
      signal: controller.signal,
      current: () =>
        this.controller === controller && !controller.signal.aborted,
      finish,
      cancel: () => {
        controller.abort();
        finish();
      },
    };
  }
}
