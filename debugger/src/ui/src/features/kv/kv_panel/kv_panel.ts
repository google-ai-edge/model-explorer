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

import {A11yModule} from '@angular/cdk/a11y';
import {OverlayModule} from '@angular/cdk/overlay';
import {NgTemplateOutlet} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {
  KvAnalysisContext,
  KvAnalysisLayer,
  KvFindResult,
  KvMetric,
  KvRangeBin,
  KvRangeDetailRow,
  KvSnapshotIdentity,
  KvTokenMetrics,
  KvTokenStructure,
} from '../../../data/contracts/kv';
import {ReportApiService} from '../../../data/report_api_service';
import {ReportStateService} from '../../../data/report_state_service';
import {BookmarkStore} from '../../../shared/bookmark_store/bookmark_store';
import {ExplorerInfoSection} from '../../../shared/explorer_info_section/explorer_info_section';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {FindNavigator} from '../../../shared/find_navigator/find_navigator';
import {formatSignificant} from '../../../shared/format/format';
import {
  OverlayDialog,
  OverlayPanel,
} from '../../../shared/overlay_dialog/overlay_dialog';
import {ResizeHandle} from '../../../shared/resize_handle/resize_handle';
import {
  KV_BOOKMARK_CODEC,
  SavedPoint,
  kvBookmarkKey,
  kvBookmarkLabel,
} from './kv_bookmarks';
import {KvDataController} from './kv_data';
import {KvFindController} from './kv_find_controller';
import {KvHeadInspector, KvHeadSelection} from './kv_head_inspector';
import {
  KV_DETAIL_METRICS,
  KvMetricValues,
  formatKvMetric,
} from './kv_metric_values';
import {KvRangeChart} from './kv_range_chart';
import {KV_RANGE_MAX_BINS} from './kv_range_geometry';
import {KvRangeSelection} from './kv_range_types';
import {
  kvContextRange,
  restoreKvView,
  type KvHeadViewState,
  type KvViewIdentity,
  type KvViewSnapshot,
} from './kv_view_state';
import {KvViewStore} from './kv_view_store';

const DEFAULT_HEAD_VIEW: KvHeadViewState = {
  view: 'index',
  numericalExpanded: true,
  channelsExpanded: true,
};
const METRICS: {id: KvMetric; label: string}[] = [
  {id: 'relative_l2', label: 'Relative L2'},
  {id: 'max_abs', label: 'Max |Δ|'},
  {id: 'cosine_distance', label: 'Cosine distance'},
];

@Component({
  selector: 'kv-panel',
  standalone: true,
  imports: [
    OverlayDialog,
    OverlayPanel,
    FindNavigator,
    ResizeHandle,
    FormsModule,
    NgTemplateOutlet,
    KvRangeChart,
    KvHeadInspector,
    KvMetricValues,
    ExplorerInfoSection,
    ExplorerInfoValue,
    OverlayModule,
    A11yModule,
    MatIconModule,
    MatButtonModule,
    MatTooltipModule,
  ],
  templateUrl: './kv_panel.ng.html',
  styleUrl: './kv_panel.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvPanel {
  readonly state = inject(ReportStateService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly data = new KvDataController(inject(ReportApiService));
  private readonly views = inject(KvViewStore);
  private readonly resolvedIdentity = signal<KvViewIdentity | null>(null);
  readonly stateNotice = signal('');
  readonly entryMissing = signal(false);
  readonly metrics = METRICS;
  readonly detailMetrics = KV_DETAIL_METRICS;
  readonly formatDetailMetric = formatKvMetric;
  readonly analysis = this.data.metadata.data;
  readonly error = this.data.metadata.error;
  readonly loading = this.data.metadata.loading;
  readonly retry = signal(0);
  readonly selectedContextId = signal('');
  readonly kind = signal('key');
  readonly metric = signal<KvMetric>('relative_l2');
  readonly colorScale = signal<'auto' | 'thresholds'>('auto');
  readonly head = signal('max');
  readonly view = signal<'heatmap' | 'trends'>('heatmap');
  readonly layer = signal(0);
  readonly position = signal(0);
  readonly column = signal(false);
  readonly showDetails = signal(
    window.matchMedia('(min-width: 801px)').matches,
  );
  readonly popup = signal<'context' | 'display' | 'find' | null>(null);
  draftScale: 'auto' | 'thresholds' = 'auto';
  draftMetric: KvMetric = 'relative_l2';
  draftKind = 'key';
  draftHead = 'max';
  readonly selectedTurn = computed(
    () =>
      this.state
        .session()
        ?.batches.find((b) => b.batch === this.state.batchId())?.turn ??
      this.state.session()?.turns[0]?.n ??
      1,
  );
  private readonly identity = computed<KvViewIdentity | null>(() => {
    const captureId = this.state.captureId();
    return captureId ? {captureId, turn: this.selectedTurn()} : null;
  });
  private readonly queryIdentity = computed(() => {
    const current = this.identity(),
      loaded = this.resolvedIdentity();
    return current?.captureId === loaded?.captureId &&
      current?.turn === loaded?.turn
      ? loaded
      : null;
  });
  readonly context = computed(() =>
    this.queryIdentity()
      ? this.analysis()?.contexts.find(
          (context) =>
            context.id === this.selectedContextId() &&
            context.turn === this.selectedTurn(),
        )
      : undefined,
  );
  readonly layers = computed(() =>
    (this.context()?.layers.filter((l) => l.kind === this.kind()) ?? []).sort(
      (a, b) => a.layer - b.layer,
    ),
  );
  readonly selectedLayer = computed<KvAnalysisLayer | undefined>(
    () =>
      this.layers().find((l) => l.layer === this.layer()) ??
      this.context()?.layers.find((l) => l.layer === this.layer()) ??
      this.layers()[0] ??
      this.context()?.layers[0],
  );
  readonly layerIds = computed(() => this.layers().map((l) => l.layer));
  readonly selectionEnd = signal<number | null>(null);
  readonly zoomRange = signal<KvRangeBin | null>(null);
  readonly zoomHistory = signal<KvRangeBin[]>([]);
  readonly binCapacity = signal(24);
  readonly rangeData = this.data.range.data;
  readonly rangeLoading = this.data.range.loading;
  readonly rangeError = this.data.range.error;
  readonly rangeRetry = signal(0);
  readonly scaleReference = this.data.scale.data;
  readonly scaleLoading = this.data.scale.loading;
  readonly scaleError = computed(() =>
    this.data.scale.error()
      ? 'Color scale unavailable: ' + this.data.scale.error()
      : '',
  );
  readonly chartLoading = computed(
    () =>
      this.rangeLoading() ||
      (this.colorScale() === 'auto' && this.scaleLoading()),
  );
  readonly chartError = computed(
    () =>
      this.rangeError() ||
      (this.colorScale() === 'auto' ? this.scaleError() : ''),
  );
  readonly selectionData = this.data.selection.data;
  readonly selectionLoading = this.data.selection.loading;
  readonly selectionError = this.data.selection.error;
  readonly selectionRetry = signal(0);
  readonly findData = this.data.find.data;
  readonly findLoading = this.data.find.loading;
  readonly findError = this.data.find.error;
  readonly find = new KvFindController({
    data: this.findData,
    loading: this.findLoading,
    select: (result) => {
      this.kind.set(result.kind);
      this.selectPoint(result.layer, result.position);
    },
    setResults: (show) => this.setResults(show),
    openPopup: () => this.popup.set('find'),
    closePopup: () => this.popup.set(null),
  });
  readonly findRetry = this.find.retry;
  readonly resultIndex = this.find.resultIndex;
  readonly resultTotal = this.find.resultTotal;
  readonly resultPage = this.find.resultPage;
  readonly showResults = this.find.showResults;
  readonly formula = this.find.formula;
  readonly queryError = this.find.queryError;
  get draftFormula() {
    return this.find.draftFormula;
  }
  set draftFormula(value: string) {
    this.find.draftFormula = value;
  }
  readonly fullRange = computed(() => kvContextRange(this.context()));
  readonly visibleRange = computed(() => this.zoomRange() ?? this.fullRange());
  readonly visibleBins = computed<KvRangeBin[]>(() => {
    const range = this.visibleRange();
    if (!range) return [];
    const n = Math.min(this.binCapacity(), range.end - range.start);
    return Array.from({length: n}, (_, i) => ({
      start: range.start + Math.floor((i * (range.end - range.start)) / n),
      end: range.start + Math.floor(((i + 1) * (range.end - range.start)) / n),
    }));
  });
  readonly chartBins = computed(
    () => this.rangeData()?.bins ?? this.visibleBins(),
  );
  readonly chartRows = computed(() =>
    this.colorScale() === 'auto' && !this.scaleReference()
      ? []
      : (this.rangeData()?.rows ?? []),
  );
  readonly selectedPosition = computed<number | undefined>(() => {
    const range = this.fullRange(),
      position = this.position();
    return range
      ? position >= range.start && position < range.end
        ? position
        : range.start
      : undefined;
  });
  readonly inspectedRange = computed<KvRangeSelection | null>(() => {
    const start = this.selectedPosition();
    if (start === undefined) return null;
    return {
      start,
      end: Math.min(
        this.fullRange()!.end,
        Math.max(start + 1, this.selectionEnd() ?? start + 1),
      ),
      layer: this.column() ? null : (this.selectedLayer()?.layer ?? null),
    };
  });
  readonly isRange = computed(() => {
    const range = this.inspectedRange();
    return !!range && range.end - range.start > 1;
  });
  readonly heads = computed(() =>
    Array.from(
      {
        length: Math.max(
          0,
          ...(this.context()?.layers ?? []).map((row) => row.head_count ?? 0),
        ),
      },
      (_, i) => i,
    ),
  );
  readonly results = this.find.results;
  readonly resultPages = this.find.resultPages;
  readonly pageResults = this.results;
  readonly infoExpanded = signal(true);
  readonly actionsExpanded = signal(true);
  readonly savedExpanded = signal(false);
  readonly expandedHeads = signal<Record<string, boolean>>({});
  readonly headViews = signal<Record<string, KvHeadViewState>>({});
  readonly detailHeadPages = signal<Record<string, number>>({key: 0, value: 0});
  readonly tensorExpanded = signal<Record<string, boolean>>({
    key: true,
    value: true,
  });
  readonly sides = ['ref', 'target'] as const;
  readonly tokenTensors = computed(() =>
    ['key', 'value'].map((kind) => {
      const layer = this.context()?.layers.find(
        (row) => row.layer === this.selectedLayer()?.layer && row.kind === kind,
      );
      return {
        kind,
        label: kind === 'key' ? 'Key (K)' : 'Value (V)',
        layer,
        sources: layer?.token_structure ?? {ref: null, target: null},
        reason:
          this.selectionData()?.rows.find(
            (row) => row.kind === kind && row.layer === layer?.layer,
          )?.reason ?? layer?.reason,
        comparison: this.exactComparison(kind, this.selectedLayer()?.layer),
      };
    }),
  );
  readonly columnTensors = computed(() =>
    ['key', 'value'].map((kind) => ({
      kind,
      label: kind === 'key' ? 'Key (K)' : 'Value (V)',
      rows: [...new Set(this.context()?.layers.map((row) => row.layer) ?? [])]
        .sort((a, b) => a - b)
        .map((id) => {
          const row = this.context()?.layers.find(
            (row) => row.layer === id && row.kind === kind,
          );
          return {
            layer: id,
            status: row?.status ?? 'not_captured',
            comparison: this.exactComparison(kind, id),
          };
        }),
    })),
  );
  readonly snapshotLabel = computed(() => {
    const context = this.context();
    if (!context) return 'Not captured';
    const moments: Record<string, string> = {
      prefill_pre: 'Before Prefill',
      prefill_post: 'After Prefill',
      terminal: 'Generation end',
    };
    return `Turn ${context.turn} · ${moments[context.moment] ?? context.moment ?? 'Stored cache'}`;
  });
  readonly snapshotRows = computed(() =>
    this.sides.map((side) => ({
      side,
      label: side === 'ref' ? 'Ref' : 'Target',
      value:
        (this.context()?.snapshots?.[side] ?? [])
          .map((snapshot) => this.snapshotIdentity(snapshot))
          .join('; ') || 'Not captured',
    })),
  );
  private readonly bookmarks = new BookmarkStore(KV_BOOKMARK_CODEC);
  readonly saved = computed(() =>
    this.bookmarks.read(kvBookmarkKey(this.state.captureId())),
  );
  readonly bookmarkError = this.bookmarks.error;
  readonly savedHere = computed(() =>
    this.saved().filter((p) => p.context === this.context()?.id),
  );
  readonly bookmarkSelection = computed<SavedPoint | null>(() => {
    const context = this.context(),
      position = this.selectedPosition(),
      layer = this.selectedLayer();
    return !this.isRange() &&
      context &&
      position !== undefined &&
      (this.column() || layer)
      ? {
          context: context.id,
          position,
          layer: this.column() ? null : layer!.layer,
        }
      : null;
  });
  readonly isSaved = computed(() => {
    const selection = this.bookmarkSelection();
    return (
      !!selection &&
      this.saved().some(
        (point) => this.bookmarkKey(point) === this.bookmarkKey(selection),
      )
    );
  });
  readonly panelWidth = signal(350);
  readonly workspaceWidth = signal(window.innerWidth);
  readonly resultsOverlay = computed(() => this.workspaceWidth() < 755);
  readonly detailsOverlay = computed(() => this.workspaceWidth() <= 800);
  readonly panelMaxWidth = computed(() =>
    Math.max(
      100,
      Math.floor(this.workspaceWidth()) - (this.detailsOverlay() ? 40 : 390),
    ),
  );
  readonly panelMinWidth = computed(() => Math.min(220, this.panelMaxWidth()));
  readonly actualPanelWidth = computed(() =>
    this.clampPanelWidth(this.panelWidth()),
  );
  readonly canShowBothPanels = computed(
    () => this.workspaceWidth() - this.actualPanelWidth() - 365 >= 390,
  );
  readonly resizing = signal(false);
  readonly chartColor = (value: number | null) => this.color(value);
  readonly chartFormat = (value: number | null) => this.format(value);
  readonly rangeTensors = computed(() =>
    ['key', 'value'].map((kind) => ({
      kind,
      label: kind === 'key' ? 'Key (K)' : 'Value (V)',
      rows: (this.selectionData()?.rows ?? [])
        .filter((row) => row.kind === kind)
        .sort((a, b) => a.layer - b.layer),
    })),
  );
  constructor() {
    inject(DestroyRef).onDestroy(() => {
      this.saveView();
      this.data.destroy();
    });
    effect((onCleanup) => {
      const observer = new ResizeObserver(([entry]) =>
        this.workspaceWidth.set(entry.contentRect.width),
      );
      observer.observe(this.host.nativeElement);
      onCleanup(() => observer.disconnect());
    });
    effect(() => {
      if (this.showDetails() && this.showResults() && !this.canShowBothPanels())
        this.showResults.set(false);
    });
    effect((onCleanup) => {
      const identity = this.identity();
      this.state.revision();
      this.retry();
      untracked(() => {
        this.saveView();
        const saved = identity ? this.views.read(identity) : undefined;
        this.resolvedIdentity.set(null);
        this.popup.set(null);
        onCleanup(
          this.data.loadMetadata(identity, (analysis) => {
            if (!identity) return;
            const entry = this.state.kvEntry();
            const restored = restoreKvView(
              saved,
              analysis,
              identity,
              entry,
              window.matchMedia('(min-width: 801px)').matches,
            );
            this.applyView(restored.state);
            this.stateNotice.set(restored.notice);
            this.entryMissing.set(restored.entryMissing);
            this.resolvedIdentity.set(identity);
            // An entry is consumed once its capture resolves; one from another capture is stale.
            if (
              entry &&
              (entry.sessionId !== identity.captureId ||
                entry.turn === identity.turn)
            )
              this.state.kvEntry.set(null);
          }),
        );
      });
    });
    effect(() => {
      this.context()?.id;
      this.formula();
      this.resultPage.set(0);
      this.resultTotal.set(0);
      this.find.reset();
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        range = this.fullRange(),
        kind = this.kind(),
        head = this.head(),
        enabled = this.colorScale() === 'auto';
      this.rangeRetry();
      onCleanup(
        this.data.loadScale(
          identity && contextId && range && enabled
            ? {...identity, contextId, range, kind, head, bins: 1, formula: ''}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        range = this.visibleRange(),
        bins = this.binCapacity(),
        kind = this.kind(),
        head = this.head(),
        formula = this.formula();
      this.rangeRetry();
      onCleanup(
        this.data.loadRange(
          identity && contextId && range
            ? {...identity, contextId, range, bins, kind, head, formula}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        selection = this.inspectedRange();
      this.selectionRetry();
      onCleanup(
        this.data.loadSelection(
          identity && contextId && selection
            ? {...identity, contextId, selection}
            : null,
        ),
      );
    });
    effect((onCleanup) => {
      const identity = this.queryIdentity(),
        contextId = this.context()?.id,
        formula = this.formula(),
        page = this.resultPage();
      this.findRetry();
      onCleanup(
        this.data.loadFind(
          identity && contextId && formula
            ? {...identity, contextId, formula, page}
            : null,
          (data) => untracked(() => this.find.receive(data, page)),
        ),
      );
    });
  }
  private captureView(): KvViewSnapshot {
    return {
      version: 1,
      contextId: this.selectedContextId(),
      kind: this.kind(),
      metric: this.metric(),
      colorScale: this.colorScale(),
      head: this.head(),
      view: this.view(),
      formula: this.formula(),
      layer: this.layer(),
      position: this.position(),
      selectionEnd: this.selectionEnd(),
      column: this.column(),
      zoomRange: this.zoomRange(),
      zoomHistory: this.zoomHistory(),
      panelWidth: this.panelWidth(),
      showDetails: this.showDetails(),
      showResults: this.showResults(),
      infoExpanded: this.infoExpanded(),
      actionsExpanded: this.actionsExpanded(),
      savedExpanded: this.savedExpanded(),
      expandedHeads: this.expandedHeads(),
      headViews: this.headViews(),
      detailHeadPages: this.detailHeadPages(),
      tensorExpanded: this.tensorExpanded(),
    };
  }
  private saveView() {
    const identity = this.resolvedIdentity();
    if (identity && !this.entryMissing())
      this.views.save(identity, this.captureView());
  }
  private applyView(view: KvViewSnapshot) {
    this.selectedContextId.set(view.contextId);
    this.kind.set(view.kind);
    this.metric.set(view.metric);
    this.colorScale.set(view.colorScale);
    this.head.set(view.head);
    this.view.set(view.view);
    this.formula.set(view.formula);
    this.layer.set(view.layer);
    this.position.set(view.position);
    this.selectionEnd.set(view.selectionEnd);
    this.column.set(view.column);
    this.zoomRange.set(view.zoomRange);
    this.zoomHistory.set(view.zoomHistory);
    this.panelWidth.set(view.panelWidth);
    this.showDetails.set(view.showDetails);
    this.showResults.set(view.showResults);
    this.infoExpanded.set(view.infoExpanded);
    this.actionsExpanded.set(view.actionsExpanded);
    this.savedExpanded.set(view.savedExpanded);
    this.expandedHeads.set(view.expandedHeads);
    this.headViews.set(view.headViews);
    this.detailHeadPages.set(view.detailHeadPages);
    this.tensorExpanded.set(view.tensorExpanded);
  }
  snapshotIdentity(snapshot: KvSnapshotIdentity) {
    const identity =
      snapshot.snapshot_id !== null
        ? `Snapshot ${snapshot.snapshot_id}`
        : (snapshot.signature ?? snapshot.id);
    return (
      [
        snapshot.runtime,
        identity,
        snapshot.forward_id !== null ? `Forward ${snapshot.forward_id}` : null,
        snapshot.step !== null ? `Step ${snapshot.step}` : null,
        snapshot.edge,
      ]
        .filter(Boolean)
        .join(' · ') || 'Not captured'
    );
  }
  rowContainsPosition(row: KvAnalysisLayer, position: number) {
    return (
      row.cells.some((cell) => cell.position === position) ||
      [row.token_structure?.ref, row.token_structure?.target].some(
        (source) =>
          source?.status === 'ok' &&
          source.position_start !== null &&
          source.position_count !== null &&
          position >= source.position_start &&
          position < source.position_start + source.position_count,
      )
    );
  }
  structureAvailable(source: KvTokenStructure | null) {
    const position = this.selectedPosition();
    return (
      !!source &&
      source.status === 'ok' &&
      position !== undefined &&
      source.position_start !== null &&
      source.position_count !== null &&
      position >= source.position_start &&
      position < source.position_start + source.position_count
    );
  }
  structureShape(source: KvTokenStructure | null) {
    return this.structureAvailable(source) && source?.shape
      ? '[' + source.shape.join(', ') + ']'
      : 'Unavailable';
  }
  structureDtype(source: KvTokenStructure | null) {
    return this.structureAvailable(source)
      ? (source?.dtype ?? 'Not captured')
      : 'Unavailable';
  }
  structureStatus(source: KvTokenStructure | null) {
    if (!source) return 'Not captured';
    if (source.status !== 'ok')
      return source.reason ?? source.status.replaceAll('_', ' ');
    const position = this.selectedPosition();
    if (
      position === undefined ||
      source.position_start === null ||
      source.position_count === null ||
      position < source.position_start ||
      position >= source.position_start + source.position_count
    )
      return 'Token position not captured';
    return 'Captured';
  }
  structureHeadCount(kind: string) {
    const sources = this.tokenTensors().find(
      (tensor) => tensor.kind === kind,
    )?.sources;
    return sources
      ? Math.max(
          0,
          ...this.sides.map((side) =>
            this.structureAvailable(sources[side])
              ? (sources[side]?.head_count ?? 0)
              : 0,
          ),
        )
      : 0;
  }
  headPages(kind: string) {
    return Math.max(1, Math.ceil(this.structureHeadCount(kind) / 64));
  }
  headPage(kind: string) {
    return Math.min(
      this.detailHeadPages()[kind] ?? 0,
      this.headPages(kind) - 1,
    );
  }
  visibleHeads(kind: string) {
    const start = this.headPage(kind) * 64;
    return Array.from(
      {length: Math.min(64, this.structureHeadCount(kind) - start)},
      (_, i) => start + i,
    );
  }
  moveHeadPage(kind: string, delta: number) {
    this.detailHeadPages.update((pages) => ({
      ...pages,
      [kind]: Math.max(
        0,
        Math.min(this.headPages(kind) - 1, this.headPage(kind) + delta),
      ),
    }));
  }
  headOpen(kind: string, head: number) {
    return this.expandedHeads()[`${kind}:${head}`] ?? false;
  }
  toggleHead(kind: string, head: number) {
    this.expandedHeads.update((state) => ({
      ...state,
      [`${kind}:${head}`]: !this.headOpen(kind, head),
    }));
  }
  headView(kind: string, head: number) {
    return this.headViews()[`${kind}:${head}`] ?? DEFAULT_HEAD_VIEW;
  }
  setHeadView(kind: string, head: number, patch: Partial<KvHeadViewState>) {
    this.headViews.update((state) => ({
      ...state,
      [`${kind}:${head}`]: {...this.headView(kind, head), ...patch},
    }));
  }
  headSelection(kind: string, head: number): KvHeadSelection | null {
    const context = this.context(),
      layer = this.selectedLayer(),
      position = this.selectedPosition();
    return context && layer && position !== undefined
      ? {
          turn: this.selectedTurn(),
          contextId: context.id,
          layer: layer.layer,
          kind,
          position,
          head,
        }
      : null;
  }
  setTensorExpanded(kind: string, expanded: boolean) {
    this.tensorExpanded.update((state) => ({...state, [kind]: expanded}));
  }
  columnRows(kind: string) {
    return (
      this.columnTensors().find((tensor) => tensor.kind === kind)?.rows ?? []
    );
  }
  exactComparison(
    kind: string,
    layer: number | undefined,
  ): KvTokenMetrics | undefined {
    if (this.isRange()) return undefined;
    const row = this.selectionData()?.rows.find(
        (row) => row.kind === kind && row.layer === layer,
      ),
      position = this.selectedPosition();
    if (!row || position === undefined) return undefined;
    if (row.token_metrics?.[0]?.position === position)
      return row.token_metrics[0];
    return {
      position,
      status: row.status,
      metrics: {
        cosine_similarity: row.metrics.cosine_similarity.value,
        relative_l2: row.metrics.relative_l2.value,
        rmse: row.metrics.rmse.value,
        max_abs: row.metrics.max_abs.value,
      },
      metric_status: {
        cosine_similarity: row.metrics.cosine_similarity.status,
        relative_l2: row.metrics.relative_l2.status,
        rmse: row.metrics.rmse.status,
        max_abs: row.metrics.max_abs.status,
      },
    };
  }
  selectionLabel() {
    const range = this.inspectedRange();
    return range
      ? range.end - range.start === 1
        ? String(range.start)
        : `${range.start}–${range.end - 1}`
      : 'Not captured';
  }
  rangeMetricTitle(
    row: KvRangeDetailRow,
    key: (typeof this.detailMetrics)[number]['key'],
  ) {
    const metric = row.metrics[key],
      range = this.inspectedRange();
    return `${metric.valid_count} / ${range ? range.end - range.start : 0} valid positions · ${metric.status.replaceAll('_', ' ')}`;
  }
  setCapacity(capacity: number) {
    this.binCapacity.set(
      Math.max(1, Math.min(KV_RANGE_MAX_BINS, Math.floor(capacity))),
    );
  }
  zoomTo(range: KvRangeBin) {
    const current = this.visibleRange();
    if (!current || this.chartLoading() || this.chartError()) return;
    const next = {
      start: Math.max(current.start, range.start),
      end: Math.min(current.end, range.end),
    };
    if (
      next.end <= next.start ||
      (next.start === current.start && next.end === current.end)
    )
      return;
    this.zoomHistory.update((history) => [...history, current]);
    this.zoomRange.set(next);
  }
  undoZoom() {
    const history = this.zoomHistory();
    if (!history.length) return;
    this.zoomRange.set(history[history.length - 1]);
    this.zoomHistory.set(history.slice(0, -1));
  }
  selectRange(selection: KvRangeSelection) {
    if (this.chartLoading() || this.chartError()) return;
    this.find.reset();
    this.position.set(selection.start);
    this.selectionEnd.set(selection.end);
    this.column.set(selection.layer === null);
    if (selection.layer !== null) this.layer.set(selection.layer);
  }
  metricLabel(value = this.metric()) {
    return METRICS.find((m) => m.id === value)!.label;
  }
  headLabel(value = this.head()) {
    return value === 'max'
      ? 'Max over heads'
      : value === 'mean'
        ? 'Mean over heads'
        : `Head ${value}`;
  }
  format(value: number | null | undefined, metric = this.metric()) {
    return value == null
      ? 'Unavailable'
      : metric === 'relative_l2'
        ? formatSignificant(value * 100) + '%'
        : formatSignificant(value);
  }
  readonly scaleMax = computed(() =>
    (this.scaleReference()?.rows ?? []).reduce(
      (max, row) =>
        row.bins.reduce(
          (value, bin) => Math.max(value, bin.metrics[this.metric()].max ?? 0),
          max,
        ),
      0.000001,
    ),
  );
  paletteColor(percent: number) {
    return `color-mix(in srgb,var(--heat-high) ${Math.min(100, Math.max(0, percent))}%,var(--heat-low))`;
  }
  color(value: number | null) {
    if (value == null) return 'var(--kv-missing)';
    if (this.colorScale() === 'auto')
      return this.paletteColor((value / this.scaleMax()) * 100);
    const limits =
      this.metric() === 'relative_l2'
        ? [0.0001, 0.001, 0.01, 0.1]
        : [0.00001, 0.0001, 0.001, 0.01];
    return this.paletteColor(limits.filter((n) => value >= n).length * 25);
  }
  readonly scale = computed(() =>
    this.colorScale() === 'auto'
      ? this.scaleReference()
        ? [0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => {
            return formatSignificant(
              f * this.scaleMax() * (this.metric() === 'relative_l2' ? 100 : 1),
              3,
            );
          })
        : []
      : this.metric() === 'relative_l2'
        ? ['0', '0.01', '0.1', '1', '10', '+']
        : ['0', '1e−5', '1e−4', '1e−3', '0.01', '+'],
  );
  chooseLayer(layer: number) {
    this.find.reset();
    this.layer.set(layer);
  }
  selectPoint(layer: number, position: number | undefined) {
    if (position === undefined) return;
    this.find.reset();
    this.layer.set(layer);
    this.position.set(position);
    this.selectionEnd.set(null);
    this.column.set(false);
  }
  selectColumn(position: number) {
    this.find.reset();
    this.position.set(position);
    this.selectionEnd.set(null);
    this.column.set(true);
  }
  selectResult(result: KvFindResult, index: number) {
    this.find.select(result, index);
  }
  setResults(show: boolean) {
    if (show && !this.canShowBothPanels()) this.setDetails(false);
    this.showResults.set(show);
  }
  setDetails(show: boolean) {
    if (show && !this.canShowBothPanels()) this.showResults.set(false);
    this.showDetails.set(show);
  }
  moveResult(delta: number) {
    this.find.move(delta);
  }
  firstResult() {
    this.find.first();
  }
  openDisplay() {
    this.draftScale = this.colorScale();
    this.draftMetric = this.metric();
    this.draftKind = this.kind();
    this.draftHead = this.head();
    this.popup.set('display');
  }
  applyDisplay() {
    this.colorScale.set(this.draftScale);
    this.metric.set(this.draftMetric);
    this.kind.set(this.draftKind);
    this.head.set(this.draftHead);
    this.popup.set(null);
  }
  openFind() {
    this.find.open();
  }
  applyFind() {
    this.find.apply();
  }
  clearFind() {
    this.find.clear();
  }
  closePopup() {
    this.popup.set(null);
  }
  chooseTurn(value: number) {
    // Explicit navigation may change shared Turn; ordinary view restoration never does.
    this.state.selectTurn(value);
  }
  chooseContext(context: KvAnalysisContext) {
    const analysis = this.analysis(),
      identity = this.identity();
    if (!analysis || !identity || context.turn !== identity.turn) return;
    const restored = restoreKvView(
      {
        ...this.captureView(),
        contextId: context.id,
        position: kvContextRange(context)?.start ?? 0,
        column: false,
        selectionEnd: null,
        zoomRange: null,
        zoomHistory: [],
      },
      analysis,
      identity,
    );
    this.applyView(restored.state);
    this.stateNotice.set('');
    this.entryMissing.set(false);
    this.popup.set(null);
  }
  statusLabel(status: string) {
    return status === 'ok' ? 'Comparable' : status.replace(/_/g, ' ');
  }
  bookmarkKey(point: SavedPoint) {
    return KV_BOOKMARK_CODEC.identity(point);
  }
  readonly bookmarkLabel = kvBookmarkLabel;
  toggleBookmark() {
    const point = this.bookmarkSelection();
    if (!point) return;
    const next = this.isSaved()
      ? this.saved().filter(
          (p) => this.bookmarkKey(p) !== this.bookmarkKey(point),
        )
      : [...this.saved(), point];
    this.bookmarks.set(kvBookmarkKey(this.state.captureId()), next);
  }
  openSaved(point: SavedPoint) {
    const context = this.analysis()?.contexts.find(
      (context) =>
        context.id === point.context && context.turn === this.selectedTurn(),
    );
    if (
      !context ||
      !context.layers.some(
        (row) =>
          (point.layer === null || row.layer === point.layer) &&
          this.rowContainsPosition(row, point.position),
      )
    ) {
      this.stateNotice.set(
        'This bookmarked KV location is no longer available. The bookmark was kept.',
      );
      return;
    }
    this.chooseContext(context);
    const candidates = (this.context()?.layers ?? []).filter(
      (row) =>
        (point.layer === null || row.layer === point.layer) &&
        this.rowContainsPosition(row, point.position),
    );
    if (!candidates.some((row) => row.kind === this.kind()) && candidates[0])
      this.kind.set(candidates[0].kind);
    if (point.layer === null) this.selectColumn(point.position);
    else this.selectPoint(point.layer, point.position);
  }
  private clampPanelWidth(width: number) {
    return Math.max(
      this.panelMinWidth(),
      Math.min(this.panelMaxWidth(), Math.round(width)),
    );
  }
}
