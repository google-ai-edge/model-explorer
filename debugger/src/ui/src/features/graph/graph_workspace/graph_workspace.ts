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

import {TitleCasePipe} from '@angular/common';
import {
  afterNextRender,
  ChangeDetectionStrategy,
  Component,
  computed,
  DestroyRef,
  effect,
  ElementRef,
  HostListener,
  inject,
  Injector,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatIconModule} from '@angular/material/icon';
import type {ComparisonRow} from '../../../data/contracts/types';
import {ReportStateService} from '../../../data/report_state_service';
import {
  BookmarkCodec,
  BookmarkStore,
} from '../../../shared/bookmark_store/bookmark_store';
import {FindNavigator} from '../../../shared/find_navigator/find_navigator';
import {
  loadTokenPlotly,
  PlotHost,
  PlotlyApi,
} from '../../../shared/plotly_loader';
import {ResizeHandle} from '../../../shared/resize_handle/resize_handle';
import {ArchitectureView} from '../architecture_view/architecture_view';
import {ExecutionGraph} from '../execution_graph/execution_graph';
import {ExplicitPairs} from '../explicit_pairs/explicit_pairs';
import {GraphDetails} from '../graph_details/graph_details';
import {GraphInspectionService} from '../graph_inspection_service';
import {
  emptyGraphQuery,
  graphAnchor,
  graphMetricValue,
  GraphQuery,
  graphQueryCount,
  matchesGraphQuery,
} from './graph_query';
import {
  resolveGraphView,
  type GraphViewport,
  type GraphViewState,
} from './graph_view_state';
import {GraphViewStore} from './graph_view_store';

interface Bookmark {
  capture: string;
  batch: number;
  layer: number;
  semantic: string;
  metric: string;
  execution: boolean;
}
const GRAPH_BOOKMARKS_KEY = 'debugger.graph.bookmarks';
const GRAPH_BOOKMARK_CODEC: BookmarkCodec<Bookmark> = {
  parse(value) {
    const x = value as Partial<Bookmark> | null;
    return x &&
      typeof x.capture === 'string' &&
      Number.isInteger(x.batch) &&
      Number.isInteger(x.layer) &&
      typeof x.semantic === 'string'
      ? (value as Bookmark)
      : null;
  },
  identity: (value) => JSON.stringify(value),
};
type Panel = 'display' | 'find' | 'observation' | 'bookmarks' | null;

@Component({
  selector: 'graph-workspace',
  standalone: true,
  imports: [
    FindNavigator,
    ResizeHandle,
    FormsModule,
    TitleCasePipe,
    MatIconModule,
    ArchitectureView,
    ExecutionGraph,
    GraphDetails,
    ExplicitPairs,
  ],
  providers: [GraphInspectionService],
  templateUrl: './graph_workspace.ng.html',
  styleUrl: './graph_workspace.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphWorkspace {
  readonly state = inject(ReportStateService);
  readonly inspection = inject(GraphInspectionService);
  private readonly views = inject(GraphViewStore);
  readonly restoredViewport = signal<GraphViewport | null>(null);
  viewport: GraphViewport | null = null;
  private viewScope: {capture: string; batch: number} | null = null;
  private lastContext: {layer: number; batch: number; semantic: string} | null =
    null;
  private viewRevision = 0;
  private bookmarkNavigation: Bookmark | null = null;
  private savedNavigationScope: {capture: string; batch: number} | null = null;
  private readonly bookmarkStore = new BookmarkStore(GRAPH_BOOKMARK_CODEC);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly injector = inject(Injector);
  readonly architecture = viewChild(ArchitectureView);
  readonly execution = viewChild(ExecutionGraph);
  readonly plot = viewChild<ElementRef<PlotHost>>('plot');
  readonly explicitDialog =
    viewChild<ElementRef<HTMLDialogElement>>('explicitDialog');
  readonly panel = signal<Panel>(null);
  readonly executionOpen = signal(false);
  readonly floating = signal(false);
  readonly detailsOpen = signal(window.innerWidth > 800);
  readonly resultsOpen = signal(false);
  readonly trendsOpen = signal(true);
  readonly trendHeight = signal(120);
  readonly detailsWidth = signal(350);
  readonly dockHeight = signal(66);
  private readonly hostWidth = signal(window.innerWidth);
  private readonly hostHeight = signal(window.innerHeight);
  readonly maximumDetailsWidth = computed(() =>
    Math.max(300, Math.min(560, this.hostWidth() - 340)),
  );
  readonly maximumTrendHeight = computed(() =>
    Math.max(80, Math.min(320, this.hostHeight() - this.dockHeight() - 230)),
  );
  readonly searchOpen = signal(false);
  readonly executionSearch = signal('');
  readonly plotError = signal('');
  readonly query = signal<GraphQuery>(emptyGraphQuery());
  draftQuery = emptyGraphQuery();
  draftDisplay = {metric: 'CosSim', trends: true, details: true};
  draftBatch = 0;
  readonly bookmarks = computed(() =>
    this.bookmarkStore.read(GRAPH_BOOKMARKS_KEY),
  );
  private panelTrigger?: HTMLElement;
  private resizeObserver?: ResizeObserver;
  private plotly?: PlotlyApi;
  private plotRevision = 0;
  private destroyed = false;
  private boundPlot?: PlotHost;
  readonly rows = computed(() =>
    [...(this.state.comparison()?.rows ?? [])].sort(
      (a, b) => a.layer - b.layer || a.anchor.localeCompare(b.anchor),
    ),
  );
  readonly results = computed(() =>
    this.rows().filter((row) =>
      matchesGraphQuery(row, this.state.semantic(), this.query()),
    ),
  );
  readonly filterCount = computed(() => graphQueryCount(this.query()));
  readonly outputValues = computed(() =>
    this.state.semantic()!.layers.map((_, layer) =>
      graphMetricValue(
        this.rows().find(
          (row) => row.layer === layer && row.anchor === 'output',
        ),
        this.state.metric(),
      ),
    ),
  );
  readonly hasOutputMetrics = computed(() =>
    this.outputValues().some((value) => value != null),
  );
  readonly selectedRow = computed(() => {
    const context = this.inspection.context();
    return context
      ? this.rows().find(
          (row) =>
            row.layer === context.layer &&
            'anchor:' + row.anchor === context.semantic,
        )
      : undefined;
  });
  readonly resultIndex = computed(() =>
    this.results().findIndex((row) => row === this.selectedRow()),
  );
  readonly selection = computed(() => {
    const context = this.inspection.context();
    if (!context || context.layer !== this.state.layer()) return null;
    return context.semantic.startsWith('anchor:')
      ? {kind: 'tensor' as const, id: context.semantic.slice(7)}
      : {kind: 'architecture' as const, id: context.semantic};
  });
  readonly title = computed(() => {
    const context = this.inspection.context();
    const definition = this.state.definition();
    return context
      ? (definition.nodes.find((n) => n.id === context.semantic)?.label ??
          definition.anchors.find((a) => 'anchor:' + a.id === context.semantic)
            ?.label ??
          definition.anchors.find((a) => 'anchor:' + a.id === context.semantic)
            ?.semantic ??
          context.semantic)
      : 'Architecture';
  });
  readonly currentBookmarks = computed(() =>
    this.bookmarks().filter((b) => b.capture === this.state.captureId()),
  );
  readonly searchResults = computed(() => {
    const details = this.inspection.details();
    const query = this.executionSearch().trim().toLowerCase();
    if (!details) return [];
    const runs = this.state.preview.runs;
    const results: {
      key: string;
      side: 'ref' | 'target';
      id: string;
      graphId: string;
      label: string;
      kind: 'tensor' | 'operation';
    }[] = [];
    for (const tensor of details.tensors) {
      if (
        query &&
        ![tensor.id, tensor.node, tensor.output, tensor.graph]
          .join(' ')
          .toLowerCase()
          .includes(query)
      )
        continue;
      const side =
        tensor.run === runs[0]?.id
          ? 'ref'
          : tensor.run === runs[1]?.id
            ? 'target'
            : null;
      if (side)
        results.push({
          key: 'tensor:' + tensor.id,
          side,
          id: tensor.id,
          graphId: tensor.graph,
          label: tensor.id,
          kind: 'tensor',
        });
    }
    if (!this.inspection.mapping())
      for (const [index, execution] of details.executions.entries()) {
        const side =
          execution.id === runs[0]?.id
            ? 'ref'
            : execution.id === runs[1]?.id
              ? 'target'
              : index === 0
                ? 'ref'
                : index === 1
                  ? 'target'
                  : null;
        if (!side) continue;
        for (const graph of execution.graphs)
          for (const node of graph.nodes) {
            if (
              query &&
              ![node.id, node.label, node.namespace]
                .join(' ')
                .toLowerCase()
                .includes(query)
            )
              continue;
            results.push({
              key: side + ':' + graph.id + ':' + node.id,
              side,
              id: node.id,
              graphId: graph.id,
              label: node.label + ' · ' + node.id,
              kind: 'operation',
            });
          }
      }
    return results.slice(0, 100);
  });

  constructor() {
    const capture = this.state.captureId(),
      batch = this.state.batchId();
    this.viewScope = capture ? {capture, batch} : null;
    this.restoreView();
    effect(() => {
      const currentCapture = this.state.captureId(),
        currentBatch = this.state.batchId(),
        context = this.inspection.context();
      untracked(() => {
        if (!this.viewScope) return;
        if (currentCapture !== capture) {
          // The old component may run before ReportPage removes it. Persist its
          // original identity once, before later reset effects clear its UI.
          this.saveView();
          this.viewScope = null;
          this.viewRevision++;
          return;
        }
        if (currentBatch !== this.viewScope.batch) {
          const saved = this.savedNavigationScope;
          if (
            saved?.capture !== capture ||
            saved.batch !== this.viewScope.batch
          )
            this.saveView();
          this.savedNavigationScope = null;
          this.viewScope = {capture: currentCapture!, batch: currentBatch};
          this.lastContext = null;
          this.executionOpen.set(false);
          this.restoreView(true);
          return;
        }
        if (
          context?.batch === currentBatch &&
          context.layer === this.state.layer()
        )
          this.lastContext = {...context};
      });
    });
    effect(() => {
      this.state.captureId();
      this.state.batchId();
      this.state.layer();
      untracked(() => {
        this.executionOpen.set(false);
        this.floating.set(false);
        this.searchOpen.set(false);
        this.executionSearch.set('');
      });
    });
    effect(() => {
      this.plot();
      this.state.comparison();
      this.state.layer();
      this.state.metric();
      this.state.dark();
      this.trendsOpen();
      this.trendHeight();
      untracked(() => void this.drawTrends());
    });
    afterNextRender(() => {
      const main = this.host.nativeElement.querySelector('.analysis-main');
      const dock = this.host.nativeElement.querySelector('.dock-area');
      this.resizeObserver = new ResizeObserver(() => {
        this.hostWidth.set(this.host.nativeElement.clientWidth);
        this.hostHeight.set(this.host.nativeElement.clientHeight);
        if (dock)
          this.dockHeight.set(
            Math.ceil(dock.getBoundingClientRect().height) + 4,
          );
        this.ensureNarrowPanels();
        if (window.innerWidth > 800)
          this.detailsWidth.update((value) =>
            Math.min(value, this.maximumDetailsWidth()),
          );
        this.trendHeight.update((value) =>
          Math.min(value, this.maximumTrendHeight()),
        );
        void this.drawTrends();
      });
      if (main) this.resizeObserver.observe(main);
      if (dock) this.resizeObserver.observe(dock);
    });
    inject(DestroyRef).onDestroy(() => {
      this.saveView();
      this.destroyed = true;
      this.viewRevision++;
      this.plotRevision++;
      this.resizeObserver?.disconnect();
      if (this.boundPlot) this.plotly?.purge(this.boundPlot);
    });
  }
  private ensureNarrowPanels() {
    if (window.innerWidth <= 800 && this.detailsOpen() && this.resultsOpen())
      this.resultsOpen.set(false);
  }
  private saveView() {
    if (this.viewScope)
      this.views.save(
        this.viewScope.capture,
        this.viewScope.batch,
        this.captureView(),
      );
  }
  private restoreView(resetViewport = false) {
    const scope = this.viewScope,
      model = this.state.semantic();
    if (!scope || !model) return;
    const revision = ++this.viewRevision,
      layer = this.state.layer(),
      bookmark = this.bookmarkNavigation;
    const explicit =
      bookmark?.capture === scope.capture && bookmark.batch === scope.batch;
    const saved = resolveGraphView(
      this.views.read(scope.capture, scope.batch),
      model,
      layer,
      this.state.metrics,
    );
    if (saved && !explicit) {
      this.query.set(saved.query);
      this.state.metric.set(saved.metric);
      this.detailsOpen.set(saved.details);
      this.resultsOpen.set(
        saved.results && (window.innerWidth > 800 || !saved.details),
      );
      this.trendsOpen.set(saved.trends);
      this.detailsWidth.set(saved.detailsWidth);
      this.trendHeight.set(saved.trendHeight);
    }
    const viewport = explicit ? null : (saved?.viewport ?? null);
    this.viewport = viewport;
    this.restoredViewport.set(
      viewport ?? (resetViewport ? {layer, zoom: 1, left: 0, top: 0} : null),
    );
    if (!saved?.context || explicit) return;
    const context = {...saved.context, batch: scope.batch};
    afterNextRender(
      () => {
        if (
          this.destroyed ||
          revision !== this.viewRevision ||
          scope.capture !== this.state.captureId() ||
          scope.batch !== this.state.batchId() ||
          layer !== this.state.layer()
        )
          return;
        this.lastContext = context;
        this.inspection.select(context);
        this.executionOpen.set(saved.execution);
      },
      {injector: this.injector},
    );
  }
  private captureView(): GraphViewState {
    const context =
      this.lastContext?.batch === this.viewScope?.batch
        ? this.lastContext
        : null;
    return {
      version: 1,
      query: {...this.query()},
      metric: this.state.metric(),
      details: this.detailsOpen(),
      results: this.resultsOpen(),
      trends: this.trendsOpen(),
      detailsWidth: this.detailsWidth(),
      trendHeight: this.trendHeight(),
      context: context
        ? {layer: context.layer, semantic: context.semantic}
        : null,
      execution: this.executionOpen(),
      viewport: this.viewport,
    };
  }
  openPanel(panel: Exclude<Panel, null>, event: Event) {
    if (this.panel() === panel) {
      this.closePanel();
      return;
    }
    this.panelTrigger = event.currentTarget as HTMLElement;
    this.draftQuery = {...this.query()};
    this.draftDisplay = {
      metric: this.state.metric(),
      trends: this.trendsOpen(),
      details: this.detailsOpen(),
    };
    this.draftBatch = this.state.batchId();
    this.panel.set(panel);
    queueMicrotask(() =>
      this.host.nativeElement
        .querySelector<HTMLElement>('.popup select,.popup input,.popup button')
        ?.focus(),
    );
  }
  closePanel() {
    this.panel.set(null);
    this.panelTrigger?.focus();
  }
  applyPanel() {
    if (this.panel() === 'display') {
      this.state.metric.set(this.draftDisplay.metric);
      this.trendsOpen.set(this.draftDisplay.trends);
      this.detailsOpen.set(this.draftDisplay.details);
      this.ensureNarrowPanels();
    } else if (this.panel() === 'find') {
      this.query.set({...this.draftQuery});
      this.resultsOpen.set(true);
      if (window.innerWidth <= 800) this.detailsOpen.set(false);
    } else if (this.panel() === 'observation')
      this.state.selectBatch(this.draftBatch);
    this.closePanel();
  }
  invalidQuery() {
    return (
      this.draftQuery.threshold.trim() !== '' &&
      !Number.isFinite(Number(this.draftQuery.threshold))
    );
  }
  clearFilters() {
    this.query.set(emptyGraphQuery());
  }
  rowName(row: ComparisonRow) {
    const anchor = graphAnchor(this.state.semantic(), row);
    return anchor?.label || anchor?.semantic || row.anchor;
  }
  selectLayer(layer: number) {
    if (!this.inspection.mapping()) this.state.selectLayer(layer);
  }
  inspect(semantic: string) {
    if (this.inspection.mapping()) return;
    this.inspection.select({
      layer: this.state.layer(),
      batch: this.state.batchId(),
      semantic,
    });
  }
  openExecution(semantic: string) {
    this.inspect(semantic);
    this.executionOpen.set(true);
    this.searchOpen.set(false);
  }
  backToArchitecture() {
    if (this.inspection.mapping()) this.inspection.cancelMapping();
    this.executionOpen.set(false);
    this.floating.set(false);
    this.searchOpen.set(false);
  }
  selectRow(row: ComparisonRow) {
    if (this.inspection.mapping()) return;
    const capture = this.state.captureId(),
      batch = this.state.batchId();
    this.state.selectLayer(row.layer);
    afterNextRender(
      () => {
        if (
          this.destroyed ||
          capture !== this.state.captureId() ||
          batch !== this.state.batchId() ||
          row.layer !== this.state.layer()
        )
          return;
        this.inspection.select({
          layer: row.layer,
          batch: this.state.batchId(),
          semantic: 'anchor:' + row.anchor,
        });
        this.executionOpen.set(row.anchor !== 'output');
        if (row.anchor === 'output')
          this.architecture()?.control('focus', 'anchor:' + row.anchor);
      },
      {injector: this.injector},
    );
    if (window.innerWidth <= 800) this.resultsOpen.set(false);
  }
  locate(direction: number) {
    const rows = this.results(),
      index = this.resultIndex();
    const row =
      rows[
        direction === 0
          ? 0
          : Math.max(
              0,
              Math.min(rows.length - 1, index < 0 ? 0 : index + direction),
            )
      ];
    if (row) this.selectRow(row);
  }
  chooseSearch(result: ReturnType<GraphWorkspace['searchResults']>[number]) {
    if (result.kind === 'tensor')
      this.inspection.selectTensor({side: result.side, id: result.id});
    else
      this.inspection.selectOperation({
        side: result.side,
        nodeId: result.id,
        graphId: result.graphId,
      });
    this.searchOpen.set(false);
  }
  toggleResults() {
    this.resultsOpen.update((value) => !value);
    if (window.innerWidth <= 800 && this.resultsOpen())
      this.detailsOpen.set(false);
  }
  toggleDetails() {
    this.detailsOpen.update((value) => !value);
    if (window.innerWidth <= 800 && this.detailsOpen())
      this.resultsOpen.set(false);
  }
  @HostListener('document:keydown.escape') escape() {
    // The native modal dialog owns Escape while it is open.
    if (this.explicitDialog()?.nativeElement.open) return;
    if (this.panel()) this.closePanel();
    else if (this.searchOpen()) this.searchOpen.set(false);
    else if (this.inspection.mapping()) this.inspection.cancelMapping();
    else if (this.floating()) this.floating.set(false);
  }
  @HostListener('document:pointerdown', ['$event']) outside(
    event: PointerEvent,
  ) {
    if (
      this.panel() &&
      !this.host.nativeElement
        .querySelector('.dock-area')
        ?.contains(event.target as Node)
    )
      this.closePanel();
  }
  addBookmark() {
    const capture = this.state.captureId(),
      context = this.inspection.context();
    if (!capture) return;
    const bookmark = {
      capture,
      batch: this.state.batchId(),
      layer: this.state.layer(),
      semantic: context?.semantic ?? '',
      metric: this.state.metric(),
      execution: this.executionOpen(),
    };
    const key = GRAPH_BOOKMARK_CODEC.identity;
    this.bookmarkStore.set(GRAPH_BOOKMARKS_KEY, [
      ...this.bookmarks().filter((item) => key(item) !== key(bookmark)),
      bookmark,
    ]);
  }
  removeBookmark(bookmark: Bookmark) {
    const key = GRAPH_BOOKMARK_CODEC.identity;
    this.bookmarkStore.set(
      GRAPH_BOOKMARKS_KEY,
      this.bookmarks().filter((item) => key(item) !== key(bookmark)),
    );
  }
  restoreBookmark(bookmark: Bookmark) {
    if (this.inspection.mapping()) return;
    // Save before the bookmark changes the shared batch, layer and metric.
    this.saveView();
    this.savedNavigationScope = this.viewScope;
    this.bookmarkNavigation = bookmark;
    this.viewRevision++;
    this.state.selectBatch(bookmark.batch);
    this.state.selectLayer(bookmark.layer);
    if (this.state.metrics.includes(bookmark.metric))
      this.state.metric.set(bookmark.metric);
    this.closePanel();
    afterNextRender(
      () => {
        this.savedNavigationScope = null;
        if (this.bookmarkNavigation !== bookmark) return;
        this.bookmarkNavigation = null;
        if (
          this.destroyed ||
          bookmark.capture !== this.state.captureId() ||
          bookmark.batch !== this.state.batchId() ||
          bookmark.layer !== this.state.layer()
        )
          return;
        if (this.state.metrics.includes(bookmark.metric))
          this.state.metric.set(bookmark.metric);
        if (bookmark.semantic)
          this.inspection.select({
            layer: this.state.layer(),
            batch: this.state.batchId(),
            semantic: bookmark.semantic,
          });
        this.executionOpen.set(bookmark.execution && !!bookmark.semantic);
      },
      {injector: this.injector},
    );
  }
  readonly explicitOpen = signal(false);
  showExplicit() {
    this.explicitOpen.set(true);
    this.explicitDialog()?.nativeElement.showModal();
  }
  async drawTrends() {
    const host = this.plot()?.nativeElement;
    if (!host || !this.trendsOpen() || !host.clientWidth || this.destroyed)
      return;
    const revision = ++this.plotRevision;
    try {
      const plotly = await loadTokenPlotly();
      if (this.destroyed || revision !== this.plotRevision) return;
      this.plotly = plotly;
      const layers = this.state.semantic()!.layers.map((_, index) => index),
        metric = this.state.metric();
      const values = this.outputValues();
      const style = getComputedStyle(this.host.nativeElement),
        color = (key: string) => style.getPropertyValue(key).trim();
      await plotly.react(
        host,
        [
          {
            type: 'scatter',
            mode: 'lines+markers',
            x: layers,
            y: values,
            connectgaps: false,
            line: {color: color('--graph-accent'), width: 1.5},
            marker: {
              size: layers.map((layer) =>
                layer === this.state.layer() ? 10 : 5,
              ),
              color: color('--graph-accent'),
            },
            hovertemplate:
              'Layer %{x}<br>' + metric + ': %{y:.5g}<extra></extra>',
          },
        ],
        {
          height: this.trendHeight(),
          margin: {l: 48, r: 24, t: 12, b: 28},
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          showlegend: false,
          font: {
            family: style.fontFamily,
            size: 11,
            color: color('--graph-muted'),
          },
          xaxis: {
            tickmode: 'array',
            tickvals: layers,
            ticktext: layers.map((layer) => 'L' + layer),
            fixedrange: true,
            showgrid: false,
            zeroline: false,
          },
          yaxis: {
            title: {text: metric, font: {size: 11}},
            fixedrange: true,
            gridcolor: color('--graph-line'),
            zeroline: false,
          },
        },
        {displayModeBar: false, responsive: true},
      );
      if (this.destroyed) return;
      if (this.boundPlot !== host) {
        this.boundPlot = host;
        host.on('plotly_click', (event) => {
          const point = (event['points'] as {x: number}[] | undefined)?.[0];
          if (point) this.selectLayer(point.x);
        });
      }
      this.plotError.set('');
    } catch (error) {
      if (!this.destroyed && revision === this.plotRevision)
        this.plotError.set(String(error));
    }
  }
}
