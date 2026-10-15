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
import {ReportStateService} from '../../../data/report_state_service';
import {
  WorkspaceStateService,
  type GraphNavigationTarget,
} from '../../../app/workspace_state_service';
import {ArchitectureView} from '../architecture_view/architecture_view';
import {ExecutionGraph} from '../execution_graph/execution_graph';
import {GraphInspectionService} from '../graph_inspection_service';
import {GraphDetails} from '../graph_details/graph_details';
import {GraphViewStore} from './graph_view_store';
import {resolveGraphView, type GraphViewState, type GraphViewport} from './graph_view_state';
import {ResizeHandle} from '../../../shared/resize_handle/resize_handle';

@Component({
  selector: 'graph-workspace',
  standalone: true,
  imports: [
    ResizeHandle,
    FormsModule,
    MatIconModule,
    ArchitectureView,
    ExecutionGraph,
    GraphDetails,
  ],
  providers: [GraphInspectionService],
  templateUrl: './graph_workspace.ng.html',
  styleUrl: './graph_workspace.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphWorkspace {
  readonly state = inject(ReportStateService);
  private readonly workspace = inject(WorkspaceStateService);
  readonly inspection = inject(GraphInspectionService);
  private readonly views = inject(GraphViewStore);
  readonly restoredViewport = signal<GraphViewport | null>(null);
  viewport: GraphViewport | null = null;
  private viewScope: {capture: string; batch: number} | null = null;
  private lastContext: {layer: number; batch: number; semantic: string} | null = null;
  private viewRevision = 0;
  private contextualNavigation: GraphNavigationTarget | null = null;
  readonly navigationMessage = signal('');
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly injector = inject(Injector);
  readonly architecture = viewChild(ArchitectureView);
  readonly execution = viewChild(ExecutionGraph);
  readonly executionOpen = signal(false);
  readonly floating = signal(false);
  readonly detailsOpen = signal(window.innerWidth > 800);
  readonly detailsWidth = signal(350);
  private readonly hostWidth = signal(window.innerWidth);
  readonly maximumDetailsWidth = computed(() =>
    Math.max(300, Math.min(560, this.hostWidth() - 340)),
  );
  readonly searchOpen = signal(false);
  readonly executionSearch = signal('');
  private resizeObserver?: ResizeObserver;
  private destroyed = false;
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
          definition.anchors.find((a) => 'anchor:' + a.id === context.semantic)?.label ??
          definition.anchors.find((a) => 'anchor:' + a.id === context.semantic)?.semantic ??
          context.semantic)
      : 'Architecture';
  });
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
        tensor.run === runs[0]?.id ? 'ref' : tensor.run === runs[1]?.id ? 'target' : null;
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
              ![node.id, node.label, node.namespace].join(' ').toLowerCase().includes(query)
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
    const target = this.workspace.graphTarget();
    if (target) this.workspace.graphTarget.set(null);
    if (target?.capture === capture && target.batch === batch) {
      this.contextualNavigation = target;
      this.selectNavigationLayer(target);
    }
    this.restoreView();
    if (this.contextualNavigation) this.restoreNavigation(this.contextualNavigation);
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
          this.navigationMessage.set('');
          this.saveView();
          this.viewScope = {capture: currentCapture!, batch: currentBatch};
          this.lastContext = null;
          this.executionOpen.set(false);
          this.restoreView(true);
          return;
        }
        if (context?.batch === currentBatch && context.layer === this.state.layer())
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
    afterNextRender(() => {
      const main = this.host.nativeElement.querySelector('.analysis-main');
      this.resizeObserver = new ResizeObserver(() => {
        this.hostWidth.set(this.host.nativeElement.clientWidth);
        if (window.innerWidth > 800)
          this.detailsWidth.update((value) => Math.min(value, this.maximumDetailsWidth()));
      });
      if (main) this.resizeObserver.observe(main);
    });
    inject(DestroyRef).onDestroy(() => {
      this.saveView();
      this.destroyed = true;
      this.viewRevision++;
      this.resizeObserver?.disconnect();
    });
  }
  private saveView() {
    if (this.viewScope)
      this.views.save(this.viewScope.capture, this.viewScope.batch, this.captureView());
  }
  private restoreView(resetViewport = false) {
    const scope = this.viewScope,
      model = this.state.semantic();
    if (!scope || !model) return;
    const revision = ++this.viewRevision,
      layer = this.state.layer();
    const navigation = this.contextualNavigation;
    const explicit = navigation?.capture === scope.capture && navigation.batch === scope.batch;
    const saved = resolveGraphView(
      this.views.read(scope.capture, scope.batch),
      model,
      layer,
      this.state.metrics,
    );
    if (saved && !explicit) {
      this.state.metric.set(saved.metric);
      this.detailsOpen.set(saved.details);
      this.detailsWidth.set(saved.detailsWidth);
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
  private selectNavigationLayer(target: GraphNavigationTarget) {
    const layers = this.state.semantic()?.layers ?? [];
    if (Number.isInteger(target.layer) && target.layer >= 0 && target.layer < layers.length)
      this.state.selectLayer(target.layer);
  }
  /** Explicit entry wins over cached graph selection, without inventing a tensor mapping. */
  private restoreNavigation(target: GraphNavigationTarget) {
    afterNextRender(
      () => {
        if (
          this.destroyed ||
          this.contextualNavigation !== target ||
          target.capture !== this.state.captureId() ||
          target.batch !== this.state.batchId()
        )
          return;
        this.contextualNavigation = null;
        const model = this.state.semantic(),
          layer = model?.layers[target.layer],
          definition = layer ? model?.semantic_graph[layer.def] : undefined;
        if (definition && target.layer !== this.state.layer()) return;
        this.inspection.close();
        this.lastContext = null;
        this.executionOpen.set(false);
        this.navigationMessage.set('');
        if (!definition) {
          this.navigationMessage.set(`Layer ${target.layer} is not available in this layer graph.`);
        } else if (target.anchor) {
          if (definition.anchors.some((anchor) => anchor.id === target.anchor)) {
            const context = {
              layer: target.layer,
              batch: target.batch,
              semantic: 'anchor:' + target.anchor,
            };
            this.lastContext = context;
            this.inspection.select(context);
            this.architecture()?.control('focus', context.semantic);
          } else {
            this.navigationMessage.set(
              `Sync point ${target.anchor} is not mapped in this layer graph.`,
            );
          }
        }
      },
      {injector: this.injector},
    );
  }
  private captureView(): GraphViewState {
    const context = this.lastContext?.batch === this.viewScope?.batch ? this.lastContext : null;
    return {
      version: 1,
      metric: this.state.metric(),
      details: this.detailsOpen(),
      detailsWidth: this.detailsWidth(),
      context: context ? {layer: context.layer, semantic: context.semantic} : null,
      execution: this.executionOpen(),
      viewport: this.viewport,
    };
  }
  selectLayer(layer: number) {
    if (this.inspection.mapping()) return;
    const previous = this.state.layer();
    this.state.selectLayer(layer);
    if (this.state.layer() !== previous) this.navigationMessage.set('');
  }
  inspect(semantic: string) {
    if (this.inspection.mapping()) return;
    this.navigationMessage.set('');
    this.inspection.select({layer: this.state.layer(), batch: this.state.batchId(), semantic});
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
  chooseSearch(result: ReturnType<GraphWorkspace['searchResults']>[number]) {
    if (result.kind === 'tensor') this.inspection.selectTensor({side: result.side, id: result.id});
    else
      this.inspection.selectOperation({
        side: result.side,
        nodeId: result.id,
        graphId: result.graphId,
      });
    this.searchOpen.set(false);
  }
  toggleDetails() {
    this.detailsOpen.update((value) => !value);
  }
  @HostListener('document:keydown.escape') escape() {
    if (this.searchOpen()) this.searchOpen.set(false);
    else if (this.inspection.mapping()) this.inspection.cancelMapping();
    else if (this.floating()) this.floating.set(false);
  }
}
