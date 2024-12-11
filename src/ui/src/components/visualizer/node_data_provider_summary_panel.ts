/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
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

import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  DestroyRef,
  effect,
  Input,
  OnChanges,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTooltipModule} from '@angular/material/tooltip';
import {debounceTime} from 'rxjs/operators';
import {AppService} from './app_service';
import {NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX} from './common/consts';
import {GroupNode, ModelGraph, OpNode} from './common/model_graph';
import {
  AggregatedStat,
  NodeDataProviderRunData,
  NodeDataProviderValueInfo,
} from './common/types';
import {
  genSortedValueInfos,
  getRunName,
  isGroupNode,
  isOpNode,
} from './common/utils';
import {InfoPanelService, SortingDirection} from './info_panel_service';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {Paginator} from './paginator';

interface Row {
  // Node id.
  id: string;
  // Node label.
  label: string;
  index: number;
  cols: Col[];
  isInput?: boolean;
  isOutput?: boolean;
}

interface ChildrenStatRow {
  // Node id.
  id: string;
  // Node label.
  label: string;
  index: number;
  colValues: number[];
  colStrs: string[];
  colHidden: boolean[];
}

interface StatRow {
  stat: string;
  values: number[];
}

interface Stat {
  min: number;
  max: number;
  sum: number;
  count: number;
}

interface Col {
  // tslint:disable-next-line:no-any Allow arbitrary types.
  value: any;
  strValue: string;
  bgColor: string;
  textColor: string;
}

interface RunItem {
  runId: string;
  runName: string;
  done: boolean;
  error?: string;
  hideInAggregatedStatsTable?: boolean;
}

interface ChildrenStatsCol {
  colIndex: number;
  runIndex: number;
  label: string;
  hideInChildrenStatsTable?: boolean;
  multiLineHeader?: boolean;
}

const CHILDREN_STATS = ['Sum %'];

/** The panel to show node data provider summary for certain layouer. */
@Component({
  standalone: true,
  selector: 'node-data-provider-summary-panel',
  imports: [
    CommonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
    Paginator,
    ReactiveFormsModule,
  ],
  templateUrl: 'node_data_provider_summary_panel.ng.html',
  styleUrls: ['./node_data_provider_summary_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NodeDataProviderSummaryPanel implements OnChanges {
  @Input({required: true}) paneId!: string;
  @Input('rootGroupNodeId') rootGroupNodeId?: string;
  @ViewChild('paginator') paginator?: Paginator;
  @ViewChild('childrenStatsPaginator') childrenStatsPaginator?: Paginator;

  readonly childrenStatsTableNodeFilter = new FormControl<string>('');
  readonly resultsTableNodeFilter = new FormControl<string>('');

  curRows?: Row[];
  curPageRows: Row[] = [];
  savedCurRows?: Row[];
  curStatRows: StatRow[] = [];
  curChildrenStatRows: ChildrenStatRow[] = [];
  curPageChildrenStatRows: ChildrenStatRow[] = [];
  savedChildrenStatRows: ChildrenStatRow[] = [];
  runItems: RunItem[] = [];
  curSelectedRunId = '';
  orderedNodes: OpNode[] = [];
  childrenStatsCols: ChildrenStatsCol[] = [];
  tablePageSize = 50;

  private curModelGraph?: ModelGraph;
  private prevModelGraph?: ModelGraph;
  private prevRunsKey = '';
  // "Model graph collection + id + root group node id" to ordered nodes.
  private readonly orderedNodesCache: Record<string, OpNode[]> = {};

  constructor(
    private readonly appService: AppService,
    private readonly destroyRef: DestroyRef,
    private readonly infoPanelService: InfoPanelService,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
    private readonly changeDetectorRef: ChangeDetectorRef,
  ) {
    // For testing.
    const params = new URLSearchParams(document.location.search);
    if (params.has('nodeDataProviderDataSummaryTablePageSize')) {
      this.tablePageSize = Number(
        params.get('nodeDataProviderDataSummaryTablePageSize'),
      );
    }

    // Update the currently selected run id.
    effect(() => {
      const modelGraph = this.appService.getPaneById(this.paneId)?.modelGraph;
      if (!modelGraph) {
        return;
      }
      const selectedRun =
        this.nodeDataProviderExtensionService.getSelectedRunForModelGraph(
          this.paneId,
          modelGraph,
        );
      this.curSelectedRunId = selectedRun?.runId || '';
      this.changeDetectorRef.markForCheck();
    });

    effect(() => {
      this.curModelGraph = this.appService.getPaneById(this.paneId)?.modelGraph;
      const runs = this.curModelGraph
        ? this.nodeDataProviderExtensionService.getRunsForModelGraph(
            this.curModelGraph,
          )
        : [];

      let modelGraphChanged = false;
      let runsChanged = false;
      if (this.prevModelGraph !== this.curModelGraph) {
        this.prevModelGraph = this.curModelGraph;
        modelGraphChanged = true;
      }
      const curRunsKey = this.getRunsKey(runs);
      if (this.prevRunsKey !== curRunsKey) {
        this.prevRunsKey = curRunsKey;
        runsChanged = true;
      }

      if (this.curModelGraph && (modelGraphChanged || runsChanged)) {
        // Update run items in the index panel.
        this.runItems = [];
        const runs = this.nodeDataProviderExtensionService.getRunsForModelGraph(
          this.curModelGraph,
        );
        for (const run of runs) {
          this.runItems.push({
            runId: run.runId,
            runName: this.getRunName(run),
            done: run.done,
            error: run.error,
            hideInAggregatedStatsTable: (run.nodeDataProviderData ?? {})[
              this.curModelGraph.id
            ]?.hideInAggregatedStatsTable,
          });
        }
        this.changeDetectorRef.markForCheck();

        this.infoPanelService.curSortingRunIndex = Math.min(
          this.infoPanelService.curSortingRunIndex,
          runs.length - 1,
        );
        this.paginator?.reset();
        this.genOrderedNodes();
        this.populateResultsTable();
        this.infoPanelService.curChildrenStatSortingColIndex = Math.min(
          this.infoPanelService.curChildrenStatSortingColIndex,
          this.childrenStatsCols.length - 1,
        );
        this.childrenStatsPaginator?.reset();
      }
    });

    // Handle changes on children stats table node filter.
    this.childrenStatsTableNodeFilter.valueChanges
      .pipe(debounceTime(150), takeUntilDestroyed(this.destroyRef))
      .subscribe((text) => {
        this.handleChildrenStatsTableFilterChanged();
      });

    // Handle changes on results table node filter.
    this.resultsTableNodeFilter.valueChanges
      .pipe(debounceTime(150), takeUntilDestroyed(this.destroyRef))
      .subscribe((text) => {
        this.handleResultsTableFilterChanged();
      });
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['rootGroupNodeId']) {
      this.paginator?.reset();
      this.childrenStatsPaginator?.reset();
      this.genOrderedNodes();
      this.populateResultsTable();
    }
  }

  getIconName(runItem: RunItem): string {
    return this.isRunItemSelected(runItem) ? 'visibility' : 'visibility_off';
  }

  getVisibleToggleTooltip(runItem: RunItem): string {
    return this.isRunItemSelected(runItem)
      ? 'Visualizing in graph'
      : 'Click to visualize in graph';
  }

  isRunItemSelected(runItem: RunItem): boolean {
    return runItem.runId === this.curSelectedRunId;
  }

  handleChildrenStatsTablePaginatorChanged(curPageIndex: number) {
    this.curPageChildrenStatRows = this.curChildrenStatRows.slice(
      curPageIndex * this.tablePageSize,
      (curPageIndex + 1) * this.tablePageSize,
    );
    this.changeDetectorRef.markForCheck();
  }

  handleTablePaginatorChanged(curPageIndex: number) {
    if (this.curRows == null) {
      this.curPageRows = [];
    } else {
      this.curPageRows = this.curRows.slice(
        curPageIndex * this.tablePageSize,
        (curPageIndex + 1) * this.tablePageSize,
      );
    }
    this.changeDetectorRef.markForCheck();
  }

  handleClickHeader(colIndex: number) {
    if (this.infoPanelService.curSortingRunIndex === colIndex) {
      this.infoPanelService.curSortingDirection = this.nextSortingDirection(
        this.curSortingDirection,
      );
    } else {
      this.infoPanelService.curSortingDirection = colIndex < 0 ? 'asc' : 'desc';
    }

    this.infoPanelService.curSortingRunIndex = colIndex;
    this.sortAndFiltertRows();

    this.paginator?.reset();
    this.handleTablePaginatorChanged(0);
  }

  handleClickChildrenStatsHeader(colIndex: number) {
    if (this.infoPanelService.curChildrenStatSortingColIndex === colIndex) {
      this.infoPanelService.curChildrenStatSortingDirection =
        this.nextSortingDirection(
          this.infoPanelService.curChildrenStatSortingDirection,
        );
    } else {
      this.infoPanelService.curChildrenStatSortingDirection =
        colIndex < 0 ? 'asc' : 'desc';
    }

    this.infoPanelService.curChildrenStatSortingColIndex = colIndex;
    this.sortAndFilterChildrenStatsRows();

    this.childrenStatsPaginator?.reset();
    this.handleChildrenStatsTablePaginatorChanged(0);
  }

  handleClickToggleVisibility(runItem: RunItem, event: Event) {
    event.stopPropagation();

    if (this.isRunItemSelected(runItem)) {
      return;
    }
    this.appService.setSelectedNodeDataProviderRunId(
      this.paneId,
      runItem.runId,
    );
  }

  handleClickDelete(runItem: RunItem) {
    if (!this.curModelGraph) {
      return;
    }

    this.nodeDataProviderExtensionService.deleteRun(runItem.runId);
    this.appService.deleteShowOnNodeItemType([
      `${NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX}${runItem.runName}`,
    ]);
  }

  handleClickNodeLabel(nodeId: string) {
    this.appService.curToLocateNodeInfo.set({
      nodeId,
      // Locate node in the main renderer in the current pane. Its renderer id
      // is the same as the pane id.
      rendererId: this.paneId,
      isGroupNode: false,
    });
  }

  handleToggleExpandCollapseStatsTable(tableContainer: HTMLElement) {
    if (!this.infoPanelService.statsTableCollapsed) {
      tableContainer.style.maxHeight = `${tableContainer.offsetHeight}px`;
    } else {
      tableContainer.style.maxHeight = `${tableContainer.scrollHeight}px`;
    }
    this.changeDetectorRef.markForCheck();

    setTimeout(() => {
      this.infoPanelService.statsTableCollapsed =
        !this.infoPanelService.statsTableCollapsed;
      this.changeDetectorRef.markForCheck();

      if (!this.infoPanelService.statsTableCollapsed) {
        setTimeout(() => {
          tableContainer.style.maxHeight = 'fit-content';
        }, 150);
      }
    });
  }

  handleToggleExpandCollapseChildrenStatsTable(tableContainer: HTMLElement) {
    if (!this.infoPanelService.childrenStatsTableCollapsed) {
      tableContainer.style.maxHeight = `${tableContainer.offsetHeight}px`;
    } else {
      tableContainer.style.maxHeight = `${tableContainer.scrollHeight}px`;
    }
    this.changeDetectorRef.markForCheck();

    setTimeout(() => {
      this.infoPanelService.childrenStatsTableCollapsed =
        !this.infoPanelService.childrenStatsTableCollapsed;
      this.changeDetectorRef.markForCheck();

      if (!this.infoPanelService.childrenStatsTableCollapsed) {
        setTimeout(() => {
          tableContainer.style.maxHeight = 'fit-content';
        }, 150);
      }
    });
  }

  handleToggleExpandCollapseNodeDataTable(tableContainer: HTMLElement) {
    if (!this.infoPanelService.nodeDataTableCollapsed) {
      tableContainer.style.maxHeight = `${tableContainer.offsetHeight}px`;
    } else {
      tableContainer.style.maxHeight = `${tableContainer.scrollHeight}px`;
    }
    this.changeDetectorRef.markForCheck();

    setTimeout(() => {
      this.infoPanelService.nodeDataTableCollapsed =
        !this.infoPanelService.nodeDataTableCollapsed;
      this.changeDetectorRef.markForCheck();

      if (!this.infoPanelService.nodeDataTableCollapsed) {
        setTimeout(() => {
          tableContainer.style.maxHeight = 'fit-content';
        }, 150);
      }
    });
  }

  handleChildrenStatsTableFilterChanged() {
    this.childrenStatsPaginator?.reset();
    this.sortAndFilterChildrenStatsRows();
    this.handleChildrenStatsTablePaginatorChanged(0);
  }

  handleResultsTableFilterChanged() {
    this.paginator?.reset();
    this.sortAndFiltertRows();
    this.handleTablePaginatorChanged(0);
  }

  handleClearStatsTableFilter(formControl: FormControl<string>) {
    if (formControl === this.childrenStatsTableNodeFilter) {
      this.childrenStatsPaginator?.reset();
    } else if (formControl === this.resultsTableNodeFilter) {
      this.paginator?.reset();
    }

    formControl.reset();
  }

  getStatValue(value: number): string {
    if (
      value === Number.POSITIVE_INFINITY ||
      value === Number.NEGATIVE_INFINITY ||
      isNaN(value)
    ) {
      return '-';
    }
    return `${value}`;
  }

  getHideStatsTableCol(index: number): boolean {
    return this.runItems[index]?.hideInAggregatedStatsTable === true;
  }

  trackByRunId(index: number, runItem: RunItem): string {
    return runItem.runId;
  }

  trackByNodeId(index: number, row: {id: string}): string {
    return row.id;
  }

  trackByStat(index: number, row: StatRow): string {
    return row.stat;
  }

  get showResults(): boolean {
    return this.runItems.some((runItem) => runItem.done);
  }

  get rowsCount(): number {
    return this.curRows == null ? 0 : this.curRows.length;
  }

  get childrenStatRowsCount(): number {
    return this.curChildrenStatRows.length;
  }

  get statsTableTitleIcon(): string {
    return this.statsTableCollapsed ? 'arrow_right' : 'arrow_drop_down';
  }

  get statsTableTitle(): string {
    if (this.rootGroupNodeId == null) {
      return 'Aggregated stats';
    }
    return 'Aggregated stats in selected layer';
  }

  get statsTableCollapsed(): boolean {
    return this.infoPanelService.statsTableCollapsed;
  }

  get childrenStatsTableTitleIcon(): string {
    return this.childrenStatsTableCollapsed ? 'arrow_right' : 'arrow_drop_down';
  }

  get childrenStatsTableTitle(): string {
    if (this.rootGroupNodeId == null) {
      return 'Root-level nodes stats';
    }
    return 'Child nodes stats in selected layer';
  }

  get childrenStatsTableCollapsed(): boolean {
    return this.infoPanelService.childrenStatsTableCollapsed;
  }

  get nodeDataTableTitleIcon(): string {
    return this.nodeDataTableCollapsed ? 'arrow_right' : 'arrow_drop_down';
  }

  get nodeDataTableTitle(): string {
    if (this.rootGroupNodeId == null) {
      return 'Node data';
    }
    return 'Node data in selected layer';
  }

  get nodeDataTableCollapsed(): boolean {
    return this.infoPanelService.nodeDataTableCollapsed;
  }

  get curSortingDirection(): SortingDirection {
    return this.infoPanelService.curSortingDirection;
  }

  get curSortingRunIndex(): number {
    return this.infoPanelService.curSortingRunIndex;
  }

  get curChildrenStatSortingDirection(): SortingDirection {
    return this.infoPanelService.curChildrenStatSortingDirection;
  }

  get curChildrenStatSortingColIndex(): number {
    return this.infoPanelService.curChildrenStatSortingColIndex;
  }

  get showStatsTable(): boolean {
    if (!this.curModelGraph) {
      return false;
    }

    const runs = this.nodeDataProviderExtensionService.getRunsForModelGraph(
      this.curModelGraph,
    );
    let hide = true;
    for (const run of runs) {
      if (!run.nodeDataProviderData) {
        continue;
      }
      const curData = run.nodeDataProviderData[this.curModelGraph.id];
      if (!curData.hideInAggregatedStatsTable) {
        hide = false;
        break;
      }
    }
    return !hide;
  }

  get showChildrenStatsTable(): boolean {
    if (!this.curModelGraph) {
      return false;
    }

    const runs = this.nodeDataProviderExtensionService.getRunsForModelGraph(
      this.curModelGraph,
    );
    let hide = true;
    for (const run of runs) {
      if (!run.nodeDataProviderData) {
        continue;
      }
      const curData = run.nodeDataProviderData[this.curModelGraph.id];
      if (!curData.hideInChildrenStatsTable) {
        hide = false;
        break;
      }
    }
    return !hide;
  }

  private genOrderedNodes() {
    if (!this.curModelGraph) {
      return;
    }

    const cacheKey = this.getOrderedNodesCacheKey();
    const cachedOrderedNodes = this.orderedNodesCache[cacheKey];
    if (cachedOrderedNodes != null) {
      this.orderedNodes = cachedOrderedNodes;
    } else {
      const rootGroupNode: GroupNode | undefined =
        this.rootGroupNodeId == null
          ? undefined
          : (this.curModelGraph.nodesById[this.rootGroupNodeId] as GroupNode);
      let nodeIdsInRootGroupNode = new Set<string>();
      if (rootGroupNode != null) {
        nodeIdsInRootGroupNode = new Set<string>(
          rootGroupNode.descendantsOpNodeIds || [],
        );
      }
      this.orderedNodes = this.curModelGraph.nodes.filter(
        (node) =>
          isOpNode(node) &&
          !node.hideInLayout &&
          node.id !== 'GraphInputs' &&
          node.id !== 'GraphOutputs' &&
          (rootGroupNode == null || nodeIdsInRootGroupNode.has(node.id)),
      ) as OpNode[];
      this.orderedNodesCache[cacheKey] = this.orderedNodes;
    }
  }

  private populateResultsTable() {
    if (!this.curModelGraph || this.orderedNodes.length === 0) {
      return;
    }
    const runs = this.nodeDataProviderExtensionService.getRunsForModelGraph(
      this.curModelGraph,
    );

    this.curStatRows = [
      {stat: 'Min', values: []},
      {stat: 'Max', values: []},
      {stat: 'Sum', values: []},
      {stat: 'Avg', values: []},
    ];
    const stats: Stat[] = [];
    for (let i = 0; i < runs.length; i++) {
      stats.push({
        min: Number.POSITIVE_INFINITY,
        max: Number.NEGATIVE_INFINITY,
        sum: 0,
        count: 0,
      });
    }

    this.curRows = [];
    for (let i = 0; i < this.orderedNodes.length; i++) {
      const node = this.orderedNodes[i];
      const nodeId = node.id;
      const cols: Col[] = [];
      for (let j = 0; j < runs.length; j++) {
        const run = runs[j];
        const curResults = run.results || {};
        const nodeResult = (curResults[this.curModelGraph.id] || {})[nodeId];
        // tslint:disable-next-line:no-any Allow arbitrary types.
        const value: any = nodeResult?.value;
        const strValue = nodeResult?.strValue || '-';
        const bgColor = nodeResult?.bgColor || '';
        const textColor = nodeResult?.textColor || 'black';
        cols.push({value, strValue, bgColor, textColor});

        // Update stats.
        if (value != null && typeof value === 'number') {
          const curStat = stats[j];
          // min.
          curStat.min = Math.min(value, curStat.min);
          // max.
          curStat.max = Math.max(value, curStat.max);
          // count.
          curStat.count++;
          // sum.
          curStat.sum += value;
        }
      }
      const incomingEdges = node.incomingEdges || [];
      const isInput =
        incomingEdges.length === 0 ||
        incomingEdges.some((edge) => edge.sourceNodeId === 'GraphInputs');
      const outgoingEdges = node.outgoingEdges || [];
      const isOutput =
        outgoingEdges.length === 0 ||
        outgoingEdges.some((edge) => edge.targetNodeId === 'GraphOutputs');
      this.curRows.push({
        id: nodeId,
        index: i,
        isInput,
        isOutput,
        label: this.curModelGraph.nodesById[nodeId].label || '?',
        cols,
      });
    }
    this.savedCurRows = [...this.curRows];
    this.sortAndFiltertRows();
    this.handleTablePaginatorChanged(0);

    // Populate stat rows.
    this.curStatRows[0].values = stats.map((stat) => stat.min);
    this.curStatRows[1].values = stats.map((stat) => stat.max);
    this.curStatRows[2].values = stats.map((stat) => stat.sum);
    this.curStatRows[3].values = stats.map((stat) => stat.sum / stat.count);

    // Hide stat values based on hideAggregatedStats.
    const allStats: AggregatedStat[] = ['min', 'max', 'sum', 'avg'];
    for (let i = 0; i < runs.length; i++) {
      const run = runs[i];
      const statsToHide: AggregatedStat[] =
        run.nodeDataProviderData?.[this.curModelGraph.id]
          ?.hideAggregatedStats ?? [];
      for (let j = 0; j < allStats.length; j++) {
        const stat = allStats[j];
        if (statsToHide.includes(stat)) {
          // Set the value to positive infinity so that it will be displayed as
          // '-' in the table. See `getStatValue()`.
          this.curStatRows[j].values[i] = Number.POSITIVE_INFINITY;
        }
      }
    }

    // Generate children stats columns.
    this.childrenStatsCols = [];
    let childrenStatColIndex = 0;
    const groupNode = this.curModelGraph.nodesById[
      this.rootGroupNodeId ?? ''
    ] as GroupNode;
    const runIdToValueInfos: Record<string, NodeDataProviderValueInfo[]> = {};
    for (let i = 0; i < runs.length; i++) {
      const run = runs[i];
      let childrenStats = CHILDREN_STATS;
      let valueInfos: NodeDataProviderValueInfo[] = [];
      let multiLineHeader = false;
      if (
        (run.nodeDataProviderData ?? {})[this.curModelGraph.id]
          ?.showLabelCountColumnsInChildrenStatsTable
      ) {
        valueInfos = genSortedValueInfos(
          groupNode,
          this.curModelGraph,
          (run.results ?? {})[this.curModelGraph.id],
        ).sort((a, b) => a.label.localeCompare(b.label));
        runIdToValueInfos[run.runId] = valueInfos;
        childrenStats = valueInfos.map((valueInfo) => `#${valueInfo.label}`);
        multiLineHeader = true;
      }
      for (const childrenStat of childrenStats) {
        let label = childrenStat;
        if (runs.length > 1) {
          if (multiLineHeader) {
            label = `${this.getRunName(runs[i])}\n${childrenStat}`;
          } else {
            label = `${this.getRunName(runs[i])} • ${childrenStat}`;
          }
        }
        this.childrenStatsCols.push({
          colIndex: childrenStatColIndex,
          runIndex: i,
          label,
          hideInChildrenStatsTable:
            runs[i].nodeDataProviderData?.[this.curModelGraph.id]
              ?.hideInChildrenStatsTable,
          multiLineHeader,
        });
        childrenStatColIndex++;
      }
    }

    // Populate children stats rows.
    this.curChildrenStatRows = [];
    const nsChildrenIds = this.rootGroupNodeId
      ? (this.curModelGraph.nodesById[this.rootGroupNodeId] as GroupNode)
          .nsChildrenIds || []
      : this.curModelGraph.rootNodes.map((node) => node.id);
    for (let i = 0; i < nsChildrenIds.length; i++) {
      const nodeId = nsChildrenIds[i];
      const node = this.curModelGraph.nodesById[nodeId];
      const colValues: number[] = [];
      const colStrs: string[] = [];
      const colHidden: boolean[] = [];
      for (let runIndex = 0; runIndex < runs.length; runIndex++) {
        const run = runs[runIndex];
        const curResults = run.results || {};
        // Sum pct.
        if (!runIdToValueInfos[run.runId]) {
          let sumPct = 0;
          let hasValue = false;
          if (isOpNode(node)) {
            const nodeResult = (curResults[this.curModelGraph.id] || {})[
              nodeId
            ];
            const value = nodeResult?.value;
            if (value != null && typeof value === 'number') {
              sumPct = (value / stats[runIndex].sum) * 100;
              hasValue = true;
            }
          } else if (isGroupNode(node)) {
            let layerSum = 0;
            const childrenIds = node.descendantsOpNodeIds || [];
            for (const childNodeId of childrenIds) {
              const nodeResult = (curResults[this.curModelGraph.id] || {})[
                childNodeId
              ];
              const value = nodeResult?.value;
              if (value != null && typeof value === 'number') {
                layerSum += value;
                hasValue = true;
              }
            }
            sumPct = (layerSum / stats[runIndex].sum) * 100;
          }
          colValues.push(sumPct);
          colStrs.push(hasValue ? sumPct.toFixed(1) : '-');
          colHidden.push(
            run.nodeDataProviderData?.[this.curModelGraph.id]
              ?.hideInChildrenStatsTable === true,
          );
        }
        // Label counts.
        else {
          const valueInfos = runIdToValueInfos[run.runId];
          const curResults = run.results || {};
          const nodeResult = (curResults[this.curModelGraph.id] || {})[nodeId];
          const value = nodeResult?.value || '';
          for (const valueInfo of valueInfos) {
            let count = 0;
            if (isOpNode(node)) {
              if (valueInfo.label === value) {
                count = 1;
              }
            } else if (isGroupNode(node)) {
              const childrenIds = node.descendantsOpNodeIds || [];
              for (const childNodeId of childrenIds) {
                const nodeResult = (curResults[this.curModelGraph.id] || {})[
                  childNodeId
                ];
                const childValue = nodeResult?.value || '';
                if (childValue === valueInfo.label) {
                  count++;
                }
              }
            }
            colValues.push(count);
            colStrs.push(`${count}`);
            colHidden.push(
              run.nodeDataProviderData?.[this.curModelGraph.id]
                ?.hideInChildrenStatsTable === true,
            );
          }
        }
      }
      this.curChildrenStatRows.push({
        id: nodeId,
        label: node.label,
        index: i,
        colValues,
        colStrs,
        colHidden,
      });
    }
    this.savedChildrenStatRows = [...this.curChildrenStatRows];
    this.sortAndFilterChildrenStatsRows();
    this.handleChildrenStatsTablePaginatorChanged(0);

    this.changeDetectorRef.markForCheck();
  }

  private nextSortingDirection(direction: SortingDirection) {
    switch (direction) {
      case 'desc':
        return 'asc';
      case 'asc':
        return 'desc';
      default:
        return direction;
    }
  }

  private sortAndFiltertRows() {
    this.curRows = [...(this.savedCurRows || [])];

    // Filter.
    const regexText = (this.resultsTableNodeFilter.value || '').trim();
    if (regexText !== '') {
      try {
        const regex = new RegExp(regexText, 'i');
        this.curRows = this.curRows.filter((row) => regex.test(row.label));
      } catch {
        return;
      }
    }

    // Sort.
    this.curRows.sort((a, b) => {
      const v1 = this.getCellValue(a, this.infoPanelService.curSortingRunIndex);
      const v2 = this.getCellValue(b, this.infoPanelService.curSortingRunIndex);
      return this.compareValue(
        v1,
        v2,
        this.infoPanelService.curSortingDirection,
      );
    });
  }

  private sortAndFilterChildrenStatsRows() {
    this.curChildrenStatRows = [...(this.savedChildrenStatRows || [])];

    // Filter.
    const regexText = (this.childrenStatsTableNodeFilter.value || '').trim();
    if (regexText !== '') {
      try {
        const regex = new RegExp(regexText, 'i');
        this.curChildrenStatRows = this.curChildrenStatRows.filter((row) =>
          regex.test(row.label),
        );
      } catch {
        return;
      }
    }

    // Sort.
    this.curChildrenStatRows.sort((a, b) => {
      const v1 = this.getChildrenStatsColValue(
        a,
        this.infoPanelService.curChildrenStatSortingColIndex,
      );
      const v2 = this.getChildrenStatsColValue(
        b,
        this.infoPanelService.curChildrenStatSortingColIndex,
      );
      return this.compareValue(
        v1,
        v2,
        this.infoPanelService.curChildrenStatSortingDirection,
      );
    });
  }

  private compareValue(
    v1: number | string | undefined,
    v2: number | string | undefined,
    direction: SortingDirection,
  ): number {
    if (v1 == null && v2 == null) {
      return 0;
    } else if (v1 == null && v2 != null) {
      return direction === 'asc' ? -1 : 1;
    } else if (v1 != null && v2 == null) {
      return direction === 'asc' ? 1 : -1;
    } else if (typeof v1 === 'number' && typeof v2 === 'number') {
      return direction === 'asc' ? v1 - v2 : v2 - v1;
    } else {
      const strV1 = JSON.stringify(v1);
      const strV2 = JSON.stringify(v2);
      return direction === 'asc'
        ? strV1.localeCompare(strV2)
        : strV2.localeCompare(strV1);
    }
  }

  private getCellValue(
    row: Row,
    colIndex: number,
  ): string | number | undefined {
    switch (colIndex) {
      case -2:
        return row.index;
      case -1:
        return row.label;
      default:
        return row.cols[colIndex].value;
    }
  }

  private getChildrenStatsColValue(
    row: ChildrenStatRow,
    colIndex: number,
  ): string | number | undefined {
    switch (colIndex) {
      case -2:
        return row.index;
      case -1:
        return row.label;
      default:
        return row.colValues[colIndex];
    }
  }

  private getOrderedNodesCacheKey(): string {
    return `${this.curModelGraph?.collectionLabel}___${this.curModelGraph?.id}___${this.rootGroupNodeId}`;
  }

  private getRunsKey(runs: NodeDataProviderRunData[]): string {
    return runs
      .map((run) => {
        const parts: string[] = [];
        parts.push(run.runId);
        parts.push(String(run.done));
        const results = run.results || {};
        parts.push(String(Object.keys(results).length));
        return parts.join('__');
      })
      .join(',');
  }

  private getRunName(run: NodeDataProviderRunData): string {
    return getRunName(run, this.curModelGraph);
  }
}
