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

import {ConnectedPosition, OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  DestroyRef,
  effect,
  ElementRef,
  HostBinding,
  Input,
  QueryList,
  ViewChildren,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatTooltipModule} from '@angular/material/tooltip';
import {combineLatest, fromEvent} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {Bubble} from '../bubble/bubble';

import {AppService} from './app_service';
import {TENSOR_TAG_METADATA_KEY, TENSOR_VALUES_KEY} from './common/consts';
import {GroupNode, ModelGraph, ModelNode, OpNode} from './common/model_graph';
import {
  CommandType,
  IncomingEdge,
  KeyValue,
  KeyValueList,
  KeyValuePairs,
  NodeAttributeValueType,
  NodeDataProviderRunInfo,
  NodeIdsNodeAttributeValue,
  OutgoingEdge,
  SearchMatchAttr,
  SearchMatchInputMetadata,
  SearchMatchOutputMetadata,
  SearchMatchType,
  SearchResults,
  SpecialNodeAttributeValue,
} from './common/types';
import {
  getNamespaceLabel,
  getRunName,
  isGroupNode,
  isOpNode,
} from './common/utils';
import {ExpandableInfoText} from './expandable_info_text';
import {HoverableLabel} from './hoverable_label';
import {InfoPanelService} from './info_panel_service';
import {genIoTreeData, IoTree, TreeNode} from './io_tree';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {NodeDataProviderSummaryPanel} from './node_data_provider_summary_panel';
import {Paginator} from './paginator';
import {SplitPaneService} from './split_pane_service';

interface InfoSection {
  label: SectionLabel;
  sectionType: 'graph' | 'op' | 'group';
  items: InfoItem[];
}

enum SectionLabel {
  GRAPH_INFO = 'Graph info',
  NODE_INFO = 'Node info',
  LAYER_INFO = 'Layer info',
  LAYER_ATTRS = 'Layer attributes',
  ATTRIBUTES = 'Attributes',
  NODE_DATA_PROVIDERS = 'Node data providers',
  IDENTICAL_GROUPS = 'Identical groups',
  INPUTS = 'inputs',
  OUTPUTS = 'outputs',
  GROUP_INPUTS = 'layer inputs',
  GROUP_OUTPUTS = 'layer outputs',
}

interface InfoItem {
  id?: string;
  section: InfoSection;
  label: string;
  value: string;
  canShowOnNode?: boolean;
  showOnNode?: boolean;
  bigText?: boolean;
  bgColor?: string;
  textColor?: string;
  loading?: boolean;
  specialValue?: SpecialNodeAttributeValue;
}

interface OutputItem {
  index: number;
  tensorTag: string;
  sourceOpNode: OpNode;
  outputId: string;
  metadataList: OutputItemMetadata[];
  showSourceOpNode?: boolean;
}

interface OutputItemMetadata extends KeyValue {
  connectedNodes?: OpNode[];
}

interface InputItem {
  index: number;
  opNode: OpNode;
  metadataList: KeyValueList;
  targetOpNode?: OpNode;
}

const MIN_WIDTH = 64;
const SIDE_PANEL_WIDTH_ANIMATION_DURATION = 150;
const DEFAULT_WIDTH = 370;

/** The info panel component that shows info for selected element. */
@Component({
  standalone: true,
  selector: 'info-panel',
  imports: [
    Bubble,
    CommonModule,
    ExpandableInfoText,
    HoverableLabel,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatSlideToggleModule,
    MatTooltipModule,
    Paginator,
    IoTree,
    NodeDataProviderSummaryPanel,
  ],
  providers: [InfoPanelService],
  templateUrl: './info_panel.ng.html',
  styleUrls: ['./info_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class InfoPanel {
  @Input({required: true}) paneId!: string;
  @ViewChildren('inputValueContent')
  inputValueContents = new QueryList<ElementRef<HTMLElement>>();

  readonly NodeAttributeValueType = NodeAttributeValueType;

  private curModelGraph?: ModelGraph;
  private curSelectedNode?: ModelNode;
  private readonly curShowOnOpNodeInfoIds = new Set<string>();
  private readonly curShowOnOpNodeAttrIds = new Set<string>();
  private readonly curShowOnGroupNodeInfoIds = new Set<string>();
  private readonly curShowOnNodeDataProviderRuns: Record<
    string,
    NodeDataProviderRunInfo
  > = {};
  private curSearchResults: SearchResults | undefined = undefined;

  @HostBinding('style.width.px') width = DEFAULT_WIDTH;
  @HostBinding('style.min-width.px') minWidth = DEFAULT_WIDTH;

  sections: InfoSection[] = [];
  inputItems: InputItem[] = [];
  inputItemsForCurPage: InputItem[] = [];
  outputItems: OutputItem[] = [];
  outputItemsForCurPage: OutputItem[] = [];
  groupInputItems: InputItem[] = [];
  groupInputItemsForCurPage: InputItem[] = [];
  groupOutputItems: OutputItem[] = [];
  groupOutputItemsForCurPage: OutputItem[] = [];
  identicalGroupNodes: GroupNode[] = [];
  identicalGroupsData?: TreeNode[];
  curRendererId = '';
  curInputsCount = 0;
  curOutputsCount = 0;
  curGroupInputsCount = 0;
  curGroupOutputsCount = 0;
  resizing = false;
  hide = false;

  readonly ioPageSize: number;
  readonly SectionLabel = SectionLabel;
  readonly outputMetadataConnectedTo = 'connects to';
  readonly inputMetadataValuesKey = 'values';
  readonly inputMetadataNamespaceKey = 'namespace';
  readonly locatorTooltip = 'Click: locate\nAlt+click: select';
  readonly getNamespaceLabel = getNamespaceLabel;

  readonly constValuesPopupSize: OverlaySizeConfig = {
    minWidth: 100,
    minHeight: 0,
    maxWidth: 600,
  };

  readonly constValuesPopupPosition: ConnectedPosition[] = [
    {
      originX: 'start',
      originY: 'top',
      overlayX: 'end',
      overlayY: 'top',
      offsetX: -32,
    },
  ];

  readonly outputConnectsToNamespacePopupPosition: ConnectedPosition[] = [
    {
      originX: 'start',
      originY: 'top',
      overlayX: 'end',
      overlayY: 'top',
      offsetX: -4,
    },
  ];

  private curSearchAttrMatches: SearchMatchAttr[] = [];
  private curSearchInputMatches: SearchMatchInputMetadata[] = [];
  private curSearchOutputMatches: SearchMatchOutputMetadata[] = [];
  private savedWidth = 0;

  constructor(
    private readonly appService: AppService,
    private readonly destroyRef: DestroyRef,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly infoPanelService: InfoPanelService,
    private readonly splitPaneService: SplitPaneService,
  ) {
    this.ioPageSize = this.appService.testMode ? 5 : 25;

    // Handle selected node changes.
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      if (!pane || !pane.modelGraph) {
        return;
      }
      this.curModelGraph = pane.modelGraph;
      this.curRendererId = pane.selectedNodeInfo?.rendererId || '';
      const selectedNodeId = pane.selectedNodeInfo?.nodeId || '';
      if (this.curSelectedNodeId === selectedNodeId) {
        return;
      }
      this.handleNodeSelected(selectedNodeId);
      setTimeout(() => {
        this.splitPaneService.resetInputOutputHiddenIds();
        this.handleSearchResultsChanged();
      });
    });

    // Handle search results changes.
    effect(() => {
      const pane = this.appService.getPaneById(this.paneId);
      if (!pane || !pane.modelGraph) {
        return;
      }
      if (this.curSearchResults === pane.searchResults) {
        return;
      }
      this.curSearchResults = pane.searchResults;
      this.handleSearchResultsChanged();
    });

    effect(() => {
      this.nodeDataProviderExtensionService.runs();
      this.genInfoData();
      this.changeDetectorRef.markForCheck();

      setTimeout(() => {
        this.updateInputValueContentsExpandable();
      });
    });

    // React to commands.
    this.appService.command
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((command) => {
        // Ignore commands not for this pane.
        if (
          command.paneIndex !== this.appService.getPaneIndexById(this.paneId)
        ) {
          return;
        }

        // Handle commands.
        switch (command.type) {
          case CommandType.COLLAPSE_INFO_PANEL:
            this.setHideInfoPanel(true);
            break;
          case CommandType.SHOW_INFO_PANEL:
            this.setHideInfoPanel(false);
            break;
          default:
            break;
        }
      });
  }

  isSearchMatchedAttrId(attrId: string): boolean {
    return (
      this.curSearchAttrMatches.find(
        (match) => match.matchedAttrId === attrId,
      ) != null
    );
  }

  isSearchMatchedInputValue(value: string): boolean {
    if (!this.curSearchInputMatches) {
      return false;
    }
    return (
      this.curSearchInputMatches.find((match) => match.matchedText === value) !=
      null
    );
  }

  isSearchMatchedOutputValue(value: string): boolean {
    if (!this.curSearchOutputMatches) {
      return false;
    }
    return (
      this.curSearchOutputMatches.find(
        (match) => match.matchedText === value,
      ) != null
    );
  }

  handleMouseDownResizer(event: MouseEvent) {
    event.preventDefault();

    document.body.style.cursor = 'ew-resize';

    const move = fromEvent<MouseEvent>(document, 'mousemove');
    const up = fromEvent<MouseEvent>(window, 'mouseup');
    const savedWidth = this.width;

    // Hit position.
    const hitPtX = event.clientX;
    this.resizing = true;
    this.changeDetectorRef.markForCheck();

    combineLatest([move])
      .pipe(takeUntil(up))
      .subscribe({
        next: ([moveEvent]) => {
          // Calculate delta.
          const delta = moveEvent.clientX - hitPtX;
          this.width = Math.max(MIN_WIDTH, savedWidth - delta);
          this.minWidth = this.width;
          this.changeDetectorRef.markForCheck();
        },
        complete: () => {
          document.body.style.cursor = 'default';
          this.resizing = false;
          this.changeDetectorRef.markForCheck();
        },
      });
  }

  handleInputPaginatorChanged(curPageIndex: number) {
    this.inputItemsForCurPage = this.inputItems.slice(
      curPageIndex * this.ioPageSize,
      (curPageIndex + 1) * this.ioPageSize,
    );
    this.changeDetectorRef.markForCheck();

    setTimeout(() => {
      this.updateInputValueContentsExpandable();
    });
  }

  handleOutputPaginatorChanged(curPageIndex: number) {
    this.outputItemsForCurPage = this.outputItems.slice(
      curPageIndex * this.ioPageSize,
      (curPageIndex + 1) * this.ioPageSize,
    );
    this.changeDetectorRef.markForCheck();
  }

  handleGroupInputPaginatorChanged(curPageIndex: number) {
    this.groupInputItemsForCurPage = this.groupInputItems.slice(
      curPageIndex * this.ioPageSize,
      (curPageIndex + 1) * this.ioPageSize,
    );
    this.changeDetectorRef.markForCheck();

    setTimeout(() => {
      this.updateInputValueContentsExpandable();
    });
  }

  handleGroupOutputPaginatorChanged(curPageIndex: number) {
    this.groupOutputItemsForCurPage = this.groupOutputItems.slice(
      curPageIndex * this.ioPageSize,
      (curPageIndex + 1) * this.ioPageSize,
    );
    this.changeDetectorRef.markForCheck();
  }

  handleIdenticalGroupsPaginatorChanged(curPageIndex: number) {
    this.identicalGroupsData = genIoTreeData(
      this.identicalGroupNodes.slice(
        curPageIndex * this.ioPageSize,
        (curPageIndex + 1) * this.ioPageSize,
      ),
      [],
      'incoming',
      this.curSelectedNode?.id || '',
    );
  }

  toggleHideInfoPanel() {
    this.hide = !this.hide;
    let targetWidth = 0;
    if (this.hide) {
      this.savedWidth = this.width;
    } else {
      targetWidth = this.savedWidth;
    }
    this.animateSidePanelWidth(targetWidth);
  }

  setHideInfoPanel(hide: boolean) {
    this.hide = hide;
    let targetWidth = 0;
    if (this.hide) {
      this.savedWidth = this.width;
    } else {
      targetWidth = this.savedWidth;
    }
    this.animateSidePanelWidth(targetWidth, 0);
  }

  handleToggleSection(sectionName: SectionLabel, sectionEle?: HTMLElement) {
    if (!sectionEle) return;

    const collapsed = this.isSectionCollapsed(sectionName);
    if (!collapsed) {
      sectionEle.style.maxHeight = `${sectionEle.offsetHeight}px`;
      sectionEle.style.overflow = 'hidden';
    } else {
      sectionEle.style.maxHeight = `${sectionEle.scrollHeight}px`;
    }
    this.changeDetectorRef.markForCheck();

    setTimeout(() => {
      if (this.infoPanelService.collapsedSectionNames.has(sectionName)) {
        this.infoPanelService.collapsedSectionNames.delete(sectionName);
      } else {
        this.infoPanelService.collapsedSectionNames.add(sectionName);
      }
      this.changeDetectorRef.markForCheck();

      setTimeout(() => {
        if (!this.isSectionCollapsed(sectionName)) {
          sectionEle.style.overflow = 'visible';
          sectionEle.style.maxHeight = 'fit-content';
        } else {
          sectionEle.style.overflow = 'hidden';
        }
      }, 150);
    });
  }

  isSectionCollapsed(sectionName: SectionLabel): boolean {
    return this.infoPanelService.collapsedSectionNames.has(sectionName);
  }

  getSectionToggleIcon(sectionName: SectionLabel): string {
    return this.isSectionCollapsed(sectionName)
      ? 'chevron_right'
      : 'expand_more';
  }

  handleLocateNode(nodeId: string, event: MouseEvent) {
    this.appService.curToLocateNodeInfo.set({
      nodeId,
      rendererId: this.curRendererId,
      isGroupNode: false,
      select: event.altKey,
    });
  }

  handleToggleInputOpNodeVisibility(
    nodeId: string,
    allItems: InputItem[],
    event: MouseEvent,
  ) {
    event.stopPropagation();

    if (event.altKey) {
      this.splitPaneService.setInputOpNodeVisible(
        nodeId,
        allItems.map((item) => item.opNode.id),
      );
    } else {
      this.splitPaneService.toggleInputOpNodeVisibility(nodeId);
    }
  }

  getInputOpNodeToggleVisible(nodeId: string): boolean {
    return this.splitPaneService.getInputOpNodeVisible(nodeId);
  }

  getInputOpNodeToggleVisibilityIcon(nodeId: string): string {
    return this.getInputOpNodeToggleVisible(nodeId)
      ? 'visibility'
      : 'visibility_off';
  }

  getInputOpNodeToggleVisibilityTooltip(nodeId: string): string {
    return this.getInputOpNodeToggleVisible(nodeId)
      ? 'Click to hide highlight'
      : 'Click to show highlight';
  }

  handleToggleOutputVisibility(
    item: OutputItem,
    items: OutputItem[],
    event: MouseEvent,
  ) {
    event.stopPropagation();

    if (event.altKey) {
      this.splitPaneService.setOutputVisible(
        item.sourceOpNode.id,
        item.outputId,
        items.map((item) => ({
          nodeId: item.sourceOpNode.id,
          outputId: item.outputId,
        })),
      );
    } else {
      this.splitPaneService.toggleOutputVisibility(
        item.sourceOpNode.id,
        item.outputId,
      );
    }
  }

  getOutputToggleVisible(item: OutputItem): boolean {
    return this.splitPaneService.getOutputVisible(
      item.sourceOpNode.id,
      item.outputId,
    );
  }

  getOutputToggleVisibilityIcon(item: OutputItem): string {
    return this.getOutputToggleVisible(item) ? 'visibility' : 'visibility_off';
  }

  getOutputToggleVisibilityTooltip(item: OutputItem): string {
    return this.getOutputToggleVisible(item)
      ? 'Click to hide highlight'
      : 'Click to show highlight';
  }

  getInputName(item: InputItem): string {
    const tensorTagItem = item.metadataList.find(
      (item) => item.key === TENSOR_TAG_METADATA_KEY,
    );

    return tensorTagItem
      ? `${tensorTagItem.value} (${item.opNode.label})`
      : item.opNode.label;
  }

  getInputTensorTag(item: InputItem): string {
    const tensorTagItem = item.metadataList.find(
      (item) => item.key === TENSOR_TAG_METADATA_KEY,
    );
    return tensorTagItem?.value ?? '';
  }

  getOutputName(item: OutputItem): string {
    return item.tensorTag === '' ? 'output' : item.tensorTag;
  }

  getShowMetadata(metadata: KeyValue): boolean {
    return !metadata.key.startsWith('__');
  }

  getHasConnectedToNodes(item: OutputItem): boolean {
    const connectedNodesMetadataItem = item.metadataList.find(
      (metadataItem) => metadataItem.key === this.outputMetadataConnectedTo,
    );
    return (connectedNodesMetadataItem?.connectedNodes || []).length > 0;
  }

  getSectionDisplayLabel(sectionLabel: SectionLabel): string {
    if (sectionLabel === SectionLabel.NODE_DATA_PROVIDERS) {
      return this.nodeDataProviderPanelTitle;
    }
    return sectionLabel;
  }

  trackByItemIdOrLabel(index: number, item: InfoItem): string {
    return item.id || item.label;
  }

  get canShowGraphInfo(): boolean {
    return this.curModelGraph != null && this.curSelectedNode == null;
  }

  get showNodeDataProviderSummary(): boolean {
    if (
      !this.curModelGraph ||
      this.appService.config()?.hideNodeDataInInfoPanel
    ) {
      return false;
    }

    return (
      (this.curSelectedNode == null || isGroupNode(this.curSelectedNode)) &&
      this.nodeDataProviderExtensionService.getRunsForModelGraph(
        this.curModelGraph,
      ).length > 0
    );
  }

  get curSelectedNodeId(): string | undefined {
    return this.curSelectedNode ? this.curSelectedNode.id : undefined;
  }

  get showInputPaginator(): boolean {
    return (
      this.inputItems.length > this.ioPageSize &&
      !this.isSectionCollapsed(SectionLabel.INPUTS)
    );
  }

  get showOutputPaginator(): boolean {
    return (
      this.outputItems.length > this.ioPageSize &&
      !this.isSectionCollapsed(SectionLabel.OUTPUTS)
    );
  }

  get showGroupInputPaginator(): boolean {
    return (
      this.groupInputItems.length > this.ioPageSize &&
      !this.isSectionCollapsed(SectionLabel.GROUP_INPUTS)
    );
  }

  get showGroupOutputPaginator(): boolean {
    return (
      this.groupOutputItems.length > this.ioPageSize &&
      !this.isSectionCollapsed(SectionLabel.GROUP_OUTPUTS)
    );
  }

  get showIdenticalGroupsPaginator(): boolean {
    return (
      this.identicalGroupNodes.length > this.ioPageSize &&
      !this.isSectionCollapsed(SectionLabel.IDENTICAL_GROUPS)
    );
  }

  get hideToggleTooltip(): string {
    return this.hide ? 'Show info panel' : 'Hide info panel';
  }

  get hideToggleIconName(): string {
    return this.hide ? 'chevron_left' : 'chevron_right';
  }

  get nodeDataProviderPanelTitle(): string {
    return (
      this.appService.config()?.renameNodeDataProviderPanelTitleTo ??
      SectionLabel.NODE_DATA_PROVIDERS
    );
  }

  private handleNodeSelected(nodeId: string) {
    if (!this.curModelGraph || !nodeId) {
      this.curSelectedNode = undefined;
    } else {
      this.curSelectedNode = this.curModelGraph.nodesById[nodeId];
    }
    this.genInfoData();
    this.changeDetectorRef.markForCheck();

    setTimeout(() => {
      this.updateInputValueContentsExpandable();
    });
  }

  private genInfoData() {
    this.sections = [];
    this.inputItems = [];
    this.outputItems = [];
    this.groupInputItems = [];
    this.groupOutputItems = [];
    this.identicalGroupNodes = [];
    this.identicalGroupsData = undefined;

    if (this.canShowGraphInfo) {
      this.genInfoDataForGraph();
    } else if (this.curSelectedNode) {
      if (isOpNode(this.curSelectedNode)) {
        this.genInfoDataForSelectedOpNode();
        this.genInputsOutputsData();
      } else if (isGroupNode(this.curSelectedNode)) {
        this.genInfoDataForSelectedGroupNode();
        if (this.appService.config()?.highlightLayerNodeInputsOutputs) {
          this.genGroupInputsOutputsData();
        }
      }
    }
  }

  private genInfoDataForGraph() {
    if (!this.curModelGraph) {
      return;
    }

    // Section for basic graph data.
    const graphSection: InfoSection = {
      label: SectionLabel.GRAPH_INFO,
      sectionType: 'graph',
      items: [],
    };
    this.sections.push(graphSection);

    // Custom attributes.
    const graphAttributes = this.curModelGraph.groupNodeAttributes?.[''];
    if (graphAttributes) {
      for (const key of Object.keys(graphAttributes)) {
        graphSection.items.push({
          section: graphSection,
          label: key,
          value: graphAttributes[key],
        });
      }
    }

    // Node count.
    let nodeCount = 0;
    let layerCount = 0;
    for (const node of this.curModelGraph.nodes) {
      if (isOpNode(node) && !node.hideInLayout) {
        nodeCount++;
      } else if (isGroupNode(node)) {
        layerCount++;
      }
    }
    graphSection.items.push(
      {
        section: graphSection,
        label: 'op node count',
        value: String(nodeCount),
      },
      {
        section: graphSection,
        label: 'layer count',
        value: String(layerCount),
      },
    );
  }

  private genInfoDataForSelectedOpNode() {
    if (!this.curModelGraph || !this.curSelectedNode) {
      return;
    }

    const opNode = this.curSelectedNode as OpNode;
    const config = this.appService.config();

    // Section for basic node data.
    const nodeSection: InfoSection = {
      label: SectionLabel.NODE_INFO,
      sectionType: 'op',
      items: [],
    };
    this.sections.push(nodeSection);

    // Node op.
    let label = config?.renameNodeInfoOpNameTo ?? 'op name';
    nodeSection.items.push({
      section: nodeSection,
      label,
      value: `${opNode.label}`,
    });
    // Node id.
    label = 'id';
    nodeSection.items.push({
      section: nodeSection,
      label,
      value: opNode.id,
      canShowOnNode: true,
      showOnNode: this.curShowOnOpNodeInfoIds.has(label),
    });
    // Node namespace.
    label = 'namespace';
    nodeSection.items.push({
      section: nodeSection,
      label,
      value: getNamespaceLabel(opNode),
      canShowOnNode: true,
      showOnNode: this.curShowOnOpNodeInfoIds.has(label),
    });

    // Filter out node info keys specified in the config.
    const nodeInfoKeysToHide = config?.nodeInfoKeysToHide ?? [];
    nodeSection.items = nodeSection.items.filter(
      (item) =>
        !(
          config && nodeInfoKeysToHide.some((regex) => item.label.match(regex))
        ),
    );

    // Section for attrs.
    if (Object.keys(opNode.attrs || {}).length > 0) {
      const attrSection: InfoSection = {
        label: SectionLabel.ATTRIBUTES,
        sectionType: 'op',
        items: [],
      };
      const attrs = opNode.attrs || {};
      for (const key of Object.keys(attrs)) {
        // Ignore reserved keys.
        if (key.startsWith('__')) {
          continue;
        }
        const value = attrs[key];
        const strValue = typeof value === 'string' ? value : '';
        const specialValue: SpecialNodeAttributeValue | undefined =
          typeof value === 'string' ? undefined : value;
        attrSection.items.push({
          section: attrSection,
          label: key,
          value: strValue,
          canShowOnNode: true,
          showOnNode: this.curShowOnOpNodeAttrIds.has(key),
          specialValue,
        });
      }
      if (attrSection.items.length > 0) {
        this.sections.push(attrSection);
      }
    }

    // Section for node data providers.
    const runs = this.nodeDataProviderExtensionService.getRunsForModelGraph(
      this.curModelGraph,
    );
    if (runs.length > 0) {
      const nodeDataProvidersSection: InfoSection = {
        label: SectionLabel.NODE_DATA_PROVIDERS,
        sectionType: 'op',
        items: [],
      };
      this.sections.push(nodeDataProvidersSection);
      for (const run of runs) {
        const nodeResult = ((run.results || {})[this.curModelGraph.id] || {})[
          opNode.id
        ];
        if (this.appService.config()?.hideEmptyNodeDataEntries && !nodeResult) {
          continue;
        }
        const strValue = nodeResult?.strValue || '-';
        const bgColor = nodeResult?.bgColor || 'transparent';
        const textColor = nodeResult?.textColor || 'black';
        nodeDataProvidersSection.items.push({
          id: run.runId,
          section: nodeDataProvidersSection,
          label: getRunName(run, this.curModelGraph),
          value: strValue,
          canShowOnNode: run.done,
          showOnNode: this.curShowOnNodeDataProviderRuns[run.runId] != null,
          bgColor,
          textColor,
          loading: !run.done,
        });
      }
    }
  }

  private genInputsOutputsData() {
    if (!this.curModelGraph || !this.curSelectedNode) {
      return;
    }

    // Inputs.
    const selectedOpNode = this.curSelectedNode as OpNode;
    const incomingEdges = selectedOpNode.incomingEdges || [];

    this.inputItems = [];
    for (let i = 0; i < incomingEdges.length; i++) {
      const edge = incomingEdges[i];
      const metadataList = this.genInputMetadataList(selectedOpNode, edge);
      const sourceOpNode = this.curModelGraph?.nodesById[
        edge.sourceNodeId
      ] as OpNode;
      this.inputItems.push({
        index: i,
        opNode: sourceOpNode,
        metadataList,
      });
    }

    this.curInputsCount = this.inputItems.length;
    this.inputItemsForCurPage = this.inputItems.slice(0, this.ioPageSize);

    // Outputs.
    this.outputItems = [];
    const outputsMetadata = selectedOpNode.outputsMetadata || {};
    const outgoingEdges = selectedOpNode.outgoingEdges || [];
    let index = 0;
    for (const outputId of Object.keys(outputsMetadata)) {
      // The connected nodes.
      const connectedNodes = outgoingEdges
        .filter((edge) => edge.sourceNodeOutputId === outputId)
        .map(
          (edge) => this.curModelGraph!.nodesById[edge.targetNodeId],
        ) as OpNode[];

      // Metadata list.
      const {metadataList, tensorTag} = this.genOutputMetadataList(
        outgoingEdges,
        outputsMetadata[outputId],
        connectedNodes,
      );

      this.outputItems.push({
        index,
        tensorTag,
        outputId,
        sourceOpNode: selectedOpNode,
        metadataList,
      });
      index++;
    }
    this.curOutputsCount = this.outputItems.length;
    this.outputItemsForCurPage = this.outputItems.slice(0, this.ioPageSize);
  }

  private genGroupInputsOutputsData() {
    if (!this.curModelGraph || !this.curSelectedNode) {
      return;
    }

    // Inputs.
    const selectedGroupNode = this.curSelectedNode as GroupNode;
    const seenInputNodeIds = new Set<string>();

    this.groupInputItems = [];
    let index = 0;
    for (const nodeId of selectedGroupNode.descendantsOpNodeIds || []) {
      const descendantIds = new Set<string>(
        selectedGroupNode.descendantsOpNodeIds || [],
      );
      const opNode = this.curModelGraph?.nodesById[nodeId] as OpNode;
      const incomingEdges = opNode.incomingEdges || [];

      for (const edge of incomingEdges) {
        const sourceOpNode = this.curModelGraph?.nodesById[
          edge.sourceNodeId
        ] as OpNode;

        // Ignore if the source op node is within the layer.
        if (descendantIds.has(sourceOpNode.id)) {
          continue;
        }

        // Dedup.
        if (seenInputNodeIds.has(sourceOpNode.id)) {
          continue;
        }
        seenInputNodeIds.add(sourceOpNode.id);

        const metadataList = this.genInputMetadataList(opNode, edge);
        this.groupInputItems.push({
          index: index++,
          opNode: sourceOpNode,
          metadataList,
          targetOpNode: opNode,
        });
      }
    }

    this.curGroupInputsCount = this.groupInputItems.length;
    this.groupInputItemsForCurPage = this.groupInputItems.slice(
      0,
      this.ioPageSize,
    );

    // Outputs.
    this.groupOutputItems = [];
    index = 0;
    for (const nodeId of selectedGroupNode.descendantsOpNodeIds || []) {
      const descendantIds = new Set<string>(
        selectedGroupNode.descendantsOpNodeIds || [],
      );
      const opNode = this.curModelGraph?.nodesById[nodeId] as OpNode;

      const outputsMetadata = opNode.outputsMetadata || {};
      const outgoingEdges = opNode.outgoingEdges || [];
      for (const outputId of Object.keys(outputsMetadata)) {
        // The connected nodes.
        const connectedNodes = outgoingEdges
          .filter((edge) => !descendantIds.has(edge.targetNodeId))
          .filter((edge) => edge.sourceNodeOutputId === outputId)
          .map(
            (edge) => this.curModelGraph!.nodesById[edge.targetNodeId],
          ) as OpNode[];
        if (connectedNodes.length === 0) {
          continue;
        }

        // Metadata list.
        const {metadataList, tensorTag} = this.genOutputMetadataList(
          outgoingEdges,
          outputsMetadata[outputId],
          connectedNodes,
        );

        this.groupOutputItems.push({
          index,
          tensorTag,
          outputId,
          sourceOpNode: opNode,
          metadataList,
          showSourceOpNode: true,
        });
        index++;
      }
    }

    this.curGroupOutputsCount = this.groupOutputItems.length;
    this.groupOutputItemsForCurPage = this.groupOutputItems.slice(
      0,
      this.ioPageSize,
    );
  }

  private genInputMetadataList(
    opNode: OpNode,
    edge: IncomingEdge,
  ): KeyValueList {
    const sourceOpNode = this.curModelGraph?.nodesById[
      edge.sourceNodeId
    ] as OpNode;
    const metadata =
      (opNode.inputsMetadata || {})[edge.targetNodeInputId] || {};
    // Merge the corresponding output metadata with the current input
    // metadata.
    const sourceNodeOutputMetadata = {
      ...((sourceOpNode.outputsMetadata || {})[edge.sourceNodeOutputId] || {}),
    };
    for (const key of Object.keys(sourceNodeOutputMetadata)) {
      if (metadata[key] == null && key !== TENSOR_TAG_METADATA_KEY) {
        metadata[key] = sourceNodeOutputMetadata[key];
      }
    }
    // Sort by key.
    const metadataList: KeyValueList = [];
    Object.entries(metadata).forEach(([key, value]) => {
      metadataList.push({key, value});
    });
    metadataList.sort((a, b) => a.key.localeCompare(b.key));
    // Add namespace to metadata.
    metadataList.push({
      key: this.inputMetadataNamespaceKey,
      value: getNamespaceLabel(sourceOpNode),
    });
    // Add tensor values to metadata if existed.
    const attrs = sourceOpNode.attrs || {};
    if (attrs[TENSOR_VALUES_KEY]) {
      const value = attrs[TENSOR_VALUES_KEY];
      if (typeof value === 'string') {
        metadataList.push({
          key: this.inputMetadataValuesKey,
          value,
        });
      }
    }

    // Filter out hidden input metadata keys.
    const config = this.appService.config();
    const inputMetadataKeysToHide = config?.inputMetadataKeysToHide ?? [];
    return metadataList.filter((item) => {
      return !(
        config && inputMetadataKeysToHide.some((regex) => item.key.match(regex))
      );
    });
  }

  private genOutputMetadataList(
    outgoingEdges: OutgoingEdge[],
    outputMetadata: KeyValuePairs,
    connectedNodes: OpNode[],
  ) {
    let metadataList: OutputItemMetadata[] = [];
    let tensorTag = '';
    for (const metadataKey of Object.keys(outputMetadata)) {
      const value = outputMetadata[metadataKey];
      if (metadataKey === TENSOR_TAG_METADATA_KEY) {
        tensorTag = value;
      }
      // Hide all metadata keys that start with '__'.
      if (metadataKey.startsWith('__')) {
        continue;
      }
      metadataList.push({
        key: metadataKey,
        value,
      });
    }
    metadataList.sort((a, b) => a.key.localeCompare(b.key));

    metadataList.push({
      key: this.outputMetadataConnectedTo,
      value: '',
      connectedNodes,
    });

    // Filter out hidden output metadata keys.
    const config = this.appService.config();
    const outputMetadataKeysToHide = config?.outputMetadataKeysToHide ?? [];
    metadataList = metadataList.filter((item) => {
      return !(
        config &&
        outputMetadataKeysToHide.some((regex) => item.key.match(regex))
      );
    });

    return {metadataList, tensorTag};
  }

  private genInfoDataForSelectedGroupNode() {
    if (!this.curModelGraph || !this.curSelectedNode) {
      return;
    }

    const groupNode = this.curSelectedNode as GroupNode;

    // Section for basic node data.
    const nodeSection: InfoSection = {
      label: SectionLabel.LAYER_INFO,
      sectionType: 'group',
      items: [],
    };
    this.sections.push(nodeSection);

    // Label.
    let label = 'name';
    nodeSection.items.push({
      section: nodeSection,
      label: 'name',
      value: groupNode.label,
    });
    // Namespace.
    label = 'namespace';
    nodeSection.items.push({
      section: nodeSection,
      label,
      value: getNamespaceLabel(groupNode),
      canShowOnNode: true,
      showOnNode: this.curShowOnGroupNodeInfoIds.has(label),
    });
    // Number of direct child nodes.
    label = '#children';
    nodeSection.items.push({
      section: nodeSection,
      label,
      value: String((groupNode.nsChildrenIds || []).length),
      canShowOnNode: true,
      showOnNode: this.curShowOnGroupNodeInfoIds.has(label),
    });
    // Number of descendant op nodes.
    label = '#descendants';
    nodeSection.items.push({
      section: nodeSection,
      label,
      value: String((groupNode.descendantsNodeIds || []).length),
      canShowOnNode: true,
      showOnNode: this.curShowOnGroupNodeInfoIds.has(label),
    });

    // Filter out hidden node info keys.
    const config = this.appService.config();
    const nodeInfoKeysToHide = config?.nodeInfoKeysToHide ?? [];
    nodeSection.items = nodeSection.items.filter((item) => {
      return !(
        config && nodeInfoKeysToHide.some((regex) => item.label.match(regex))
      );
    });

    // Section for custom attributes.
    const groupAttributes =
      this.curModelGraph.groupNodeAttributes?.[
        groupNode.id.replace('___group___', '')
      ];
    if (groupAttributes) {
      const attrsSection: InfoSection = {
        label: SectionLabel.LAYER_ATTRS,
        sectionType: 'group',
        items: [],
      };
      this.sections.push(attrsSection);
      for (const key of Object.keys(groupAttributes)) {
        attrsSection.items.push({
          section: nodeSection,
          label: key,
          value: groupAttributes[key],
        });
      }
    }

    // Section for identical groups.
    if (groupNode.identicalGroupIndex != null) {
      this.identicalGroupNodes = this.curModelGraph.nodes.filter(
        (node) =>
          isGroupNode(node) &&
          node.identicalGroupIndex === groupNode.identicalGroupIndex,
      ) as GroupNode[];
      this.identicalGroupsData = genIoTreeData(
        this.identicalGroupNodes.slice(0, this.ioPageSize),
        [],
        'incoming',
        groupNode.id,
      );
    }
  }

  private handleSearchResultsChanged() {
    if (!this.curSelectedNode || !this.curSearchResults) {
      return;
    }

    // Collect different types of matches.
    const nodeId = this.curSelectedNode.id;
    const nodeSearchResults = this.curSearchResults.results[nodeId] || [];
    const inputMatches: SearchMatchInputMetadata[] = [];
    const outputMatches: SearchMatchOutputMetadata[] = [];
    const attrMatches: SearchMatchAttr[] = [];
    for (const match of nodeSearchResults) {
      switch (match.type) {
        case SearchMatchType.INPUT_METADATA:
          inputMatches.push(match);
          break;
        case SearchMatchType.OUTPUT_METADATA:
          outputMatches.push(match);
          break;
        case SearchMatchType.ATTRIBUTE:
          attrMatches.push(match);
          break;
        default:
          break;
      }
    }

    // Update info panel itself to highlight matched attributes.
    this.curSearchAttrMatches = attrMatches;
    this.curSearchOutputMatches = outputMatches;
    this.curSearchInputMatches = inputMatches;
    this.changeDetectorRef.markForCheck();
  }

  private animateSidePanelWidth(
    targetWidth: number,
    duration = SIDE_PANEL_WIDTH_ANIMATION_DURATION,
  ) {
    const startTs = Date.now();
    const startWidth = this.width;
    const animate = () => {
      const elapsed = Date.now() - startTs;
      let t = this.appService.testMode ? 1 : Math.min(1, elapsed / duration);
      // ease out sine.
      t = Math.sin((t * Math.PI) / 2);
      const curWidth = startWidth + (targetWidth - startWidth) * t;
      this.width = curWidth;
      this.minWidth = curWidth;
      this.changeDetectorRef.markForCheck();

      if (t >= 1) {
        this.width = targetWidth;
        this.minWidth = targetWidth;
        this.changeDetectorRef.markForCheck();
        return;
      }

      requestAnimationFrame(animate);
    };
    animate();
  }

  private updateInputValueContentsExpandable() {
    for (let i = 0; i < this.inputValueContents.length; i++) {
      const valueContent = this.inputValueContents.get(i)?.nativeElement;
      if (!valueContent) {
        continue;
      }
      if (valueContent.scrollHeight > valueContent.offsetHeight) {
        valueContent.classList.add('expandable');
      }
    }
  }
}
