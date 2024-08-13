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
  effect,
  ElementRef,
  HostBinding,
  Input,
  QueryList,
  ViewChildren,
} from '@angular/core';
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
  KeyValue,
  KeyValueList,
  KeyValuePairs,
  NodeDataProviderRunInfo,
  SearchMatchAttr,
  SearchMatchInputMetadata,
  SearchMatchOutputMetadata,
  SearchMatchType,
  SearchResults,
} from './common/types';
import {getNamespaceLabel, isGroupNode, isOpNode} from './common/utils';
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
}

interface OutputItem {
  index: number;
  tensorTag: string;
  outputId: string;
  metadataList: OutputItemMetadata[];
}

interface OutputItemMetadata extends KeyValue {
  connectedNodes?: OpNode[];
}

interface FlatInputItem {
  index: number;
  opNode: OpNode;
  metadataList: KeyValueList;
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
  flatInputItems: FlatInputItem[] = [];
  outputItems: OutputItem[] = [];
  outputItemsForCurPage: OutputItem[] = [];
  identicalGroupNodes: GroupNode[] = [];
  identicalGroupsData?: TreeNode[];
  curRendererId = '';
  curInputsCount = 0;
  curOutputsCount = 0;
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
  private inputSourceNodes: OpNode[] = [];
  private inputMetadataList: KeyValuePairs[] = [];
  private savedWidth = 0;

  constructor(
    private readonly appService: AppService,
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
    const curInputSourceNodes = this.inputSourceNodes.slice(
      curPageIndex * this.ioPageSize,
      (curPageIndex + 1) * this.ioPageSize,
    );
    const curInputMetadataList = this.inputMetadataList.slice(
      curPageIndex * this.ioPageSize,
      (curPageIndex + 1) * this.ioPageSize,
    );
    this.flatInputItems = this.genInputFlatItems(
      curPageIndex * this.ioPageSize,
      curInputSourceNodes,
      curInputMetadataList,
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

  handleToggleSection(sectionName: string, sectionEle?: HTMLElement) {
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

  isSectionCollapsed(sectionName: string): boolean {
    return this.infoPanelService.collapsedSectionNames.has(sectionName);
  }

  getSectionToggleIcon(sectionName: string): string {
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
    allItems: FlatInputItem[],
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
    outputId: string,
    items: OutputItem[],
    event: MouseEvent,
  ) {
    event.stopPropagation();

    if (event.altKey) {
      this.splitPaneService.setOutputVisible(
        outputId,
        items.map((item) => item.outputId),
      );
    } else {
      this.splitPaneService.toggleOutputVisibility(outputId);
    }
  }

  getOutputToggleVisible(outputId: string): boolean {
    return this.splitPaneService.getOutputVisible(outputId);
  }

  getOutputToggleVisibilityIcon(outputId: string): string {
    return this.getOutputToggleVisible(outputId)
      ? 'visibility'
      : 'visibility_off';
  }

  getOutputToggleVisibilityTooltip(outputId: string): string {
    return this.getOutputToggleVisible(outputId)
      ? 'Click to hide highlight'
      : 'Click to show highlight';
  }

  getInputName(item: FlatInputItem): string {
    const tensorTagItem = item.metadataList.find(
      (item) => item.key === TENSOR_TAG_METADATA_KEY,
    );

    return tensorTagItem
      ? `${tensorTagItem.value} (${item.opNode.label})`
      : item.opNode.label;
  }

  getInputTensorTag(item: FlatInputItem): string {
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

  trackBySectionLabel(index: number, section: InfoSection): string {
    return section.label;
  }

  trackByItemIdOrLabel(index: number, item: InfoItem): string {
    return item.id || item.label;
  }

  get canShowGraphInfo(): boolean {
    return this.curModelGraph != null && this.curSelectedNode == null;
  }

  get showNodeDataProviderSummary(): boolean {
    if (!this.curModelGraph) {
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
      this.inputSourceNodes.length > this.ioPageSize &&
      !this.isSectionCollapsed(SectionLabel.INPUTS)
    );
  }

  get showOutputPaginator(): boolean {
    return (
      this.outputItems.length > this.ioPageSize &&
      !this.isSectionCollapsed(SectionLabel.OUTPUTS)
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
    this.flatInputItems = [];
    this.inputSourceNodes = [];
    this.inputMetadataList = [];
    this.outputItems = [];
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
  }

  private genInfoDataForSelectedOpNode() {
    if (!this.curModelGraph || !this.curSelectedNode) {
      return;
    }

    const opNode = this.curSelectedNode as OpNode;

    // Section for basic node data.
    const nodeSection: InfoSection = {
      label: SectionLabel.NODE_INFO,
      sectionType: 'op',
      items: [],
    };
    this.sections.push(nodeSection);
    // Node op.
    nodeSection.items.push({
      section: nodeSection,
      label: 'op name',
      value: `${opNode.label}`,
    });
    // Node id.
    let label = 'id';
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
        attrSection.items.push({
          section: attrSection,
          label: key,
          value: attrs[key],
          canShowOnNode: true,
          showOnNode: this.curShowOnOpNodeAttrIds.has(key),
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
        const strValue = nodeResult?.strValue || '-';
        const bgColor = nodeResult?.bgColor || 'transparent';
        const textColor = nodeResult?.textColor || 'black';
        nodeDataProvidersSection.items.push({
          id: run.runId,
          section: nodeDataProvidersSection,
          label: run.runName,
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

    this.inputMetadataList = [];
    this.inputSourceNodes = [];
    this.flatInputItems = [];
    for (const edge of incomingEdges) {
      const sourceOpNode = this.curModelGraph?.nodesById[
        edge.sourceNodeId
      ] as OpNode;
      this.inputSourceNodes.push(sourceOpNode);
      const metadata =
        (selectedOpNode.inputsMetadata || {})[edge.targetNodeInputId] || {};
      // Merge the corresponding output metadata with the current input
      // metadata.
      const sourceNodeOutputMetadata = {
        ...((sourceOpNode.outputsMetadata || {})[edge.sourceNodeOutputId] ||
          {}),
      };
      for (const key of Object.keys(sourceNodeOutputMetadata)) {
        if (metadata[key] == null && key !== TENSOR_TAG_METADATA_KEY) {
          metadata[key] = sourceNodeOutputMetadata[key];
        }
      }
      this.inputMetadataList.push(metadata);
    }
    this.curInputsCount = this.inputSourceNodes.length;
    if (incomingEdges.length > 0) {
      const curInputSourceNodes = this.inputSourceNodes.slice(
        0,
        this.ioPageSize,
      );
      const curInputMetadataList = this.inputMetadataList.slice(
        0,
        this.ioPageSize,
      );
      this.flatInputItems = this.genInputFlatItems(
        0,
        curInputSourceNodes,
        curInputMetadataList,
      );
    }

    // Outputs.
    this.outputItems = [];
    const outputsMetadata = selectedOpNode.outputsMetadata || {};
    const outgoingEdges = selectedOpNode.outgoingEdges || [];
    let index = 0;
    for (const outputId of Object.keys(outputsMetadata)) {
      // The metadata for the current output tensor.
      const metadataList: OutputItemMetadata[] = [];
      let tensorTag = '';
      for (const metadataKey of Object.keys(outputsMetadata[outputId])) {
        const value = outputsMetadata[outputId][metadataKey];
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

      // The connected nodes.
      const connectedNodes = outgoingEdges
        .filter((edge) => edge.sourceNodeOutputId === outputId)
        .map(
          (edge) => this.curModelGraph!.nodesById[edge.targetNodeId],
        ) as OpNode[];
      metadataList.push({
        key: this.outputMetadataConnectedTo,
        value: '',
        connectedNodes,
      });
      this.outputItems.push({
        index,
        tensorTag,
        outputId,
        metadataList,
      });
      index++;
    }
    this.curOutputsCount = this.outputItems.length;
    this.outputItemsForCurPage = this.outputItems.slice(0, this.ioPageSize);
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
    nodeSection.items.push({
      section: nodeSection,
      label: 'name',
      value: groupNode.label,
    });
    // Namespace.
    let label = 'namespace';
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

  private animateSidePanelWidth(targetWidth: number) {
    const startTs = Date.now();
    const startWidth = this.width;
    const animate = () => {
      const elapsed = Date.now() - startTs;
      let t = this.appService.testMode
        ? 1
        : Math.min(1, elapsed / SIDE_PANEL_WIDTH_ANIMATION_DURATION);
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

  private genInputFlatItems(
    startIndex: number,
    inputSourceNodes: OpNode[],
    inputMetadataList: KeyValuePairs[],
  ): FlatInputItem[] {
    const flatInputItems: FlatInputItem[] = [];
    for (let i = 0; i < inputSourceNodes.length; i++) {
      const sourceNode = inputSourceNodes[i];
      const metadataList: KeyValueList = [];
      Object.entries(inputMetadataList[i]).forEach(([key, value]) => {
        metadataList.push({key, value});
      });
      metadataList.sort((a, b) => a.key.localeCompare(b.key));
      metadataList.push({
        key: this.inputMetadataNamespaceKey,
        value: getNamespaceLabel(inputSourceNodes[i]),
      });
      const attrs = sourceNode.attrs || {};
      if (attrs[TENSOR_VALUES_KEY]) {
        metadataList.push({
          key: this.inputMetadataValuesKey,
          value: attrs[TENSOR_VALUES_KEY],
        });
      }
      flatInputItems.push({
        index: i + startIndex,
        opNode: sourceNode,
        metadataList,
      });
    }
    return flatInputItems;
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
