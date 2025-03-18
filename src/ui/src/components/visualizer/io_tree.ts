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
import {FlatTreeControl} from '@angular/cdk/tree';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {
  MatTooltipModule,
  type TooltipPosition,
} from '@angular/material/tooltip';
import {
  MatTree,
  MatTreeFlatDataSource,
  MatTreeFlattener,
  MatTreeModule,
} from '@angular/material/tree';

import {Bubble} from '../bubble/bubble';

import {AppService} from './app_service';
import {TENSOR_TAG_METADATA_KEY, TENSOR_VALUES_KEY} from './common/consts';
import {ModelNode, NodeType, OpNode} from './common/model_graph';
import {
  KeyValuePairs,
  SearchMatches,
  SearchMatchInputMetadata,
  SearchMatchOutputMetadata,
  SearchMatchType,
} from './common/types';
import {isOpNode} from './common/utils';
import {MENU_ANIMATIONS} from './ui_utils';

/** Holds data for a node in the tree. */
export interface TreeNode {
  label: string;
  boldLabel?: string;
  nodeId?: string;
  node?: ModelNode;
  type: 'incoming' | 'outgoing';
  highlightGroupLabel?: boolean;
  children?: TreeNode[];
  showLocator?: boolean;
  highlight?: boolean;
  isGroupNode?: boolean;
  metadata?: KeyValuePairs;
  extraData?: SearchMatches;
}

interface FlatTreeNode {
  nodeId?: string;
  node?: ModelNode;
  expandable: boolean;
  label: string;
  boldLabel?: string;
  level: number;
  showLocator?: boolean;
  highlight?: boolean;
  metadata?: KeyValuePairs;
  extraData?: SearchMatches;
}

/** The tree that shows the inputs/outputs data. */
@Component({
  standalone: true,
  selector: 'io-tree',
  imports: [
    Bubble,
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    MatTreeModule,
  ],
  templateUrl: './io_tree.ng.html',
  styleUrls: ['./io_tree.scss'],
  animations: MENU_ANIMATIONS,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class IoTree implements OnChanges {
  @Input('data') data?: TreeNode[];
  @Input('solidBackground') solidBackground = false;
  @Input('rendererId') rendererId = '';
  @Input('tooltipPosition') tooltipPosition: TooltipPosition = 'left';
  @Input('showLocator') showLocator = true;
  @Input('colorBoldNodeLabel') colorBoldNodeLabel = false;
  @Output() readonly onClose = new EventEmitter<{}>();

  @ViewChild('tree') tree!: MatTree<TreeNode>;

  readonly constValuesPopupSize: OverlaySizeConfig = {
    minWidth: 100,
  };

  readonly constValuesPopupPosition: ConnectedPosition[] = [
    {
      originX: 'start',
      originY: 'top',
      overlayX: 'end',
      overlayY: 'top',
    },
  ];

  readonly locatorTooltip = 'Click: locate\nAlt+click: select';

  private readonly transformer = (
    node: TreeNode,
    level: number,
  ): FlatTreeNode => {
    return {
      nodeId: node.nodeId,
      node: node.node,
      expandable: !!node.children && node.children.length > 0,
      label: node.label,
      boldLabel: node.boldLabel,
      level,
      showLocator: node.showLocator,
      highlight: node.highlight,
      metadata: node.metadata,
      extraData: node.extraData,
    };
  };
  private curSearchMatches:
    | SearchMatchInputMetadata[]
    | SearchMatchOutputMetadata[] = [];

  readonly treeItemPadding = 8;

  treeControl = new FlatTreeControl<FlatTreeNode>(
    (node) => node.level,
    (node) => node.expandable,
  );

  treeFlattener = new MatTreeFlattener(
    this.transformer,
    (node) => node.level,
    (node) => node.expandable,
    (node) => node.children,
  );

  dataSource = new MatTreeFlatDataSource(this.treeControl, this.treeFlattener);

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
  ) {}

  ngOnChanges(changes: SimpleChanges) {
    if (this.data != null) {
      this.dataSource.data = this.data;
      this.treeControl.expandAll();
    }
  }

  updateData(data: TreeNode[]) {
    this.data = data;
    this.dataSource.data = this.data;
    this.treeControl.expandAll();
    this.changeDetectorRef.markForCheck();
  }

  updateSearchResults(
    matches: SearchMatchInputMetadata[] | SearchMatchOutputMetadata[],
  ) {
    this.curSearchMatches = matches;
    this.changeDetectorRef.markForCheck();
  }

  handleLocateNode(treeNode: FlatTreeNode, select: boolean) {
    if (!treeNode.showLocator) {
      return;
    }

    this.appService.curToLocateNodeInfo.set({
      nodeId: treeNode.nodeId || '',
      rendererId: this.rendererId,
      isGroupNode: treeNode.expandable,
      select,
    });
    this.onClose.emit({});
  }

  hasChild(unused: number, node: FlatTreeNode): boolean {
    return node.expandable;
  }

  hasMetadata(node: FlatTreeNode): boolean {
    return node.metadata != null && Object.keys(node.metadata).length > 0;
  }

  getSortedMetadataKeys(node: FlatTreeNode): string[] {
    return (
      Object.keys(node.metadata || {})
        // Hide all metadata keys that start with '__'.
        .filter((key) => !key.startsWith('__'))
        .sort()
    );
  }

  showHoverForValuesLabel(node: FlatTreeNode): boolean {
    const modelNode = node.node;
    if (!modelNode) {
      return false;
    }

    if (isOpNode(modelNode)) {
      const attrs = modelNode.attrs || {};
      if (attrs[TENSOR_VALUES_KEY]) {
        return attrs[TENSOR_VALUES_KEY] !== 'DATA_ELIDED';
      }
    }

    return false;
  }

  getMaxConstValueCount(): number {
    return this.appService.config()?.maxConstValueCount ?? 0;
  }

  getConstValues(node: FlatTreeNode): string {
    const modelNode = node.node;
    if (!modelNode) {
      return '';
    }

    if (isOpNode(modelNode)) {
      const attrs = modelNode.attrs || {};
      const value = attrs[TENSOR_VALUES_KEY];
      if (value && typeof value === 'string') {
        return value;
      }
      return '<empty>';
    }

    return '';
  }

  getExtraLabelTooltip(matchType: string) {
    switch (matchType) {
      case SearchMatchType.NODE_LABEL:
        return 'Node label matched';
      case SearchMatchType.ATTRIBUTE:
        return 'Node attribute(s) matched';
      case SearchMatchType.INPUT_METADATA:
        return 'Input(s) matched';
      case SearchMatchType.OUTPUT_METADATA:
        return 'Output(s) matched';
      default:
        return '';
    }
  }

  isSearchMatched(value: string, boldValue?: string): boolean {
    if (!this.curSearchMatches) {
      return false;
    }
    return (
      this.curSearchMatches.find(
        (match) =>
          match.matchedText === value || match.matchedText === boldValue,
      ) != null
    );
  }
}

/** Generates io tree data for given nodes. */
export function genIoTreeData(
  nodes: ModelNode[],
  metadataList: KeyValuePairs[],
  type: 'incoming' | 'outgoing',
  nodeIdToHighlight?: string,
  extraDataList: SearchMatches[] = [],
): TreeNode[] {
  let root: TreeNode = {
    label: '<root>',
    children: [],
    type,
    isGroupNode: true,
  };
  const hiddenNodes: Array<{
    node: OpNode;
    index: number;
    metadata: KeyValuePairs;
  }> = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const metadata = metadataList[i];
    const extraData = extraDataList[i];
    if (isOpNode(node) && node.hideInLayout) {
      hiddenNodes.push({node, index: i, metadata});
    } else {
      const curTreeNode = populateTreeStructureFromNamespace(
        node.savedNamespace || node.namespace,
        root,
      );
      if (curTreeNode) {
        if (curTreeNode.children == null) {
          curTreeNode.children = [];
        }
        const targetNode = curTreeNode.children.find(
          (child) => child.label === node.label && child.isGroupNode,
        );
        if (targetNode == null) {
          const curTreeNodeChild: TreeNode = {
            label: node.label,
            boldLabel: getTensorTag(metadata),
            nodeId: node.id,
            node,
            type,
            showLocator: true,
            highlight: node.id === nodeIdToHighlight,
            isGroupNode: node.nodeType === NodeType.GROUP_NODE,
            metadata,
          };
          if (extraData != null) {
            curTreeNodeChild.extraData = extraData;
          }
          curTreeNode.children.push(curTreeNodeChild);
        } else {
          targetNode.nodeId = node.id;
          targetNode.showLocator = true;
          targetNode.extraData = extraData;
        }
      }
    }
  }
  root = compressTreeStructure(root);
  const treeNodes: TreeNode[] = [root];

  // Add a sub-tree for hidden nodes.
  if (hiddenNodes.length > 0) {
    let rootLabel = 'weights';
    for (const item of hiddenNodes) {
      if (!item.node.label.toLowerCase().includes('const')) {
        rootLabel = '<hidden>';
        break;
      }
    }
    treeNodes.push({
      label: rootLabel,
      children: hiddenNodes.map((item) => {
        return {
          label: item.node.label,
          boldLabel: getTensorTag(item.metadata),
          nodeId: item.node.id,
          node: item.node,
          type,
          showLocator: false,
          highlight: item.node.id === nodeIdToHighlight,
          isGroupNode: false,
          metadata: metadataList[item.index],
        };
      }),
      type,
      isGroupNode: false,
    });
  }
  return treeNodes;
}

function getTensorTag(metadata: KeyValuePairs | undefined): string | undefined {
  if (!metadata) {
    return undefined;
  }
  return metadata[TENSOR_TAG_METADATA_KEY];
}

function populateTreeStructureFromNamespace(
  namespace: string,
  rootNode: TreeNode,
): TreeNode | undefined {
  const nsParts = !namespace ? ['<root>'] : ['<root>', ...namespace.split('/')];
  let curTreeNode: TreeNode | undefined;
  const curNsParts: string[] = [];
  for (const nsPart of nsParts) {
    if (nsPart !== '<root>') {
      curNsParts.push(nsPart);
    }

    if (nsPart === '<root>') {
      curTreeNode = rootNode;
    } else {
      if (curTreeNode && !curTreeNode.children) {
        curTreeNode.children = [];
      }
      const targetNode = curTreeNode!.children!.find(
        (child) => child.label === nsPart,
      );

      if (targetNode == null) {
        const treeNode: TreeNode = {
          label: nsPart,
          children: [],
          type: rootNode.type,
          isGroupNode: true,
        };
        curTreeNode!.children!.push(treeNode);
        curTreeNode = treeNode;
      } else {
        curTreeNode = targetNode;
      }
    }
  }

  return curTreeNode;
}

function compressTreeStructure(root: TreeNode): TreeNode {
  // Turn a tree such as:
  //
  // [a]
  //   [b]
  //     [c]
  //       x1
  //       x2
  //
  // into:
  //
  // [a/b/c]
  //   x1
  //   x2
  let curNode = root;
  const seenNsParts: string[] = [];
  while (true) {
    seenNsParts.push(curNode.label);
    if (
      curNode?.children &&
      curNode.children.length === 1 &&
      (curNode.children[0].children || []).length > 0 &&
      !curNode.showLocator
    ) {
      curNode = curNode.children[0];
    } else {
      break;
    }
  }
  if (curNode !== root) {
    curNode.label = seenNsParts.join(' / ');
  }
  return curNode;
}
