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

import {OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  Input,
  OnChanges,
  SimpleChanges,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

import {BubbleClick} from '../bubble/bubble_click';

import {ModelNode} from './common/model_graph';
import {genIoTreeData, IoTree, TreeNode} from './io_tree';

/**
 * A clickable button that shows the given list of nodes in a tree.
 */
@Component({
  standalone: true,
  selector: 'node-list-viewer',
  imports: [BubbleClick, CommonModule, IoTree, MatIconModule],
  templateUrl: './node_list_viewer.ng.html',
  styleUrls: ['./node_list_viewer.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NodeListViewer implements OnChanges {
  @Input() nodes: ModelNode[] = [];
  @Input() rendererId = '';
  @Input() labelSuffix = 'node';

  readonly popupSize: OverlaySizeConfig = {
    minWidth: 320,
    maxWidth: 640,
    minHeight: 0,
  };

  curIoTreeData: TreeNode[] = [];

  ngOnChanges(changes: SimpleChanges) {
    if (changes['nodes']) {
      this.curIoTreeData = genIoTreeData(this.nodes, [], 'incoming');
    }
  }

  get label(): string {
    const numNodes = this.nodes.length;
    return `${numNodes} ${this.labelSuffix}${numNodes === 1 ? '' : 's'}`;
  }
}
