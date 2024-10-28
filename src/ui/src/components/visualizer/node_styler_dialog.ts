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

import {DragDropModule} from '@angular/cdk/drag-drop';
import {OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  computed,
  effect,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';
import {setAnchorHref} from 'safevalues/dom';

import {Bubble} from '../bubble/bubble';

import {AppService} from './app_service';
import {COLOR_NAME_TO_HEX} from './common/consts';
import {ModelNode} from './common/model_graph';
import {
  NodeQuery,
  NodeQueryType,
  NodeStylerRule,
  SearchMatchType,
  SearchNodeType,
} from './common/types';
import {getNodeStyleValue} from './common/utils';
import {ComplexQueries} from './complex_queries';
import {NodeListViewer} from './node_list_viewer';
import {
  ALL_STYLES,
  NodeStylerService,
  Style,
  StyleType,
} from './node_styler_service';

interface SearchMatchTypeOption {
  type: SearchMatchType;
  tooltip: string;
}

interface SearchNodeTypeOption {
  type: SearchNodeType;
  label: string;
}

/**
 * The dialog where users set rules to style nodes.
 */
@Component({
  standalone: true,
  selector: 'node-styler-dialog',
  imports: [
    Bubble,
    CommonModule,
    ComplexQueries,
    DragDropModule,
    MatButtonModule,
    MatDialogModule,
    MatIconModule,
    MatMenuModule,
    MatTooltipModule,
    NodeListViewer,
  ],
  templateUrl: './node_styler_dialog.ng.html',
  styleUrls: ['./node_styler_dialog.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NodeStylerDialog {
  readonly rules;
  readonly hasNonEmptyNodeStylerRules;
  readonly hasRules = computed(() => this.rules().length > 0);
  readonly StyleType = StyleType;
  readonly NodeStylerQueryType = NodeQueryType;

  readonly allStyles = ALL_STYLES;
  readonly allSearchMatchTypeOptions: SearchMatchTypeOption[] = [
    {
      type: SearchMatchType.NODE_LABEL,
      tooltip: 'Match label',
    },
    {
      type: SearchMatchType.ATTRIBUTE,
      tooltip: 'Match attributes',
    },
    {
      type: SearchMatchType.INPUT_METADATA,
      tooltip: 'Match inputs (op node only)',
    },
    {
      type: SearchMatchType.OUTPUT_METADATA,
      tooltip: 'Match outputs (op node only)',
    },
  ];
  readonly allSearchNodeTypes: SearchNodeTypeOption[] = [
    {
      type: SearchNodeType.OP_NODES,
      label: 'Ops only',
    },
    {type: SearchNodeType.LAYER_NODES, label: 'Layers only'},
    {type: SearchNodeType.OP_AND_LAYER_NODES, label: 'Ops & layers'},
  ];
  readonly allQueryTypes = [
    {type: NodeQueryType.REGEX, label: 'Regex'},
    {
      type: NodeQueryType.ATTR_VALUE_RANGE,
      label: 'Attribute value range',
    },
  ];
  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
    maxWidth: 340,
  };

  private curMatchedNodes: Record<number, Record<number, ModelNode[]>> = {};

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly nodeStylerService: NodeStylerService,
  ) {
    this.rules = this.nodeStylerService.rules;
    this.hasNonEmptyNodeStylerRules =
      this.nodeStylerService.hasNonEmptyNodeStylerRules;
    effect(() => {
      this.curMatchedNodes = this.nodeStylerService.matchedNodes();
      this.changeDetectorRef.markForCheck();
    });
  }

  handleClickAddRule() {
    this.nodeStylerService.addNewRule();
  }

  handleClickExportRules() {
    const link = document.createElement('a');
    link.download = 'node_styler_rules.json';
    const dataUrl = `data:text/json;charset=utf-8, ${encodeURIComponent(
      JSON.stringify(this.rules(), null, 2),
    )}`;
    setAnchorHref(link, dataUrl);
    link.click();
  }

  handleClickImportRules(input: HTMLInputElement) {
    if (!input.files || input.files.length === 0) {
      return;
    }

    const fileReader = new FileReader();
    fileReader.onload = (event) => {
      const rules = JSON.parse(
        event.target?.result as string,
      ) as NodeStylerRule[];
      this.nodeStylerService.updateRules(rules);
    };
    fileReader.readAsText(input.files[0]);
  }

  handleQueriesUpdated(ruleIndex: number, queries: NodeQuery[]) {
    this.nodeStylerService.updateQueries(ruleIndex, queries);
  }

  handleToggleStyle(index: number, style: Style, checked?: boolean) {
    this.nodeStylerService.toggleStyle(index, style, checked);
  }

  handleStyleColorChanged(index: number, style: Style, color: string) {
    let hexColor = color;
    if (color.startsWith('rgb')) {
      hexColor = this.rgbToHex(color);
    } else if (!color.startsWith('#')) {
      hexColor = COLOR_NAME_TO_HEX[color] || color;
    }
    this.nodeStylerService.updateStyleValue(index, style, hexColor);
  }

  handleNumberChanged(index: number, style: Style, strNumber: string) {
    let n = Number(strNumber);
    if (isNaN(n)) {
      return;
    }
    n = Math.min(10, Math.max(0.001, n));
    this.nodeStylerService.updateStyleValue(index, style, `${n}`);
  }

  handleMoveUpRule(index: number) {
    this.nodeStylerService.moveUpRule(index);
  }

  handleMoveDownRule(index: number) {
    this.nodeStylerService.moveDownRule(index);
  }

  handleDuplicateRule(index: number) {
    this.nodeStylerService.duplicateRule(index);
  }

  handleDeleteRule(index: number) {
    this.nodeStylerService.deleteRule(index);
  }

  getIsStyleEnabled(rule: NodeStylerRule, style: Style): boolean {
    return rule.styles[style.id] != null;
  }

  getSerializedStyleValue(rule: NodeStylerRule, style: Style): string {
    return getNodeStyleValue(rule, style.id);
  }

  getMatchedNodes(ruleIndex: number, paneIndex: number): ModelNode[] {
    return (this.curMatchedNodes[ruleIndex] || {})[paneIndex] || [];
  }

  get panesCount(): number {
    return this.appService.panes().length;
  }

  get leftPaneRendererId(): string {
    return this.appService.panes()[0].id;
  }

  get rightPaneRendererId(): string {
    return this.appService.panes()[1].id;
  }

  private rgbToHex(rgb: string) {
    const match = rgb.match(
      /^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+\.{0,1}\d*))?\)$/,
    );
    if (match) {
      const hex = match
        .slice(1, 4)
        .map((n) => Number(n).toString(16).padStart(2, '0'))
        .join('');

      if (hex[0] === hex[1] && hex[2] === hex[3] && hex[4] === hex[5]) {
        return `#${hex[0]}${hex[2]}${hex[4]}`;
      } else {
        return `#${hex}`;
      }
    }
    return 'unknown';
  }
}
