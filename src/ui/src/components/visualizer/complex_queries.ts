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
  EventEmitter,
  Input,
  OnInit,
  Output,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ModelNode} from './common/model_graph';
import {
  NodeAttrValueRangeQuery,
  NodeQuery,
  NodeQueryType,
  NodeRegexQuery,
  NodeTypeQuery,
  SearchMatchType,
  SearchNodeType,
} from './common/types';

interface SearchMatchTypeOption {
  type: SearchMatchType;
  tooltip: string;
}

interface SearchNodeTypeOption {
  type: SearchNodeType;
  label: string;
}

/**
 * A component where users can specify complex queries and see matching results.
 */
@Component({
  standalone: true,
  selector: 'complex-queries',
  imports: [CommonModule, MatIconModule, MatMenuModule, MatTooltipModule],
  templateUrl: './complex_queries.ng.html',
  styleUrls: ['./complex_queries.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ComplexQueries implements OnInit {
  @Input({required: true}) queries: NodeQuery[] = [];

  @Output() readonly queriesUpdated = new EventEmitter<NodeQuery[]>();

  readonly NodeQueryType = NodeQueryType;

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
      label: 'Op nodes only',
    },
    {type: SearchNodeType.LAYER_NODES, label: 'Layer nodes only'},
    {type: SearchNodeType.OP_AND_LAYER_NODES, label: 'Op and layer nodes'},
  ];
  readonly allQueryTypes = [
    {type: NodeQueryType.REGEX, label: 'Regex'},
    {
      type: NodeQueryType.ATTR_VALUE_RANGE,
      label: 'Attribute value range',
    },
  ];

  curQueries: NodeQuery[] = [];
  curMatchedNodes: Record<number, ModelNode[]> = {};

  constructor(private readonly changeDetectorRef: ChangeDetectorRef) {}

  ngOnInit() {
    this.curQueries = JSON.parse(JSON.stringify(this.queries)) as NodeQuery[];
  }

  setMatchedNodes(matchedNodes: Record<number, ModelNode[]>) {
    this.curMatchedNodes = matchedNodes;
    this.changeDetectorRef.markForCheck();
  }

  handleRegexQueryChanged(queryIndex: number, regex: string) {
    // Update query.
    const query = this.curQueries[queryIndex] as NodeRegexQuery;
    query.queryRegex = regex.trim();

    // Notify changes.
    this.queriesUpdated.emit([...this.curQueries]);
  }

  handleAttrValueRangeQueryChanged(
    queryIndex: number,
    attrName: string,
    strMin: string,
    strMax: string,
  ) {
    // Update query.
    let min = Number.NEGATIVE_INFINITY;
    let max = Number.POSITIVE_INFINITY;
    if (strMin !== '' && !isNaN(Number(strMin))) {
      min = Number(strMin);
    }
    if (strMax !== '' && !isNaN(Number(strMax))) {
      max = Number(strMax);
    }
    const query = this.curQueries[queryIndex] as NodeAttrValueRangeQuery;
    query.attrName = attrName;
    query.min = min;
    query.max = max;

    // Notify changes.
    this.queriesUpdated.emit([...this.curQueries]);
  }

  handleNodeTypeChanged(queryIndex: number, nodeType: string) {
    // Update.
    const query = this.curQueries[queryIndex] as NodeTypeQuery;
    query.nodeType = nodeType as SearchNodeType;

    // Notify changes.
    this.queriesUpdated.emit([...this.curQueries]);
  }

  handleToggleMatchType(queryIndex: number, matchType: SearchMatchType) {
    if (
      this.getDisableMatchType(
        this.curQueries[queryIndex] as NodeRegexQuery,
        matchType,
      )
    ) {
      return;
    }

    // Update query.
    const query = this.curQueries[queryIndex] as NodeRegexQuery;
    const matchTypeIndex = query.matchTypes.indexOf(matchType);
    if (matchTypeIndex >= 0) {
      query.matchTypes.splice(matchTypeIndex, 1);
    } else {
      query.matchTypes.push(matchType);
    }

    // Notify changes.
    this.queriesUpdated.emit([...this.curQueries]);
  }

  handleDeleteQuery(queryIndex: number) {
    // Delete query.
    this.curQueries.splice(queryIndex, 1);

    // Notify changes.
    this.queriesUpdated.emit([...this.curQueries]);
  }

  handleAddQuery(queryType: NodeQueryType) {
    // Add query.
    switch (queryType) {
      case NodeQueryType.REGEX:
        this.curQueries.push({
          type: queryType,
          queryRegex: '',
          matchTypes: [SearchMatchType.NODE_LABEL],
        });
        break;
      case NodeQueryType.ATTR_VALUE_RANGE:
        this.curQueries.push({
          type: queryType,
          attrName: '',
          min: Number.NEGATIVE_INFINITY,
          max: Number.POSITIVE_INFINITY,
        });
        break;
      case NodeQueryType.NODE_TYPE:
        this.curQueries.push({
          type: queryType,
          nodeType: SearchNodeType.OP_NODES,
        });
        break;
      default:
        break;
    }

    // Notify changes.
    this.queriesUpdated.emit([...this.curQueries]);
  }

  getIsMatchTypeSelected(
    query: NodeRegexQuery,
    matchType: SearchMatchType,
  ): boolean {
    return query.matchTypes.includes(matchType);
  }

  getDisableMatchType(
    query: NodeRegexQuery,
    matchType: SearchMatchType,
  ): boolean {
    return query.matchTypes.length === 1 && query.matchTypes[0] === matchType;
  }

  getAttrValueRangeString(value: number | null): string {
    return value == null ||
      value === Number.NEGATIVE_INFINITY ||
      value === Number.POSITIVE_INFINITY
      ? ''
      : `${value}`;
  }

  getShowDeleteQueryButton(query: NodeQuery): boolean {
    return query.type !== NodeQueryType.NODE_TYPE;
  }
}
