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
} from '@angular/core';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import * as d3 from 'd3';
import * as three from 'three';

import {ModelEdge, ModelGraph, NodeType, OpNode} from './common/model_graph';
import {Point} from './common/types';
import {generateCurvePoints} from './common/utils';
import {EdgeOverlaysService} from './edge_overlays_service';
import {Logo} from './logo';
import {SplitPaneService} from './split_pane_service';
import {SubgraphSelectionService} from './subgraph_selection_service';
import {WebglRenderer} from './webgl_renderer';

const THREE = three;

/**
 * The component that runs the benchmark of the webgl rendering engine.
 */
@Component({
  standalone: true,
  selector: 'benchmark-runner',
  imports: [
    CommonModule,
    Logo,
    MatButtonModule,
    ReactiveFormsModule,
    WebglRenderer,
  ],
  providers: [EdgeOverlaysService, SubgraphSelectionService, SplitPaneService],
  templateUrl: './benchmark_runner.ng.html',
  styleUrls: ['./benchmark_runner.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class BenchmarkRunner {
  readonly curNodeCount = new FormControl<number>(10000);
  readonly curEdgeCount = new FormControl<number>(5000);
  readonly curColorize = new FormControl<boolean>(false);

  generating = false;
  modelGraph?: ModelGraph;

  constructor(private readonly changeDetectorRef: ChangeDetectorRef) {}

  handleClickStart() {
    this.generating = true;
    this.changeDetectorRef.detectChanges();

    setTimeout(() => {
      this.genModelGraph();
    }, 100);
  }

  private genModelGraph() {
    const nodeCount = this.curNodeCount.value as number;
    const edgeCount = this.curEdgeCount.value as number;
    const graphSize = Math.max(500, Math.floor(Math.sqrt(nodeCount) * 300));

    const nodes: OpNode[] = [];
    for (let i = 0; i < nodeCount; i++) {
      const id = `n${i}`;
      const x = Math.random() * graphSize;
      const y = Math.random() * graphSize;
      const width = 150 + (Math.random() - 0.5) * 60;
      const height = 50 + (Math.random() - 0.5) * 30;
      const node: OpNode = {
        nodeType: NodeType.OP_NODE,
        id,
        label: id,
        namespace: '',
        level: 0,
        x,
        y,
        globalX: 0,
        globalY: 0,
        width,
        height,
      };
      if (this.curColorize.value) {
        node.style = {
          backgroundColor: `rgb(${Math.floor(
            Math.random() * 255,
          )}, ${Math.floor(Math.random() * 255)}, ${Math.floor(
            Math.random() * 255,
          )})`,
        };
      }
      nodes.push(node);
    }

    const nodesById: {[key: string]: OpNode} = {};
    for (const node of nodes) {
      nodesById[node.id] = node;
    }

    const edgesByGroupNodeIds: {[id: string]: ModelEdge[]} = {'': []};
    for (let i = 0; i < edgeCount; i++) {
      const fromNodeId = `n${Math.floor(Math.random() * nodeCount)}`;
      const toNodeId = `n${Math.floor(Math.random() * nodeCount)}`;
      const fromNode = nodesById[fromNodeId];
      const toNode = nodesById[toNodeId];
      const id = `${fromNodeId}_${toNodeId}`;
      const x1 = fromNode.x! + fromNode.width! / 2;
      const y1 = fromNode.y! + fromNode.height! / 2;
      const x2 = toNode.x! + toNode.width! / 2;
      const y2 = toNode.y! + toNode.height! / 2;
      const points: Point[] = [
        {
          x: x1,
          y: y1,
        },
        {
          x: (x1 + x2) / 2 + (Math.random() - 0.5) * 100,
          y: (y1 + y2) / 2 + (Math.random() - 0.5) * 100,
        },
        {
          x: x2,
          y: y2,
        },
      ];
      const curvePoints = generateCurvePoints(
        points,
        d3.line,
        d3.curveMonotoneY,
        THREE,
        true,
      );
      const edge: ModelEdge = {
        id,
        fromNodeId,
        toNodeId,
        points,
        curvePoints,
      };
      edgesByGroupNodeIds[''].push(edge);
    }

    // Generate model graph.
    this.modelGraph = {
      id: 'benchmark_graph',
      collectionLabel: 'benchmark_collection',
      nodes,
      nodesById,
      rootNodes: nodes,
      edgesByGroupNodeIds,
      maxDescendantOpNodeCount: 0,
      minDescendantOpNodeCount: 0,
      layoutGraphEdges: {},
    };
    this.changeDetectorRef.detectChanges();
  }
}
