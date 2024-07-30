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
import {Component, OnInit} from '@angular/core';

import {GraphCollection} from '../visualizer/common/input_graph';
import {RendererType} from '../visualizer/common/types';
import {VisualizerConfig} from '../visualizer/common/visualizer_config';
import {ModelGraphVisualizer} from '../visualizer/model_graph_visualizer';

const COCOSSD_TFLITE_GRAPHS_JSON_URL =
  'https://storage.googleapis.com/tfweb/model-graph-vis-v2-test-models/coco-ssd.tflite%20(10).json';
const COCOSSD_TF_GRAPHS_JSON_URL =
  'https://storage.googleapis.com/tfweb/model-graph-vis-v2-test-models/coco-ssd-tf.json';

/**
 * The demo page that uses visualizer to visualize a graphs json file.
 */
@Component({
  selector: 'demo-page',
  standalone: true,
  imports: [CommonModule, ModelGraphVisualizer],
  templateUrl: './demo_page.ng.html',
  styleUrl: './demo_page.scss',
})
export class DemoPage implements OnInit {
  graphCollections: GraphCollection[] = [];

  config: VisualizerConfig = {
    nodeLabelsToHide: ['Const', 'pseudo_const', 'ReadVariableOp'],
    defaultRenderer: RendererType.WEBGL,
    maxConstValueCount: 16,
    enableSubgraphSelection: true,
    enableExportToResource: true,
  };

  ngOnInit() {
    Promise.all([
      this.fetch(COCOSSD_TFLITE_GRAPHS_JSON_URL),
      this.fetch(COCOSSD_TF_GRAPHS_JSON_URL),
    ]).then((collections) => {
      this.graphCollections = collections;
    });
  }

  private fetch(url: string): Promise<GraphCollection> {
    return new Promise<GraphCollection>((resolve) => {
      fetch(url).then((resp) => {
        resp.json().then((json) => {
          if (json['label'] == null && json['graphs'] == null) {
            resolve({label: 'unnamed collection', graphs: json});
          } else {
            resolve(json as GraphCollection);
          }
        });
      });
    });
  }
}
