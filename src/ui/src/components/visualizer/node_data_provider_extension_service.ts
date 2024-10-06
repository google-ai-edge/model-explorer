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

import {computed, Injectable, signal} from '@angular/core';
import {toObservable} from '@angular/core/rxjs-interop';

import {IS_EXTERNAL} from '../../common/flags';

import {AppService} from './app_service';
import {COLOR_NAME_TO_HEX} from './common/consts';
import {ModelGraph} from './common/model_graph';
import {
  NodeDataProviderData,
  NodeDataProviderResultProcessedData,
  NodeDataProviderRunData,
  ReadFileResp,
  ThresholdItem,
} from './common/types';
import {genUid, isOpNode} from './common/utils';

interface Rgb {
  r: number;
  g: number;
  b: number;
}

interface ProcessedGradientItem {
  stop: number;
  bgColor?: Rgb;
  textColor?: Rgb;
}

declare interface ExternalReadFileResp {
  content: string;
  error?: string;
}

const READ_TEXT_FILE_API_PATH = '/api/v1/read_text_file';
const LOAD_NODE_DATA_API_PATH = '/api/v1/load_node_data';

/**
 * A service to manage all node data provider extension executions.
 */
@Injectable()
export class NodeDataProviderExtensionService {
  // Indexed by run id.
  readonly runs = signal<Record<string, NodeDataProviderRunData>>({});

  readonly remoteSourceLoading = signal<boolean>(false);

  private readonly leftPaneModelGraph$ = toObservable(
    computed(() => this.appService.panes()[0].modelGraph),
  );

  private readonly rightPaneModelGraph$ = toObservable(
    computed(() => {
      const panes = this.appService.panes();
      if (panes.length <= 1) {
        return undefined;
      }
      return panes[1].modelGraph;
    }),
  );

  constructor(private readonly appService: AppService) {
    this.leftPaneModelGraph$.subscribe((modelGraph) => {
      if (modelGraph == null) {
        return;
      }
      this.handleModelGraphInPaneChanged(modelGraph, 0);
    });

    this.rightPaneModelGraph$.subscribe((modelGraph) => {
      if (modelGraph == null) {
        return;
      }
      this.handleModelGraphInPaneChanged(modelGraph, 1);
    });
  }

  addRun(
    runId: string,
    runName: string,
    extensionId: string,
    modelGraph: ModelGraph,
    nodeDataProviderData?: NodeDataProviderData,
    clearExisting = false,
    remotePath?: string,
  ) {
    const collectionId = modelGraph.collectionLabel;

    // Update run data for the model graph.
    this.runs.update((runs) => {
      // Clear runs with the same collection id.
      if (clearExisting) {
        const runIdsToRemove: string[] = [];
        for (const [runId, run] of Object.entries(runs)) {
          if (clearExisting && run.collectionId === collectionId) {
            runIdsToRemove.push(runId);
          }
        }
        for (const runId of runIdsToRemove) {
          delete runs[runId];
        }
      }

      // Add this run, and process the given model graph right away.
      runs[runId] = {
        runId,
        runName,
        done: nodeDataProviderData == null ? false : true,
        results:
          nodeDataProviderData == null
            ? undefined
            : {
                [modelGraph.id]: this.processNodeDataProviderDataForGraph(
                  modelGraph,
                  nodeDataProviderData,
                ),
              },
        extensionId,
        collectionId,
        remotePath,
        nodeDataProviderData,
      };

      // Select it if a pane is showing this model graph.
      const panes = this.appService.panes();
      for (const pane of panes) {
        if (
          pane.modelGraph?.id === modelGraph.id &&
          pane.modelGraph?.collectionLabel === modelGraph.collectionLabel
        ) {
          this.appService.setSelectedNodeDataProviderRunId(pane.id, runId);
        }
      }

      return {...runs};
    });
  }

  updateRunResults(
    runId: string,
    data: NodeDataProviderData,
    modelGraph: ModelGraph,
    error?: string,
  ) {
    this.runs.update((runs) => {
      const run = runs[runId];
      run.done = true;
      run.nodeDataProviderData = data;
      if (run.results == null) {
        run.results = {};
      }
      run.results[modelGraph.id] = this.processNodeDataProviderDataForGraph(
        modelGraph,
        data,
      );
      if (error) {
        run.error = error;
      }
      return {...runs};
    });
  }

  async addRunFromRemoteSource(path: string, modelGraph: ModelGraph) {
    this.remoteSourceLoading.set(true);

    const fileNameParts = path.split('/');
    let fileName = fileNameParts[fileNameParts.length - 1];

    // Call API to read file content.
    let url = `/read_file?path=${path}`;
    if (IS_EXTERNAL) {
      if (path.startsWith('node_data://')) {
        const partsStr = path.replace('node_data://', '');
        const parts = partsStr.split('/');
        fileName = parts[0];
        const index = Number(parts[1]);
        url = `${LOAD_NODE_DATA_API_PATH}?node_data_index=${index}`;
      } else {
        url = `${READ_TEXT_FILE_API_PATH}?path=${path}`;
      }
    }
    const runId = genUid();
    this.addRun(runId, fileName, '', modelGraph, undefined, false, path);
    const resp = await fetch(url);
    if (!resp.ok) {
      this.updateRunResults(
        runId,
        {[modelGraph.id]: {results: {}}},
        modelGraph,
        `Failed to load JSON file "${path}"`,
      );
      this.remoteSourceLoading.set(false);
      return;
    }

    // Update run with results.
    if (IS_EXTERNAL) {
      const json = JSON.parse(await resp.text()) as ExternalReadFileResp;
      if (json.error) {
        this.updateRunResults(
          runId,
          {[modelGraph.id]: {results: {}}},
          modelGraph,
          `Failed to process JSON file. ${json.error}`,
        );
      } else {
        try {
          this.updateRunResults(
            runId,
            this.getNodeDataProviderData(json.content, modelGraph),
            modelGraph,
          );
          this.notifyRemoteNodeDataChanges();
        } catch (e) {
          this.updateRunResults(
            runId,
            {[modelGraph.id]: {results: {}}},
            modelGraph,
            `Failed to process JSON file. ${e}`,
          );
        }
      }
    } else {
      const json = JSON.parse(
        (await resp.text()).replace(")]}'\n", ''),
      ) as ReadFileResp;
      try {
        this.updateRunResults(
          runId,
          this.getNodeDataProviderData(json.content, modelGraph),
          modelGraph,
        );
        this.notifyRemoteNodeDataChanges();
      } catch (e) {
        this.updateRunResults(
          runId,
          {[modelGraph.id]: {results: {}}},
          modelGraph,
          `Failed to process JSON file. ${e}`,
        );
      }
    }

    this.remoteSourceLoading.set(false);
  }

  deleteRun(runId: string) {
    this.runs.update((runs) => {
      delete runs[runId];
      return {...runs};
    });
    this.notifyRemoteNodeDataChanges();

    // Update selection.
    for (const pane of this.appService.panes()) {
      if (pane.selectedNodeDataProviderRunId === runId) {
        const runs = this.getRunsForModelGraph(pane.modelGraph!);
        this.appService.setSelectedNodeDataProviderRunId(
          pane.id,
          runs.length > 0 ? runs[0].runId : undefined,
        );
      }
    }
  }

  getSelectedRunForModelGraph(
    paneId: string,
    modelGraph: ModelGraph,
  ): NodeDataProviderRunData | undefined {
    const selectedRunId =
      this.appService.getSelectedNodeDataProviderRunId(paneId);
    if (!selectedRunId) {
      return undefined;
    }
    const runs = this.getRunsForModelGraph(modelGraph);
    return runs.find((run) => run.runId === selectedRunId);
  }

  getRunsForModelGraph(modelGraph: ModelGraph): NodeDataProviderRunData[] {
    const ret: NodeDataProviderRunData[] = [];
    const runs = this.runs();
    for (const run of Object.values(runs)) {
      if (run.collectionId !== modelGraph.collectionLabel) {
        continue;
      }
      const nodeDataProviderData = run.nodeDataProviderData;
      if (!nodeDataProviderData) {
        continue;
      }
      if (nodeDataProviderData[modelGraph.id] != null) {
        ret.push(run);
      }
    }
    return ret;
  }

  private processNodeDataProviderDataForGraph(
    modelGraph: ModelGraph,
    nodeDataProviderData: NodeDataProviderData,
  ): Record<string, NodeDataProviderResultProcessedData> {
    this.genOutputTensorIdToNodeIdMap(modelGraph);

    const results: Record<string, NodeDataProviderResultProcessedData> = {};
    const graphData = nodeDataProviderData[modelGraph.id];
    if (!graphData) {
      return {};
    }

    // Preprocess gradient.
    const processedGradientItems: ProcessedGradientItem[] = [];
    for (const gradientItem of graphData.gradient || []) {
      const processedGradientItem: ProcessedGradientItem = {
        stop: gradientItem.stop,
      };
      if (gradientItem.bgColor != null) {
        processedGradientItem.bgColor = this.getRgbFromColor(
          gradientItem.bgColor,
          '#ffffff',
        );
      }
      if (gradientItem.textColor != null) {
        processedGradientItem.textColor = this.getRgbFromColor(
          gradientItem.textColor,
          '#000000',
        );
      }
      processedGradientItems.push(processedGradientItem);
    }
    processedGradientItems.sort((a, b) => a.stop - b.stop);
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    if (processedGradientItems.length > 0) {
      for (const {value} of Object.values(graphData.results)) {
        if (typeof value === 'number') {
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
    }

    for (const id of Object.keys(graphData.results)) {
      const respResult = graphData.results[id];
      // Try to calculate bgColor from the given thresholds and gradient.
      if (typeof respResult.value === 'number' && respResult.bgColor == null) {
        respResult.bgColor = this.getBgColor(
          respResult.value,
          graphData.thresholds || [],
          processedGradientItems,
          min,
          max,
        );
      }

      // Try to calculate textColor from the given thresholds and gradient.
      if (
        typeof respResult.value === 'number' &&
        respResult.textColor == null
      ) {
        respResult.textColor = this.getTextColor(
          respResult.value,
          graphData.thresholds || [],
          processedGradientItems,
          min,
          max,
        );
      }

      // Check if we need to set textColor to white for darker bgColor.
      //
      // See: https://gist.github.com/Myndex/e1025706436736166561d339fd667493
      if (
        (respResult.textColor == null || respResult.textColor === '') &&
        respResult.bgColor != null
      ) {
        const rgb = this.getRgbFromColor(respResult.bgColor!, '#ffffff');
        if (rgb != null) {
          const luminance =
            Math.pow(rgb.r / 255.0, 2.2) * 0.2126 +
            Math.pow(rgb.g / 255.0, 2.2) * 0.7152 +
            Math.pow(rgb.b / 255.0, 2.2) * 0.0722;
          if (luminance < 0.38) {
            respResult.textColor = '#ffffff';
          }
        }
      }

      // Process value.
      let strValue = '-';
      const resultValue = respResult.value;
      if (
        typeof resultValue === 'number' ||
        typeof resultValue === 'boolean' ||
        typeof resultValue === 'string'
      ) {
        strValue = `${resultValue}`;
      } else {
        strValue = JSON.stringify(resultValue);
      }
      strValue = strValue;

      // Find the node id for the result id in outputTensorIdToNodeId map.
      const nodeId = (modelGraph.outputTensorIdToNodeId || {})[id];
      const indexId = `${nodeId ?? id}`;
      const existingResult = results[indexId];
      if (!existingResult) {
        results[indexId] = {
          ...respResult,
          strValue,
          allValues: {[id]: respResult.value},
        };
      } else {
        // Accumulate all results that map to this node in the `allValues` field
        // indexed by their original id.
        const allValues = existingResult.allValues;
        allValues[id] = respResult.value;
        results[indexId] = {
          value: respResult.value,
          bgColor: respResult.bgColor,
          textColor: respResult.textColor,

          allValues,
          strValue: `${existingResult.strValue}, ${strValue}`,
        };
      }
    }

    return results;
  }

  private getBgColor(
    value: number,
    thresholds: ThresholdItem[],
    gradient: ProcessedGradientItem[],
    min: number,
    max: number,
  ): string {
    if (gradient.length > 0) {
      return this.getColorFromGradient(
        value,
        gradient,
        min,
        max,
        true,
        'transparent',
      );
    } else {
      for (const thrshold of thresholds) {
        if (value <= thrshold.value) {
          return thrshold.bgColor;
        }
      }
    }
    return 'transparent';
  }

  private getTextColor(
    value: number,
    thresholds: ThresholdItem[],
    gradient: ProcessedGradientItem[],
    min: number,
    max: number,
  ): string {
    if (gradient.length > 0) {
      return this.getColorFromGradient(value, gradient, min, max, false, '');
    } else {
      for (const thrshold of thresholds) {
        if (value <= thrshold.value) {
          return thrshold.textColor || '';
        }
      }
    }
    return '';
  }

  private getColorFromGradient(
    value: number,
    gradient: ProcessedGradientItem[],
    min: number,
    max: number,
    isBgColor: boolean,
    defaultColor: string,
  ): string {
    const targetStop = (value - min) / (max - min);
    for (let i = 0; i < gradient.length - 1; i++) {
      const curItem = gradient[i];
      const nextItem = gradient[i + 1];
      const curColor = isBgColor ? curItem.bgColor : curItem.textColor;
      const nextColor = isBgColor ? nextItem.bgColor : nextItem.textColor;
      if (targetStop >= curItem.stop && targetStop <= nextItem.stop) {
        if (curColor == null || nextColor == null) {
          return defaultColor;
        }
        const ratio =
          (targetStop - curItem.stop) / (nextItem.stop - curItem.stop);
        const r = Math.floor(curColor.r + (nextColor.r - curColor.r) * ratio);
        const g = Math.floor(curColor.g + (nextColor.g - curColor.g) * ratio);
        const b = Math.floor(curColor.b + (nextColor.b - curColor.b) * ratio);
        return `#${this.numToHex(r)}${this.numToHex(g)}${this.numToHex(b)}`;
      }
    }
    return defaultColor;
  }

  private genOutputTensorIdToNodeIdMap(modelGraph: ModelGraph) {
    if (modelGraph.outputTensorIdToNodeId != null) {
      return;
    }

    modelGraph.outputTensorIdToNodeId = {};
    for (const node of modelGraph.nodes) {
      if (isOpNode(node)) {
        const outputMetadata = node.outputsMetadata || {};
        for (const outputId of Object.keys(outputMetadata)) {
          const curMetadata = outputMetadata[outputId];
          const tensorName = curMetadata['tensor_name'];
          if (tensorName != null) {
            modelGraph.outputTensorIdToNodeId[tensorName] = node.id;
          }
        }
      }
    }
  }

  private handleModelGraphInPaneChanged(
    modelGraph: ModelGraph,
    paneIndex: number,
  ) {
    // Process node data provider results for the model graph.
    const runsForModelGraph = this.getRunsForModelGraph(modelGraph);
    if (runsForModelGraph.length > 0) {
      for (const run of runsForModelGraph) {
        if (run.results == null) {
          run.results = {};
        }
        if (
          run.results[modelGraph.id] == null &&
          run.nodeDataProviderData != null
        ) {
          run.results[modelGraph.id] = this.processNodeDataProviderDataForGraph(
            modelGraph,
            run.nodeDataProviderData,
          );
        }
      }

      // Set the first run as the selected run.
      this.appService.setSelectedNodeDataProviderRunId(
        this.appService.panes()[paneIndex].id,
        runsForModelGraph[0].runId,
      );
    }
  }

  private getRgbFromColor(
    color: string,
    defaultColor: string,
  ): Rgb | undefined {
    let hexColor = color;
    if (!color.startsWith('#')) {
      hexColor = COLOR_NAME_TO_HEX[color];
    }
    if (!hexColor) {
      hexColor = defaultColor;
    }
    hexColor = hexColor.replace('#', '');
    return {
      r: this.hexStrToInt(hexColor.substring(0, 2)),
      g: this.hexStrToInt(hexColor.substring(2, 4)),
      b: this.hexStrToInt(hexColor.substring(4, 6)),
    };
  }

  private numToHex(x: number): string {
    const hex = x.toString(16);
    return hex.length === 1 ? `0${hex}` : hex;
  }

  private hexStrToInt(hex: string): number {
    if (/^[a-fA-F0-9]+$/.test(hex)) {
      // Needed to parse hexadecimal.
      // tslint:disable-next-line:ban
      return parseInt(hex, 16);
    }
    return 255;
  }

  private getNodeDataProviderData(str: string, modelGraph: ModelGraph) {
    // tslint:disable-next-line:no-any Allow arbitrary types.
    const jsonObj = JSON.parse(str) as any;
    let data: NodeDataProviderData = {};

    // The json file doesn't have graph id as key. Use the current graph
    // as the only key.
    if (jsonObj['results'] != null && jsonObj['results']['results'] == null) {
      if (modelGraph) {
        data[modelGraph.id] = jsonObj;
      }
    }
    // File with graph id as keys.
    else {
      data = jsonObj;
    }

    return data;
  }

  private notifyRemoteNodeDataChanges() {
    const remoteNodeDataPaths = Object.values(this.runs())
      .filter((run) => run.remotePath != null)
      .map((run) => run.remotePath!);
    this.appService.remoteNodeDataPaths.set(remoteNodeDataPaths);
  }
}
