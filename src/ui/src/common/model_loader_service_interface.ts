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

import {WritableSignal} from '@angular/core';

import {GraphCollection} from '../components/visualizer/common/input_graph';
import type { KeyValue, NodeDataProviderData } from '../components/visualizer/common/types';

import {ModelItem} from './types';

export type ChangesPerNode = Record<string, KeyValue[]>;
export type ChangesPerGraphAndNode = Record<string, ChangesPerNode>;

export interface ExecutionCommand {
  stdout: string;
  log_file: string;
  perf_trace?: string;
  perf_data?: NodeDataProviderData
}

/** The interface of model load service. */
export interface ModelLoaderServiceInterface {
  loadModels(modelItems: ModelItem[]): Promise<void>;
  loadModel(modelItems: ModelItem): Promise<GraphCollection[]>;
  executeModel(modelItem: ModelItem): Promise<ExecutionCommand | undefined>;
  overrideModel(modelItem: ModelItem, graphCollection: GraphCollection, fieldsToUpdate: ChangesPerNode): Promise<GraphCollection | undefined>;
  get loadedGraphCollections(): WritableSignal<GraphCollection[] | undefined>;
  get models(): WritableSignal<ModelItem[]>;
  get changesToUpload(): WritableSignal<ChangesPerGraphAndNode>;
  getOptimizationPolicies(extensionId: string): string[];
  get selectedOptimizationPolicy(): WritableSignal<string>;
  get graphErrors(): WritableSignal<string[] | undefined>;
  get hasChangesToUpload(): boolean;
}
