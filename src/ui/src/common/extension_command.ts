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

import {Graph, GraphCollection,} from '../components/visualizer/common/input_graph';
import type { NodeDataProviderData } from '../components/visualizer/common/types';
import type { ChangesPerNode } from './model_loader_service_interface';

/** A command sent to extension. */
export declare interface ExtensionCommand {
  cmdId: string;
  extensionId: string;
  modelPath: string;
  settings: Record<string, any>;
  deleteAfterConversion: boolean;
}

interface ExtensionGraphResponse<G extends Array<unknown>> {
  graphs: G;
  graphCollections?: never;
  error?: never;
}

interface ExtensionCollectionResponse<C extends Array<unknown>> {
  graphs?: never;
  graphCollections: C;
  error?: never;
}

interface ExtensionErrorResponse<E extends unknown = string> {
  graphs?: never;
  graphCollections?: never;
  error: E;
}

/** A response received from the extension. */
export type ExtensionResponse<G extends Array<unknown> = Graph[], C extends Array<unknown> = GraphCollection[], E extends unknown = string> = ExtensionGraphResponse<G> | ExtensionCollectionResponse<C> | ExtensionErrorResponse<E>;

/** Adapter's "convert" command. */
export declare interface AdapterConvertCommand extends ExtensionCommand {
  cmdId: 'convert';
  modelPath: string;
  // tslint:disable-next-line:no-any Allow arbitrary types.
  settings: Record<string, any>;
  // Whether to delete the model file at `modelPath` after conversion is done.
  deleteAfterConversion: boolean;
  perf_trace?: string;
  perf_data?: NodeDataProviderData;
}

/** Adapter's "convert" command response. */
export type AdapterConvertResponse = ExtensionResponse;

/** Adapter's "override" command. */
export declare interface AdapterOverrideCommand extends ExtensionCommand {
  cmdId: 'override';
  settings: {
    graphs: Graph[];
    changes: ChangesPerNode;
  };
}

/** Adapter's "override" command response. */
export type AdapterOverrideResponse = ExtensionResponse<[{
  success: boolean;
}], never>;

/** Adapter's "execute" command. */
export declare interface AdapterExecuteCommand extends ExtensionCommand {
  cmdId: 'execute';
}

/** Adapter's "execute" results inside the response. */
export interface AdapterExecuteResults {}

/** Adapter's "execute" command response. */
export type AdapterExecuteResponse = ExtensionResponse<[], never>;

/** Adapter's "status check" command. */
export declare interface AdapterStatusCheckCommand extends ExtensionCommand {
  cmdId: 'status_check';
}

/** Adapter's "status check" results inside the response. */
export interface AdapterStatusCheckResults {
  isDone: boolean;
  progress: number;
  total?: number;
  timeElapsed?: number;
  currentStatus?: string;
  error?: string;
  stdout?: string;
  log_file?: string;
}

/** Adapter's "status check" command response. */
export type AdapterStatusCheckResponse = ExtensionResponse<[AdapterStatusCheckResults], never>;
